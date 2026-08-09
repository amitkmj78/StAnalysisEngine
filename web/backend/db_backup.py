import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from starlette.concurrency import run_in_threadpool

from services.backup_service import (
    BACKUP_TABLES,
    create_backup_dump,
    list_dump_contents,
    remove_dump,
    run_real_restore_test,
)
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ.get("BACKUP_S3_BUCKET", "stanalysisengine-db-backups")


def _s3_client():
    # Uses the credentials already stored in .env (AWS_ACCESS_KEY_ID /
    # AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION) — boto3 picks these up
    # from the environment automatically, no explicit key passing needed.
    # Note: this key has broader permissions than this feature strictly
    # needs (it's the same key used for the EC2 deploy tooling) — a
    # properly scoped IAM role/user couldn't be created here because the
    # underlying AWS account's credentials don't have iam:CreateRole.
    return boto3.client("s3")


def _tmp_dump_path(prefix: str) -> Path:
    # Deliberately a bare /tmp path, not a private tempfile.TemporaryDirectory()
    # — pg_dump/pg_restore run as the postgres OS user (via sudo), a
    # different user than this app process, and would be denied write/
    # traverse access to a 0700 directory owned by the app's own user.
    # /tmp itself is world-writable (sticky bit), so postgres can create
    # or read files placed directly there regardless of which OS user
    # this process runs as.
    return Path(f"/tmp/{prefix}-{uuid.uuid4().hex[:8]}.dump")


async def run_backup() -> dict:
    """
    NFR-03: dumps the published-record + PIT-store tables, uploads to S3,
    and runs the free structural-integrity check (list_dump_contents —
    parses the dump's own table of contents, no DB connection needed)
    on every single run. Persists a backup_runs row either way, so a
    failed or incomplete backup is visible, not silently missing.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    s3_key = f"backups/{timestamp}.dump"
    dump_path = _tmp_dump_path("backup")

    try:
        try:
            await run_in_threadpool(create_backup_dump, dump_path)
        except Exception as e:
            logger.warning("NFR-03 backup: pg_dump failed: %s", e)
            return await _persist_run(s3_key=None, size_bytes=None, tables_verified=[], error=str(e))

        size_bytes = dump_path.stat().st_size
        tables_verified: list[str] = []
        try:
            tables_verified = await run_in_threadpool(list_dump_contents, dump_path)
        except Exception as e:
            logger.warning("NFR-03 backup: structural verification failed: %s", e)

        try:
            s3 = _s3_client()
            await run_in_threadpool(s3.upload_file, str(dump_path), S3_BUCKET, s3_key)
        except Exception as e:
            logger.warning("NFR-03 backup: S3 upload failed: %s", e)
            return await _persist_run(
                s3_key=None, size_bytes=size_bytes, tables_verified=tables_verified,
                error=f"S3 upload failed: {e}",
            )
    finally:
        await run_in_threadpool(remove_dump, dump_path)

    return await _persist_run(s3_key=s3_key, size_bytes=size_bytes, tables_verified=tables_verified, error=None)


async def run_restore_test(s3_key: str | None = None) -> dict:
    """
    NFR-03's actual "restore-tested" requirement: downloads a backup
    (the given key, or the most recent one if omitted), restores it into
    a genuine throwaway database, counts rows per table, and compares
    against the live tables — a much stronger guarantee than the
    structural check alone, since it proves the data actually comes back,
    not just that the file parses.
    """
    if s3_key is None:
        async with service_conn() as conn:
            row = await conn.fetchrow(
                "SELECT s3_key FROM backup_runs WHERE s3_key IS NOT NULL ORDER BY started_at_utc DESC LIMIT 1"
            )
        if row is None:
            return {"restore_succeeded": False, "error": "No backup available to test.", "row_counts": {}, "all_match": False}
        s3_key = row["s3_key"]

    dump_path = _tmp_dump_path("restore")
    try:
        try:
            s3 = _s3_client()
            await run_in_threadpool(s3.download_file, S3_BUCKET, s3_key, str(dump_path))
            # Downloaded by this app's own OS user — no sudo needed to
            # loosen permissions, just make sure postgres (a different
            # OS user, invoked via sudo in run_real_restore_test) can
            # read it back.
            await run_in_threadpool(os.chmod, dump_path, 0o644)
        except Exception as e:
            logger.warning("NFR-03 restore test: S3 download failed: %s", e)
            return {"restore_succeeded": False, "error": f"S3 download failed: {e}", "row_counts": {}, "all_match": False}

        result = await run_in_threadpool(run_real_restore_test, dump_path)
    finally:
        try:
            dump_path.unlink(missing_ok=True)
        except OSError:
            pass

    async with service_conn() as conn:
        await conn.execute(
            """
            UPDATE backup_runs SET restore_test_run = true, restore_test_passed = $2, restore_test_row_counts = $3
            WHERE s3_key = $1
            """,
            s3_key, result["all_match"], _row_counts_json(result.get("row_counts", {})),
        )

    return {**result, "s3_key": s3_key}


async def get_backup_status() -> dict:
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM backup_runs ORDER BY started_at_utc DESC LIMIT 10")
    runs = []
    for r in rows:
        run = dict(r)
        # asyncpg returns jsonb as a raw string by default — parse it so
        # the frontend gets real JSON, not a double-encoded string.
        if isinstance(run.get("restore_test_row_counts"), str):
            run["restore_test_row_counts"] = json.loads(run["restore_test_row_counts"])
        runs.append(run)
    return {"recent_runs": runs, "backup_tables": BACKUP_TABLES}


async def _persist_run(
    s3_key: str | None, size_bytes: int | None, tables_verified: list[str], error: str | None
) -> dict:
    structural_check_passed = bool(tables_verified) and set(tables_verified) == set(BACKUP_TABLES)
    async with service_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO backup_runs (s3_key, size_bytes, tables_verified, structural_check_passed, error)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            s3_key, size_bytes, tables_verified, structural_check_passed, error,
        )
    return dict(row)


def _row_counts_json(row_counts: dict) -> str:
    return json.dumps(row_counts)
