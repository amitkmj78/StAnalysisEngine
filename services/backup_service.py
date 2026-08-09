"""
NFR-03: backups of the published record and PIT store, restore-tested.
Every operation here shells out to `sudo -u postgres ...` — the app
process runs as an unprivileged OS user with passwordless sudo scoped to
postgres operations, the same access pattern already used for schema
migrations throughout this project's SSH-driven ops workflow, just
invoked from the app itself instead of a human at a terminal. This gives
genuine CREATE/DROP DATABASE rights for restore-testing without granting
the app's normal DB roles (app_user/app_service) any elevated privilege.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# NFR-03's literal scope: "the published record and PIT store" — not a
# full database dump. These are exactly the tables that matter for TR-1
# through TR-6's honesty guarantees; user data (portfolios, trades, etc.)
# is a separate concern with its own backup story, not folded in here.
BACKUP_TABLES = [
    "published_signals",
    "signal_outcomes",
    "pit_prices",
    "pit_universe_membership",
    "pit_fundamentals",
    "backtest_runs",
]

DB_NAME = "stanalysisengine"
SCRATCH_DB_NAME = "nfr03_restore_verify"


def create_backup_dump(out_path: Path) -> None:
    """pg_dump the NFR-03 table set, custom format (compressed, restorable
    via pg_restore), scoped to exactly BACKUP_TABLES."""
    cmd = ["sudo", "-u", "postgres", "pg_dump", "-d", DB_NAME, "--format=custom", "--no-owner", "--no-privileges"]
    for table in BACKUP_TABLES:
        cmd += ["-t", table]
    cmd += ["-f", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300, text=True)


def list_dump_contents(dump_path: Path) -> list[str]:
    """
    Structural integrity check that needs no database connection at all —
    just parses the dump file's own table of contents. Every automated
    backup run gets this for free; it catches the most common backup
    failure mode (a truncated/corrupt/empty dump file) without the
    overhead of a full restore.
    """
    result = subprocess.run(
        ["sudo", "-u", "postgres", "pg_restore", "--list", str(dump_path)],
        check=True, capture_output=True, timeout=60, text=True,
    )
    tables = []
    for line in result.stdout.splitlines():
        for table in BACKUP_TABLES:
            if f"TABLE DATA public {table} " in line or line.strip().endswith(f"TABLE DATA public {table}"):
                tables.append(table)
    return tables


def run_real_restore_test(dump_path: Path) -> dict:
    """
    NFR-03's actual "restore-tested" requirement: create a genuine throwaway
    database, restore the dump into it for real, count rows per table, and
    compare against the live source — then tear the scratch database down.
    This is a stronger guarantee than list_dump_contents (which only proves
    the file parses, not that it actually restores and the data is intact).
    """
    subprocess.run(
        ["sudo", "-u", "postgres", "dropdb", "--if-exists", SCRATCH_DB_NAME],
        check=True, capture_output=True, timeout=30, text=True,
    )
    subprocess.run(
        ["sudo", "-u", "postgres", "createdb", SCRATCH_DB_NAME],
        check=True, capture_output=True, timeout=30, text=True,
    )
    try:
        subprocess.run(
            ["sudo", "-u", "postgres", "pg_restore", "-d", SCRATCH_DB_NAME, str(dump_path)],
            check=True, capture_output=True, timeout=300, text=True,
        )

        row_counts: dict[str, dict] = {}
        for table in BACKUP_TABLES:
            restored = _count_rows(SCRATCH_DB_NAME, table)
            live = _count_rows(DB_NAME, table)
            row_counts[table] = {"restored": restored, "live": live, "match": restored == live}

        all_match = all(v["match"] for v in row_counts.values())
        return {"restore_succeeded": True, "row_counts": row_counts, "all_match": all_match}
    except subprocess.CalledProcessError as e:
        logger.warning("NFR-03 restore test failed: %s", e.stderr)
        return {"restore_succeeded": False, "error": e.stderr, "row_counts": {}, "all_match": False}
    finally:
        subprocess.run(
            ["sudo", "-u", "postgres", "dropdb", "--if-exists", SCRATCH_DB_NAME],
            check=False, capture_output=True, timeout=30, text=True,
        )


def _count_rows(db_name: str, table: str) -> int:
    result = subprocess.run(
        ["sudo", "-u", "postgres", "psql", "-d", db_name, "-t", "-c", f"SELECT count(*) FROM {table}"],
        check=True, capture_output=True, timeout=30, text=True,
    )
    return int(result.stdout.strip())
