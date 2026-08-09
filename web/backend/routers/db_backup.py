from fastapi import APIRouter, Depends, Query

from web.backend.admin import require_admin
from web.backend.db_backup import get_backup_status, run_backup, run_restore_test

router = APIRouter(
    prefix="/api/v1/db-backup",
    tags=["db-backup"],
    dependencies=[Depends(require_admin)],
)


@router.get("/status")
async def backup_status():
    """NFR-03: the last 10 backup attempts, each showing whether the free
    structural check passed and — for whichever runs have had a real
    restore test performed — whether that passed too."""
    return await get_backup_status()


@router.post("/backup-now")
async def backup_now():
    """Manual trigger for the same backup the scheduler runs nightly —
    for verifying the pipeline, not routine use. No enable-gate: taking a
    backup is never itself risky or irreversible."""
    return await run_backup()


@router.post("/restore-test-now")
async def restore_test_now(s3_key: str | None = Query(None, description="Specific backup to test; defaults to the most recent.")):
    """NFR-03's actual "restore-tested" step, on demand: restores a real
    backup into a throwaway database and compares row counts against the
    live tables — the same test the quarterly schedule runs automatically,
    available here to verify it works without waiting for the calendar."""
    return await run_restore_test(s3_key=s3_key)
