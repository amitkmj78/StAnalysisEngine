import json
import os
import threading
import time
import uuid
from datetime import datetime

# Jobs persisted to disk so they survive backend restarts (uvicorn --reload,
# a crash mid-deploy) and the frontend can resume polling a job it lost track of.
_STATE_DIR = os.path.join(os.path.expanduser("~"), ".stanalysisengine")
_JOBS_FILE = os.path.join(_STATE_DIR, "deploy_jobs.json")

_LOG_SAVE_EVERY = 10  # persist to disk every N log lines, not on every single line


def _load_jobs() -> dict:
    try:
        with open(_JOBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_jobs(jobs: dict) -> None:
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        with open(_JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(jobs, f, default=str)
    except Exception:
        pass


JOBS: dict[str, dict] = _load_jobs()


def new_job(action: str) -> tuple[str, dict]:
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "action": action,
        "status": "running",
        "logs": [f"[{datetime.utcnow().strftime('%H:%M:%S')}] {action}"],
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
    }
    JOBS[job_id] = job
    _save_jobs(JOBS)
    return job_id, job


def log(job: dict, line: str) -> None:
    job["logs"].append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {line}")
    if len(job["logs"]) % _LOG_SAVE_EVERY == 0:
        _save_jobs(JOBS)


def finish(job: dict, ok: bool) -> None:
    job["status"] = "success" if ok else "error"
    job["finished_at"] = datetime.utcnow().isoformat()
    _save_jobs(JOBS)


def get_job(job_id: str) -> dict | None:
    return JOBS.get(job_id)


def cancel_job(job_id: str) -> bool:
    job = JOBS.get(job_id)
    if job is None or job["status"] != "running":
        return False
    # Best-effort — the worker thread itself isn't forcibly killed (SSH/boto3
    # calls aren't cleanly interruptible), this just stops the UI from waiting
    # on it and frees the "one job at a time" lock the frontend enforces.
    job["logs"].append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] ⚠ Cancelled by user")
    finish(job, False)
    return True


def _recover_interrupted_jobs() -> None:
    changed = False
    for job in JOBS.values():
        if job.get("status") == "running":
            job["status"] = "error"
            job["finished_at"] = datetime.utcnow().isoformat()
            job["logs"].append(
                f"[{datetime.utcnow().strftime('%H:%M:%S')}] ⚠ Backend restarted — job interrupted"
            )
            changed = True
    if changed:
        _save_jobs(JOBS)


_recover_interrupted_jobs()


def _watchdog() -> None:
    """Mark jobs running longer than 90 minutes as failed — well above the
    longest legitimate step here (npm build + apt installs), so this only
    fires if a worker thread genuinely hung without reaching its except block."""
    while True:
        time.sleep(60)
        now = datetime.utcnow()
        for job in list(JOBS.values()):
            if job.get("status") != "running":
                continue
            try:
                started = datetime.fromisoformat(job.get("started_at", ""))
                if (now - started).total_seconds() > 5400:
                    job["logs"].append(
                        f"[{now.strftime('%H:%M:%S')}] ⚠ Watchdog: job exceeded 90-minute limit — marked failed"
                    )
                    finish(job, False)
            except Exception:
                pass


threading.Thread(target=_watchdog, daemon=True, name="deploy-watchdog").start()
