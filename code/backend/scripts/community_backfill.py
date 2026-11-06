"""Run the community stage on the projects that are done, without the worker (which would also mine the queued ones).

Run in the API container with the current code mounted:
docker compose run --rm --no-deps -v ./backend/app:/app/backend/app -v ./backend/scripts:/app/backend/scripts \
    api python scripts/community_backfill.py
The stage row is marked done, so a worker started later does not recompute it.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from sqlalchemy import select

from app.db import engine, init_db, session_scope
from app.models import Project, StageRun, utcnow
from app.pipeline import ProjectRef, StageContext, ensure_stage_runs
from app.stages.community import CommunityStage

PROCESSES = int(os.getenv("PROCESSES", "48"))


def reset_pool():
    engine.dispose(close=False)  # connections are not shared with the parent process


def one(pid: int) -> tuple[int, float, str]:
    start = time.time()
    with session_scope() as s:
        ensure_stage_runs(s, pid)
        s.flush()
        run = s.scalar(select(StageRun).where(StageRun.project_id == pid, StageRun.stage == "community"))
        if run.status == "done":
            return pid, 0.0, "skipped"
        p = s.get(Project, pid)
        run.status, run.started_at = "running", utcnow()
        CommunityStage().run(StageContext(s, ProjectRef(p.id, p.owner, p.name, p.since), run))
        run.status, run.finished_at = "done", utcnow()
    return pid, time.time() - start, "done"


def main():
    init_db()
    with session_scope() as s:
        pids = list(s.scalars(select(Project.id).where(Project.status == "done").order_by(Project.id)))
    print(f"{len(pids)} done projects, {PROCESSES} processes", file=sys.stderr, flush=True)
    t0, n, failed = time.time(), 0, []
    with ProcessPoolExecutor(PROCESSES, initializer=reset_pool) as pool:
        futures = {pool.submit(one, pid): pid for pid in pids}
        for f in as_completed(futures):
            n += 1
            try:
                f.result()
            except Exception as e:  # reported, the others go on
                failed.append(futures[f])
                print(f"project {futures[f]} failed: {e!r}", file=sys.stderr, flush=True)
            if n % 100 == 0 or n == len(pids):
                el = time.time() - t0
                print(f"{n}/{len(pids)} in {el / 60:.1f} min, ~{el / n * (len(pids) - n) / 60:.0f} min left",
                      file=sys.stderr, flush=True)  # fmt: skip
    print(f"done; failed: {failed}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
