"""Split the contributions elsewhere of developers who moved to other repositories by destination (same owner, own
repositories, other owners), for the developers already checked by post_exit before the split was stored.

Only sustained developers (>= MIN_MONTHS active months) of done projects with public contributions elsewhere are
queried again, over the same 12 months as their first check. Run in the API container with the current code:
docker compose run --rm --no-deps -e PYTHONPATH=/app/backend -v ./backend/app:/app/backend/app \
    -v ./backend/scripts:/app/backend/scripts api python scripts/after_exit_destinations.py
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

from sqlalchemy import func, select

from app.db import init_db, session_scope
from app.github import GitHubClient
from app.models import Developer, DeveloperMonth, Project, utcnow
from app.stages.context import AFTER_EXIT_MONTHS, PostExitStage, add_months, destinations

MIN_MONTHS = int(os.getenv("MIN_MONTHS", "6"))
THREADS = int(os.getenv("THREADS", "8"))


def check(item):
    did, login, repo, start = item
    ctx = SimpleNamespace(github=GitHubClient(), log=lambda msg, level="info": print(msg, file=sys.stderr, flush=True))
    stop = min(add_months(start, AFTER_EXIT_MONTHS), utcnow()) - timedelta(seconds=1)
    user = PostExitStage.fetch(ctx, login, start, stop)
    split = None if user is None else destinations(user["contributionsCollection"], repo, login)
    with session_scope() as s:
        dev = s.get(Developer, did)
        dev.after_exit_same_owner, dev.after_exit_own, dev.after_exit_other_owner = split or (None, None, None)
    return split is not None


def main():
    init_db()
    with session_scope() as s:
        months = (
            select(DeveloperMonth.developer_id, func.count().label("n"))
            .group_by(DeveloperMonth.developer_id).subquery()
        )  # fmt: skip
        todo = s.execute(
            select(Developer.id, Developer.login, Project.owner + "/" + Project.name, Developer.after_exit_from)
            .join(Project, Project.id == Developer.project_id)
            .join(months, months.c.developer_id == Developer.id)
            .where(Project.status == "done", ~Developer.is_bot, Developer.login.is_not(None),
                   Developer.after_exit_contributions > 0, Developer.after_exit_same_owner.is_(None),
                   months.c.n >= MIN_MONTHS)
        ).tuples().all()  # fmt: skip
    print(f"{len(todo)} developers to split", file=sys.stderr, flush=True)
    t0, ok = time.time(), 0
    with ThreadPoolExecutor(THREADS) as pool:
        for n, done in enumerate(pool.map(check, todo), 1):
            ok += done
            if n % 100 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)} ({ok} split) in {el / 60:.1f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left",
                      file=sys.stderr, flush=True)  # fmt: skip
    print("finished", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
