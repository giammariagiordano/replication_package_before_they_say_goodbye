"""Results of the paper (RQ1-RQ3) on the projects whose pipeline is done, as JSON on stdout.

Run in the API container: docker compose exec -T api python - < backend/scripts/paper_results.py > results.json
Archived repositories are excluded and only developers with a sustained involvement are analyzed (MIN_MONTHS);
-e PAPER_CUTOFF="<UTC time>" restricts the analysis to the projects done by then, to reproduce the results while
mining continues. -e PAPER_SENSITIVITY=/data/anger_probabilities.csv runs the sensitivity analyses of the Threats to
Validity instead (abandonment and involvement thresholds, explicit anger).
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from app import dataset as dataset_mod
from app import survival, trajectories
from app.db import rows_in, session_scope
from app.models import Alias, Comment, Commit, Developer, DeveloperMonth, Issue, Project

BASE = ["core", "centrality", "off_hours_share", "commits", "comments", "reviews", "cum_active_months",
        "truck_factor_member_tv", "project_active_devs", "project_trend"]  # fmt: skip
PSYCH = ["anger_share", "joy_share", "love_share", "sadness_share", "fear_share", "surprise_share",
         "negative_emitted_share", "anger_received", "we_share_12m", "we_unknown", "thanks_received",
         "reactions_pos_received"]  # fmt: skip
# Community bond beyond pronouns (app.community): an extended model adds them to the full one, on the same intervals.
COMMUNITY = ["citizenship_share", "newcomer_share", "accommodation_12m", "accommodation_unknown"]
# Collinearity of the community measures with the structural role, the activity and belonging.
COLLINEARITY = ["core", "centrality", "comments", "reviews", "we_share_12m", *COMMUNITY]
# Projects done by this time (UTC), so that the results can be reproduced while mining continues; unset = all done.
CUTOFF = os.getenv("PAPER_CUTOFF")
# Developer selection (paper, Dataset): sustained involvement. Developers enter the risk set after this active month,
# and RQ1 compares leavers and stayers with at least this many active months.
MIN_MONTHS = int(os.getenv("PAPER_MIN_MONTHS", "6"))
# Path of the anger probabilities (anger_probabilities.py): runs the sensitivity analyses instead of the main ones.
SENSITIVITY = os.getenv("PAPER_SENSITIVITY")
# Set (after scripts/after_exit_destinations.py): where the developers who moved elsewhere went, instead of the rest.
DESTINATIONS = os.getenv("PAPER_DESTINATIONS")


def jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def quartiles(s: pd.Series) -> dict:
    q = s.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    return dict(zip(["min", "q1", "median", "q3", "max"], (float(v) for v in q)))


def describe(db, pids: list[int]) -> dict:
    """Per-project descriptive statistics (Table: dataset)."""
    counts = {}
    for name, model in (("commits", Commit), ("issues", Issue), ("comments", Comment)):
        counts[name] = pd.Series(dict(db.execute(
            select(model.project_id, func.count()).where(model.project_id.in_(pids)).group_by(model.project_id)
        ).all()))  # fmt: skip
    p = pd.DataFrame(counts).reindex(pids).rename_axis("id").reset_index()
    since = db.execute(select(func.min(Project.since), func.max(Project.since)).where(Project.id.in_(pids))).one()
    dm = pd.DataFrame(
        db.execute(
            select(DeveloperMonth.project_id, DeveloperMonth.developer_id, DeveloperMonth.month, DeveloperMonth.reviews)
            .join(Developer, Developer.id == DeveloperMonth.developer_id)
            .where(~Developer.is_bot, DeveloperMonth.project_id.in_(pids))
        ).all(),
        columns=["project_id", "developer_id", "month", "reviews"],
    )
    per_dev = dm.groupby(["project_id", "developer_id"]).size()
    out = {
        "projects": len(pids),
        "developers": int(per_dev.size),
        "people": int(dm["developer_id"].nunique()),
        "developer_months": len(dm),
        "first_month": dm["month"].min(),
        "last_month": dm["month"].max(),
        "github_since": [str(v) for v in since],
        "per_project": {
            "commits": quartiles(p["commits"].fillna(0)),
            "issues_prs": quartiles(p["issues"].fillna(0)),
            "comments_reviews": quartiles(
                p.set_index("id")["comments"].fillna(0).add(dm.groupby("project_id")["reviews"].sum(), fill_value=0)
            ),
            "developers": quartiles(per_dev.groupby(level=0).size()),
            "developers_2plus_months": quartiles((per_dev >= 2).groupby(level=0).sum()),
            "observed_months": quartiles(
                dm.groupby("project_id")["month"].agg(
                    lambda m: (int(m.max()[:4]) * 12 + int(m.max()[5:])) - (int(m.min()[:4]) * 12 + int(m.min()[5:])) + 1
                )
            ),
        },
    }
    return out


def developers(db, pids: list[int]) -> dict:
    """Developer funnel (developer-project pairs) and, for the sustained developers who abandoned a project, whether
    they were active elsewhere on GitHub in the following 12 months."""
    dm = pd.DataFrame(
        db.execute(
            select(DeveloperMonth.project_id, DeveloperMonth.developer_id, DeveloperMonth.month, Developer.login,
                   Developer.after_exit_contributions, Developer.after_exit_private)
            .join(Developer, Developer.id == DeveloperMonth.developer_id)
            .where(~Developer.is_bot, DeveloperMonth.project_id.in_(pids))
        ).all(),
        columns=["project_id", "developer_id", "month", "login", "public", "private"],
    )  # fmt: skip
    dm["midx"] = dm["month"].map(lambda m: int(m[:4]) * 12 + int(m[5:]))
    dev = dm.groupby(["project_id", "developer_id"]).agg(
        months=("midx", "size"), last=("midx", "max"), login=("login", "first"), public=("public", "first"),
        private=("private", "first"))  # fmt: skip
    dev["end"] = dm.groupby("project_id")["midx"].max().reindex(dev.index.get_level_values(0)).to_numpy()
    sustained = dev[dev["months"] >= MIN_MONTHS]
    # Abandoned: no activity in the 12 months after the last active month, within the project's observed activity.
    left = sustained[sustained["end"] - sustained["last"] >= 12]
    checked = left.dropna(subset=["public"])
    return {
        "pairs": len(dev),
        "pairs_2plus_months": int((dev["months"] >= 2).sum()),
        "pairs_sustained": len(sustained),
        "people_sustained": int(sustained["login"].str.lower().nunique() + sustained["login"].isna().sum()),
        "projects_with_sustained": int(sustained.index.get_level_values(0).nunique()),
        "sustained_months": quartiles(sustained["months"]),
        "abandoned": len(left),
        "after_exit_checked": len(checked),
        "after_exit_active_elsewhere": int(((checked["public"] + checked["private"].fillna(0)) > 0).sum()),
    }


def community_coverage(db, pids: list[int]) -> dict:
    """How many messages and developer-months the community measures cover (paper, Operationalization)."""
    dm = pd.DataFrame(
        db.execute(
            select(DeveloperMonth.psych_messages, DeveloperMonth.thread_messages, DeveloperMonth.others_messages,
                   DeveloperMonth.newcomer_messages, DeveloperMonth.xent_messages)
            .join(Developer, Developer.id == DeveloperMonth.developer_id)
            .where(~Developer.is_bot, DeveloperMonth.project_id.in_(pids), DeveloperMonth.xent_messages.is_not(None))
        ).all(),
        columns=["psych", "thread", "others", "newcomer", "scored"],
    ).fillna(0)  # fmt: skip
    return {
        "developer_months": len(dm),
        "messages_with_words": float(dm["psych"].sum()),
        "messages_scored": float(dm["scored"].sum()),
        "months_with_scored": float((dm["scored"] > 0).mean()),
        "thread_messages": float(dm["thread"].sum()),
        "others_share": float(dm["others"].sum() / dm["thread"].sum()),
        "newcomer_share": float(dm["newcomer"].sum() / dm["thread"].sum()),
        "months_with_thread": float((dm["thread"] > 0).mean()),
    }


def concordance(intervals: pd.DataFrame, fit: dict) -> float:
    """Harrell's C for time-varying covariates: each event interval against the non-event intervals at risk in the
    same project and month, compared on the linear predictor."""
    lp = np.zeros(len(intervals))
    for c in fit["coefficients"]:
        x = intervals[c["covariate"]].to_numpy(float)
        if "log1p" in c["transform"]:
            x = np.log1p(x)
        lp += c["coef"] * x  # standardization does not change the ranking
    d = pd.DataFrame({"g": intervals["project"].astype(str) + "|" + intervals["stop"].astype(str),
                      "event": intervals["event"].to_numpy(), "lp": lp})  # fmt: skip
    conc = ties = pairs = 0.0
    for _, g in d[d["g"].isin(d.loc[d["event"] == 1, "g"])].groupby("g"):
        ctrl = np.sort(g.loc[g["event"] == 0, "lp"].to_numpy())
        if not len(ctrl):
            continue
        for v in g.loc[g["event"] == 1, "lp"]:
            lo, hi = np.searchsorted(ctrl, v, "left"), np.searchsorted(ctrl, v, "right")
            conc += lo
            ties += hi - lo
            pairs += len(ctrl)
    return float((conc + 0.5 * ties) / pairs) if pairs else float("nan")


def cox(intervals: pd.DataFrame, covariates: list[str]) -> dict:
    """Cox model with concordance, and Benjamini-Hochberg q-values over the psychological signals of the model
    (missing indicators included), as reported in the paper."""
    from scipy.stats import false_discovery_control

    fit = survival.fit_cox(intervals, covariates, cluster="person")
    fit["concordance"] = concordance(intervals, fit)
    signals = [c for c in fit["coefficients"] if c["covariate"] in PSYCH + COMMUNITY and c["p"] is not None]
    for c, q in zip(signals, false_discovery_control([c["p"] for c in signals]) if signals else []):
        c["q"] = float(q)
    return fit


def collinearity(intervals: pd.DataFrame, covariates: list[str]) -> dict:
    """Spearman correlations between the COLLINEARITY covariates, and the variance inflation factors of all the
    covariates of a model, on the intervals it is fitted on (counts log-transformed as in the model)."""
    x = intervals[covariates].astype(float).copy()
    for c in covariates:
        if next(v for v in survival.COVARIATES if v.key == c).count:
            x[c] = np.log1p(x[c])
    rho = x[[c for c in COLLINEARITY if c in covariates]].corr(method="spearman")
    vif = np.diag(np.linalg.pinv(x.corr().to_numpy()))
    return {"spearman": {a: {b: float(rho.loc[a, b]) for b in rho.columns} for a in rho.index},
            "vif": dict(zip(covariates, (float(v) for v in vif)))}  # fmt: skip


def extended(iv: pd.DataFrame, full: dict) -> dict:
    """Full model plus the community measures, on the intervals where they are measured, with the full model refitted
    on the same intervals for the likelihood-ratio test."""
    from scipy import stats as sps

    sub = iv.dropna(subset=COMMUNITY)
    if sub.empty:
        return {"error": "community stage not computed"}
    same = cox(sub, BASE + PSYCH) if len(sub) < len(iv) else full
    ext = cox(sub, BASE + PSYCH + COMMUNITY)
    lr = 2 * (ext["log_likelihood"] - same["log_likelihood"])
    return {"intervals": len(sub), "full_same_intervals": same if same is not full else None, "extended": ext,
            "lr_full_vs_extended": {"statistic": lr, "df": len(COMMUNITY), "p": float(sps.chi2.sf(lr, len(COMMUNITY)))},
            "collinearity": collinearity(sub, BASE + PSYCH + COMMUNITY)}  # fmt: skip


def destinations(db, pids: list[int], k: int = 12, min_months: int = MIN_MONTHS) -> dict:
    """Where the developers who moved elsewhere went: another community (repositories of other owners), or close to
    the project (other repositories of its owner, or their own repositories only); developers with only private
    contributions have an unknown destination. Counts, and a cause-specific model for each destination, in which the
    other moves and the departures from GitHub are censored."""
    split = dict(
        (did, "other" if other else "same_owner" if same else "own" if own else "private")
        for did, same, own, other in db.execute(
            select(Developer.id, Developer.after_exit_same_owner, Developer.after_exit_own, Developer.after_exit_other_owner)
            .where(Developer.after_exit_same_owner.is_not(None))
        ).tuples()
    )  # fmt: skip
    iv, st = survival.build_intervals(db, pids, k, 3, min_months, "moved", True)
    iv = iv.dropna(subset=BASE + PSYCH)
    movers = iv.loc[iv["event"] == 1, "sid"]
    dest = movers.map(lambda sid: split.get(int(sid.split(":")[1]), "unsplit"))
    out = {"moves": len(movers), "destinations": dest.value_counts().to_dict()}
    for name, kinds in (("other_community", {"other"}), ("close", {"same_owner", "own"})):
        event = iv["sid"].map(lambda sid: split.get(int(sid.split(":")[1])) in kinds) & (iv["event"] == 1)
        out[name] = cox(iv.assign(event=event.astype(int)), BASE + PSYCH)
        log(f"destinations {name}: {int(event.sum())} events")
    return out


def analyze(db, pids: list[int], k: int = 12, min_months: int = MIN_MONTHS) -> dict:
    """RQ1-RQ3 with abandonment threshold k (months) and developers with at least min_months active months."""
    res = {}
    # RQ1: event-time trajectories of leavers vs matched stayers.
    rows, stats = trajectories.event_time_frame(db, pids, k, 12, 3, min_months, 2)
    log(f"trajectories frame: {len(rows)} rows, {stats}")
    res["rq1"] = {"stats": stats, "trajectories": trajectories.trajectories(rows)}
    # Project fixed effects with thousands of projects are too large for the mixed model: the signals are
    # demeaned by project instead (within transformation), with the stayers matched within each project.
    demeaned = rows.copy()
    for s in trajectories.SIGNALS:
        demeaned[s] = demeaned[s] - demeaned.groupby("project")[s].transform("mean")
    demeaned["project"] = "all"
    res["rq1"]["models"] = trajectories.fit_models(demeaned)
    log("RQ1 done")

    # RQ2: base vs full model on the same intervals.
    iv, st = survival.build_intervals(db, pids, k, 3, min_months, "abandonment", True)
    log(f"abandonment intervals: {len(iv)}, {st}")
    iv = iv.dropna(subset=BASE + PSYCH)
    base, full = cox(iv, BASE), cox(iv, BASE + PSYCH)
    lr = 2 * (full["log_likelihood"] - base["log_likelihood"])
    from scipy import stats as sps

    res["rq2"] = {"build": st, "base": base, "full": full,
                  "lr_base_vs_full": {"statistic": lr, "df": len(PSYCH), "p": float(sps.chi2.sf(lr, len(PSYCH)))}}  # fmt: skip
    res["rq2"]["community"] = extended(iv, full)
    log("RQ2 done")

    # RQ3: cause-specific hazards.
    res["rq3"] = {}
    for outcome in ("disengagement", "moved"):
        iv, st = survival.build_intervals(db, pids, k, 3, min_months, outcome, True)
        iv = iv.dropna(subset=BASE + PSYCH)
        full = cox(iv, BASE + PSYCH)
        res["rq3"][outcome] = {"build": st, "full": full, "community": extended(iv, full)}
        log(f"RQ3 {outcome} done")
    return res


def explicit_anger(db, path: str):
    """Developer-month frame in which anger counts only messages with an anger probability >= 0.5, from the
    probabilities written by anger_probabilities.py; messages below it are removed from the anger counts, expressed and
    received, exactly as the aggregation stage counts them."""
    p = pd.read_csv(path)
    weak = p[p["p"] < 0.5]
    dev_of = dict(db.execute(select(Alias.id, Alias.developer_id)).all())
    bots = set(db.scalars(select(Developer.id).where(Developer.is_bot)))
    owner = {(pid, n): dev_of.get(a) for pid, n, a in db.execute(select(Issue.project_id, Issue.number, Issue.alias_id))}
    ids = weak.loc[weak["kind"] == "issue", "id"].tolist()
    issues = rows_in(db, select(Issue.project_id, Issue.alias_id, Issue.created_at), Issue.id, ids)
    ids = weak.loc[weak["kind"] == "comment", "id"].tolist()
    comments = rows_in(db, select(Comment.project_id, Comment.issue_number, Comment.alias_id, Comment.created_at), Comment.id, ids)
    month = lambda dt: f"{dt.year:04d}-{dt.month:02d}"  # noqa: E731
    delta = []  # (project, developer, month, expressed, received)
    for pid, alias, created in issues:
        if (dev := dev_of.get(alias)) is not None and created is not None:
            delta.append((pid, dev, month(created), 1, 0))
    for pid, number, alias, created in comments:
        dev = dev_of.get(alias)
        if created is None:
            continue
        if dev is not None:
            delta.append((pid, dev, month(created), 1, 0))
        own = owner.get((pid, number))
        if own is not None and own != dev and dev is not None and dev not in bots:  # received from a person
            delta.append((pid, own, month(created), 0, 1))
    delta = pd.DataFrame(delta, columns=["project_id", "developer_id", "month", "d_emit", "d_recv"])
    delta = delta.groupby(["project_id", "developer_id", "month"], as_index=False).sum()
    log(f"explicit anger: {len(weak)} of {len(p)} anger labels below 0.5 removed")
    original = dataset_mod.developer_month_frame

    def frame(db, project_ids=None, abandon_months=None):
        df = original(db, project_ids, abandon_months)
        df = df.merge(delta, on=["project_id", "developer_id", "month"], how="left")
        df[["d_emit", "d_recv"]] = df[["d_emit", "d_recv"]].fillna(0)
        df["anger_emitted"] = df["anger_emitted"] - df["d_emit"]
        df["anger_received"] = df["anger_received"] - df["d_recv"]
        df["anger_share"] = df["anger_emitted"] / df["emotion_messages"].replace(0, np.nan)
        df["anger_received_share"] = df["anger_received"] / df["emotion_received"].replace(0, np.nan)
        return df.drop(columns=["d_emit", "d_recv"])

    return frame


def main() -> None:
    with session_scope() as db:
        q = select(Project.id).where(Project.status == "done")
        if CUTOFF:
            q = q.where(Project.finished_at <= datetime.fromisoformat(CUTOFF))
        pids = [p for (p,) in db.execute(q).all()]
        mined = len(pids)
        # Everyone "abandons" a project that was shut down.
        archived = survival.archived_projects(db, pids)
        pids = [p for p in pids if p not in archived]
        log(f"{mined} done projects, {len(pids)} not archived")
        if DESTINATIONS:
            print(json.dumps({"destinations": destinations(db, pids)}, default=jsonable), flush=True)
            return
        if SENSITIVITY:
            # Threats to Validity: explicit anger, abandonment threshold, involvement threshold. One JSON line per
            # analysis, printed as soon as it is done.
            frame = explicit_anger(db, SENSITIVITY)
            original = survival.developer_month_frame
            survival.developer_month_frame = trajectories.developer_month_frame = frame
            log("sensitivity explicit_anger")
            print(json.dumps({"explicit_anger": analyze(db, pids)}, default=jsonable), flush=True)
            survival.developer_month_frame = trajectories.developer_month_frame = original
            for name, k, m in (("abandon_6", 6, MIN_MONTHS), ("abandon_18", 18, MIN_MONTHS),
                               ("involvement_3", 12, 3), ("involvement_12", 12, 12)):  # fmt: skip
                log(f"sensitivity {name}")
                print(json.dumps({name: analyze(db, pids, k, m)}, default=jsonable), flush=True)
            return
        else:
            res = {"projects": {"mined": mined, "archived": len(archived), "analyzed": len(pids)},
                   "developers": developers(db, pids), "dataset": describe(db, pids),
                   "community_coverage": community_coverage(db, pids)}  # fmt: skip
            log(f"dataset described: {res['developers']}")
            res.update(analyze(db, pids))
    print(json.dumps(res, default=jsonable))


if __name__ == "__main__":
    main()
