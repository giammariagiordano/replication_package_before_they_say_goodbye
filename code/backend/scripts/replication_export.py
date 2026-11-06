"""Analysis-ready, pseudonymized tables of the paper for the replication package, written to /out.

Run in the API container, with the output folder and the anger probabilities (anger_probabilities.py) available:
  docker compose run --rm --no-deps -T -v <replication>/data:/out api python scripts/replication_export.py [part ...]
Parts: main (default: tables of RQ1-RQ3), developers, sensitivity, messages; "all" exports every part.

No names, logins, e-mails or message texts are exported. Developers are identified by their salted pseudonym
(unique within a project) and, for the cluster-robust standard errors, by a salted pseudonym of the person;
messages by their public GitHub ids, from which their text can be retrieved.
The tables follow the paper's settings: non-archived projects whose pipeline is done, abandonment after 12 months
of inactivity, covariates over the previous 3 months, developers with at least 6 active months.
"""

import hashlib
import hmac
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from app import survival, trajectories
from app.dataset import developer_month_frame
from app.db import rows_in, session_scope
from app.export import projects_frame
from app.models import Alias, Comment, DeveloperMonth, Developer, Issue, Project
from app.privacy import get_salt
from app.psych import grateful, pronouns
from app.stages.bots import BOT_PATTERN, KNOWN_BOTS
from app.stages.text import clean_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_results import BASE, COMMUNITY, PSYCH, explicit_anger  # noqa: E402

OUT = Path("/out")
ANGER_PROBABILITIES = Path("/data/anger_probabilities.csv")
K, WINDOW, MIN_MONTHS = 12, 3, 6
# Toxicity is out of the study: its columns are empty and left out.
DROP = ["project_id", "developer_id", "toxic_emitted", "toxic_received", "toxic_threads", "toxic_emitted_share",
        "toxic_received_share", "threads"]  # fmt: skip
# Threats to Validity: (abandonment threshold, minimum active months, explicit anger only).
VARIANTS = {"abandon_6": (6, MIN_MONTHS, False), "abandon_18": (18, MIN_MONTHS, False),
            "involvement_3": (K, 3, False), "involvement_12": (K, 12, False), "explicit_anger": (K, MIN_MONTHS, True)}  # fmt: skip
# Chinese, Japanese, Korean, or Cyrillic characters (Threats to Validity: messages not in English).
NON_LATIN = re.compile("[一-鿿぀-ヿ가-힯Ѐ-ӿ]")
OUTCOMES = (("abandonment", "abandonment"), ("disengagement", "left_github"), ("moved", "moved"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def write(df: pd.DataFrame, name: str, folder: Path = OUT) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv.gz"
    df.to_csv(path, index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})
    log(f"{path.relative_to(OUT)}: {len(df)} rows, {path.stat().st_size / 1e6:.1f} MB")
    return {"rows": len(df), "columns": list(df.columns), "bytes": path.stat().st_size}


class Export:
    def __init__(self, db):
        self.db = db
        pids = [p for (p,) in db.execute(select(Project.id).where(Project.status == "done")).all()]
        self.pids = [p for p in pids if p not in survival.archived_projects(db, pids)]
        log(f"{len(self.pids)} non-archived projects")
        self.salt = get_salt(db)
        devs = db.execute(
            select(Developer.id, Developer.project_id, Project.owner, Project.name, Developer.pseudonym)
            .join(Project, Project.id == Developer.project_id).where(Developer.project_id.in_(self.pids))
        ).all()  # fmt: skip
        self.pseud = {did: pseud for did, _, _, _, pseud in devs}
        self.project = {pid: f"{owner}/{name}" for _, pid, owner, name, _ in devs}
        # Internal subject id "project_id:developer_id" -> "owner/name:pseudonym", as in the published tables.
        self.subject = {f"{pid}:{did}": f"{owner}/{name}:{pseud}" for did, pid, owner, name, pseud in devs}
        # Where the developers who moved elsewhere went (paper, Disengagement).
        self.destination = {
            did: "other_owner" if other else "same_owner" if same else "own" if own else "private"
            for did, same, own, other in db.execute(
                select(Developer.id, Developer.after_exit_same_owner, Developer.after_exit_own,
                       Developer.after_exit_other_owner).where(Developer.after_exit_same_owner.is_not(None))
            ).all()
        }  # fmt: skip

    def person(self, key: str) -> str:
        """Salted hash of a person across projects, long enough that two people never share it (the developers'
        pseudonyms are unique only within a project)."""
        return hmac.new(self.salt.encode(), f"person|{key}".encode(), hashlib.sha256).hexdigest()[:16]

    def intervals(self, k: int, min_months: int, folder: Path, columns: list[str] | None = None) -> dict:
        meta = {}
        for outcome, name in OUTCOMES:
            iv, stats = survival.build_intervals(self.db, self.pids, k, WINDOW, min_months, outcome, True)
            if outcome == "moved":
                did = iv["sid"].str.split(":").str[1].astype(int)
                iv["destination"] = did.map(self.destination).where(iv["event"] == 1)
            iv["person"] = iv["person"].map(self.person)
            iv["sid"] = iv["sid"].map(self.subject)
            iv = iv.drop(columns=["developer"])
            if columns:
                iv = iv[[c for c in columns if c in iv.columns]]
            meta[f"intervals_{name}"] = {**write(iv, f"intervals_{name}", folder), "build": stats}
        rows, stats = trajectories.event_time_frame(self.db, self.pids, k, 12, 3, min_months, 2)
        rows["sid"] = rows["sid"].map(self.subject)
        keep = ["sid", "project", "group", "set", "month", "rel", "bin", "leaver", "log_messages", *trajectories.SIGNALS]
        meta["event_time"] = {**write(rows[keep], "event_time", folder), "build": stats}
        return meta

    def main(self) -> dict:
        meta = {"projects": write(projects_frame(self.db, self.pids).drop(columns=["pipeline_status", "mined_at"], errors="ignore"), "projects")}
        months = developer_month_frame(self.db, self.pids, K).drop(columns=DROP, errors="ignore")
        meta["developer_months"] = write(months, "developer_months")
        return {**meta, **self.intervals(K, MIN_MONTHS, OUT)}

    def sensitivity(self) -> dict:
        """The tables of the main analyses under the thresholds of the Threats to Validity, with the model columns."""
        columns = ["sid", "person", "project", "episode", "start", "stop", "event", *BASE, *PSYCH, *COMMUNITY, "destination"]
        meta, original = {}, survival.developer_month_frame
        for name, (k, m, explicit) in VARIANTS.items():
            log(f"sensitivity {name}")
            if explicit:
                frame = explicit_anger(self.db, str(ANGER_PROBABILITIES))
                survival.developer_month_frame = trajectories.developer_month_frame = frame
            try:
                meta[name] = {"abandon_months": k, "min_active_months": m, "explicit_anger": explicit,
                              "files": self.intervals(k, m, OUT / "sensitivity" / name, columns)}  # fmt: skip
            finally:
                survival.developer_month_frame = trajectories.developer_month_frame = original
        return meta

    def developers(self) -> dict:
        """One row per developer and project, bots included: bot rule, activity, and outcome after the exit."""
        rule_of = {}  # alias -> first bot rule of the paper that flags it: (i) type, (ii) [bot], (iii) known, (iv) pattern
        for aid, did, gh_type, login, name, email in rows_in(
            self.db, select(Alias.id, Alias.developer_id, Alias.gh_type, Alias.login, Alias.name, Alias.email),
            Alias.developer_id, list(self.pseud),
        ):  # fmt: skip
            values = [v.strip().lower() for v in (login, name, (email or "").split("@")[0]) if v]
            rule = ("i" if gh_type == "Bot" else "ii" if any(v.endswith("[bot]") for v in values)
                    else "iii" if any(v in KNOWN_BOTS for v in values)
                    else "iv" if any(BOT_PATTERN.search(v) for v in values) else None)  # fmt: skip
            if rule and (did not in rule_of or rule < rule_of[did]):
                rule_of[did] = rule
        aliases = pd.Series([did for (did,) in rows_in(self.db, select(Alias.developer_id), Alias.developer_id, list(self.pseud))]).value_counts()
        rows = self.db.execute(
            select(Developer.id, Developer.project_id, Developer.login, Developer.is_bot, Developer.after_exit_from,
                   Developer.after_exit_contributions, Developer.after_exit_private, Developer.after_exit_repos)
            .where(Developer.project_id.in_(self.pids))
        ).all()  # fmt: skip
        months = pd.DataFrame(self.db.execute(
            select(DeveloperMonth.developer_id, DeveloperMonth.month).where(DeveloperMonth.project_id.in_(self.pids))
        ).all(), columns=["did", "month"]).groupby("did")["month"].agg(["size", "min", "max"])  # fmt: skip
        out = []
        for did, pid, login, is_bot, checked, public, private, repos in rows:
            m = months.loc[did] if did in months.index else None
            out.append({
                "project": self.project[pid], "developer": self.pseud[did],
                # the same person across projects, as in the clustered standard errors of the survival models
                "person": self.person(f"gh:{login.lower()}" if login else f"{pid}:{did}"),
                "is_bot": int(is_bot),
                "bot_rule": rule_of.get(did) if is_bot else None, "aliases": int(aliases.get(did, 0)),
                "active_months": int(m["size"]) if m is not None else 0,
                "first_month": m["min"] if m is not None else None, "last_month": m["max"] if m is not None else None,
                # GitHub's contribution record in the 12 months after the last active month (checked for abandoners)
                "after_exit_checked": int(checked is not None),
                "after_exit_public": public, "after_exit_private": private, "after_exit_repos": repos,
                "destination": self.destination.get(did),
            })  # fmt: skip
        return {"developers": write(pd.DataFrame(out), "developers")}

    def messages(self) -> dict:
        """One row per issue, pull request, comment, and review: public GitHub ids, author pseudonym, labels, and the
        lexical counts of the paper, without text. Anger probabilities only for the messages labeled with anger."""
        p = pd.read_csv(ANGER_PROBABILITIES)
        anger_p = {(k, i): v for k, i, v in zip(p["kind"], p["id"], p["p"])}
        dev_of = dict(self.db.execute(select(Alias.id, Alias.developer_id)).all())
        bots = set(self.db.scalars(select(Developer.id).where(Developer.is_bot)))
        out = []
        for pid in self.pids:
            opener = {}
            for iid, number, is_pr, alias, created, title, body, emo, sent, pos, neg in self.db.execute(
                select(Issue.id, Issue.number, Issue.is_pr, Issue.alias_id, Issue.created_at, Issue.title, Issue.body,
                       Issue.emotions, Issue.sentiment, Issue.reactions_pos, Issue.reactions_neg).where(Issue.project_id == pid)
            ):  # fmt: skip
                dev = dev_of.get(alias)
                opener[number] = dev
                words, we, i = pronouns(clean_text(body))
                out.append((pid, "pull_request" if is_pr else "issue", number, None, created, dev, emo,
                            anger_p.get(("issue", iid)), sent, words, we, i, None, pos, neg, None,
                            int(bool(NON_LATIN.search(body or "")))))  # fmt: skip
            for cid, gh_id, kind, number, alias, created, body, emo, sent, pos, neg in self.db.execute(
                select(Comment.id, Comment.gh_id, Comment.kind, Comment.issue_number, Comment.alias_id, Comment.created_at,
                       Comment.body, Comment.emotions, Comment.sentiment, Comment.reactions_pos, Comment.reactions_neg)
                .where(Comment.project_id == pid)
            ):  # fmt: skip
                dev = dev_of.get(alias)
                text = clean_text(body)
                words, we, i = pronouns(text)
                owner = opener.get(number)
                # a reply received by the opener of the item, as in the aggregation stage
                reply = owner is not None and owner != dev
                thanks = int(grateful(text)) if reply and dev is not None and dev not in bots else None
                out.append((pid, kind, number, gh_id, created, dev, emo, anger_p.get(("comment", cid)), sent, words, we,
                            i, thanks, pos, neg, owner if reply else None, int(bool(NON_LATIN.search(body or "")))))  # fmt: skip
        df = pd.DataFrame(out, columns=["pid", "kind", "number", "gh_id", "created", "dev", "emotions", "anger_p",
                                        "sentiment", "words", "we_words", "i_words", "grateful_reply", "reactions_pos",
                                        "reactions_neg", "reply_to", "non_latin"])  # fmt: skip
        df.insert(0, "project", df["pid"].map(self.project))
        df["month"] = pd.to_datetime(df["created"]).dt.strftime("%Y-%m")
        # "" (classified, no emotion) would read back as missing, like None (not classified)
        df["emotions"] = df["emotions"].replace("", "none")
        df["author"] = df["dev"].map(self.pseud)
        df["author_is_bot"] = df["dev"].isin(bots).astype(int)
        df["reply_to"] = df["reply_to"].map(self.pseud)
        df["gh_id"] = df["gh_id"].astype("Int64")
        cols = ["project", "kind", "number", "gh_id", "month", "author", "author_is_bot", "emotions", "anger_p", "sentiment",
                "words", "we_words", "i_words", "reply_to", "grateful_reply", "reactions_pos", "reactions_neg", "non_latin"]  # fmt: skip
        return {"messages": write(df[cols], "messages")}


def main() -> None:
    parts = sys.argv[1:] or ["main"]
    if parts == ["all"]:
        parts = ["main", "developers", "sensitivity", "messages"]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "metadata.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta["settings"] = {"abandon_months": K, "window_months": WINDOW, "min_active_months": MIN_MONTHS}
    meta.setdefault("files", {})
    with session_scope() as db:
        export = Export(db)
        for part in parts:
            result = getattr(export, part)()
            if part == "sensitivity":
                meta["sensitivity"] = result
            else:
                meta["files"].update(result)
            path.write_text(json.dumps(meta, indent=1, default=str))
    log("done")


if __name__ == "__main__":
    main()
