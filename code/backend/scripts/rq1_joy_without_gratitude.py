"""RQ1 robustness: does the decline of joy before the exit hold once gratitude is set aside?

On GitHub, many joyful messages are thanks ("thanks for the help!"), so the decline of joy might mean that leavers
thank less, e.g. because they ask for less help, rather than that they are less engaged. The replication package
contains no message text, so the gratitude of each message is known only in part:

- excluding grateful replies (exact): messages.csv.gz flags the replies to other people's items that thank or
  appreciate (grateful_reply); joy is recomputed per developer-month without them.
- excluding all thanks (bound): developer_months.csv.gz counts every message of the developer that expresses
  gratitude (thanks_emitted), but not which of them are joyful. We remove all of them both from the joyful messages
  and from the classified ones, i.e. we assume that every grateful message is joyful, which removes as much joy as
  gratitude could account for.

  python scripts/rq1_joy_without_gratitude.py ../../data   # from code/backend: writes ../../data/rq1_joy_without_gratitude.json

Both variants replace joy_share in data/event_time.csv.gz and refit the main RQ1 model (within developer) and the
random-intercept model, with the same transformation as the main analysis (centering on the project mean).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import trajectories  # noqa: E402


def last(models: list[dict], signal: str = "joy_share") -> dict:
    m = next(m for m in models if m["signal"] == signal)
    out = {k: m["interactions"][-1][k] for k in ("coef", "lower", "upper", "p")}
    out["window_3_5"] = {k: m["interactions"][-2][k] for k in ("coef", "p")}
    return out


def fit(rows: pd.DataFrame, joy: pd.Series) -> dict:
    r = rows.assign(joy_share=joy)
    r["joy_share"] = r["joy_share"] - r.groupby("project")["joy_share"].transform("mean")
    r["project"] = "all"
    only = {"joy_share": "Joy"}
    signals, trajectories.SIGNALS = trajectories.SIGNALS, only  # fit joy alone: no correction across signals
    try:
        return {"within": last(trajectories.fit_within(r)), "random_intercept": last(trajectories.fit_models(r)),
                "developer_months": int(r["joy_share"].notna().sum())}  # fmt: skip
    finally:
        trajectories.SIGNALS = signals


def main(data: Path) -> None:
    rows = pd.read_csv(data / "event_time.csv.gz")
    rows["developer"] = rows["sid"].str.rsplit(":", n=1).str[1]
    key = ["project", "developer", "month"]

    m = pd.read_csv(data / "messages.csv.gz", usecols=["project", "author", "month", "author_is_bot", "emotions",
                                                       "grateful_reply"])  # fmt: skip
    m = m[(m["author_is_bot"] == 0) & (m["emotions"] != "-")].rename(columns={"author": "developer"})
    m["joy"] = m["emotions"].str.contains("joy")
    m["grateful"] = m["grateful_reply"] == 1
    agg = m.groupby(key).agg(classified=("joy", "size"), joy=("joy", "sum"), grateful=("grateful", "sum"),
                             grateful_joy=("grateful", lambda g: (g & m.loc[g.index, "joy"]).sum())).reset_index()  # fmt: skip
    dm = pd.read_csv(data / "developer_months.csv.gz", usecols=[*key, "emotion_messages", "joy_emitted", "thanks_emitted"])
    x = rows[key].merge(agg, on=key, how="left").merge(dm, on=key, how="left")
    assert len(x) == len(rows)

    # exact: without the grateful replies
    n = x["classified"] - x["grateful"]
    joy_no_replies = ((x["joy"] - x["grateful_joy"]) / n).where(n > 0)
    # bound: without all the messages that express gratitude, assumed joyful
    thanks = np.minimum(x["thanks_emitted"].fillna(0), x["emotion_messages"])
    n = x["emotion_messages"] - thanks
    joy_no_thanks = ((x["joy_emitted"] - np.minimum(thanks, x["joy_emitted"])) / n).where(n > 0)

    check = (rows["joy_share"] - x["joy_emitted"] / x["emotion_messages"]).abs().max()
    res = {
        "check_max_abs_difference_original_joy": float(check),
        "share_of_classified_messages_grateful_reply": float(x["grateful"].sum() / x["classified"].sum()),
        "share_of_joyful_messages_grateful_reply": float(x["grateful_joy"].sum() / x["joy"].sum()),
        "share_of_classified_messages_thanks": float(thanks.sum() / x["emotion_messages"].sum()),
        "original": fit(rows, rows["joy_share"]),
        "without_grateful_replies": fit(rows, joy_no_replies.to_numpy()),
        "without_all_thanks_bound": fit(rows, joy_no_thanks.to_numpy()),
    }
    (data / "rq1_joy_without_gratitude.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
