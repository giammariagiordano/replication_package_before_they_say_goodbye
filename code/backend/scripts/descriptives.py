"""Descriptive numbers of the paper (Dataset and Operationalization, Table of the dataset, outcomes after the exit)
from the tables of the replication package, without the database.

  python scripts/descriptives.py ../../data      # from code/backend; writes ../../data/descriptives.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

K, MIN_MONTHS = 12, 6


def midx(month: pd.Series) -> pd.Series:
    return month.str[:4].astype(int) * 12 + month.str[5:7].astype(int)


def quartiles(s: pd.Series) -> dict:
    q = s.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    return dict(zip(["min", "q1", "median", "q3", "max"], (float(v) for v in q)))


def main(data: Path) -> None:
    projects = pd.read_csv(data / "projects.csv.gz")
    devs = pd.read_csv(data / "developers.csv.gz")
    dm = pd.read_csv(data / "developer_months.csv.gz")
    msg = pd.read_csv(data / "messages.csv.gz", dtype={"emotions": str})
    out = {"projects": len(projects)}

    # Developers and bots (Dataset)
    bots = devs[devs["is_bot"] == 1]
    out["developers"] = {"all": len(devs), "bots": len(bots), "bot_share": len(bots) / len(devs),
                         "bots_by_rule": bots["bot_rule"].value_counts().to_dict(), "humans": int((devs["is_bot"] == 0).sum()),
                         "developer_months": len(dm)}  # fmt: skip

    # Developer selection: sustained involvement
    dm["midx"] = midx(dm["month"])
    pair = dm.groupby(["project", "developer"]).agg(months=("midx", "size"), last=("midx", "max"))
    pair["end"] = dm.groupby("project")["midx"].max().reindex(pair.index.get_level_values(0)).to_numpy()
    sustained = pair[pair["months"] >= MIN_MONTHS]
    person = devs.set_index(["project", "developer"])["person"]
    out["selection"] = {"single_month_share": float((pair["months"] == 1).mean()), "sustained_pairs": len(sustained),
                        "sustained_people": int(person.reindex(sustained.index).nunique()),
                        "projects_with_sustained": int(sustained.index.get_level_values(0).nunique()),
                        "sustained_active_months": quartiles(sustained["months"])}  # fmt: skip

    # Table of the dataset, per project
    per_dev = pair["months"]
    reviews = dm.groupby("project")["reviews"].sum()
    p = projects.set_index("project")
    out["per_project"] = {
        "commits": quartiles(p["commits"].fillna(0)),
        "issues_prs": quartiles((p["issues"] + p["pull_requests"]).fillna(0)),
        # all comments, reviews included, plus the reviews counted per developer and month (as in paper_results.py)
        "comments_reviews": quartiles((p["comments"] + p["reviews"]).fillna(0).add(reviews, fill_value=0)),
        "developers": quartiles(per_dev.groupby(level=0).size()),
        "developers_sustained": quartiles((per_dev >= MIN_MONTHS).groupby(level=0).sum()),
        "observed_months": quartiles(dm.groupby("project")["midx"].agg(lambda m: m.max() - m.min() + 1)),
    }

    # Lexical measures, recognition, citizenship, accommodation (Operationalization)
    human = msg[(msg["author_is_bot"] == 0) & msg["author"].notna()]
    worded = human[human["words"] > 0]
    out["lexical"] = {"messages_with_words": int(dm["psych_messages"].sum()), "we_words": int(dm["we_words"].sum()),
                      "i_words": int(dm["i_words"].sum()),
                      "we_share": float(dm["we_words"].sum() / (dm["we_words"].sum() + dm["i_words"].sum())),
                      "developer_months_with_pronouns": float(((dm["we_words"] + dm["i_words"]) > 0).mean()),
                      "replies_received": int(dm["replies_received"].sum()),
                      "grateful_share": float(dm["thanks_received"].sum() / dm["replies_received"].sum()),
                      "reacted_items": int(dm["reacted_items"].sum()),
                      "positive_reactions": int(dm["reactions_pos_received"].sum()),
                      "messages_table_with_words": len(worded)}  # fmt: skip
    measured = dm[dm["xent_messages"].notna()]
    out["community"] = {"scored_messages": int(measured["xent_messages"].sum()),
                        "scored_share": float(measured["xent_messages"].sum() / measured["psych_messages"].sum()),
                        "developer_months_scored": float((measured["xent_messages"] > 0).mean()),
                        "thread_messages": int(measured["thread_messages"].sum()),
                        "others_share": float(measured["others_messages"].sum() / measured["thread_messages"].sum()),
                        "newcomer_share": float(measured["newcomer_messages"].sum() / measured["thread_messages"].sum())}  # fmt: skip

    # Emotion labels: explicit anger and messages not in English (Threats to Validity)
    # emotions: "none" = classified without emotions, "-" = nothing to classify, missing = not classified
    classified = msg[msg["emotions"].notna() & (msg["emotions"] != "-")]
    anger = classified[classified["emotions"].str.contains("anger")]
    out["emotions"] = {"classified_messages": len(classified), "anger_labels": len(anger),
                       "anger_below_0_5_share": float((anger["anger_p"] < 0.5).mean()),
                       "non_latin_share": float(classified["non_latin"].mean())}  # fmt: skip

    # Abandonment and outcome after the exit (RQ3)
    left = sustained[sustained["end"] - sustained["last"] >= K]
    after = devs.set_index(["project", "developer"]).reindex(left.index)
    checked = after[after["after_exit_public"].notna()]
    elsewhere = (checked["after_exit_public"] + checked["after_exit_private"].fillna(0)) > 0
    out["after_exit"] = {"abandoned": len(left), "checked": len(checked), "active_elsewhere": int(elsewhere.sum()),
                         "active_elsewhere_share": float(elsewhere.mean())}  # fmt: skip

    (data / "descriptives.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
