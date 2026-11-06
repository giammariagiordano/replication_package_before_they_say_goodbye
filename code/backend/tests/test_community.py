"""Community bond: citizenship counts and linguistic accommodation on hand-made conversations."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.community import MIN_CORPUS, accommodation, citizenship
from app.dataset import accommodation_counts, psych_ratios

JAN = datetime(2024, 1, 10)


def test_citizenship_splits_own_others_and_newcomers_items():
    first = {1: datetime(2020, 1, 1), 2: datetime(2020, 1, 1), 3: JAN - timedelta(days=5)}
    threads = {10: (1, JAN), 11: (2, JAN), 12: (3, JAN)}  # 12 opened by a newcomer; 13 by a bot (absent)
    messages = [(1, JAN, 10), (1, JAN, 10), (1, JAN, 11), (1, JAN, 12), (1, JAN, 13), (2, JAN, 10)]
    out = citizenship(threads, messages, first)
    assert out[(1, "2024-01")] == {"thread_messages": 4, "others_messages": 2, "newcomer_messages": 1}
    assert out[(2, "2024-01")] == {"thread_messages": 1, "others_messages": 1, "newcomer_messages": 0}


def corpus(words: str, n: int, who: int):
    return [(who, JAN, words) for _ in range(n)]


def test_accommodation_scores_against_the_others_only():
    jargon = "please rebase the branch on main and squash the fixup commits before we merge"
    others = corpus(jargon, MIN_CORPUS // 14 + 1, 1) + corpus(jargon, 5, 2)
    alike = [(3, JAN, jargon)]
    novel = [(4, JAN, "honestly the weather in lisbon was lovely and the pastries were delicious too")]
    out = accommodation(others + alike + novel)
    (n3, h3), (n4, h4) = out[(3, "2024-01")], out[(4, "2024-01")]
    assert n3 == n4 == 1 and h3 < h4
    # Leave-self-out: a developer who wrote everything has no community to compare with.
    alone = accommodation(corpus(jargon, MIN_CORPUS // 14 + 1, 9))
    assert alone == {}


def test_short_messages_and_small_corpora_are_not_scored():
    assert accommodation([(1, JAN, "lgtm thanks"), (2, JAN, "fix the build please now")]) == {}


def test_accommodation_is_relative_to_the_peers_of_the_month():
    df = pd.DataFrame({
        "project_id": [1, 1, 1, 1], "month": ["2024-01"] * 3 + ["2024-02"],
        "xent_messages": [2.0, 1.0, 0.0, 3.0], "xent_sum": [10.0, 8.0, 0.0, 9.0],
        "others_messages": [1.0, 0, 0, 0], "thread_messages": [4.0, 0, 0, 0],
    })  # fmt: skip
    for c in ("we_words", "i_words", "words", "thanks_emitted", "psych_messages", "thanks_received", "replies_received",
              "reactions_pos_received", "reactions_neg_received", "reacted_items", "emotion_messages", "anger_received",
              "emotion_received", "newcomer_messages", *(f"{e}_emitted" for e in ("anger", "joy", "love", "sadness", "fear", "surprise"))):  # fmt: skip
        df[c] = 0.0
    accommodation_counts(df)
    psych_ratios(df)
    # developer 0: own 5 bits/word, peers 8 -> +3; developer 1: own 8, peers 5 -> -3
    assert df["accommodation"].iloc[0] == 3.0 and df["accommodation"].iloc[1] == -3.0
    assert np.isnan(df["accommodation"].iloc[2])  # nothing scored
    assert df["accommodation_messages"].iloc[3] == 0 and np.isnan(df["accommodation"].iloc[3])  # no peers that month
    assert df["citizenship_share"].iloc[0] == 0.25 and np.isnan(df["citizenship_share"].iloc[1])
