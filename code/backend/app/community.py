"""The developer's bond with the community beyond pronouns: citizenship behavior and linguistic accommodation.

- Citizenship (organizational citizenship behavior, Organ 1988; its decline precedes turnover, Chen et al. 1998):
  the share of the developer's messages that go to other people's issues and pull requests, i.e. helping others
  rather than following up on their own items, and the share that goes to newcomers' items. Shares, not counts:
  volume is already captured by activity and centrality.
- Linguistic accommodation (Danescu-Niculescu-Mizil et al., "No country for old members", WWW 2013): how close the
  developer's wording is to the project's current language. For every project and month, a bigram language model is
  trained on the messages written by the *other* human developers in that month and the previous two; the
  cross-entropy of each of the developer's messages under it (first EVAL_WORDS words) measures how unexpected their
  wording is to the community. Cross-entropy also depends on how much text the project produces, so the analyses
  compare it with that of the other developers of the same project and month (app.dataset).

Messages are cleaned as for the other lexical measures (app.stages.text.clean_text). English only.
"""

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from .psych import WORD

NEWCOMER_DAYS = 90  # an item is a newcomer's if opened within this many days of its author's first activity
LM_MONTHS = 3  # months of others' messages in the language model: the message's month and the previous two
EVAL_WORDS = 30  # words of each message scored, so that long messages do not weigh more (DNM 2013)
MIN_WORDS = 10  # shorter messages are not scored
MIN_CORPUS = 2000  # words of others' messages needed for a language model
LAMBDA = 0.7  # weight of the bigram estimate, interpolated with the add-one unigram one (Jelinek-Mercer)
START = "<s>"


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


def midx(dt: datetime) -> int:
    return dt.year * 12 + dt.month - 1


def month_str(m: int) -> str:
    return f"{m // 12:04d}-{m % 12 + 1:02d}"


# -- citizenship ------------------------------------------------------------------------------------------------


def citizenship(
    threads: dict[int, tuple[int, datetime]],
    messages: list[tuple[int, datetime, int]],
    first_activity: dict[int, datetime],
) -> dict[tuple[int, str], Counter]:
    """Per (developer, month): `thread_messages`, the messages written in issues/PRs opened by a human developer
    (own or others', description included), `others_messages`, those in items opened by someone else, and
    `newcomer_messages`, those in items opened by someone else within NEWCOMER_DAYS of their first activity.

    `threads`: issue number -> (human author, opened at); items opened by bots or unknown accounts are absent.
    `messages`: (human author, written at, issue number) for descriptions, comments and reviews."""
    out: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for dev, when, number in messages:
        thread = threads.get(number)
        if thread is None or when is None:
            continue
        owner, opened = thread
        c = out[(dev, month_str(midx(when)))]
        c["thread_messages"] += 1
        if owner != dev:
            c["others_messages"] += 1
            first = first_activity.get(owner)
            c["newcomer_messages"] += first is not None and opened - first <= timedelta(days=NEWCOMER_DAYS)
    return out


# -- linguistic accommodation -----------------------------------------------------------------------------------


class _Counts:
    """Unigram, bigram and context counts of a set of messages."""

    __slots__ = ("uni", "bi", "ctx", "n")

    def __init__(self):
        self.uni, self.bi, self.ctx, self.n = Counter(), Counter(), Counter(), 0

    def add(self, toks: list[str]):
        prev = START
        for w in toks:
            self.uni[w] += 1
            self.bi[(prev, w)] += 1
            self.ctx[prev] += 1
            prev = w
        self.n += len(toks)


def _cross_entropy(toks: list[str], window: list[_Counts], own: list[_Counts], vocab: int) -> float:
    """Bits per word of `toks` under the language model of `window` minus `own` (the developer's messages)."""
    total = lambda attr, key: sum(getattr(c, attr).get(key, 0) for c in window) - sum(getattr(c, attr).get(key, 0) for c in own)  # noqa: E731
    n = sum(c.n for c in window) - sum(c.n for c in own)
    bits, prev = 0.0, START
    for w in toks:
        p = (total("uni", w) + 1) / (n + vocab + 1)  # add-one, +1 for unseen words
        ctx = total("ctx", prev)
        if ctx:
            p = LAMBDA * total("bi", (prev, w)) / ctx + (1 - LAMBDA) * p
        bits -= math.log2(p)
        prev = w
    return bits / len(toks)


def accommodation(messages: list[tuple[int, datetime, str]]) -> dict[tuple[int, str], tuple[int, float]]:
    """Per (developer, month): (scored messages, sum of their cross-entropies in bits per word) under the language
    model of the other developers' messages in the month and the previous LM_MONTHS - 1. Lower = closer to the
    community's language. `messages` are (human author, written at, cleaned text)."""
    monthly: dict[int, _Counts] = defaultdict(_Counts)
    personal: dict[tuple[int, int], _Counts] = defaultdict(_Counts)
    scored: dict[tuple[int, int], list[list[str]]] = defaultdict(list)
    for dev, when, text in messages:
        toks = tokens(text)
        if not toks or when is None:
            continue
        m = midx(when)
        monthly[m].add(toks)
        personal[(dev, m)].add(toks)
        if len(toks) >= MIN_WORDS:
            scored[(dev, m)].append(toks[:EVAL_WORDS])
    empty = _Counts()
    out = {}
    vocabs: dict[int, int] = {}
    for (dev, m), msgs in scored.items():
        months = range(m - LM_MONTHS + 1, m + 1)
        window = [monthly.get(x, empty) for x in months]
        own = [personal.get((dev, x), empty) for x in months]
        if sum(c.n for c in window) - sum(c.n for c in own) < MIN_CORPUS:
            continue
        if m not in vocabs:
            vocabs[m] = len(set().union(*(c.uni for c in window)))
        h = [_cross_entropy(t, window, own, vocabs[m]) for t in msgs]
        out[(dev, month_str(m))] = (len(h), sum(h))
    return out
