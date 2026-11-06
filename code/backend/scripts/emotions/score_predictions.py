"""Recompute the table of the emotion classifiers from the predictions in the replication package, without a GPU.

The gold standards cannot be redistributed: download them and build the GitHub test set as in training
    git clone https://github.com/vcu-swim-lab/SE-Emotion-Study external/se-emotion-src
    (cd external && PYTHONPATH=../backend python ../backend/scripts/emotions/prepare_data.py)   # gh-emotions/gh_clean.csv
and get the MSR 2020 polarity gold standard (Novielli et al.) as external/gold/github_gold.csv. Then
    python scripts/emotions/score_predictions.py <data>/classifier external/gh-emotions/gh_clean.csv external/gold/github_gold.csv

Thresholds are those of each model (scores.json). Polarity: positive if only love/joy, negative if only
anger/sadness/fear, neutral if none, mixed (always wrong) otherwise, as in validate_github.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ["anger", "joy", "love", "sadness", "fear", "surprise"]
POSITIVE, NEGATIVE = {"love", "joy"}, {"anger", "sadness", "fear"}
CLASSES = ["positive", "neutral", "negative"]


def f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    return (2 * tp / (2 * tp + fp + fn) if tp else 0.0), precision


def polarity(labels: list[str]) -> str:
    pos, neg = bool(POSITIVE & set(labels)), bool(NEGATIVE & set(labels))
    return "mixed" if pos and neg else "positive" if pos else "negative" if neg else "neutral"


def main(folder: Path, gh_path: Path, msr_path: Path) -> None:
    scores = json.loads((folder / "scores.json").read_text())
    gh = pd.read_csv(gh_path)
    gh = gh[gh["split"] == "test"]
    msr = pd.read_csv(msr_path, sep=";").rename(columns={"ID": "id"})
    for name, published in scores.items():
        th = np.array([published["thresholds"][e] for e in EMOTIONS])
        pred = pd.read_csv(folder / f"{name}_gh_test.csv").merge(gh, on="id")
        y = pred[[e.capitalize() for e in EMOTIONS]].to_numpy().astype(bool)
        p = pred[[f"p_{e}" for e in EMOTIONS]].to_numpy() >= th
        per = {e: f1(y[:, j], p[:, j]) for j, e in enumerate(EMOTIONS)}
        micro = f1(y.ravel(), p.ravel())[0]
        pm = pd.read_csv(folder / f"{name}_msr2020.csv").merge(msr[["id", "Polarity"]], on="id")
        probs = pm[[f"p_{e}" for e in EMOTIONS]].to_numpy() >= th
        guess = np.array([polarity([e for e, on in zip(EMOTIONS, row) if on]) for row in probs])
        gold = pm["Polarity"].to_numpy()
        macro = np.mean([f1(gold == c, guess == c)[0] for c in CLASSES])
        print(f"{name:17s} micro-F1 {micro:.3f} (published {published['gh_test']['micro_f1']}), "
              f"anger {per['anger'][0]:.3f}, joy {per['joy'][0]:.3f}, fear {per['fear'][0]:.3f} (precision {per['fear'][1]:.3f}), "
              f"polarity macro-F1 {macro:.3f} (published {published['polarity']['macro_f1']})")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
