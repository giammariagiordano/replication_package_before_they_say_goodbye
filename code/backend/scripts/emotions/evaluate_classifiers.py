"""Scores of the RoBERTa emotion models as deployed, and their predictions, for the paper and the replication package.

For each model (single model with a validation split, 5-fold ensemble, final model): F1 per emotion and micro-F1 on
the official GitHub test set of Imran et al. (ASE 2022), and macro-F1 and kappa of the derived polarity on the 7,122
comments of Novielli et al. (MSR 2020), as in validate_github.py. The predictions are written with the datasets' ids
and without texts, which the datasets do not allow to redistribute.

Run from external/ with external/toxbert-venv, after train_roberta_final.py:
    HF_HOME=$PWD/hf-cache toxbert-venv/bin/python ../backend/scripts/emotions/evaluate_classifiers.py
Outputs classifier-eval/: scores.json, <model>_gh_test.csv, <model>_msr2020.csv (probability per emotion).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_roberta import EMOTIONS, device, scores  # noqa: E402
from validate_github import polarity  # noqa: E402

OUT = Path("classifier-eval")
MODELS = {"validation_split": "roberta-emotions", "cv_ensemble": "roberta-emotions-cv", "final": "roberta-emotions-final"}
MAX_LEN = 128


@torch.no_grad()
def probabilities(model_dir: Path, texts: list[str], dev) -> np.ndarray:
    paths = sorted(model_dir.glob("fold_*")) or [model_dir]  # an ensemble averages its folds, as the runner does
    tok = AutoTokenizer.from_pretrained(paths[0])
    out = []
    for p in paths:
        model = AutoModelForSequenceClassification.from_pretrained(p).to(dev).eval()
        probs = []
        for i in range(0, len(texts), 64):
            enc = tok(texts[i : i + 64], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            probs.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
        out.append(np.concatenate(probs))
    return np.mean(out, axis=0)


def main():
    OUT.mkdir(exist_ok=True)
    dev = device()
    gh = pd.read_csv("gh-emotions/gh_clean.csv").fillna({"text": ""})
    test = gh[gh["split"] == "test"].reset_index(drop=True)
    msr = pd.read_csv("gold/github_gold.csv", sep=";")
    results = {}
    for name, folder in MODELS.items():
        model_dir = Path(folder)
        th = json.loads((model_dir / "thresholds.json").read_text())
        assert list(th) == [e.lower() for e in EMOTIONS] or list(th) == EMOTIONS, (list(th), EMOTIONS)
        t = np.array(list(th.values()))
        pt = probabilities(model_dir, test["text"].tolist(), dev)
        test_scores = scores(test[EMOTIONS].to_numpy(), (pt >= t).astype(int))
        pm = probabilities(model_dir, msr["Text"].astype(str).tolist(), dev)
        names = list(th)
        pred = [polarity([names[j] for j in np.flatnonzero(row >= t)]) for row in pm]
        rep = classification_report(msr["Polarity"], pred, labels=["positive", "neutral", "negative"], output_dict=True,
                                    zero_division=0)  # fmt: skip
        results[name] = {"thresholds": th, "gh_test": test_scores,
                         "polarity": {"n": len(msr), "macro_f1": round(rep["macro avg"]["f1-score"], 3),
                                      "kappa": round(cohen_kappa_score(msr["Polarity"], pred), 3)}}  # fmt: skip
        cols = [f"p_{e}" for e in names]
        pd.concat([test[["id"]], pd.DataFrame(pt, columns=cols)], axis=1).to_csv(OUT / f"{name}_gh_test.csv", index=False, float_format="%.5f")
        pd.concat([msr[["ID"]].rename(columns={"ID": "id"}), pd.DataFrame(pm, columns=cols)], axis=1).to_csv(
            OUT / f"{name}_msr2020.csv", index=False, float_format="%.5f")
        print(name, "test micro-F1", test_scores["micro_f1"], "polarity", results[name]["polarity"], flush=True)
    (OUT / "scores.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
