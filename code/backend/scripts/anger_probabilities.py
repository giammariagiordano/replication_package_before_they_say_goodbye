"""Anger probability of every message labelled as anger, for the sensitivity analysis on explicit anger (paper, Threats
to Validity): the decision threshold of anger is low (thresholds.json), so a message with probability >= 0.5 is always
labelled as anger, and only those messages need their probability.

Run in a worker container (GPU when available), with the workers stopped or idle:
  docker compose run --rm --no-deps -T worker python - < backend/scripts/anger_probabilities.py
Writes /data/anger_probabilities.csv (kind = issue | comment, id, p), read by paper_results.py.
"""

import csv
import json
import subprocess
import sys
import time

from sqlalchemy import select

from app.config import settings
from app.db import session_scope
from app.models import Comment, Issue, Project
from app.stages.text import clean_text

OUT = "/data/anger_probabilities.csv"
RUNNER = """
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_dir = Path(sys.argv[1])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
anger = list(json.loads((model_dir / "thresholds.json").read_text())).index("anger")
torch.set_grad_enabled(False)
print("READY", dev, flush=True)
for line in sys.stdin:
    texts = json.loads(line)
    enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)  # as in training
    print(json.dumps(torch.sigmoid(model(**enc).logits)[:, anger].float().cpu().tolist()), flush=True)
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    with session_scope() as db:
        done = select(Project.id).where(Project.status == "done")
        rows = [("issue", i, f"{t or ''}. {b or ''}") for i, t, b in db.execute(
            select(Issue.id, Issue.title, Issue.body).where(Issue.project_id.in_(done), Issue.emotions.like("%anger%")))]
        rows += [("comment", i, b) for i, b in db.execute(
            select(Comment.id, Comment.body).where(Comment.project_id.in_(done), Comment.emotions.like("%anger%")))]  # fmt: skip
    rows = [(k, i, clean_text(t)) for k, i, t in rows]  # cleaned as by the emotions stage
    rows.sort(key=lambda r: len(r[2]))  # length-sorted batches waste less padding
    log(f"{len(rows)} messages labelled as anger")

    with open("/tmp/anger_runner.py", "w") as f:
        f.write(RUNNER)
    base = settings.external_dir
    proc = subprocess.Popen([str(base / "toxbert-venv" / "bin" / "python"), "/tmp/anger_runner.py",
                             str(base / "roberta-emotions-final")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # fmt: skip
    log(proc.stdout.readline().strip())
    with open(OUT, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["kind", "id", "p"])
        for start in range(0, len(rows), 64):
            batch = rows[start : start + 64]
            proc.stdin.write(json.dumps([t for _, _, t in batch]) + "\n")
            proc.stdin.flush()
            for (kind, i, _), p in zip(batch, json.loads(proc.stdout.readline())):
                out.writerow([kind, i, round(p, 4)])
            if start % 12800 == 0:
                log(f"{start}/{len(rows)}")
    proc.stdin.close()
    proc.wait()
    log(f"written {OUT}")


main()
