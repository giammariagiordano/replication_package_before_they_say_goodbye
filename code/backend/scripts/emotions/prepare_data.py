"""Build the training data of the emotion models: gh-emotions/{so_clean,gh_clean,aug_clean}.csv.

Run from external/ with the backend's interpreter (it reuses the pipeline's clean_text):
    PYTHONPATH=../backend python ../backend/scripts/emotions/prepare_data.py

Inputs
- emotxt-gold/Emotions_GoldSandard_andAnnotation.xlsx: Stack Overflow gold standard (Novielli et al., MSR 2018;
  github.com/collab-uniba/EmotionDatasetMSR18), one sheet per emotion over the same 4,800 posts; a post has the
  emotion when the sheet's "Gold Label" is set.
- se-emotion-src/dataset/experiment_dataset/ann-{train,test}.csv: GitHub comments (Imran et al., ASE 2022;
  github.com/vcu-swim-lab/SE-Emotion-Study), official 1,600/400 split, original text in `comment`.
- se-emotion-src/dataset/Augmented/polarity-dataset.csv: their polarity-targeted augmentation of GitHub-train
  (no header: id, text, Anger, Love, Fear, Joy, Sadness, Surprise).

Every text is cleaned with the pipeline's clean_text, as in use. Outputs: id, text, [split,] Anger ... Surprise.
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

from app.stages.text import clean_text

EMOTIONS = ["Anger", "Joy", "Love", "Sadness", "Fear", "Surprise"]
OUT = Path("gh-emotions")


def read_xlsx(path):
    """{sheet name: rows as {column letter: value}} without openpyxl (the file has only strings and numbers)."""
    z = zipfile.ZipFile(path)
    strings = [re.sub(r"<[^>]+>", "", s) for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S)]
    unescape = lambda s: s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")  # noqa: E731
    workbook = z.read("xl/workbook.xml").decode()
    rels = z.read("xl/_rels/workbook.xml.rels").decode()
    targets = dict(re.findall(r'<Relationship [^>]*?Id="([^"]+)"[^>]*?Target="([^"]+)"', rels))
    targets |= {k: v for v, k in re.findall(r'<Relationship [^>]*?Target="([^"]+)"[^>]*?Id="([^"]+)"', rels)}
    sheets = {}
    for name, rid in re.findall(r'<sheet [^>]*?name="([^"]+)"[^>]*?r:id="([^"]+)"', workbook):
        xml = z.read("xl/" + targets[rid].lstrip("/").removeprefix("xl/")).decode()
        rows = []
        for row in re.findall(r"<row [^>]*>(.*?)</row>", xml, re.S):
            cells = {}
            for col, attr, inner in re.findall(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', row, re.S):
                v = re.search(r"<v>(.*?)</v>", inner or "", re.S) or re.search(r"<t[^>]*>(.*?)</t>", inner or "", re.S)
                v = v.group(1) if v else ""
                cells[col] = unescape(strings[int(v)] if 't="s"' in attr and v else v)
            rows.append(cells)
        sheets[name] = rows
    return sheets


def stack_overflow():
    sheets = read_xlsx("emotxt-gold/Emotions_GoldSandard_andAnnotation.xlsx")
    frames = []
    for emotion in EMOTIONS:
        rows = sheets[f"{emotion}_all"]
        assert rows[0].get("D") == "Text" and rows[0].get("H") == "Gold Label", rows[0]
        # post ids restart in each annotation group (A-D) and set (Pilot, Second, Third)
        frames.append(pd.DataFrame({"id": [f'{r["A"]}-{r["B"]}-{r["C"]}' for r in rows[1:]], "raw": [r.get("D", "") for r in rows[1:]],
                                    emotion: [int(bool(r.get("H", "").strip())) for r in rows[1:]]}).set_index("id"))  # fmt: skip
    so = frames[0]
    for f in frames[1:]:
        assert so.index.equals(f.index) and (so["raw"] == f["raw"]).all(), "sheets must list the same posts"
        so[f.columns[-1]] = f[f.columns[-1]]
    assert so.index.is_unique and len(so) == 4800
    so = so.reset_index()
    so["id"] = "so-" + so["id"]
    so["text"] = so["raw"].map(clean_text)
    return so[["id", "text", *EMOTIONS]]


def github():
    base = Path("se-emotion-src/dataset/experiment_dataset")
    parts = []
    for split in ("train", "test"):
        df = pd.read_csv(base / f"ann-{split}.csv")
        df["split"] = split
        parts.append(df)
    gh = pd.concat(parts, ignore_index=True)
    assert gh["id"].is_unique and len(gh) == 2000
    gh["text"] = gh["comment"].fillna("").map(clean_text)
    return gh[["id", "text", "split", *EMOTIONS]]


def augmented():
    cols = ["id", "text", "Anger", "Love", "Fear", "Joy", "Sadness", "Surprise"]
    aug = pd.read_csv("se-emotion-src/dataset/Augmented/polarity-dataset.csv", header=None, names=cols)
    aug["text"] = aug["text"].fillna("").map(clean_text)
    return aug[["id", "text", *EMOTIONS]]


def main():
    OUT.mkdir(exist_ok=True)
    for name, df in (("so_clean", stack_overflow()), ("gh_clean", github()), ("aug_clean", augmented())):
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} texts", df[EMOTIONS].sum().to_dict())


if __name__ == "__main__":
    main()
