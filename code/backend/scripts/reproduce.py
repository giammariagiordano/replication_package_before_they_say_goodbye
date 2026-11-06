"""Reproduce the results of the paper (RQ1-RQ3 and the sensitivity analyses) from the CSV tables of the replication
package, without the database.

  pip install -r requirements.txt
  python scripts/reproduce.py ../../data         # from code/backend: writes ../../data/results.json
  python scripts/reproduce.py ../../data --main  # only the main analyses, about two minutes

The models are fitted with the same functions as the paper (app.survival.fit_cox, app.trajectories.fit_models).
The sensitivity analyses of the Threats to Validity use the tables in data/sensitivity/<variant>/.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_results import BASE, PSYCH, cox, extended, jsonable  # noqa: E402

from app import trajectories  # noqa: E402


def load(folder: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{name}.csv.gz")


def rq1(folder: Path) -> dict:
    rows = load(folder, "event_time")
    for s in trajectories.SIGNALS:  # within-project transformation, as in the paper
        rows[s] = rows[s] - rows.groupby("project")[s].transform("mean")
    rows["project"] = "all"
    return {"models": trajectories.fit_models(rows)}


def analyze(folder: Path) -> dict:
    """RQ1, RQ2 (base, full, extended model), and RQ3 (cause-specific full and extended models) on one set of tables."""
    res = {"rq1": rq1(folder)}
    iv = load(folder, "intervals_abandonment").dropna(subset=BASE + PSYCH)
    base, full = cox(iv, BASE), cox(iv, BASE + PSYCH)
    lr = 2 * (full["log_likelihood"] - base["log_likelihood"])
    res["rq2"] = {"base": base, "full": full, "community": extended(iv, full),
                  "lr_base_vs_full": {"statistic": lr, "df": len(PSYCH), "p": float(sps.chi2.sf(lr, len(PSYCH)))}}  # fmt: skip
    res["rq3"] = {}
    for name in ("left_github", "moved"):
        d = load(folder, f"intervals_{name}").dropna(subset=BASE + PSYCH)
        f = cox(d, BASE + PSYCH)
        res["rq3"][name] = {"full": f, "community": extended(d, f)}
    return res


def robustness(folder: Path) -> dict:
    """Threats to Validity: the full model on core and peripheral risk intervals; RQ3: where the movers went."""
    out = {"strata": {}}
    covs = [c for c in BASE if c != "core"] + PSYCH
    for name in ("abandonment", "left_github", "moved"):
        d = load(folder, f"intervals_{name}").dropna(subset=BASE + PSYCH)
        out["strata"][name] = {k: cox(d[m], covs) for k, m in (("core", d["core"] >= 1), ("peripheral", d["core"] < 1))}
    d = load(folder, "intervals_moved").dropna(subset=BASE + PSYCH)
    moves = d[d["event"] == 1]
    # moves with only private contributions have no public split: empty destination
    counts = moves["destination"].fillna("private_only").value_counts().to_dict()
    out["destinations"] = {"moves": len(moves), "counts": counts}
    for dest, kinds in (("other_community", {"other_owner"}), ("close", {"same_owner", "own"})):
        ev = ((d["event"] == 1) & d["destination"].isin(kinds)).astype(int)
        out["destinations"][dest] = cox(d.assign(event=ev), BASE + PSYCH)
    return out


def hr(fit: dict, covariate: str) -> str:
    c = next(c for c in fit["coefficients"] if c["covariate"] == covariate)
    return f"{c['hr']:.2f}" + ("*" if (c.get("q") or 1) < 0.05 else "")


def summary(name: str, res: dict) -> str:
    """One line per set of tables: the estimates of the sensitivity table (* = Benjamini-Hochberg q < .05)."""
    joy = next(r for r in res["rq1"]["models"] if r["signal"] == "joy_share")["interactions"][-1]
    full, ext = res["rq2"]["full"], res["rq2"]["community"]["extended"]
    cells = [f"joy beta {joy['coef']:.3f}" + ("*" if joy["q"] < 0.05 else "")]
    cells += [f"{c} {hr(full, c)}" for c in ("anger_share", "anger_received", "fear_share", "we_share_12m",
                                             "reactions_pos_received", "thanks_received")]  # fmt: skip
    cells += [f"{c} {hr(ext, c)}" for c in ("citizenship_share", "newcomer_share", "accommodation_12m")]
    cells += [f"we moved {hr(res['rq3']['moved']['full'], 'we_share_12m')}",
              f"joy left_github {hr(res['rq3']['left_github']['full'], 'joy_share')}",
              f"abandonments {full['events']}"]  # fmt: skip
    return f"{name}: " + ", ".join(cells)


def main(data: Path, main_only: bool) -> None:
    res = analyze(data)
    res.update(robustness(data))
    lines = [summary("main", res)]
    print(lines[0], file=sys.stderr)
    if not main_only and (data / "sensitivity").exists():
        res["sensitivity"] = {}
        for folder in sorted(p for p in (data / "sensitivity").iterdir() if p.is_dir()):
            res["sensitivity"][folder.name] = analyze(folder)
            lines.append(summary(folder.name, res["sensitivity"][folder.name]))
            print(lines[-1], file=sys.stderr)
    (data / "results.json").write_text(json.dumps(res, indent=1, default=jsonable))
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"), "--main" in sys.argv)
