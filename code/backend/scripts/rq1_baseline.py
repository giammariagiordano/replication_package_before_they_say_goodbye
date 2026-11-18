"""RQ1 robustness to regression toward the mean: does the divergence of leavers from stayers hold once the initial
level of each signal (9-11 months before the (pseudo-)exit) is taken into account?

  python scripts/rq1_baseline.py ../../data     # from code/backend: writes ../../data/rq1_baseline.json

Three analyses, on the project-centered signals of data/event_time.csv.gz, as in the main RQ1 models:
- ANCOVA (subject level): mean signal 0-2 months before the exit ~ leaver + baseline (mean 9-11 months before) +
  log messages in both windows, with standard errors clustered by project. The leaver coefficient is the difference
  between leavers and stayers in the last window at equal baseline. The unadjusted change score (0-2 minus 9-11) is
  reported for comparison; it corresponds to the leaver x last-window interaction of the main model.
- Mixed model with baseline: the developer-months 0-8 months before the exit, with the leaver effect and the effect
  of the baseline allowed to differ by window, log messages, and a random intercept per developer x project.
- Within-developer model: the main model with developer x project fixed effects instead of the random intercept,
  so that each developer is compared only with themself; it needs no baseline window.
- Baseline-matched pairs: the main mixed model on the leaver-stayer pairs of the same matched set whose baselines
  differ by at most CALIPER standard deviations of the signal.
p-values of the last-window effect are corrected across signals with Benjamini-Hochberg.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.trajectories import OPTIMIZERS, SIGNALS  # noqa: E402

CALIPER = 0.25  # baseline-matched pairs: maximum difference, in standard deviations of the subject-level baseline
BASE_BIN, LAST_BIN = -9, 0


def mixed(formula: str, d: pd.DataFrame):
    """REML mixed model with a random intercept per subject: the converged fit with the highest likelihood."""
    best = None
    for method in OPTIMIZERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = smf.mixedlm(formula, d, groups=d["sid"]).fit(reml=True, method=method)
        except Exception:
            continue
        if f.converged and np.isfinite(f.llf) and (best is None or f.llf > best.llf):
            best = f
    return best


def term(fit, name: str) -> dict:
    return {"coef": float(fit.params[name]), "se": float(fit.bse[name]), "p": float(fit.pvalues[name])}


def analyze(rows: pd.DataFrame, s: str) -> dict:
    d = rows.dropna(subset=[s])
    w = d.groupby(["sid", "bin"]).agg(y=(s, "mean"), lm=("log_messages", "mean")).unstack("bin")
    meta = d.groupby("sid")[["leaver", "project", "set"]].first()
    subj = pd.DataFrame({"base": w[("y", BASE_BIN)], "last": w[("y", LAST_BIN)],
                         "lm_base": w[("lm", BASE_BIN)], "lm_last": w[("lm", LAST_BIN)]}).join(meta).dropna()  # fmt: skip
    out = {"signal": s, "subjects_both_windows": int(len(subj)), "leavers": int(subj["leaver"].sum())}

    # ANCOVA and unadjusted change score, clustered by project
    cl = {"groups": pd.factorize(subj["project"])[0]}
    anc = smf.ols("last ~ leaver + base + lm_base + lm_last", subj).fit(cov_type="cluster", cov_kwds=cl)
    chg = smf.ols("I(last - base) ~ leaver + lm_base + lm_last", subj).fit(cov_type="cluster", cov_kwds=cl)
    out["ancova"] = {**term(anc, "leaver"), "baseline_slope": float(anc.params["base"])}
    out["change_score"] = term(chg, "leaver")
    out["baseline_difference"] = term(smf.ols("base ~ leaver", subj).fit(cov_type="cluster", cov_kwds=cl), "leaver")

    # Mixed model on the windows after the baseline, with window-specific leaver and baseline effects
    m = d[d["bin"] > BASE_BIN].merge(subj[["base"]], left_on="sid", right_index=True)
    fit = mixed(f"{s} ~ 0 + C(bin) + C(bin):leaver + C(bin):base + log_messages", m)
    out["mixed_with_baseline"] = (
        {f"leaver_bin_{b}": term(fit, f"C(bin)[{b}]:leaver") for b in sorted(m["bin"].unique())} if fit else None
    )

    # Within-developer model: developer x project fixed effects instead of the random intercept, so that leavers and
    # stayers are compared only with themselves and the composition of the windows cannot drive the interactions
    # (few leavers write in the 9-11 window). Subjects with a single developer-month carry no within variation.
    x = d.copy()
    for b in sorted(x["bin"].unique()):
        if b != BASE_BIN:
            x[f"b{b}"], x[f"l{b}"] = (x["bin"] == b).astype(float), ((x["bin"] == b) & (x["leaver"] == 1)).astype(float)
    cols = [c for c in x.columns if c[:1] in "bl" and c[1:].lstrip("-").isdigit()] + ["log_messages"]
    x = x[x.groupby("sid")["sid"].transform("size") > 1]
    within = x[[s, *cols]] - x.groupby("sid")[[s, *cols]].transform("mean")
    fe = sm.OLS(within[s], within[cols]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["sid"])[0]})
    out["fixed_effects"] = {"subjects": int(x["sid"].nunique()), "leavers": int(x.groupby("sid")["leaver"].first().sum()),
                            **{f"leaver_bin_{b}": term(fe, f"l{b}") for b in sorted(x["bin"].unique()) if b != BASE_BIN}}  # fmt: skip

    # Main model on baseline-matched leaver-stayer pairs of the same matched set
    sd = subj["base"].std()
    keep = set()
    for _, g in subj.groupby("set"):
        lv, st = g[g["leaver"] == 1], g[g["leaver"] == 0]
        for sid, b in lv["base"].items():
            close = st.index[(st["base"] - b).abs() <= CALIPER * sd]
            if len(close):
                keep.add(sid)
                keep.update(close)
    p = d[d["sid"].isin(keep)]
    fit = mixed(f"{s} ~ C(bin, Treatment({BASE_BIN})) * leaver + log_messages", p)
    out["baseline_matched"] = {"subjects": len(keep), "leavers": int(p.groupby("sid")["leaver"].first().sum()),
                               "interaction_last": term(fit, f"C(bin, Treatment({BASE_BIN}))[T.{LAST_BIN}]:leaver") if fit else None,  # fmt: skip
                               "baseline_difference": term(smf.ols("base ~ leaver", subj.loc[sorted(keep)]).fit(), "leaver")}  # fmt: skip
    return out


def main(data: Path) -> None:
    rows = pd.read_csv(data / "event_time.csv.gz")
    for s in SIGNALS:  # within-project transformation, as in the main RQ1 models
        rows[s] = rows[s] - rows.groupby("project")[s].transform("mean")
    res = [analyze(rows, s) for s in SIGNALS]
    for key, get in (("ancova", lambda r: r["ancova"]), ("change_score", lambda r: r["change_score"]),
                     ("mixed_with_baseline", lambda r: (r["mixed_with_baseline"] or {}).get(f"leaver_bin_{LAST_BIN}")),
                     ("fixed_effects", lambda r: r["fixed_effects"][f"leaver_bin_{LAST_BIN}"]),
                     ("baseline_matched", lambda r: r["baseline_matched"]["interaction_last"])):  # fmt: skip
        idx = [i for i, r in enumerate(res) if get(r)]
        for i, q in zip(idx, false_discovery_control([get(res[i])["p"] for i in idx])):
            get(res[i])["q"] = float(q)
    (data / "rq1_baseline.json").write_text(json.dumps({"caliper_sd": CALIPER, "signals": res}, indent=1))
    for r in res:
        a, c, mx, bm = r["ancova"], r["change_score"], (r["mixed_with_baseline"] or {}).get(f"leaver_bin_{LAST_BIN}"), r["baseline_matched"]
        print(f"{r['signal']:24s} n={r['subjects_both_windows']:4d} base diff {r['baseline_difference']['coef']:+.4f} "
              f"(p {r['baseline_difference']['p']:.3f}) | change {c['coef']:+.4f} q {c['q']:.3f} | ancova {a['coef']:+.4f} "
              f"q {a['q']:.3f} slope {a['baseline_slope']:.2f} | mixed {mx['coef']:+.4f} q {mx['q']:.3f} | "
              f"FE {r['fixed_effects'][f'leaver_bin_{LAST_BIN}']['coef']:+.4f} q {r['fixed_effects'][f'leaver_bin_{LAST_BIN}']['q']:.3f} | "
              f"matched n={bm['subjects']} {bm['interaction_last']['coef']:+.4f} q {bm['interaction_last']['q']:.3f} "
              f"base diff {bm['baseline_difference']['coef']:+.4f}")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
