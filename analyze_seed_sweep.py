"""Aggregate the three-seed class-gap sweep (and the FedFV pass) into tables.

Reads:
  phofl_half/results/C_half_s3_seed_sweep.csv
  phofl_half/results/C_half_summary.csv          (pinned-stack seed 0)
Writes:
  phofl/analysis/seed_sweep_summary.csv          mean/sd per method x pattern
  phofl/analysis/seed_sweep_class_gap.csv        per-seed rows, sorted
  phofl/analysis/seed_sweep_stack_check.csv      new stack vs pinned seed 0
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "phofl_half" / "results" / "C_half_s3_seed_sweep.csv"
PINNED = ROOT / "phofl_half" / "results" / "C_half_summary.csv"
OUT = ROOT / "phofl" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["PHPFL", "FedAPEN", "FedFV", "Standalone"]
PATTERN_ORDER = ["linear", "entropy_correlated"]
NAMES = {"PHPFL": "PHPFL", "FedAPEN": "FedAPEN", "FedFV": "FedFV",
         "Standalone": "Standalone"}
PAT = {"linear": "linear", "entropy_correlated": "entropy"}


def main():
    d = pd.read_csv(MAIN)
    d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
    d["pattern"] = pd.Categorical(d["pattern"], categories=PATTERN_ORDER, ordered=True)
    d = d.sort_values(["method", "pattern", "seed"])
    d.to_csv(OUT / "seed_sweep_class_gap.csv", index=False)

    rows = []
    for (method, pattern), g in d.groupby(["method", "pattern"], observed=True):
        rows.append({
            "method": str(method), "pattern": str(pattern), "n_seeds": len(g),
            "AM_mean": g["AM"].mean(), "AM_sd": g["AM"].std(ddof=1),
            "FM_mean": g["FM"].mean(), "FM_sd": g["FM"].std(ddof=1),
            "class_gap_mean": g["class_gap"].mean(),
            "class_gap_sd": g["class_gap"].std(ddof=1),
            "class_std_mean": g["class_std"].mean(),
            "worst_recall_mean": g["worst_class_recall"].mean(),
            "worst_class_mode": g["worst_class"].mode().iloc[0],
            "seeds": ",".join(str(int(s)) for s in sorted(g["seed"])),
        })
    s = pd.DataFrame(rows)
    s = s.sort_values(["method", "pattern"]).reset_index(drop=True)
    s.to_csv(OUT / "seed_sweep_summary.csv", index=False)

    # Stack check: seed 0 on the new stack vs the pinned-stack seed-0 runs.
    checks = []
    if PINNED.exists():
        p = pd.read_csv(PINNED)
        for _, r in d[d["seed"] == 0].iterrows():
            old = p[(p["method"] == r["method"]) & (p["pattern"] == r["pattern"])]
            if old.empty:
                continue
            o = old.iloc[0]
            checks.append({
                "method": r["method"], "pattern": r["pattern"],
                "pinned_AM": float(o["AM"]), "new_AM": float(r["AM"]),
                "d_AM": float(r["AM"]) - float(o["AM"]),
                "pinned_FM": float(o["FM"]), "new_FM": float(r["FM"]),
                "d_FM": float(r["FM"]) - float(o["FM"]),
            })
    if checks:
        pd.DataFrame(checks).to_csv(OUT / "seed_sweep_stack_check.csv", index=False)

    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))
    print()
    print(d[["method", "pattern", "seed", "AM", "FM", "class_gap",
             "worst_class", "worst_class_recall"]].round(4).to_string(index=False))
    if checks:
        print()
        print(pd.DataFrame(checks).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
