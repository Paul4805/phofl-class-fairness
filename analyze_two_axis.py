"""Analysis-only: 2-D client x class fairness report for the 4 half-scale C runs.

Support-aware: per-(client,class) test counts are loaded from the capped CIFAR
test files.  The headline worst cell uses only cells with >= 10 test samples.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
HALF = ROOT / "phofl_half"
OUT = ROOT / "phofl" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
MAT = OUT / "matrices"
MAT.mkdir(parents=True, exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SUPPORT_MIN = 10

summary = pd.read_csv(HALF / "results" / "C_half_summary.csv")
global_recall = pd.read_csv(HALF / "results" / "C_half_global_recall.csv")
summary_by_run = summary.set_index("run_id")

_counts_cache = {}
def class_counts(cid):
    if cid in _counts_cache:
        return _counts_cache[cid]
    d = np.load(HALF / "data_cap" / "Cifar10" / "test" / f"{cid}.npz",
                allow_pickle=True)["data"].tolist()
    y = np.asarray(d["y"], dtype=np.int64)
    c = np.bincount(y, minlength=10).astype(int)
    _counts_cache[cid] = c
    return c

rows = []
long_rows = []
wide_rows = []
class_rows = []
low_rows = []
support_rows = []

for _, r in summary.iterrows():
    run_id = r["run_id"]
    method = r["method"]
    pattern = r["pattern"]
    rm = pd.read_csv(HALF / "logs" / "C_half" / run_id / "round_metrics.csv")
    final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").copy()
    final = final.sort_values("client_id")
    M = final[[f"class_{k}_acc" for k in range(10)]].to_numpy(dtype=float)
    C = np.vstack([class_counts(int(cid)) for cid in final["client_id"]])
    supported = C >= SUPPORT_MIN

    mat = pd.DataFrame(M, index=final["client_id"].astype(int).to_numpy(), columns=CLASSES)
    mat.index.name = "client_id"
    mat.to_csv(MAT / f"{method}_{pattern}.csv")

    # Canonical global per-class recall for this run.
    g = global_recall[(global_recall.method == method) &
                      (global_recall.pattern == pattern)].sort_values("class_id")
    rec = g["recall"].to_numpy(dtype=float)
    best_i = int(np.nanargmax(rec))
    worst_i = int(np.nanargmin(rec))
    class_std = float(np.nanstd(rec, ddof=0))
    class_gap = float(rec[best_i] - rec[worst_i])

    # All-cell worst, and support-filtered worst.
    flat_all = np.nanargmin(M)
    wi_all, wj_all = np.unravel_index(flat_all, M.shape)
    if supported.any():
        masked = np.where(supported, M, np.nan)
        flat_sup = np.nanargmin(masked)
        wi_sup, wj_sup = np.unravel_index(flat_sup, masked.shape)
        worst_sup_recall = float(masked[wi_sup, wj_sup])
        worst_sup_client = int(final["client_id"].iloc[wi_sup])
        worst_sup_class = CLASSES[wj_sup]
    else:
        worst_sup_recall = np.nan
        worst_sup_client = -1
        worst_sup_class = ""

    n_low = int((~supported).sum())
    n_sup = int(supported.sum())
    support_rows.append({"run_id": run_id, "method": method, "pattern": pattern,
                         "total_cells": int(supported.size), "supported_cells": n_sup,
                         "low_support_cells": n_low, "support_threshold": SUPPORT_MIN})

    # Class means using only supported client-cells, for the support-filtered
    # sensitivity check.  The canonical headline uses the full global test set.
    sup_rec = np.array([
        np.nanmean(M[supported[:, j], j]) if supported[:, j].any() else np.nan
        for j in range(10)
    ], dtype=float)
    sup_best_i = int(np.nanargmax(sup_rec))
    sup_worst_i = int(np.nanargmin(sup_rec))
    sup_std = float(np.nanstd(sup_rec, ddof=0))
    sup_gap = float(sup_rec[sup_best_i] - sup_rec[sup_worst_i])

    rows.append({
        "run_id": run_id, "method": method, "pattern": pattern,
        "AM": float(summary_by_run.loc[run_id, "AM"]),
        "AM_std": float(summary_by_run.loc[run_id, "FM"]),
        "client_FM": float(r["FM"]),
        "class_std": class_std,
        "class_std_clientmean": float(np.nanstd(np.nanmean(M, axis=0), ddof=0)),
        "class_std_supported": sup_std,
        "worst_class": CLASSES[worst_i], "worst_class_recall": float(rec[worst_i]),
        "best_class": CLASSES[best_i], "best_class_recall": float(rec[best_i]),
        "class_gap": class_gap,
        "gap_over_am": class_gap / max(float(summary_by_run.loc[run_id, "AM"]), 1e-12),
        "gap_over_ceiling": class_gap / max(1.0 - float(rec[worst_i]), 1e-12),
        "supported_worst_class": CLASSES[sup_worst_i],
        "supported_worst_recall": float(sup_rec[sup_worst_i]),
        "supported_best_class": CLASSES[sup_best_i],
        "supported_best_recall": float(sup_rec[sup_best_i]),
        "supported_class_gap": sup_gap,
        "gap_over_am_supported": sup_gap / max(float(summary_by_run.loc[run_id, "AM"]), 1e-12),
        "gap_over_ceiling_supported": sup_gap / max(1.0 - float(sup_rec[sup_worst_i]), 1e-12),
        "worst_cell_recall_all": float(M[wi_all, wj_all]),
        "worst_cell_client_all": int(final["client_id"].iloc[wi_all]),
        "worst_cell_class_all": CLASSES[wj_all],
        "worst_cell_recall_supported": worst_sup_recall,
        "worst_cell_client_supported": worst_sup_client,
        "worst_cell_class_supported": worst_sup_class,
        "n_supported_cells": n_sup, "n_low_support_cells": n_low,
        "support_threshold": SUPPORT_MIN,
    })
    for i, cid in enumerate(final["client_id"].astype(int)):
        for j, cname in enumerate(CLASSES):
            long_rows.append({
                "run_id": run_id, "method": method, "pattern": pattern,
                "client_id": cid, "class_id": j, "class_name": cname,
                "recall": float(M[i, j]) if np.isfinite(M[i, j]) else np.nan,
                "n_samples": int(C[i, j]),
                "supported_ge_10": bool(C[i, j] >= SUPPORT_MIN),
            })
            if C[i, j] < SUPPORT_MIN:
                low_rows.append({
                    "run_id": run_id, "method": method, "pattern": pattern,
                    "client_id": cid, "class_id": j, "class_name": cname,
                    "recall": float(M[i, j]) if np.isfinite(M[i, j]) else np.nan,
                    "n_samples": int(C[i, j]), "excluded_from_headline": True,
                })
        wide_rows.append({
            "run_id": run_id, "method": method, "pattern": pattern,
            "client_id": cid, "client_acc": float(final["ensemble_acc"].iloc[i]),
            "AM": float(summary_by_run.loc[run_id, "AM"]),
            "AM_std": float(summary_by_run.loc[run_id, "FM"]),
            **{CLASSES[j]: (float(M[i, j]) if np.isfinite(M[i, j]) else np.nan)
               for j in range(10)},
            "client_FM": float(r["FM"]), "class_std": class_std,
            "class_std_supported": sup_std,
            "worst_class": CLASSES[worst_i], "worst_class_recall": float(rec[worst_i]),
            "best_class": CLASSES[best_i], "best_class_recall": float(rec[best_i]),
            "class_gap": class_gap,
            "gap_over_am": class_gap / max(float(summary_by_run.loc[run_id, "AM"]), 1e-12),
            "gap_over_ceiling": class_gap / max(1.0 - float(rec[worst_i]), 1e-12),
            "supported_class_gap": sup_gap,
            "gap_over_am_supported": sup_gap / max(float(summary_by_run.loc[run_id, "AM"]), 1e-12),
            "gap_over_ceiling_supported": sup_gap / max(1.0 - float(sup_rec[sup_worst_i]), 1e-12),
            "worst_cell_recall_all": float(M[wi_all, wj_all]),
            "worst_cell_recall_supported": worst_sup_recall,
            "worst_cell_client_supported": worst_sup_client,
            "worst_cell_class_supported": worst_sup_class,
            "n_supported_cells": n_sup, "n_low_support_cells": n_low,
        })
    for j, cname in enumerate(CLASSES):
        class_rows.append({
            "run_id": run_id, "method": method, "pattern": pattern,
            "class_id": j, "class_name": cname,
            "global_recall": float(rec[j]),
            "clientmean_recall": float(np.nanmean(M[:, j])),
            "supported_clientmean_recall": float(sup_rec[j]),
            "total_samples": int(C[:, j].sum()),
        })

two_axis = pd.DataFrame(rows)
two_axis.to_csv(OUT / "two_axis_run_summary.csv", index=False)
pd.DataFrame(wide_rows).to_csv(OUT / "two_axis_fairness.csv", index=False)
pd.DataFrame(long_rows).to_csv(OUT / "two_axis_client_class.csv", index=False)
pd.DataFrame(class_rows).to_csv(OUT / "two_axis_class_recall.csv", index=False)
pd.DataFrame(low_rows).to_csv(OUT / "low_support_cells.csv", index=False)
pd.DataFrame(support_rows).to_csv(OUT / "support_exclusions_summary.csv", index=False)

# Separate rankings: client-FM vs class-std. Lower is better for both.
rank = two_axis[["run_id", "method", "pattern", "client_FM", "class_std",
                 "worst_class_recall", "class_gap", "gap_over_am",
                 "gap_over_ceiling"]].copy()
rank["client_FM_rank"] = rank["client_FM"].rank(method="min", ascending=True).astype(int)
rank["class_std_rank"] = rank["class_std"].rank(method="min", ascending=True).astype(int)
rank["rank_delta"] = rank["class_std_rank"] - rank["client_FM_rank"]
rank["worst_class_rank"] = rank["worst_class_recall"].rank(method="min", ascending=False).astype(int)
rank["class_gap_rank"] = rank["class_gap"].rank(method="min", ascending=True).astype(int)
rank["gap_over_am_rank"] = rank["gap_over_am"].rank(method="min", ascending=True).astype(int)
rank["gap_over_ceiling_rank"] = rank["gap_over_ceiling"].rank(method="min", ascending=True).astype(int)
rank = rank.sort_values("client_FM_rank")
rank.to_csv(OUT / "ranking_disagreement.csv", index=False)

am_table = two_axis[["method", "pattern", "AM", "AM_std", "class_gap",
                     "gap_over_am", "gap_over_ceiling",
                     "supported_class_gap", "gap_over_am_supported",
                     "gap_over_ceiling_supported"]].copy()
am_table.to_csv(OUT / "am_normalization.csv", index=False)

# Descriptive only: n=4 is too small for an inferential rank-correlation claim.
opposite = bool((rank["rank_delta"] > 0).any() and (rank["rank_delta"] < 0).any())
pd.DataFrame([{
    "n_runs": len(rank),
    "ranking_pattern": "disagree (non-monotone; FedAPEN FM rank 3 vs class-gap rank 5)" if opposite else "not opposite",
    "inferential_test": f"none; n={len(rank)} is insufficient to test whether the order disagreement is systematic",
}]).to_csv(OUT / "ranking_summary.csv", index=False)

# Reconstruct per-class recall on the earlier smoke runs where per-client logs exist.
smoke_rows = []
for exp in ["B_agg", "C_participation"]:
    d = ROOT / "phofl" / "results" / exp
    if not d.exists():
        continue
    for jp in sorted(d.glob("*.json")):
        try:
            res = json.loads(jp.read_text(encoding="utf-8"))
            run_id = res["run_id"]
            log = ROOT / "phofl" / "logs" / exp / run_id / "round_metrics.csv"
            if not log.exists():
                continue
            rm = pd.read_csv(log)
            final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last")
            if len(final) == 0:
                final = rm[rm["round"] == rm["round"].max()].drop_duplicates("client_id")
            M = final[[f"class_{k}_acc" for k in range(10)]].to_numpy(dtype=float)
            rec = np.nanmean(M, axis=0)
            best_i = int(np.nanargmax(rec))
            worst_i = int(np.nanargmin(rec))
            smoke_rows.append({
                "exp": exp, "run_id": run_id, "method": res.get("method", ""),
                "pattern": res.get("pattern", ""), "agg_mode": res.get("agg_mode", ""),
                "n_clients": len(final), "class_std_clientmean": float(np.nanstd(rec, ddof=0)),
                "worst_class": CLASSES[worst_i], "worst_recall": float(rec[worst_i]),
                "best_class": CLASSES[best_i], "best_recall": float(rec[best_i]),
                "class_gap": float(rec[best_i] - rec[worst_i]),
            })
        except Exception as e:
            print("skip", jp, e)
pd.DataFrame(smoke_rows).to_csv(OUT / "smoke_reconstructed_class_recall.csv", index=False)

print(two_axis.to_string(index=False))
print(rank.to_string(index=False))
print(pd.DataFrame(support_rows).to_string(index=False))
print("low-support cells:", len(low_rows))
print("smoke rows:", len(smoke_rows))
