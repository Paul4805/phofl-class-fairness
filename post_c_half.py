"""Post-process the 4-run CIFAR-10 C escalation with the pre-registered tests."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path("phofl_half")
RES = ROOT / "results"
OUT = Path("phofl/escalation")
OUT.mkdir(parents=True, exist_ok=True)

client = pd.read_csv(RES / "C_half_results.csv")
summary = pd.read_csv(RES / "C_half_summary.csv")
recall = pd.read_csv(RES / "C_half_global_recall.csv")

# Main 4-run table, including the full per-class recall vector.
perclass = recall.pivot_table(index=["method", "pattern"], columns="class_id",
                              values="recall").reset_index()
perclass.columns = ["method", "pattern"] + [f"class_{i}_recall" for i in range(10)]
table = summary.merge(perclass, on=["method", "pattern"], how="left")
table["worst_class_recall"] = table[[f"class_{i}_recall" for i in range(10)]].min(axis=1)
table.to_csv(RES / "C_half_table.csv", index=False)

# Paired client vectors.
php = client[client["method"] == "PHPFL"]
lin = php[php["pattern"] == "linear"].set_index("client_id")["acc"]
ent = php[php["pattern"] == "entropy_correlated"].set_index("client_id")["acc"]
pair = pd.concat([lin, ent], axis=1, join="inner").dropna()
pair.columns = ["linear", "entropy"]

def wx(a, b):
    if len(a) < 5:
        return np.nan, np.nan, np.nan
    diff = np.asarray(b) - np.asarray(a)
    if np.allclose(diff, 0):
        return 0.0, 1.0, 0.0
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p), float(np.median(diff))

def boot_diff(x, y, stat_fn, B=10000, seed=12345):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(x), len(x))
        diffs[b] = stat_fn(x[idx]) - stat_fn(y[idx])
    lo = (np.sum(diffs <= 0) + 1) / (B + 1)
    hi = (np.sum(diffs >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi))), float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

tests = []

# Method contrasts within each pattern.
for pat in ["linear", "entropy_correlated"]:
    a = client[(client.method == "Standalone") & (client.pattern == pat)].set_index("client_id")["acc"]
    b = client[(client.method == "PHPFL") & (client.pattern == pat)].set_index("client_id")["acc"]
    z = pd.concat([a, b], axis=1, join="inner").dropna()
    stat, p, med = wx(z.iloc[:, 0], z.iloc[:, 1])
    tests.append({"metric": "accuracy", "comparison": f"PHPFL vs Standalone [{pat}]",
                  "test": "wilcoxon", "n": len(z), "statistic": stat, "p_raw": p,
                  "median_diff": med, "note": ""})

# Pattern contrasts within PHP-FL.
stat, p, med = wx(pair["linear"], pair["entropy"])
tests.append({"metric": "accuracy", "comparison": "linear vs entropy [accuracy]",
              "test": "wilcoxon", "n": len(pair), "statistic": stat, "p_raw": p,
              "median_diff": med, "note": ""})

# Dispersion.
dev_lin = (pair["linear"] - pair["linear"].mean()).abs()
dev_ent = (pair["entropy"] - pair["entropy"].mean()).abs()
stat, p, med = wx(dev_lin, dev_ent)
tests.append({"metric": "|acc - group mean|", "comparison": "linear vs entropy [dispersion]",
              "test": "wilcoxon", "n": len(pair), "statistic": stat, "p_raw": p,
              "median_diff": med, "note": "larger = more unequal"})

# Tail-shortfall analogue: per-client shortfall below its own pattern mean.
short_lin = (pair["linear"].mean() - pair["linear"]).clip(lower=0)
short_ent = (pair["entropy"].mean() - pair["entropy"]).clip(lower=0)
stat, p, med = wx(short_lin, short_ent)
tests.append({"metric": "min-accuracy tail shortfall",
              "comparison": "linear vs entropy [min tail-shortfall]",
              "test": "wilcoxon", "n": len(pair), "statistic": stat, "p_raw": p,
              "median_diff": med,
              "note": "paired shortfall analogue; min/CVaR are group scalars"})
tests.append({"metric": "CVaR10 tail shortfall",
              "comparison": "linear vs entropy [CVaR10 tail-shortfall]",
              "test": "wilcoxon", "n": len(pair), "statistic": stat, "p_raw": p,
              "median_diff": med,
              "note": "same paired shortfall; with 10 clients CVaR10 = min"})

# Direct paired bootstrap for the group-level min and CVaR10 differences.
min_p, min_mean, min_lo, min_hi = boot_diff(pair["linear"], pair["entropy"], np.min)
cvar_fn = lambda v: np.mean(np.sort(v)[:max(1, int(np.ceil(0.1 * len(v))))])
cvar_p, cvar_mean, cvar_lo, cvar_hi = boot_diff(pair["linear"], pair["entropy"], cvar_fn)
tests.append({"metric": "min-accuracy", "comparison": "linear vs entropy [min]",
              "test": "paired bootstrap", "n": len(pair), "statistic": min_mean,
              "p_raw": min_p, "median_diff": min_mean,
              "note": f"95% CI [{min_lo:.4f}, {min_hi:.4f}]; positive = drop"})
tests.append({"metric": "CVaR10", "comparison": "linear vs entropy [CVaR10]",
              "test": "paired bootstrap", "n": len(pair), "statistic": cvar_mean,
              "p_raw": cvar_p, "median_diff": cvar_mean,
              "note": f"95% CI [{cvar_lo:.4f}, {cvar_hi:.4f}]; positive = drop"})

# Advantage over Standalone.
st = client[(client.method == "Standalone") & (client.pattern == "linear")].set_index("client_id")["acc"]
adv_lin = lin - st
adv_ent = ent - st
adv = pd.concat([adv_lin, adv_ent], axis=1, join="inner").dropna()
stat, p, med = wx(adv.iloc[:, 0], adv.iloc[:, 1])
tests.append({"metric": "PHPFL - Standalone", "comparison": "linear vs entropy [advantage]",
              "test": "wilcoxon", "n": len(adv), "statistic": stat, "p_raw": p,
              "median_diff": med, "note": "positive = advantage grows under entropy"})

stats = pd.DataFrame(tests)
# Holm over the full half-scale C family.
stats["p_holm"] = np.nan
order = np.argsort(stats["p_raw"].to_numpy())
m = len(stats)
running = 0.0
for rank, idx in enumerate(order, 1):
    v = (m - rank + 1) * stats.loc[idx, "p_raw"]
    running = max(running, v)
    stats.loc[idx, "p_holm"] = min(1.0, running)
stats["significant_raw"] = stats["p_raw"] < 0.05
stats["significant_holm"] = stats["p_holm"] < 0.05
stats.to_csv(RES / "C_half_stats_full.csv", index=False)

# Pre-registered verdict (final metrics).
fm_lin = float(summary[(summary.method == "PHPFL") & (summary.pattern == "linear")]["FM"].iloc[0])
fm_ent = float(summary[(summary.method == "PHPFL") & (summary.pattern == "entropy_correlated")]["FM"].iloc[0])
rho_lin = float(summary[(summary.method == "PHPFL") & (summary.pattern == "linear")]["rho_p_acc"].iloc[0])
rho_ent = float(summary[(summary.method == "PHPFL") & (summary.pattern == "entropy_correlated")]["rho_p_acc"].iloc[0])
min_lin = float(lin.min())
min_ent = float(ent.min())
fm_rel = (fm_ent - fm_lin) / fm_lin
min_drop_pts = 100.0 * (min_lin - min_ent)
rho_diff = abs(rho_ent - rho_lin)
pinned_fm_rel = (0.174359 - 0.151070) / 0.151070
stop = (fm_rel < 0.10) or (min_drop_pts < 2.0) or (fm_rel <= 1.10 * pinned_fm_rel)
go = (fm_rel > 0.25) and (min_drop_pts > 5.0) and (rho_diff > 1.0)
decision = "FAIL" if stop else ("PASS" if go else "MARGINAL")
verdict = pd.DataFrame([{
    "FM_linear": fm_lin, "FM_entropy": fm_ent, "FM_increase_rel": fm_rel,
    "min_linear": min_lin, "min_entropy": min_ent, "min_drop_pts": min_drop_pts,
    "rho_linear": rho_lin, "rho_entropy": rho_ent, "rho_diff": rho_diff,
    "pinned_smoke_FM_increase_rel": pinned_fm_rel,
    "stop_threshold_FM_rel": 1.10 * pinned_fm_rel,
    "STOP": bool(stop), "GO": bool(go), "decision": decision,
}])
verdict.to_csv(OUT / "C_half_verdict.csv", index=False)
(OUT / "C_half_verdict.txt").write_text(decision + "\n", encoding="utf-8")
print(table.to_string(index=False))
print(stats.to_string(index=False))
print(verdict.to_string(index=False))
print(decision)
