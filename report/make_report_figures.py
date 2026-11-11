"""Generate figures for the findings PDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

summary = pd.read_csv(ROOT / "phofl/analysis/two_axis_run_summary.csv")
rank = pd.read_csv(ROOT / "phofl/analysis/ranking_disagreement.csv")
recall = pd.read_csv(ROOT / "phofl/analysis/two_axis_class_recall.csv")
am = pd.read_csv(ROOT / "phofl/analysis/am_normalization.csv")

SHORT = {
    ("PHPFL", "linear"): "PHPFL/lin",
    ("PHPFL", "entropy_correlated"): "PHPFL/ent",
    ("FedAPEN", "linear"): "FedAPEN/lin",
    ("Standalone", "linear"): "Standalone/lin",
    ("Standalone", "entropy_correlated"): "Standalone/ent",
}
summary["label"] = [SHORT[(m, p)] for m, p in zip(summary.method, summary.pattern)]
order = summary.sort_values("client_FM")["label"].tolist()
summary["label"] = pd.Categorical(summary["label"], categories=order, ordered=True)
summary = summary.sort_values("label")

# 1. Two-axis bars
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sns.barplot(data=summary, y="label", x="client_FM", ax=axes[0], color="#4C72B0")
axes[0].set_title("Client axis: client-FM")
axes[0].set_xlabel("client-FM (lower is better)")
axes[0].set_ylabel("")
for i, v in enumerate(summary["client_FM"]):
    axes[0].text(v, i, f" {v:.3f}", va="center", fontsize=8)
sns.barplot(data=summary, y="label", x="class_gap", ax=axes[1], color="#C44E52")
axes[1].set_title("Class axis: best-to-worst gap")
axes[1].set_xlabel("class gap (lower is better)")
axes[1].set_ylabel("")
for i, v in enumerate(summary["class_gap"]):
    axes[1].text(v, i, f" {v:.3f}", va="center", fontsize=8)
fig.suptitle("The two axes rank the same runs differently")
fig.tight_layout()
fig.savefig(OUT / "fig_two_axis.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# 2. Ranking scatter
fig, ax = plt.subplots(figsize=(7, 4.4))
colors = sns.color_palette("deep", n_colors=len(summary))
offsets = {
    "PHPFL/ent": (8, 8),
    "PHPFL/lin": (8, -14),
    "FedAPEN/lin": (8, 6),
    "Standalone/lin": (8, -14),
    "Standalone/ent": (8, 10),
}
for (_, row), c in zip(summary.iterrows(), colors):
    ax.scatter(row["client_FM"], row["class_gap"], s=90, color=c)
    ax.annotate(row["label"], (row["client_FM"], row["class_gap"]),
                xytext=offsets.get(row["label"], (6, 5)),
                textcoords="offset points", fontsize=9)
ax.set_xlabel("client-FM (lower is better)")
ax.set_ylabel("class gap (lower is better)")
ax.set_title("Client-FM and class gap do not order the five runs consistently")
fig.tight_layout()
fig.savefig(OUT / "fig_ranking_scatter.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# 3. Per-class heatmap
pivot = recall.pivot_table(index=["method", "pattern"], columns="class_id",
                           values="global_recall")
pivot.index = [SHORT[(m, p)] for m, p in pivot.index]
pivot.columns = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
order2 = [SHORT[(m, p)] for m, p in zip(summary.method, summary.pattern)]
pivot = pivot.loc[order2]
fig, ax = plt.subplots(figsize=(10, 3.6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            cbar_kws={"label": "recall"}, ax=ax)
ax.set_title("Per-class recall on the full CIFAR-10 test set")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(OUT / "fig_per_class_heatmap.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# 4. Ladder
steps = pd.read_csv(ROOT / "phofl_method/STEP1_summary.csv")
for i in [2, 3, 4]:
    steps = pd.concat([steps, pd.read_csv(ROOT / f"phofl_method/STEP{i}_summary.csv")], ignore_index=True)
base = pd.DataFrame([{"step": 0, "class_gap": 0.443, "gap_reduction_rel": 0.0}])
steps = pd.concat([base, steps[["step", "class_gap", "gap_reduction_rel"]]], ignore_index=True)
labels = ["baseline", "1 rehearsal", "2 confusion", "3 aggregation", "4 distillation"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
sns.barplot(data=steps, x="step", y="class_gap", ax=axes[0], color="#55A868")
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(labels, rotation=15, ha="right")
axes[0].set_title("Class gap by ladder rung")
axes[0].set_ylabel("class gap")
axes[0].set_xlabel("")
for i, v in enumerate(steps["class_gap"]):
    axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
sns.barplot(data=steps, x="step", y="gap_reduction_rel", ax=axes[1], color="#DD8452")
axes[1].set_xticks(range(5)); axes[1].set_xticklabels(labels, rotation=15, ha="right")
axes[1].set_title("Relative gap reduction")
axes[1].set_ylabel("relative reduction")
axes[1].set_xlabel("")
axes[1].axhline(0.30, ls="--", color="crimson", label="30% closure threshold")
axes[1].legend(loc="upper right", fontsize=7)
for i, v in enumerate(steps["gap_reduction_rel"]):
    axes[1].text(i, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)
fig.suptitle("No ladder rung closes the class gap")
fig.tight_layout()
fig.savefig(OUT / "fig_ladder.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# 5. Step 2 zero-sum
fig, ax = plt.subplots(figsize=(6.2, 3.6))
cats = ["dog error", "cat error", "combined"]
base_vals = [1 - 0.251, 1 - 0.434, ((1 - 0.251) + (1 - 0.434)) / 2]
meth_vals = [1 - 0.270, 1 - 0.364, ((1 - 0.270) + (1 - 0.364)) / 2]
x = np.arange(len(cats)); w = 0.35
ax.bar(x - w/2, base_vals, w, label="baseline", color="#4C72B0")
ax.bar(x + w/2, meth_vals, w, label="Step 2", color="#C44E52")
for i, (a, b) in enumerate(zip(base_vals, meth_vals)):
    ax.text(i - w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylim(0, 0.9)
ax.set_ylabel("error rate")
ax.set_title("Step 2 moves dog/cat errors but does not reduce their sum")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_step2_dogcat.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# 6. AM normalization
am["label"] = [SHORT[(m, p)] for m, p in zip(am.method, am.pattern)]
am = am.set_index("label").loc[order2].reset_index()
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(am)); w = 0.35
ax.bar(x - w/2, am["gap_over_am"], w, label="raw gap / AM", color="#4C72B0")
ax.bar(x + w/2, am["gap_over_am_supported"], w, label="support gap / AM", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(am["label"], rotation=15, ha="right")
ax.set_ylabel("gap / AM")
ax.set_title("AM normalization removes the raw Standalone advantage")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_am_norm.png", dpi=250, bbox_inches="tight")
plt.close(fig)

print("wrote figures to", OUT)

# 6. Three-seed class gap per cell
seed_rows = pd.read_csv(ROOT / "phofl/analysis/seed_sweep_class_gap.csv")
seed_sum = pd.read_csv(ROOT / "phofl/analysis/seed_sweep_summary.csv")
METHOD_ORDER = ["FedFV", "PHPFL", "Standalone", "FedAPEN"]
PAT = {"linear": "linear", "entropy_correlated": "entropy"}
COLORS = {"PHPFL": "#4C72B0", "FedAPEN": "#C44E52",
          "FedFV": "#55A868", "Standalone": "#8172B2"}
seed_sum["label"] = [f"{m} / {PAT[p]}" for m, p in
                     zip(seed_sum["method"], seed_sum["pattern"])]
seed_sum = seed_sum.sort_values("class_gap_mean").reset_index(drop=True)
order = seed_sum["label"].tolist()
seed_rows["label"] = [f"{m} / {PAT[p]}" for m, p in
                      zip(seed_rows["method"], seed_rows["pattern"])]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
for i, row in seed_sum.iterrows():
    y = len(order) - 1 - i
    pts = seed_rows[seed_rows["label"] == row["label"]]["class_gap"]
    ax.scatter(pts, [y] * len(pts), s=22, color=COLORS[row["method"]],
               alpha=0.75, zorder=3, label=None)
    ax.errorbar(row["class_gap_mean"], y, xerr=row["class_gap_sd"], fmt="D",
                color=COLORS[row["method"]], capsize=4, lw=1.6, ms=5, zorder=4)
    ax.text(0.735, y, f"{row['class_gap_mean']:.3f} +/- {row['class_gap_sd']:.3f}",
            va="center", fontsize=7.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order[::-1], fontsize=8)
ax.set_xlim(0.24, 0.90)
ax.set_xlabel("class gap (max - min per-class recall)")
ax.set_title("Three seeds per cell: dots are seeds, diamonds are mean +/- sd")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m)
           for m, c in COLORS.items()]
ax.legend(handles=handles, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig_seed_gap.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "fig_seed_gap.png")
