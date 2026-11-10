"""Figures for the confusion-matrix and resolution-probe analysis."""
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
CM = ROOT / "phofl" / "analysis" / "confusion_matrices"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SHORT = {
    "baseline": "baseline",
    "STEP1_rehearsal": "step 1 rehearsal",
    "STEP2_confusion": "step 2 confusion",
    "STEP3_worst_agg": "step 3 worst-agg",
    "STEP4_distill": "step 4 distill",
}
ORDER = list(SHORT)

sns.set_theme(style="whitegrid", context="paper")

# --------------------------------------------------------------------------
# 1. Full normalized matrices
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
for ax, run in zip(axes, ORDER):
    m = pd.read_csv(CM / f"{run}_normalized.csv", index_col=0)
    m.columns = CLASSES
    sns.heatmap(m, ax=ax, cmap="rocket_r", vmin=0, vmax=0.7,
                cbar=ax is axes[-1], square=True,
                xticklabels=[c[:4] for c in CLASSES],
                yticklabels=[c[:4] for c in CLASSES] if run == ORDER[0] else False,
                linewidths=0.2, linecolor="white")
    ax.set_title(SHORT[run], fontsize=10)
    ax.set_xlabel("predicted" if run == ORDER[2] else "")
    ax.set_ylabel("true" if run == ORDER[0] else "")
    ax.tick_params(labelsize=7)
    # outline the dog/cat block
    for (i, j) in [(5, 3), (3, 5)]:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                   edgecolor="#1f3864", lw=1.6))
fig.suptitle("Row-normalized 10x10 test confusion, global ensemble "
             "(boxed: dog<->cat)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_confusion_matrices.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# 2. Dog/cat error budget decomposition
# --------------------------------------------------------------------------
pair = pd.read_csv(ROOT / "phofl/analysis/confusion_pair_budget.csv")
dc = pair[pair["pair"] == "dog-cat"].set_index("run").loc[ORDER]
cross = dc["cross_error_rate"].to_numpy()
joint = dc["all_error_rate"].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=False)
labels = [SHORT[r] for r in ORDER]
x = np.arange(len(ORDER))
colors = ["#4C72B0", "#C44E52"]

ax = axes[0]
ax.bar(x - 0.2, cross, 0.4, label="dog<->cat cross error", color=colors[0])
ax.bar(x + 0.2, joint, 0.4, label="dog+cat joint class error", color=colors[1])
for i, (c, j) in enumerate(zip(cross, joint)):
    ax.text(i - 0.2, c, f"{c:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + 0.2, j, f"{j:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
ax.set_ylim(0, 0.8)
ax.set_ylabel("rate")
ax.set_title("Targeted losses: pair confusion vs joint class error")
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
d_a = dc["a_to_b_rate"].to_numpy()
d_b = dc["b_to_a_rate"].to_numpy()
ax.bar(x - 0.2, d_a, 0.4, label="dog -> cat", color="#55A868")
ax.bar(x + 0.2, d_b, 0.4, label="cat -> dog", color="#8172B2")
for i, (a, b) in enumerate(zip(d_a, d_b)):
    ax.text(i - 0.2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + 0.2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
ax.set_ylim(0, 0.3)
ax.set_title("Directional split of the pair confusion")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig_confusion_dogcat.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# 3. Resolution probe
# --------------------------------------------------------------------------
probe = pd.read_csv(ROOT / "phofl/analysis/resolution_probe_cifar.csv")
probe = probe.sort_values("resolution")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

ax = axes[0]
ax.plot(probe["resolution"], probe["AM"], marker="o", label="overall AM")
ax.plot(probe["resolution"], probe["dog_cat_cross_error_rate"], marker="s",
        label="dog<->cat cross error")
ax.plot(probe["resolution"], probe["dog_cat_all_error_rate"], marker="^",
        label="dog+cat joint error")
for _, r in probe.iterrows():
    ax.annotate(f"{r['AM']:.3f}", (r["resolution"], r["AM"]),
                textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax.annotate(f"{r['dog_cat_all_error_rate']:.3f}",
                (r["resolution"], r["dog_cat_all_error_rate"]),
                textcoords="offset points", xytext=(0, -12), fontsize=7, ha="center")
ax.set_xticks(probe["resolution"])
ax.set_ylim(0, 0.78)
ax.set_xlabel("input resolution (CIFAR-10, upsampled from 32)")
ax.set_ylabel("rate")
ax.set_title("Centralized probe: resolution vs dog/cat error")
ax.legend(fontsize=7, loc="center right")

ax = axes[1]
w = 0.35
res = probe["resolution"].to_numpy()
pos = np.arange(len(res))
ax.bar(pos - w / 2, probe["dog_recall"], w, label="dog recall", color="#4C72B0")
ax.bar(pos + w / 2, probe["cat_recall"], w, label="cat recall", color="#C44E52")
for i, (d, c) in enumerate(zip(probe["dog_recall"], probe["cat_recall"])):
    ax.text(pos[i] - w / 2, d, f"{d:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(pos[i] + w / 2, c, f"{c:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(pos, [f"{int(r)} px" for r in res])
ax.set_xlabel("input resolution")
ax.set_ylabel("recall")
ax.set_ylim(0, 0.8)
ax.set_title("Probe recalls fall as upsampled resolution rises")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig_resolution_probe.png", dpi=250, bbox_inches="tight")
plt.close(fig)

print("wrote three confusion/resolution figures to", OUT)
