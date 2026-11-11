"""Build a results-only compendium PDF."""
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle, HRFlowable,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG = REPORT / "figures"
OUT = REPORT / "PHPFL_All_Results.pdf"
CONTENT_W = A4[0] - 4 * cm

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
summary = pd.read_csv(ROOT / "phofl/analysis/two_axis_run_summary.csv")
rank = pd.read_csv(ROOT / "phofl/analysis/ranking_disagreement.csv")
am = pd.read_csv(ROOT / "phofl/analysis/am_normalization.csv")
pc = pd.read_csv(ROOT / "phofl/analysis/two_axis_class_recall.csv")
support = pd.read_csv(ROOT / "phofl/analysis/support_exclusions_summary.csv")
A = pd.read_csv(ROOT / "phofl/results/A_fairness.csv")
B = pd.read_csv(ROOT / "phofl/results/B_results.csv")
C = pd.read_csv(ROOT / "phofl/results/C_results.csv")
Bstats = pd.read_csv(ROOT / "phofl/results/STATS_tests.csv")
Cholm = pd.read_csv(ROOT / "phofl/results/C_holm.csv")
Cstats = pd.read_csv(ROOT / "phofl_half/results/C_half_stats_full.csv")
pinned = pd.read_csv(ROOT / "phofl/escalation/pinned_smoke_C.csv")
pinned_holm = pd.read_csv(ROOT / "phofl/escalation/pinned_smoke_holm.csv")
verdict = pd.read_csv(ROOT / "phofl/escalation/C_half_verdict.csv").iloc[0]
half_client = pd.read_csv(ROOT / "phofl_half/results/C_half_results.csv")
seed_sum = pd.read_csv(ROOT / "phofl/analysis/seed_sweep_summary.csv")
seed_sum["_o"] = seed_sum["method"].map(
    {"PHPFL": 0, "FedAPEN": 1, "FedFV": 2, "Standalone": 3})
seed_sum = seed_sum.sort_values(["_o", "pattern"]).drop(columns="_o")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SHORT = {
    ("PHPFL", "linear"): "PHPFL/lin",
    ("PHPFL", "entropy_correlated"): "PHPFL/ent",
    ("FedAPEN", "linear"): "FedAPEN/lin",
    ("Standalone", "linear"): "Standalone/lin",
    ("Standalone", "entropy_correlated"): "Standalone/ent",
}
order_methods = [
    ("PHPFL", "linear"), ("PHPFL", "entropy_correlated"),
    ("FedAPEN", "linear"), ("Standalone", "linear"),
    ("Standalone", "entropy_correlated"),
]
summary["label"] = [SHORT[(m, p)] for m, p in zip(summary.method, summary.pattern)]
summary["_o"] = [order_methods.index((m, p)) for m, p in zip(summary.method, summary.pattern)]
summary = summary.sort_values("_o").drop(columns="_o")
rank["label"] = [SHORT[(m, p)] for m, p in zip(rank.method, rank.pattern)]
am["label"] = [SHORT[(m, p)] for m, p in zip(am.method, am.pattern)]

pivot = pc.pivot_table(index=["method", "pattern"], columns="class_id",
                       values="global_recall")
pivot.index = [SHORT[(m, p)] for m, p in pivot.index]
pivot.columns = CLASSES
pivot = pivot.loc[[SHORT[(m, p)] for m, p in order_methods]].reset_index().rename(columns={"index": "run"})

steps = []
for i in range(1, 5):
    steps.append(pd.read_csv(ROOT / f"phofl_method/STEP{i}_summary.csv").iloc[0])
steps = pd.DataFrame(steps)
steps_per_class = []
for i in range(1, 5):
    d = pd.read_csv(ROOT / f"phofl_method/STEP{i}_per_class.csv")
    d.insert(0, "step", i)
    steps_per_class.append(d)
steps_per_class = pd.concat(steps_per_class, ignore_index=True)

step_client_rows = []
for i in range(1, 5):
    d = ROOT / "phofl_half/logs" / f"STEP{i}"
    # find the single run directory
    run_dir = next(p for p in d.iterdir() if p.is_dir())
    rm = pd.read_csv(run_dir / "round_metrics.csv")
    final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last")
    for _, r in final.iterrows():
        step_client_rows.append({
            "run": f"STEP{i}", "client": int(r["client_id"]), "arch": r.get("arch", ""),
            "acc": float(r["ensemble_acc"]), "rare_acc": float(r["rare_acc"]) if pd.notna(r["rare_acc"]) else "",
        })
step_client = pd.DataFrame(step_client_rows)

# ---------------------------------------------------------------------------
# Styles/helpers
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                      fontSize=9.2, leading=12.5, alignment=4, spaceAfter=4)
small = ParagraphStyle("Small", parent=body, fontSize=7.8, leading=10,
                       textColor=colors.HexColor("#444444"))
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=15, leading=18, textColor=colors.HexColor("#1f3864"),
                    spaceBefore=10, spaceAfter=6, keepWithNext=True)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=10.5, leading=13, textColor=colors.HexColor("#2f5597"),
                    spaceBefore=8, spaceAfter=3, keepWithNext=True)
title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold",
                       fontSize=25, leading=30, textColor=colors.HexColor("#1f3864"),
                       spaceAfter=8)
subtitle = ParagraphStyle("Subtitle", parent=body, fontSize=12, leading=16,
                          textColor=colors.HexColor("#3b3b3b"))
cell = ParagraphStyle("Cell", parent=body, fontSize=6.6, leading=7.8, spaceAfter=0)
cell_c = ParagraphStyle("CellC", parent=cell, alignment=1)
cell_h = ParagraphStyle("CellH", parent=cell, fontName="Helvetica-Bold",
                        textColor=colors.white)
cell_hc = ParagraphStyle("CellHC", parent=cell_h, alignment=1)
caption = ParagraphStyle("Caption", parent=small, alignment=1, spaceAfter=6)


def P(text, style=body):
    return Paragraph(text, style)


def section(text):
    return P(text, h1)


def sub(text):
    return P(text, h2)


def make_table(data, widths, header=True, align="CENTER", font=6.6):
    rows = []
    for r, row in enumerate(data):
        out = []
        for c in row:
            st = cell_hc if (header and r == 0) else (cell_c if align == "CENTER" else cell)
            out.append(Paragraph(str(c), st))
        rows.append(out)
    scale = CONTENT_W / (sum(widths) * cm)
    t = Table(rows, colWidths=[w * cm * scale for w in widths],
              repeatRows=1 if header else 0, hAlign="CENTER")
    ts = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d2e3")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864"))]
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6fb")))
    t.setStyle(TableStyle(ts))
    return t


def img(path, width=CONTENT_W * 0.92):
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def figure(path, cap, width=CONTENT_W * 0.92):
    return KeepTogether([img(path, width), P(cap, caption)])


def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d0d0d0"))
    canvas.line(2 * cm, 1.45 * cm, A4[0] - 2 * cm, 1.45 * cm)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 0.95 * cm, "PHP-FL class fairness - complete results")
    canvas.drawRightString(A4[0] - 2 * cm, 0.95 * cm, f"{doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Story
# ---------------------------------------------------------------------------
story = []
story += [Spacer(1, 2.2 * cm)]
story.append(P("PHP-FL class fairness", title))
story.append(P("Complete results compendium", subtitle))
story.append(Spacer(1, 0.3 * cm))
story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1f3864")))
story.append(Spacer(1, 0.4 * cm))
story.append(P("All committed result tables, including A/B/C, the five half-scale runs, "
               "the FedAPEN scope test, and the four-step diagnostic ladder.", body))
story.append(P("Prepared 10 November 2026", small))
story.append(P("Repository: github.com/Paul4805/phofl-class-fairness", small))
story.append(PageBreak())

# Inventory
story.append(section("Experiment inventory and status"))
inventory = [
    ["Experiment", "Status", "Key result"],
    ["Original Table-2 baseline (AM 66.94 / FM 8.07)", "Not executed", "No reproduction claim"],
    ["A: fairness metrics (Fashion-MNIST smoke)", "Complete", "FM 0.1607; tail not tail-worse"],
    ["B: aggregation reweighting (smoke)", "Complete", "Null (p 0.469-0.922)"],
    ["C: correlated participation (smoke)", "Complete", "Pattern effect loses Holm"],
    ["C pinned-stack smoke", "Complete", "Signs replicate; significance does not"],
    ["C half-scale (4 runs)", "Complete", "FAIL pre-registered escalation"],
    ["FedAPEN / linear half-scale", "Complete", "52.9-point gap; field-wide scope"],
    ["Ladder 1-4 (half-scale)", "Complete", "Max 7.9% gap reduction"],
    ["Seed-1 rerun", "Not triggered", "No step reached 30%"],
    ["FedMRL / FedFV / FedHEAL / FedAWE / FedAU", "Not run", "Out of scope"],
]
story.append(make_table(inventory, [6.5, 3.5, 7.0], align="LEFT"))
story.append(P("All half-scale numbers use 10 clients, 60 rounds, one seed, four CNNs, "
               "one local epoch, 2,500 train / 500 test per client, and the pinned stack. "
               "Absolute AM/FM are not comparable to the paper's full-scale numbers.", small))
story.append(PageBreak())

# A/B/C
story.append(section("A/B/C: Fashion-MNIST smoke results"))
Acols = ["exp", "method", "pattern", "agg_mode", "AM", "FM", "min_acc", "p10", "cvar10", "jain", "rho_p_acc"]
A2 = A[Acols].copy()
A2["exp"] = A2["exp"].map({"B_agg": "B", "C_participation": "C"})
A2["pattern"] = A2["pattern"].map({"normal": "normal", "linear": "linear", "entropy_correlated": "entropy"})
for c in ["AM", "FM", "min_acc", "p10", "cvar10", "jain", "rho_p_acc"]:
    A2[c] = A2[c].map(lambda x: f"{x:.4f}")
story.append(P("A. Fairness metrics across completed smoke runs", h2))
story.append(make_table([Acols] + A2.values.tolist(), [1.8, 2.0, 2.3, 1.8, 1.7, 1.7, 1.8, 1.8, 1.8, 1.8, 2.0], align="LEFT"))
story.append(P("A's original narrative (lowest FM but not best tail) was not confirmed: the FM winner also won min/CVaR, and rho was not positive.", small))

story.append(P("B. Aggregation reweighting", h2))
Bagg = B.groupby("agg_mode", as_index=False).agg(AM=("acc", "mean"), FM=("acc", lambda x: x.std(ddof=0)), min_acc=("acc", "min"))
Bagg = Bagg.set_index("agg_mode").loc[["plain", "inv_p", "inv_N"]].reset_index()
for c in ["AM", "FM", "min_acc"]:
    Bagg[c] = Bagg[c].map(lambda x: f"{x:.4f}")
Bstats2 = Bstats[Bstats.comparison.str.contains("plain vs inv|inv_p vs inv_N")].copy()
for c in ["statistic", "p_value", "median_diff"]:
    Bstats2[c] = Bstats2[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["agg_mode", "AM", "FM", "min"]] + Bagg.values.tolist(), [4.5, 3.8, 3.8, 3.8]))
story.append(Spacer(1, 0.15 * cm))
story.append(make_table([["comparison", "statistic", "p", "median diff"]] + Bstats2[["comparison", "statistic", "p_value", "median_diff"]].values.tolist(), [8.5, 3.0, 2.8, 3.0], align="LEFT"))

story.append(P("C. Correlated participation", h2))
Cagg = C.groupby(["pattern", "method"], as_index=False).agg(AM=("acc", "mean"), FM=("acc", lambda x: x.std(ddof=0)), min_acc=("acc", "min"))
Cagg = Cagg.sort_values(["pattern", "method"])
for c in ["AM", "FM", "min_acc"]:
    Cagg[c] = Cagg[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["pattern", "method", "AM", "FM", "min"]] + Cagg.values.tolist(), [4.5, 4.5, 3.5, 3.5, 3.5]))
story.append(Spacer(1, 0.15 * cm))
Ch = Cholm[["comparison", "raw_p", "holm_p", "loses_sig"]].copy()
for c in ["raw_p", "holm_p"]:
    Ch[c] = Ch[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["C comparison", "raw p", "Holm p", "loses sig"]] + Ch.values.tolist(), [8.5, 2.8, 2.8, 2.8], align="LEFT"))
story.append(P("The dispersion effect lost Holm significance; PHP-FL vs Standalone under entropy survived. C is closed.", small))
story.append(PageBreak())

# C escalation
story.append(section("C escalation: pre-registered thresholds and outcomes"))
v = verdict
vtab = [
    ["Quantity", "Value", "Rule"],
    ["FM increase", f"{v['FM_increase_rel']*100:+.2f}%", "GO over 25%; STOP under 10%"],
    ["Min-accuracy change", f"{v['min_drop_pts']:+.2f} pts", "GO over 5 drop; STOP under 2 drop"],
    ["Rho difference", f"{v['rho_diff']:.4f}", "GO over 1.0"],
    ["Effect vs pinned smoke", f"{v['FM_increase_rel']*100:.2f}% vs {v['pinned_smoke_FM_increase_rel']*100:.2f}%", "STOP if at most 1.1x pinned"],
    ["Decision", str(v['decision']), "STOP / GO / MARGINAL"],
]
story.append(make_table(vtab, [5.0, 5.5, 6.0], align="LEFT"))
story.append(P("Pinned-stack smoke C", h2))
pt = pinned.copy()
for c in ["AM", "FM", "min", "p10", "cvar10", "rho"]:
    pt[c] = pt[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["pattern", "method", "AM", "FM", "min", "p10", "CVaR10", "rho"]] + pt.values.tolist(), [3.4, 3.6, 2.2, 2.2, 2.2, 2.2, 2.3, 2.2]))
story.append(Spacer(1, 0.15 * cm))
ph = pinned_holm[["comparison", "raw_p", "holm_p", "loses_sig"]].copy()
for c in ["raw_p", "holm_p"]:
    ph[c] = ph[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["pinned comparison", "raw p", "Holm p", "loses sig"]] + ph.values.tolist(), [8.8, 2.8, 2.8, 2.8], align="LEFT"))
story.append(P("The pinned stack preserves the sign but not the significance of the dispersion effect. "
               "PHP-FL vs Standalone under entropy remains significant; the pattern effect does not.", small))
story.append(PageBreak())

# Five-run summary
story.append(section("Five half-scale runs: two-axis fairness"))
tbl = [["Run", "AM", "client-FM", "class_std", "class gap", "support gap", "worst class", "worst recall", "best class", "best recall"]]
for _, r in summary.iterrows():
    tbl.append([r["label"], f"{r['AM']:.4f}", f"{r['client_FM']:.4f}",
                f"{r['class_std']:.4f}", f"{r['class_gap']:.4f}",
                f"{r['supported_class_gap']:.4f}", r["worst_class"],
                f"{r['worst_class_recall']:.4f}", r["best_class"],
                f"{r['best_class_recall']:.4f}"])
story.append(make_table(tbl, [3.2, 2.0, 2.3, 2.3, 2.2, 2.4, 2.4, 2.4, 2.2, 2.2]))
story.append(P("Conservative headline: 44.3 points (PHP-FL/linear). Best client-FM run: PHP-FL/entropy at 45.6 points. FedAPEN: 52.9 points.", small))

story.append(P("Per-class recall, full CIFAR-10 test set", h2))
tbl = [["run"] + CLASSES]
for _, r in pivot.iterrows():
    tbl.append([r["run"]] + [f"{r[c]:.3f}" for c in CLASSES])
story.append(make_table(tbl, [3.4] + [1.3] * 10, font=6.0))
story.append(P("Class gap = max - min of this row. Weak region: dog/cat/deer/horse.", small))

story.append(P("Support filtering", h2))
st = support.copy()
for c in ["total_cells", "supported_cells", "low_support_cells", "support_threshold"]:
    st[c] = st[c].astype(str)
st["run"] = [SHORT[(m, p)] for m, p in zip(st.method, st.pattern)]
story.append(make_table([["run", "total cells", "supported", "low support", "threshold"]] + st[["run", "total_cells", "supported_cells", "low_support_cells", "support_threshold"]].values.tolist(), [4.0, 3.0, 3.0, 3.2, 3.0]))
story.append(P("170 low-support cells total; 45 zero-sample cells. Worst supported cell is 0.0 (client 0, cat, 11 samples).", small))

# Ranking/AM
story.append(section("Ranking disagreement and AM normalization"))
r = rank[["label", "client_FM", "client_FM_rank", "class_std", "class_std_rank", "class_gap", "class_gap_rank"]].copy()
for c in ["client_FM", "class_std", "class_gap"]:
    r[c] = r[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["run", "client-FM", "FM rank", "class_std", "std rank", "class gap", "gap rank"]] + r.values.tolist(), [3.8, 2.6, 2.2, 2.6, 2.2, 2.6, 2.2]))
story.append(P("FedAPEN is 3rd by client-FM but 5th by class gap. Standalone/linear is 4th by client-FM but 1st by class gap. n=5, observation only.", small))

am2 = am[["label", "AM", "gap_over_am", "gap_over_ceiling", "supported_class_gap", "gap_over_am_supported", "gap_over_ceiling_supported"]].copy()
for c in ["AM", "gap_over_am", "gap_over_ceiling", "supported_class_gap", "gap_over_am_supported", "gap_over_ceiling_supported"]:
    am2[c] = am2[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["run", "AM", "raw gap/AM", "raw gap/ceil", "support gap", "support gap/AM", "support gap/ceil"]] + am2.values.tolist(), [3.6, 2.2, 2.5, 2.5, 2.5, 2.7, 2.7]))
story.append(P("Standalone's raw gap advantage disappears after AM normalization (0.711 vs 0.701). The PHP-FL vs FedAPEN inversion survives (FedAPEN 3rd FM, 5th gap/AM).", small))
story.append(figure(FIG / "fig_two_axis.png", "Client-FM and class gap across the five runs."))
story.append(figure(FIG / "fig_per_class_heatmap.png", "Per-class recall matrix."))
story.append(figure(FIG / "fig_ranking_scatter.png", "Client-FM versus class gap; the ordering is not monotone."))

# Ladder
story.append(section("Diagnostic ladder: all four interventions"))
lt = [["Rung", "Intervention", "AM", "FM", "gap", "reduction", "dog", "frog", "node"]]
lt.append(["0", "PHP-FL/linear baseline", "0.6320", "0.0785", "0.443", "-", "0.251", "0.694", "-"])
names = {1: "class-balanced rehearsal", 2: "confusion-targeted loss",
         3: "worst-class aggregation", 4: "per-class distillation"}
for _, r in steps.iterrows():
    lt.append([str(int(r["step"])), names[int(r["step"])], f"{r['AM']:.4f}", f"{r['FM']:.4f}",
               f"{r['class_gap']:.4f}", f"{r['gap_reduction_rel']*100:+.1f}%",
               f"{r['dog_recall']:.4f}", f"{r['frog_recall']:.4f}",
               f"{r['worst_class']} {r['worst_class_recall']:.3f}"])
story.append(make_table(lt, [1.2, 5.8, 2.2, 2.0, 2.0, 2.0, 1.8, 1.8, 3.0], font=6.2))
story.append(P("No rung reaches the 30% closure threshold; maximum reduction 7.9%. No seed-1 rerun was triggered.", small))
story.append(figure(FIG / "fig_ladder.png", "Ladder class gap and relative reduction."))
story.append(make_table([
    ["Rung", "Observed detail"],
    ["1", "dog 0.251 -> 0.320; frog 0.694 -> 0.728; gap 0.443 -> 0.408 (7.9%)."],
    ["2", "dog->cat 0.249 -> 0.191; cat recall 0.434 -> 0.364; combined dog+cat error 0.657 -> 0.683."],
    ["3", "gamma=5.0; worst class airplane early, dog later; dog recall 0.292; gap 0.431 (2.7%)."],
    ["4", "T=1.0; best dog architecture EfficientNet-B1 0.599 -> 0.629, unchanged; gap 0.424 (4.3%)."],
], [1.5, 15.0], align="LEFT", font=6.2))
story.append(P("Step-by-step per-class results", h2))
spt = steps_per_class[["step", "class", "recall_baseline", "recall_method", "delta"]].copy()
for c in ["recall_baseline", "recall_method", "delta"]:
    spt[c] = spt[c].map(lambda x: f"{x:+.4f}")
story.append(make_table([["step", "class", "baseline", "method", "delta"]] + spt.values.tolist(), [1.5, 4.0, 3.2, 3.2, 3.2], font=6.2))
story.append(P("Step 2 zero-sum: dog->cat 0.249 -> 0.191 (-23%); cat recall 0.434 -> 0.364 (-16%); combined dog+cat error 0.657 -> 0.683 (rose).", small))
story.append(figure(FIG / "fig_step2_dogcat.png", "Step 2 shifts dog/cat errors without reducing their combined rate."))

# Stats
story.append(section("All statistical tests"))
story.append(P("Half-scale C tests (raw and Holm)", h2))
cs = Cstats[["metric", "comparison", "test", "n", "statistic", "p_raw", "p_holm", "significant_holm"]].copy()
for c in ["statistic", "p_raw", "p_holm"]:
    cs[c] = cs[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["metric", "comparison", "test", "n", "stat", "raw p", "Holm p", "Holm sig"]] + cs.values.tolist(), [3.0, 6.3, 2.8, 0.8, 1.8, 1.8, 1.8, 1.6], font=6.0))
story.append(P("B smoke tests", h2))
bs = Bstats[["metric", "comparison", "n", "statistic", "p_value", "median_diff"]].copy()
for c in ["statistic", "p_value", "median_diff"]:
    bs[c] = bs[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["metric", "comparison", "n", "stat", "p", "median diff"]] + bs.values.tolist(), [3.4, 6.5, 0.8, 2.2, 2.0, 2.0], font=6.2))
story.append(P("C smoke tests", h2))
cst = Bstats[Bstats.comparison.str.contains("Standalone|linear vs entropy")][["metric", "comparison", "n", "statistic", "p_value", "median_diff"]].copy()
for c in ["statistic", "p_value", "median_diff"]:
    cst[c] = cst[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["metric", "comparison", "n", "stat", "p", "median diff"]] + cst.values.tolist(), [3.4, 6.5, 0.8, 2.2, 2.0, 2.0], font=6.2))

# Per-client appendix
story.append(section("Appendix: per-client accuracies"))
story.append(P("Five half-scale runs", h2))
hc = half_client[["method", "pattern", "client_id", "arch", "acc", "rare_acc"]].copy()
hc["run"] = [SHORT[(m, p)] for m, p in zip(hc.method, hc.pattern)]
hc["acc"] = hc["acc"].map(lambda x: f"{x:.4f}")
hc["rare_acc"] = hc["rare_acc"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
story.append(make_table([["run", "client", "arch", "acc", "rare acc"]] + hc[["run", "client_id", "arch", "acc", "rare_acc"]].values.tolist(), [4.0, 1.5, 3.2, 2.5, 2.5]))
story.append(P("Diagnostic ladder runs", h2))
sc = step_client.copy()
sc["acc"] = sc["acc"].map(lambda x: f"{x:.4f}")
sc["rare_acc"] = sc["rare_acc"].map(lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.4f}")
story.append(make_table([["run", "client", "arch", "acc", "rare acc"]] + sc[["run", "client", "arch", "acc", "rare_acc"]].values.tolist(), [2.5, 1.5, 3.2, 2.5, 2.5]))
story.append(P("Full client x class matrices are in phofl/analysis/matrices and two_axis_client_class.csv.", small))
story.append(PageBreak())

# Confusion structure
pairb = pd.read_csv(ROOT / "phofl/analysis/confusion_pair_budget.csv")
probe = pd.read_csv(ROOT / "phofl/analysis/resolution_probe_cifar.csv")
story.append(section("Confusion structure and pair error budgets"))
story.append(P("Full 10x10 matrices, global ensemble over 10 clients, full CIFAR-10 test set", h2))
story.append(P("Cross error uses only the two off-diagonal cells; joint error is the average "
               "total error of the two classes, including errors to the other eight classes. "
               "Cross share is the pair's mutual mass over its total class error.", small))
RUN_ORDER = ["baseline", "STEP1_rehearsal", "STEP2_confusion", "STEP3_worst_agg", "STEP4_distill"]
PAIR_ORDER = ["dog-cat", "deer-horse", "automobile-truck"]
pb = pairb.set_index(["run", "pair"]).loc[[(r, p) for r in RUN_ORDER for p in PAIR_ORDER]].reset_index()
pb = pb[["run", "pair", "a_to_b_rate", "b_to_a_rate", "cross_error_rate",
         "all_error_rate", "cross_share_of_class_errors"]].copy()
for c in ["a_to_b_rate", "b_to_a_rate", "cross_error_rate", "all_error_rate",
          "cross_share_of_class_errors"]:
    pb[c] = pb[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["run", "pair", "a->b", "b->a", "cross", "joint", "cross share"]]
                        + pb.values.tolist(), [4.4, 4.4, 2.0, 2.0, 2.0, 2.0, 2.4], font=6.4))
story.append(P("Step 2 lowers dog-cat cross error 15.2% but raises joint error 3.9%; "
               "dog->bird 0.188 -> 0.213 and cat->bird 0.187 -> 0.220 absorb the mass. "
               "Step 4 gives the lowest dog+cat joint error (0.6300) with cross error up 6.1%. "
               "Step 1 halves automobile/truck directional asymmetry (0.135 -> 0.009). "
               "Deer/horse is not a two-body pair: deer->bird alone is 0.261.", small))
story.append(figure(FIG / "fig_confusion_matrices.png",
                    "Row-normalised 10x10 test confusion for baseline and the four ladder rungs."))
story.append(figure(FIG / "fig_confusion_dogcat.png",
                    "Dog/cat error budget: pair confusion falls under step 2 while joint class error rises."))
story.append(PageBreak())

story.append(section("Resolution probe (centralized)"))
story.append(P("2,000 train / 500 test images per class, one 3-conv CNN, 15 epochs, seed 0. "
               "CIFAR-10 is natively 32x32, so 64 and 96 are upsampled and add no information.", small))
pr = probe.sort_values("resolution").copy()
for c in ["AM", "dog_recall", "cat_recall", "dog_cat_cross_error_rate",
          "dog_cat_all_error_rate", "deer_horse_cross_error_rate",
          "auto_truck_cross_error_rate"]:
    pr[c] = pr[c].map(lambda x: f"{x:.4f}")
story.append(make_table([["res", "AM", "dog", "cat", "dog+cat cross", "dog+cat joint",
                          "deer+horse cross", "auto+truck cross"]]
                        + pr[["resolution", "AM", "dog_recall", "cat_recall",
                              "dog_cat_cross_error_rate", "dog_cat_all_error_rate",
                              "deer_horse_cross_error_rate", "auto_truck_cross_error_rate"]]
                        .values.tolist(), [1.4, 2.0, 1.8, 1.8, 2.6, 2.6, 2.8, 2.8], font=6.4))
story.append(figure(FIG / "fig_resolution_probe.png",
                    "Upsampled resolution degrades the probe; pair cross-confusion stays flat."))
story.append(P("Verdict", h2))
story.append(P("No confirmed resolution-limited floor and no confirmed label-ambiguity floor. "
               "The probe cannot test an information floor because 64/96 are upsampled, and it "
               "degraded as resolution rose. What the data does establish is that at the same "
               "native 32x32 input a centralized model reaches dog+cat joint error 0.455 against "
               "the federation's 0.6575, so the federated dog/cat deficit is not a missing-pixel "
               "problem. Native 96x96 STL-10 (downsampled to 64/32) is the required control. "
               "Targeted losses mainly move error between directions or out of the pair; the "
               "best joint improvement is 7.3% and no rung closes a pair.", small))
story.append(PageBreak())

# Three-seed sweep
story.append(section("Three-seed class-gap sweep"))
story.append(P("Four methods x two patterns x three seeds, one stack (torch 2.14.1), 24 runs", h2))
story.append(P("Same protocol as the half-scale runs. Class gap is max minus min per-class "
               "recall on the full CIFAR-10 test set; error bars are sample sd over seeds. "
               "The seed-0 cells were re-run on the current stack, so the three seeds are "
               "comparable; the pinned seed-0 artifacts differ by at most 0.005 AM and "
               "0.012 client-FM.", small))
st = seed_sum.copy()
st["label"] = st["method"] + " / " + st["pattern"].str.replace("_correlated", "", regex=False)
rows = []
for _, r in st.iterrows():
    rows.append([r["label"],
                 f"{r['AM_mean']:.4f} +/- {r['AM_sd']:.4f}",
                 f"{r['FM_mean']:.4f} +/- {r['FM_sd']:.4f}",
                 f"{r['class_gap_mean']:.4f} +/- {r['class_gap_sd']:.4f}",
                 f"{r['worst_recall_mean']:.3f}"])
story.append(make_table([["method / pattern", "AM", "client-FM", "class gap", "worst recall"]]
                        + rows, [5.2, 3.6, 3.6, 3.8, 2.8], font=7.0))
story.append(P("The 44.3-point seed-0 PHP-FL figure is a high draw: the same cell spans "
               "0.316-0.468. PHP-FL and Standalone are tied on the class axis (per-seed "
               "differences change sign). FedAPEN is worse than PHP-FL in 6/6 per-seed "
               "comparisons. FedFV/linear has the best mean gap and worst-class recall, but "
               "its 3.4-point margin is inside the seed noise (3/3 sign, p=0.25).",
               small))
story.append(figure(FIG / "fig_seed_gap.png",
                    "Per-seed class gap. Dots are seeds; diamonds are mean +/- sd.", CONTENT_W * 0.9))
story.append(PageBreak())

# Artifacts
story.append(section("Result artifact index"))
story.append(P("Committed result files", h2))
files = [
    ["phofl/results/A_fairness.csv", "A fairness metrics"],
    ["phofl/results/B_results.csv", "B per-client aggregation results"],
    ["phofl/results/C_results.csv", "C per-client participation results"],
    ["phofl/results/STATS_tests.csv", "A/B/C Wilcoxon tests"],
    ["phofl/results/C_holm.csv", "C raw and Holm p-values"],
    ["phofl/analysis/two_axis_run_summary.csv", "Five-run two-axis summary"],
    ["phofl/analysis/two_axis_fairness.csv", "40-row client/class table"],
    ["phofl/analysis/two_axis_client_class.csv", "500-row client-class recall + support"],
    ["phofl/analysis/two_axis_class_recall.csv", "Global per-class recall"],
    ["phofl/analysis/ranking_disagreement.csv", "FM / class-gap ranks"],
    ["phofl/analysis/am_normalization.csv", "AM-normalized gaps"],
    ["phofl/analysis/low_support_cells.csv", "170 excluded cells"],
    ["phofl_half/results/C_half_summary.csv", "Five-run summary"],
    ["phofl_half/results/C_half_stats_full.csv", "Half-scale tests + Holm"],
    ["phofl_half/results/C_half_results.csv", "50 per-client rows"],
    ["phofl_method/STEP*_summary.csv", "Ladder per-step summary"],
    ["phofl_method/STEP*_per_class.csv", "Ladder per-class deltas"],
    ["phofl_method/LADDER_SUMMARY.md", "Ladder negative result"],
    ["phofl/analysis/confusion_matrices/", "10 raw + 10 normalised 10x10 matrices"],
    ["phofl/analysis/confusion_summary.csv", "Confusion AM and dog/cat recall"],
    ["phofl/analysis/confusion_pair_budget.csv", "Directional, cross, joint, share per pair"],
    ["phofl/analysis/resolution_probe_cifar.csv", "32/64/96 centralized probe"],
    ["phofl/analysis/CONFUSION_RESOLUTION_FINDINGS.md", "Confusion and resolution analysis"],
    ["phofl_half/results/C_half_s3_seed_sweep.csv", "24-run three-seed sweep"],
    ["phofl/analysis/seed_sweep_summary.csv", "Mean/sd per method x pattern"],
    ["phofl/analysis/SEED_SWEEP_FINDINGS.md", "Three-seed findings and FedFV notes"],
]
story.append(make_table([["Path", "Contents"]] + files, [7.5, 9.5], align="LEFT", font=6.2))
story.append(P("Excluded: checkpoints, run_store, generated data, third_party, phofl_pinned, .venv-pinned. See DATA.md.", small))
story.append(Spacer(1, 0.5 * cm))
story.append(P("No method contribution is claimed. The participation hypothesis is closed. "
               "class_FM is a reporting metric, not a training objective.", body))

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    title="PHP-FL Class Fairness: Complete Results",
    author="J Paul Masillamni",
    subject="All committed results: A/B/C, five half-scale runs, FedAPEN, diagnostic ladder",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, f"{OUT.stat().st_size/2**20:.2f} MiB")
