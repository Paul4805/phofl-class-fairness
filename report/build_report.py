"""Build the detailed findings PDF for the class-fairness project."""
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle, HRFlowable,
)
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG = REPORT / "figures"
OUT = REPORT / "PHPFL_Class_Fairness_Report.pdf"
CONTENT_W = A4[0] - 4 * cm

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
summary = pd.read_csv(ROOT / "phofl/analysis/two_axis_run_summary.csv")
rank = pd.read_csv(ROOT / "phofl/analysis/ranking_disagreement.csv")
am = pd.read_csv(ROOT / "phofl/analysis/am_normalization.csv")
perclass = pd.read_csv(ROOT / "phofl/analysis/two_axis_class_recall.csv")
support = pd.read_csv(ROOT / "phofl/analysis/support_exclusions_summary.csv")
steps = []
for i in range(1, 5):
    steps.append(pd.read_csv(ROOT / f"phofl_method/STEP{i}_summary.csv").iloc[0])
steps = pd.DataFrame(steps)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SHORT = {
    ("PHPFL", "linear"): "PHPFL/lin",
    ("PHPFL", "entropy_correlated"): "PHPFL/ent",
    ("FedAPEN", "linear"): "FedAPEN/lin",
    ("Standalone", "linear"): "Standalone/lin",
    ("Standalone", "entropy_correlated"): "Standalone/ent",
}
summary["label"] = [SHORT[(m, p)] for m, p in zip(summary.method, summary.pattern)]
summary = summary.sort_values("client_FM")
rank["label"] = [SHORT[(m, p)] for m, p in zip(rank.method, rank.pattern)]
am["label"] = [SHORT[(m, p)] for m, p in zip(am.method, am.pattern)]

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
body = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4,
    leading=13.2, textColor=colors.HexColor("#1b1b1b"), alignment=TA_JUSTIFY,
    spaceAfter=5,
)
body_left = ParagraphStyle("BodyLeft", parent=body, alignment=TA_LEFT)
small = ParagraphStyle("Small", parent=body, fontSize=8.2, leading=10.8,
                       textColor=colors.HexColor("#444444"))
caption = ParagraphStyle("Caption", parent=small, alignment=TA_CENTER,
                         textColor=colors.HexColor("#555555"), spaceBefore=2,
                         spaceAfter=8)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=15.5, leading=19, textColor=colors.HexColor("#1f3864"),
                    spaceBefore=12, spaceAfter=7, keepWithNext=True)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11.5, leading=14, textColor=colors.HexColor("#2f5597"),
                    spaceBefore=10, spaceAfter=4, keepWithNext=True)
title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold",
                       fontSize=26, leading=31, textColor=colors.HexColor("#1f3864"),
                       alignment=TA_LEFT, spaceAfter=8)
subtitle = ParagraphStyle("Subtitle", parent=body, fontSize=12.5, leading=17,
                          textColor=colors.HexColor("#3b3b3b"), alignment=TA_LEFT)
cover_meta = ParagraphStyle("CoverMeta", parent=small, fontSize=9.2, leading=13,
                            textColor=colors.HexColor("#555555"))
cell = ParagraphStyle("Cell", parent=body, fontSize=7.1, leading=8.7,
                      alignment=TA_LEFT, spaceAfter=0)
cell_c = ParagraphStyle("CellC", parent=cell, alignment=TA_CENTER)
cell_h = ParagraphStyle("CellH", parent=cell, fontName="Helvetica-Bold",
                        textColor=colors.white)
cell_hc = ParagraphStyle("CellHC", parent=cell_h, alignment=TA_CENTER)
box_style = ParagraphStyle("Box", parent=body, fontSize=9.0, leading=12.5,
                           alignment=TA_LEFT, spaceAfter=0)
code = ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=8.0,
                      leading=10, alignment=TA_LEFT, backColor=colors.HexColor("#f4f4f4"))


def P(text, style=body):
    return Paragraph(text, style)


def section(number, text):
    return P(f"{number}. {text}", h1)


def subsection(text):
    return P(text, h2)


def bullet(text):
    return Paragraph(f"- {text}", ParagraphStyle(
        "Bullet", parent=body, leftIndent=12, firstLineIndent=-8, spaceAfter=2))


def make_table(data, widths, header=True, font=7.1, align="CENTER"):
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
    style = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c9d2e3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    for i in range(1 if header else 0, len(rows)):
        if (i % 2) == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6fb")))
    t.setStyle(TableStyle(style))
    return t


def img(path, width=CONTENT_W * 0.92):
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def figure(path, cap, width=CONTENT_W * 0.92):
    return KeepTogether([img(path, width), P(cap, caption)])


def box(title_text, lines):
    content = [P(f"<b>{title_text}</b>", ParagraphStyle(
        "BoxTitle", parent=body, fontSize=10, leading=13,
        textColor=colors.HexColor("#1f3864"), spaceAfter=4))]
    content += [P(x, box_style) for x in lines]
    t = Table([[content]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3fa")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9fb3d1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d0d0d0"))
    canvas.setLineWidth(0.4)
    canvas.line(2 * cm, 1.45 * cm, A4[0] - 2 * cm, 1.45 * cm)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 0.95 * cm, "Class fairness in model-heterogeneous federated learning - findings report")
    canvas.drawRightString(A4[0] - 2 * cm, 0.95 * cm, f"{doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Story
# ---------------------------------------------------------------------------
story = []

# Cover
story += [Spacer(1, 2.0 * cm)]
story.append(P("Class fairness in<br/>model-heterogeneous<br/>federated learning", title))
story.append(P("A measurement study, a diagnostic ladder, and a research roadmap", subtitle))
story.append(Spacer(1, 0.4 * cm))
story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1f3864")))
story.append(Spacer(1, 0.5 * cm))
story.append(P("Prepared 10 November 2026", cover_meta))
story.append(P("Repository: github.com/Paul4805/phofl-class-fairness", cover_meta))
story.append(P("Artifacts: phofl/analysis, phofl/paper, phofl_method, phofl_half", cover_meta))
story.append(Spacer(1, 1.1 * cm))
story.append(box("The short version", [
    "Client-FM - the usual standard deviation of per-client accuracy - hides a <b>44.3-point</b> best-to-worst class-recall spread in the PHP-FL/linear run.",
    "FedAPEN/linear has a <b>52.9-point</b> gap, so the observation is field-wide across the two MH-FL methods tested, not universal across all FL.",
    "A four-rung diagnostic ladder (rehearsal, confusion loss, worst-class aggregation, per-class distillation) reduced the gap by at most <b>7.9%</b> over one seed.",
    "Step 2 is the clearest mechanism: dog-to-cat errors fell 23%, but cat recall fell 16%, and the combined dog+cat error rose 0.657 -> 0.683.",
    "class_FM is a <b>reporting metric</b>, not a training objective. No method contribution is claimed.",
]))
story.append(PageBreak())

# Contents
story.append(P("Contents", h1))
contents = [
    ("1", "The PHP-FL paper in brief"),
    ("2", "Project arc: what was planned, what was run"),
    ("3", "Experimental setup and metric definitions"),
    ("4", "First measurement phase: A/B/C"),
    ("5", "Two-axis fairness on CIFAR-10"),
    ("6", "The diagnostic ladder"),
    ("7", "What is interesting"),
    ("8", "Limitations and what is not established"),
    ("9", "Future directions"),
    ("10", "Reproducibility and artifact map"),
]
for n, t in contents:
    story.append(P(f"<b>{n}.</b>&nbsp;&nbsp;{t}", ParagraphStyle(
        "Toc", parent=body, fontSize=9.8, leading=15, spaceAfter=1)))
story.append(PageBreak())

# Executive summary
story.append(P("Executive summary", h1))
story.append(P(
    "This report documents a measurement project that started from the PHP-FL paper and "
    "ended with a two-axis fairness critique plus a negative-result diagnostic ladder. "
    "The central claim is simple: a single client-level dispersion number is not a complete "
    "fairness report for model-heterogeneous federated learning (MH-FL). A model can look "
    "well balanced across clients while leaving a large semantic class spread."
))
story.append(P(
    "On the half-scale CIFAR-10 setup, PHP-FL/linear has client-FM 0.0785 and a class gap "
    "of 44.3 points: frog recall is 69.4% while dog recall is 25.1%. The single best "
    "client-FM run (PHP-FL/entropy, FM=0.0763) is slightly worse on the class axis at 45.6 "
    "points. FedAPEN/linear reaches 52.9 points. Because FedAPEN crosses the pre-registered "
    "30-point scope threshold, the finding is field-wide across the two MH-FL methods in "
    "this study, but it is not a universal claim about all FL."
))
story.append(P(
    "The five runs also order differently on the two axes: FedAPEN is 3rd by client-FM but "
    "5th by class gap, while Standalone/linear is 4th by client-FM but 1st by class gap. "
    "This is an observation on five runs, not a rank-correlation claim."
))
story.append(P(
    "The diagnostic ladder was designed to test whether the gap is reducible. Four standard "
    "interventions were run: class-balanced rehearsal, confusion-targeted loss, "
    "worst-class-weighted aggregation, and per-class distillation. The largest reduction was "
    "7.9%, from rehearsal. Step 2 is the most informative: it moved dog-to-cat errors in the "
    "intended direction but shifted the error into cat, leaving the combined dog+cat error "
    "slightly higher. This is consistent with an information-theoretic floor at CIFAR-10 "
    "resolution and argues for class_FM as a reporting metric, not an optimization target."
))
story.append(PageBreak())

# 1. PHP-FL summary
story.append(section(1, "The PHP-FL paper in brief"))
story.append(P(
    "PHP-FL addresses fairness in model-heterogeneous federated learning when clients have "
    "different participation probabilities. The setting is deliberately heterogeneous: four "
    "CNN architectures are assigned round-robin to 20 clients, so there is no single global "
    "model that can simply be averaged across all clients. The paper's fairness target is "
    "client-level: reduce the dispersion of per-client test accuracy under participation "
    "inconsistency."
))
story.append(subsection("1.1 Method components"))
story.append(bullet(
    "<b>Private and auxiliary models.</b> Each client has a private model w_i and a local "
    "copy g_i of a shared auxiliary model. The server maintains the global auxiliary model G. "
    "The private model carries personalization; the auxiliary model carries shared knowledge."
))
story.append(bullet(
    "<b>DEAL: dual-end alignment.</b> The local objective combines cross-entropy with two "
    "alignment terms: an MMD term between the private and shared backbone features, and a KL "
    "term between their logits. An ensemble loss mixes the two predictions. The intent is to "
    "align representations and outputs at both ends while preserving personalization."
))
story.append(bullet(
    "<b>ISPU: individual-specific parameter update.</b> Each client computes a participation "
    "ratio alpha from its cumulative participation count N_i(t). The top-alpha parameters of "
    "its auxiliary model, ranked by L1 norm, are overwritten with the corresponding global "
    "parameters. In the repository the schedule is "
    "<font name='Courier'>alpha = tau * sigmoid(k, (N_i(t)+1)/(t+1))</font> with "
    "<font name='Courier'>sigmoid(k,x)=1/(1+exp(k(x-0.5)))</font>. Rare clients therefore "
    "receive a larger alpha and more global knowledge; frequent clients receive less."
))
story.append(bullet(
    "<b>Server aggregation.</b> Active clients upload their auxiliary models; the server "
    "takes a plain mean over active clients; active clients then apply their ISPU masks. The "
    "private models are not aggregated."
))
story.append(bullet(
    "<b>Evaluation.</b> AM is the mean per-client accuracy; FM is the standard deviation of "
    "per-client accuracy. Participation patterns are uniform, normal(mu, sigma), and a "
    "shuffled linear/arithmetic schedule."
))
story.append(subsection("1.2 What the paper claims"))
story.append(P(
    "The paper claims that the combination of dual-end alignment and individual-specific "
    "updates improves client fairness under participation-probability inconsistency in "
    "heterogeneous environments. The primary evidence is the FM/AM table on CIFAR-10 and "
    "related datasets. The method is evaluated as a client-fairness method; class-level "
    "recall is not part of the reported metric."
))
story.append(subsection("1.3 What this project tests"))
story.append(P(
    "This project does not re-derive or replace PHP-FL. It asks a complementary measurement "
    "question: if a method improves client-FM, does it also improve the class axis? The "
    "repository's PHP-FL code is used as the reference implementation; the class-fairness "
    "analysis is layered on top of the same training and evaluation logs."
))
story.append(PageBreak())

# 2. Project arc
story.append(section(2, "Project arc: what was planned, what was run"))
story.append(P(
    "The project began with a much larger plan: reproduce the paper's CIFAR-10 normal-pattern "
    "baseline (AM 66.94, FM 8.07), then run aggregation, participation, cold-start, "
    "convergence, efficiency, membership-inference, and baseline-porting experiments. The "
    "scope was reduced before the full baseline matrix ran. As a result, this repository "
    "does <b>not</b> contain a successful reproduction of Table 2, and no claim of exact "
    "paper reproduction is made."
))
story.append(P(
    "The work that was actually completed followed this arc:"
))
story.append(bullet(
    "<b>Phase A/B/C (Fashion-MNIST smoke).</b> Fairness-metric recomputation, aggregation "
    "reweighting, and correlated-participation comparisons on 10 clients, 40 rounds, two "
    "backbones. These are diagnostic smoke runs, not CIFAR-10 claims."
))
story.append(bullet(
    "<b>Participation branch closed.</b> The C escalation was pre-registered, corrected with "
    "Holm, and repeated on the pinned stack. The linear-vs-entropy pattern effect did not "
    "survive correction, and the half-scale run failed the pre-registered thresholds. The "
    "participation hypothesis is closed and is not revived."
))
story.append(bullet(
    "<b>Class-fairness pivot.</b> The analysis moved to the two-axis object: client x class. "
    "Five CIFAR-10 half-scale runs were analyzed: PHP-FL linear/entropy, FedAPEN linear, and "
    "Standalone linear/entropy."
))
story.append(bullet(
    "<b>FedAPEN scope test.</b> FedAPEN/linear was added as the second MH-FL method. Its "
    "52.9-point class gap promoted the finding to field-wide across the two methods tested."
))
story.append(bullet(
    "<b>Diagnostic ladder.</b> Four interventions were tested one at a time to see whether "
    "the class gap is reducible. None closed it; the maximum reduction was 7.9%."
))
story.append(P(
    "The most important consequence of the arc is methodological: the project changed from "
    "trying to build a better method to measuring whether the reported fairness metric "
    "captures the failure it claims to address. That change is the paper."
))
story.append(PageBreak())

# 3. Setup and metrics
story.append(section(3, "Experimental setup and metric definitions"))
story.append(subsection("3.1 Half-scale CIFAR-10 setup"))
story.append(make_table([
    ["Item", "Value"],
    ["Dataset", "CIFAR-10, 10 clients, Dirichlet(alpha=0.5)"],
    ["Per-client data", "2,500 train / 500 test"],
    ["Training", "60 rounds, 1 local epoch, batch 64 (32 for heavy backbones)"],
    ["Models", "Four CNNs: GoogLeNet, DenseNet-121, EfficientNet-B1, ResNet-18"],
    ["Auxiliary", "GoogLeNet, plain mean aggregation"],
    ["Seed", "0 (one seed)"],
    ["Stack", "torch 2.2.2+cu121, numpy 1.26.4, pinned"],
    ["Evaluation", "Full 1,000-sample CIFAR-10 test set per class"],
], [4.5, 11.5], align="LEFT"))
story.append(Spacer(1, 0.25 * cm))
story.append(P(
    "This is a compute-bounded half-scale setting: roughly 8x less training than the paper's "
    "full CIFAR-10 setting. Absolute AM/FM are not comparable to the paper. The linear-vs-"
    "entropy contrast and the class-axis geometry are the usable outputs."
))
story.append(subsection("3.2 Metrics"))
story.append(bullet(
    "<b>AM:</b> mean per-client accuracy. <b>FM:</b> population standard deviation of "
    "per-client accuracy. In the artifacts, <font name='Courier'>AM_std</font> is numerically "
    "FM, not a seed-level standard error."
))
story.append(bullet(
    "<b>Class gap:</b> best per-class recall minus worst per-class recall on the full test "
    "set. <b>class_std:</b> population standard deviation of the 10 per-class recalls."
))
story.append(bullet(
    "<b>Worst-class recall:</b> minimum per-class recall. This is the tail metric that a "
    "client-level average can hide."
))
story.append(bullet(
    "<b>Support filter:</b> per-(client, class) cells with fewer than 10 test samples are "
    "excluded from worst-cell and support-filtered calculations. Each run has 34/100 such "
    "cells (170 across five runs; 45 have zero samples). The global class headline uses the "
    "full 1,000 samples per class and is not affected."
))
story.append(bullet(
    "<b>class_FM:</b> standard deviation over classes of per-class mean recall. It is the "
    "class-axis analogue of client-FM and is proposed as a reporting metric, not a training "
    "objective."
))
story.append(bullet(
    "<b>AM-normalized gap:</b> class gap divided by AM. This asks whether a smaller raw gap "
    "is simply a lower-accuracy effect. The ceiling-normalized variant divides by "
    "1 - worst-class recall."
))
story.append(PageBreak())

# 4. A/B/C
story.append(section(4, "First measurement phase: A/B/C"))
story.append(P(
    "The first phase was a Fashion-MNIST smoke study: 10 clients, 40 rounds, 1 seed, "
    "GoogLeNet and ResNet-18, 1,000 train / 200 test per client. It was designed to test the "
    "measurement pipeline and the participation branch, not to make CIFAR-10 claims."
))
story.append(subsection("4.1 A: fairness is more than std"))
story.append(make_table([
    ["Metric", "Plain PHP-FL baseline"],
    ["AM", "0.5175"],
    ["FM", "0.1607"],
    ["min accuracy", "0.2400"],
    ["10th percentile", "0.2985"],
    ["CVaR10", "0.2400"],
    ["CV", "0.3105"],
    ["Jain", "0.9120"],
    ["Spearman rho(p, accuracy)", "-0.0061"],
], [7.0, 8.0], align="LEFT"))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "The original A narrative expected PHP-FL to have the lowest FM but not the best tail. "
    "In this smoke baseline the FM winner also won min/CVaR, and rho was not positive. The "
    "narrative was not confirmed on this run. This is a negative result about the baseline, "
    "not about the CIFAR-10 paper."
))
story.append(subsection("4.2 B: aggregation reweighting was a null result"))
story.append(make_table([
    ["Aggregation", "AM", "FM", "min"],
    ["plain", "0.5175", "0.1607", "0.240"],
    ["inv_p", "0.5160", "0.1628", "0.260"],
    ["inv_N", "0.5105", "0.1615", "0.260"],
], [4.5, 3.8, 3.8, 3.8]))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "The tail minimum improved by about 2 points under inv_p and inv_N, but the paired "
    "Wilcoxon tests on client accuracy were null: plain vs inv_p p=0.922, plain vs inv_N "
    "p=0.469, inv_p vs inv_N p=0.570. The aggregation branch was written up as a null result "
    "and moved past."
))
story.append(subsection("4.3 C: correlated participation and the closed branch"))
story.append(make_table([
    ["Pattern", "Method", "AM", "FM", "min", "rho(p,acc)"],
    ["linear", "PHPFL", "0.5040", "0.1439", "0.265", "-0.758"],
    ["entropy", "PHPFL", "0.5120", "0.1805", "0.175", "+0.721"],
    ["linear", "Standalone", "0.4605", "0.1856", "0.100", "-0.794"],
], [3.4, 4.0, 3.0, 3.0, 3.0, 3.4]))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "The entropy-correlated sampler raised mean accuracy slightly but increased dispersion "
    "and collapsed the tail. The raw linear-vs-entropy dispersion test was p=0.041; after "
    "Holm correction it became p=0.164 and lost significance. PHP-FL vs Standalone under "
    "entropy survived Holm (p=0.0098), but that is a method contrast, not a pattern effect. "
    "The pinned-stack smoke run preserved the sign but not the significance of the dispersion "
    "effect."
))
story.append(P(
    "The C escalation then ran at CIFAR-10 10 clients / 60 rounds. It failed the pre-"
    "registered thresholds: FM increase -2.84% (GO required >25%), min accuracy +4.09 points "
    "(GO required a >5-point drop), rho difference 0.170 (GO required >1.0). C is closed. "
    "This report does not revive the participation hypothesis."
))
story.append(figure(ROOT / "phofl/figures/C_participation_tail.png",
                    "Fashion-MNIST smoke: entropy-correlated participation collapses the tail. "
                    "This is a smoke diagnostic, not a CIFAR-10 result.", CONTENT_W * 0.8))

# 5. Two-axis
story.append(section(5, "Two-axis fairness on CIFAR-10"))
story.append(P(
    "The central measurement result is the client-by-class matrix. Client-FM is the row-mean "
    "dispersion; class_std and worst-class recall are the column summaries. The five-run "
    "table below is the core evidence."
))
tbl = [["Run", "AM", "client-FM", "class_std", "class gap", "worst class", "worst recall"]]
for _, r in summary.iterrows():
    tbl.append([r["label"], f"{r['AM']:.4f}", f"{r['client_FM']:.4f}",
                f"{r['class_std']:.4f}", f"{r['class_gap']:.4f}",
                r["worst_class"], f"{r['worst_class_recall']:.4f}"])
story.append(make_table(tbl, [3.5, 2.5, 3.0, 3.0, 3.0, 3.2, 3.2]))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "The conservative headline is 44.3 points in PHP-FL/linear: frog 69.4%, dog 25.1%. The "
    "single best client-FM run is PHP-FL/entropy (0.0763) at 45.6 points. FedAPEN/linear is "
    "52.9 points. The failure is not a single-class artifact: dog, cat, deer, and horse form "
    "a broad weak region."
))
story.append(figure(FIG / "fig_two_axis.png",
                    "Client-FM and class gap on the same five runs. The bars are ordered by "
                    "client-FM on the left; the class axis on the right does not follow that order."))
story.append(figure(FIG / "fig_per_class_heatmap.png",
                    "Per-class recall. Rows are runs; columns are classes. The weak region is "
                    "dog/cat/deer/horse, not one isolated cell.", CONTENT_W * 0.95))
story.append(subsection("5.1 Support filtering"))
story.append(P(
    "Per-client class cells are sparse: 34/100 cells per run have fewer than 10 test samples, "
    "170 cells across five runs, including 45 zero-sample cells. These are listed in "
    "<font name='Courier'>low_support_cells.csv</font> and excluded from the worst-cell "
    "calculation. The worst supported cell is still 0.0 (client 0, cat, 11 samples) in every "
    "run. It is sparse but above threshold and is not the headline. Computing class means "
    "only from supported cells gives a 51.3-point gap for PHP-FL/linear, a 45.0-point gap for "
    "PHP-FL/entropy, and a 41.6-point gap for FedAPEN; 44.3 is conservative."
))
story.append(subsection("5.2 Rankings disagree"))
story.append(make_table([
    ["Run", "client-FM", "FM rank", "class gap", "gap rank"],
    ["PHPFL/ent", "0.0763", "1", "0.456", "4"],
    ["PHPFL/lin", "0.0785", "2", "0.443", "3"],
    ["FedAPEN/lin", "0.0839", "3", "0.529", "5"],
    ["Standalone/lin", "0.0962", "4", "0.386", "1"],
    ["Standalone/ent", "0.0985", "5", "0.389", "2"],
], [4.0, 3.2, 2.7, 3.2, 2.7]))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "The two metrics order the five runs differently. FedAPEN is 3rd by client-FM but 5th by "
    "class gap; Standalone/linear is 4th by client-FM but 1st by class gap. This is an "
    "observation on five runs, not a rank-correlation claim: n=5 is too small to test whether "
    "the disagreement is systematic."
))
story.append(figure(FIG / "fig_ranking_scatter.png",
                    "The same five runs in the client-FM / class-gap plane. The ordering is not "
                    "monotone; the two axes disagree.", CONTENT_W * 0.82))
story.append(subsection("5.3 Pattern invariance"))
story.append(P(
    "Class-std is roughly invariant to the participation pattern within a method: PHP-FL "
    "0.1334 -> 0.1349 (change 0.0015) and Standalone 0.1281 -> 0.1196 (change 0.0084). The "
    "class gap changes by about 1.3 points for PHP-FL and 0.3 points for Standalone. The "
    "class imbalance is structural; the linear-vs-entropy sampler does not materially create "
    "or remove it."
))
story.append(subsection("5.4 AM normalization and the Standalone caveat"))
story.append(make_table([
    ["Run", "AM", "raw gap/AM", "support gap/AM"],
    ["PHPFL/ent", "0.6308", "0.723", "0.713"],
    ["PHPFL/lin", "0.6320", "0.701", "0.811"],
    ["FedAPEN/lin", "0.6209", "0.852", "0.670"],
    ["Standalone/lin", "0.5429", "0.711", "0.645"],
    ["Standalone/ent", "0.5467", "0.712", "0.648"],
], [4.0, 3.0, 4.6, 4.6]))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "Standalone has a smaller raw class gap (38.6 vs 44.3), but its AM is much lower. After "
    "normalizing the raw gap by AM, Standalone's advantage disappears (0.711 vs 0.701 for "
    "PHP-FL/linear). We therefore do not claim a Standalone class-fairness advantage. The "
    "support-filtered gap/AM is lower for Standalone, but it is reported only as a "
    "sensitivity, not the headline. The PHP-FL vs FedAPEN inversion is AM-independent: "
    "FedAPEN is 3rd by client-FM and 5th by gap/AM."
))
story.append(figure(FIG / "fig_am_norm.png",
                    "AM-normalized gap. The raw Standalone advantage is removed by AM "
                    "normalization; the PHP-FL/FedAPEN inversion survives."))
story.append(subsection("5.5 FedAPEN and the field-wide scope"))
story.append(P(
    "FedAPEN/linear has a 52.9-point raw class gap (cat 25.6%, frog 78.5%) and a 41.6-point "
    "support-filtered gap. The pre-registered rule was: >30 points promotes to field-wide, "
    "<20 narrows to PHP-FL, 20-30 is mixed. At 52.9, the claim is field-wide across the two "
    "MH-FL methods tested (PHP-FL and FedAPEN), not universal across all FL."
))

# 6. Ladder
story.append(section(6, "The diagnostic ladder"))
story.append(P(
    "If a standard intervention could remove the class gap, the metric critique would be less "
    "interesting. The ladder tested four one-at-a-time interventions under the same half-scale "
    "configuration. The baseline is the existing PHP-FL/linear run."
))
tbl = [["Rung", "Intervention", "AM", "FM", "gap", "reduction", "dog", "frog"]]
base_row = ["0", "PHP-FL/linear baseline", "0.6320", "0.0785", "0.443", "-", "0.251", "0.694"]
tbl.append(base_row)
for _, r in steps.iterrows():
    names = {
        1: "class-balanced rehearsal (50/class)",
        2: "confusion-targeted loss (top-3, lambda=0.1)",
        3: "worst-class-weighted aggregation (gamma=5)",
        4: "per-class distillation (T=1.0)",
    }
    tbl.append([str(int(r["step"])), names[int(r["step"])],
                f"{r['AM']:.4f}", f"{r['FM']:.4f}", f"{r['class_gap']:.4f}",
                f"{r['gap_reduction_rel']*100:+.1f}%",
                f"{r['dog_recall']:.4f}", f"{r['frog_recall']:.4f}"])
story.append(KeepTogether([
    make_table(tbl, [1.4, 7.0, 2.3, 2.2, 2.0, 2.2, 1.8, 1.8], font=6.7),
    Spacer(1, 0.2 * cm),
]))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "No rung reached the 30% closure threshold. The maximum reduction was 7.9% from rehearsal. "
    "The ladder is therefore a negative result about reducibility, not a failed method paper."
))
story.append(figure(FIG / "fig_ladder.png",
                    "Ladder gaps and relative reduction. The 30% closure threshold is dashed; "
                    "no rung approaches it.", CONTENT_W * 0.95))
story.append(subsection("6.1 Rehearsal: broad improvement, preserved gap"))
story.append(P(
    "Step 1 stored 50 samples per class and mixed them 1:1 with fresh batches. Dog recall rose "
    "from 0.251 to 0.320 (+6.9 points) and frog from 0.694 to 0.728 (+3.4 points). The gap "
    "fell only from 0.443 to 0.408 (7.9%). Both ends improved, but the improvement was "
    "proportional rather than redistributive; the normalized gap barely moved."
))
story.append(subsection("6.2 Confusion loss: the zero-sum result"))
story.append(P(
    "Step 2 added a margin penalty on each client's top-3 validation confusion pairs "
    "(lambda=0.1, margin 1.0). Dog-to-cat errors fell from 0.249 to 0.191 (-23%). But cat "
    "recall fell from 0.434 to 0.364 (-16%), and the combined dog+cat error rate rose from "
    "0.657 to 0.683. The loss moved the decision boundary rather than removing the joint "
    "error. This is the clearest evidence for an information-theoretic floor at 32x32 "
    "CIFAR-10 resolution."
))
story.append(figure(FIG / "fig_step2_dogcat.png",
                    "Step 2 shifts dog/cat errors: dog improves, cat worsens, the combined "
                    "error does not fall.", CONTENT_W * 0.72))
story.append(subsection("6.3 Worst-class aggregation"))
story.append(P(
    "Step 3 used weights exp(-gamma * recall_i(c_worst)) with gamma=5. The globally worst "
    "class was airplane early and dog later. The formula gives the largest weight to clients "
    "with low recall on the worst class, i.e. the clients least able to represent it. The gap "
    "fell only to 0.431 (2.7%); dog recall reached 0.292. Aggregation reweighting alone does "
    "not fix a representation problem."
))
story.append(subsection("6.4 Per-class distillation"))
story.append(P(
    "Step 4 weighted DEAL's KL term by exp(r_i(c)/T) with T=1.0. Dog recall rose to 0.282, "
    "frog to 0.706, and the gap fell to 0.424 (4.3%). The best dog architecture was "
    "EfficientNet-B1 before and after; its mean dog recall rose from 0.599 to 0.629, so the "
    "best-dog architecture did not change. This is the strongest mechanism-specific result, "
    "but still far from closure."
))

# 7. Interesting results
story.append(section(7, "What is interesting"))
story.append(bullet(
    "<b>1. A 44.3-point class gap in a good client-FM run.</b> Client-FM 0.0785 coexists with "
    "frog 69.4% and dog 25.1%. The best client-FM run is worse on the class axis at 45.6 "
    "points. A leaderboard sorted by FM would hide the failure."
))
story.append(bullet(
    "<b>2. FedAPEN makes the finding field-wide across two MH-FL methods.</b> Its gap is 52.9 "
    "points, above the pre-registered 30-point scope threshold. This is not a universal FL "
    "claim; it is field-wide across PHP-FL and FedAPEN."
))
story.append(bullet(
    "<b>3. The two axes order the runs differently.</b> FedAPEN is 3rd by client-FM but 5th "
    "by class gap; Standalone/linear is 4th by client-FM but 1st by class gap. With five runs "
    "this is an observation, not a systematic correlation."
))
story.append(bullet(
    "<b>4. The class bias is structural.</b> Changing linear to entropy changes class-std by "
    "0.0015 for PHP-FL and 0.0084 for Standalone. Participation is not the main driver of the "
    "class axis in these models."
))
story.append(bullet(
    "<b>5. Four standard interventions do not close the gap.</b> Rehearsal, confusion loss, "
    "worst-class aggregation, and per-class distillation reduce it by at most 7.9% over one "
    "seed. The gap is not a simple optimization failure."
))
story.append(bullet(
    "<b>6. The confusion loss is zero-sum.</b> Dog-to-cat errors fall 23%, cat recall falls "
    "16%, and the combined error rises. That is an error-allocation signature, consistent "
    "with an information-theoretic floor at CIFAR-10 resolution."
))
story.append(bullet(
    "<b>7. Standalone's raw gap advantage is an AM effect.</b> Its gap is smaller (38.6 vs "
    "44.3), but gap/AM is 0.711 vs 0.701 after normalization. The raw claim is dropped; the "
    "support-filtered variant is a sensitivity, not a headline."
))
story.append(P(
    "The most interesting single result is not the 44-point number by itself. It is the "
    "combination of the rank disagreement and the zero-sum confusion result: the class axis "
    "is a real, separate fairness object, and one of the most natural ways to attack it "
    "moves errors rather than removing them."
))

# 8. Limitations
story.append(section(8, "Limitations and what is not established"))
story.append(bullet(
    "<b>No Table 2 reproduction.</b> The original 66.94/8.07 baseline assertion was not "
    "executed after the scope change. This repository does not claim to reproduce the paper's "
    "full CIFAR-10 numbers."
))
story.append(bullet(
    "<b>Half-scale only.</b> 10 clients, 60 rounds, one seed, four CNNs, one local epoch, "
    "2,500/500 per client. Absolute AM/FM are not comparable to the paper. Only the linear-vs-"
    "entropy contrast and the class-axis geometry are usable."
))
story.append(bullet(
    "<b>One seed.</b> No seed-level uncertainty is claimed for class-std, class gap, worst-"
    "class recall, or the rank order."
))
story.append(bullet(
    "<b>Five runs are not a field survey.</b> The field-wide scope covers two MH-FL methods "
    "(PHP-FL and FedAPEN). FedFV, FedHEAL, FedAWE, and FedAU were not ported; the original "
    "E8 branch was dropped."
))
story.append(bullet(
    "<b>Support sparsity.</b> Per-client class cells are sparse. The 10-sample threshold is a "
    "pre-declared compromise; the global class headline uses the full test set, but the "
    "client-cell tail is still noisy."
))
story.append(bullet(
    "<b>Descriptive, not causal.</b> The analysis does not identify why dog/cat/deer/horse are "
    "weak. The information-floor interpretation is a hypothesis motivated by Step 2, not a "
    "proven mechanism."
))

# 9. Future directions
story.append(section(9, "Future directions"))
story.append(P(
    "The project is now a measurement paper. The most valuable next step is not another "
    "method; it is to test the mechanism behind the zero-sum result and to make the reporting "
    "practice standard."
))
story.append(subsection("9.1 Primary: resolution sweep and pairwise error budget"))
story.append(P(
    "The Step 2 result suggests that dog and cat are confusable at 32x32. A clean test is to "
    "measure the full 10x10 confusion matrix at 32, 64, and 96 pixels on a small centralized "
    "probe. If the joint dog+cat error stays flat at 32 but falls at higher resolution, the "
    "floor is resolution-limited. If it stays flat even at higher resolution, the ambiguity is "
    "in the labels/data rather than the input resolution."
))
story.append(P(
    "The same analysis should be repeated for other class pairs (deer/horse, automobile/truck). "
    "If targeted losses always move error between directions without reducing the pair sum, "
    "the claim becomes stronger: some class gaps are an error-allocation problem, not an "
    "accuracy problem."
))
story.append(subsection("9.2 Zero-training: architecture decomposition"))
story.append(P(
    "The committed per-client logs already contain architecture labels and per-class recalls. "
    "A zero-training analysis can test whether the class gap is concentrated in specific "
    "architectures. EfficientNet-B1 was the best dog architecture (mean dog recall 0.599 "
    "baseline, 0.629 in Step 4); if other architectures contribute most of the cat/dog errors, "
    "the gap is partly an inductive-bias problem rather than federation."
))
story.append(subsection("9.3 Multi-seed and full-scale confirmation"))
story.append(P(
    "A second seed and a full-scale CIFAR-10 audit are needed before any universal claim. The "
    "paper-scale audit would use 20 clients, 3 local epochs, and 3 seeds. This is a future run; "
    "it was not executed. The half-scale one-seed results should be treated as a diagnostic, "
    "not a final measurement."
))
story.append(subsection("9.4 More MH-FL methods"))
story.append(P(
    "The field-wide scope currently rests on two methods. Porting FedFV, FedHEAL, FedAWE, and "
    "FedAU into the same heterogeneous setting would test whether class-FM is a general "
    "problem or specific to PHP-FL/FedAPEN. This is measurement, not a new method."
))
story.append(subsection("9.5 Reporting standard"))
story.append(P(
    "The practical recommendation is a three-number report plus the matrix: client-FM, "
    "class-FM, worst-class recall, and the client-by-class matrix or a link to it. class_FM "
    "should be reported, not optimized. A model that wins client-FM but loses class-FM should "
    "not be presented as unambiguously fairer."
))

# 10. Reproducibility
story.append(section(10, "Reproducibility and artifact map"))
story.append(P(
    "The public repository contains the paper source, analysis code, result CSVs, structured "
    "per-run logs, and figures. Large artifacts are excluded."
))
story.append(KeepTogether([make_table([
    ["Path", "Contents"],
    ["phofl/paper/", "Abstract through conclusion (sections 00-07)"],
    ["phofl/analysis/", "Client x class matrices, support filtering, AM normalization, ranks"],
    ["phofl_method/", "Ladder per-step CSVs, findings, LADDER_SUMMARY.md"],
    ["phofl_half/results/", "Half-scale result CSVs"],
    ["phofl_half/logs/", "Per-run round metrics and histories"],
    ["phofl/results/, phofl/logs/", "Earlier A/B/C scoped results"],
    ["phofl/figures/", "300 DPI figures"],
    ["phofl_experiments.ipynb", "Resumable A/B/C notebook"],
    ["phofl_lab.py", "Runtime harness"],
    ["analyze_two_axis.py", "Rebuilds the 2-D analysis"],
    ["run_step1.py ... run_step4.py", "Diagnostic ladder"],
], [5.0, 12.0], align="LEFT"), Spacer(1, 0.2 * cm)]))
story.append(Spacer(1, 0.2 * cm))
story.append(P(
    "Excluded: third_party clones, generated CIFAR-10 partitions, checkpoints, run_store, "
    "phofl_pinned, and .venv-pinned. DATA.md documents regeneration. The repository is at "
    "github.com/Paul4805/phofl-class-fairness; release v0.1.0 is tagged."
))
story.append(subsection("10.1 Key commands"))
story.append(P("Rebuild the committed analysis without training:", body_left))
story.append(P("python analyze_two_axis.py", code))
story.append(P("python make_final_table.py", code))
story.append(P("python -m pytest -q", code))
story.append(Spacer(1, 0.4 * cm))
story.append(box("One-line conclusion", [
    "Client-FM is a useful but incomplete fairness report. The class axis is a real, "
    "structural second dimension; it survives a participation change, it ranks methods "
    "differently, and four standard interventions do not close it. Report class_FM and "
    "worst-class recall alongside client-FM, but do not treat class_FM as an objective."
]))
story.append(Spacer(1, 0.5 * cm))
story.append(P("References", h2))
refs = [
    "PHP-FL: A Fair Federated Learning Method for Handling Client Participation Probability Inconsistencies in Heterogeneous Environments. NeurIPS 2025.",
    "Qin et al. FedAPEN: Personalized Cross-silo Federated Learning with Adaptability to Statistical Heterogeneity. KDD 2023.",
    "Li et al. Fair Resource Allocation in Federated Learning (q-FedAvg). ICLR 2020.",
    "Mohri et al. Agnostic Federated Learning (AFL). ICML 2019.",
    "Wang et al. Federated Learning with Fair Averaging (FedFV). IJCAI 2021.",
    "Cui et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.",
    "Cao et al. Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM). NeurIPS 2019.",
    "Sagawa et al. Distributionally Robust Neural Networks for Group Shifts (Group DRO). ICLR 2020.",
    "Krizhevsky. Learning Multiple Layers of Features from Tiny Images. CIFAR-10 technical report.",
]
for ref in refs:
    story.append(P(ref, small))

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    title="Class Fairness in Model-Heterogeneous Federated Learning: Findings Report",
    author="J Paul Masillamni",
    subject="Measurement study, diagnostic ladder, and research roadmap",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, f"{OUT.stat().st_size/2**20:.2f} MiB")
