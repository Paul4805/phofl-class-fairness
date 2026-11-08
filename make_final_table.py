"""Print the final 4-run C escalation tables and pre-registered verdict."""
from pathlib import Path

import pandas as pd

RES = Path("phofl_half/results")
OUT = Path("phofl/escalation")

table = pd.read_csv(RES / "C_half_table.csv")
stats = pd.read_csv(RES / "C_half_stats_full.csv")
verdict = pd.read_csv(OUT / "C_half_verdict.csv").iloc[0]
decision = (OUT / "C_half_verdict.txt").read_text(encoding="utf-8").strip()

order = [("PHPFL", "linear"), ("PHPFL", "entropy_correlated"),
         ("Standalone", "linear"), ("Standalone", "entropy_correlated")]
table["_o"] = table.apply(lambda r: order.index((r["method"], r["pattern"])), axis=1)
table = table.sort_values("_o").drop(columns="_o")

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

main = table[["method", "pattern", "AM", "FM", "min_acc", "p10", "cvar10",
              "rho_p_acc", "worst_class", "worst_class_recall"]].copy()
main = main.rename(columns={"min_acc": "min", "rho_p_acc": "rho",
                            "worst_class": "worst_cls"})
main["worst_cls"] = main["worst_cls"].map(lambda i: classes[int(i)])
for c in ["AM", "FM", "min", "p10", "cvar10", "rho", "worst_class_recall"]:
    main[c] = main[c].map(lambda x: f"{x:.4f}")
print(main.to_markdown(index=False))

pc = table[["method", "pattern"] + [f"class_{i}_recall" for i in range(10)] +
           ["worst_class", "worst_class_recall"]].copy()
pc.columns = ["method", "pattern"] + classes + ["worst_cls", "worst_recall"]
pc["worst_cls"] = pc["worst_cls"].map(lambda i: classes[int(i)])
for c in classes + ["worst_recall"]:
    pc[c] = pc[c].map(lambda x: f"{x:.4f}")
print("\n" + pc.to_markdown(index=False))

# Focused paired-test table.
keep = [
    "PHPFL vs Standalone [linear]", "PHPFL vs Standalone [entropy_correlated]",
    "linear vs entropy [accuracy]", "linear vs entropy [dispersion]",
    "linear vs entropy [min tail-shortfall]", "linear vs entropy [CVaR10 tail-shortfall]",
    "linear vs entropy [min]", "linear vs entropy [CVaR10]",
    "linear vs entropy [advantage]",
]
st = stats[stats["comparison"].isin(keep)].copy()
st = st[["metric", "comparison", "test", "n", "statistic", "p_raw", "p_holm",
         "significant_holm"]]
for c in ["statistic", "p_raw", "p_holm"]:
    st[c] = st[c].map(lambda x: f"{x:.6g}")
print("\n" + st.to_markdown(index=False))

print("\n" + decision)
