"""C escalation: CIFAR-10, 10 clients, 60 rounds, 1 seed, 4 runs.

Pinned stack is expected (torch 2.2.2+cu121 / numpy 1.26.4).  This script does
not read or write A/B artifacts; it lives entirely under phofl_half/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import phofl_lab as pl

HALF_ROOT = Path("phofl_half")
DATA_ROOT = HALF_ROOT / "data_cap"
ROUNDS = 60
CLIENTS = 10
SEED = 0
BATCH = 64
LOCAL_EPOCHS = 1
EVAL_GAP = 10
MODEL_FAMILY = "HtM4"
SCALE = "half10"
EXP = "C_half"

pl.configure(HALF_ROOT, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()


def global_test_loader():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    raw = pl.REPO_DIR / "dataset" / "Cifar10" / "rawdata"
    ds = datasets.CIFAR10(root=str(raw), train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)


def load_model_file(run_id, role, name):
    path = (HALF_ROOT / "run_store" / run_id / "Cifar10" / "PHPFL" /
            f"{role}_{name}.pt")
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return obj


@torch.no_grad()
def global_per_class_recall(result, client_df):
    loader = global_test_loader()
    run_id = result["run_id"]
    method = result["method"]
    n_total = len(loader.dataset)
    probs_sum = np.zeros((n_total, 10), dtype=np.float64)
    lam = {int(r.client_id): float(r.lambda_i) for _, r in client_df.iterrows()}
    for cid in range(CLIENTS):
        local = load_model_file(run_id, f"Client_{cid}", "model")
        local.to(pl.DEVICE).eval()
        aux = None
        w = 1.0
        if method == "PHPFL":
            aux = load_model_file(run_id, f"Client_{cid}", "global_model")
            aux.to(pl.DEVICE).eval()
            w = lam.get(cid, 0.5)
        start = 0
        for x, y in loader:
            x = x.to(pl.DEVICE)
            out = w * local(x)
            if aux is not None:
                out = out + (1.0 - w) * aux(x)
            probs_sum[start:start + len(y)] += torch.softmax(out, dim=1).cpu().numpy()
            start += len(y)
        del local
        if aux is not None:
            del aux
        pl.clear_memory()
    pred = probs_sum.argmax(axis=1)
    target = np.asarray(loader.dataset.targets, dtype=np.int64)
    rows = []
    for k in range(10):
        m = target == k
        rec = float((pred[m] == k).mean()) if m.any() else float("nan")
        rows.append({"run_id": run_id, "method": method, "pattern": result["pattern"],
                     "class_id": k, "recall": rec, "support": int(m.sum())})
    return pd.DataFrame(rows)


summary_rows = []
all_client_rows = []
recall_frames = []

for method in ["PHPFL", "Standalone"]:
    for pattern in ["linear", "entropy_correlated"]:
        p = pl.participation_probabilities(pattern, "Cifar10", CLIENTS, seed=SEED)
        print(f"[half] {method} / {pattern}")
        if method == "PHPFL":
            res = pl.run_federated(
                exp=EXP, method="PHPFL", dataset="Cifar10", pattern=pattern,
                seed=SEED, rounds=ROUNDS, num_clients=CLIENTS, batch_size=BATCH,
                local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
                valid_ratio=0.1, agg_mode="plain", aux_arch="googlenet",
                data_root=DATA_ROOT, scale=SCALE, resume=True, max_hours=8.0,
                p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False)
        else:
            res = pl.run_standalone(
                exp=EXP, dataset="Cifar10", pattern=pattern, seed=SEED,
                rounds=ROUNDS, num_clients=CLIENTS, batch_size=BATCH,
                local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP,
                data_root=DATA_ROOT, scale=SCALE, resume=True, max_hours=8.0,
                model_family=MODEL_FAMILY)
        if res.get("skipped"):
            pth = pl._result_path(EXP, method, pattern, SEED, scale=SCALE)
            res = json.loads(pth.read_text(encoding="utf-8"))
        log = HALF_ROOT / "logs" / EXP / res["run_id"] / "round_metrics.csv"
        df = pd.read_csv(log)
        final = df[df["is_final"] == 1].drop_duplicates("client_id", keep="last").copy()
        if len(final) == 0:
            final = df[df["round"] == df["round"].max()].drop_duplicates("client_id", keep="last").copy()
        fair = pl.compute_fairness(final)
        for _, r in final.iterrows():
            all_client_rows.append({
                "exp": EXP, "method": method, "pattern": pattern,
                "client_id": int(r["client_id"]), "arch": r.get("arch", ""),
                "p_i": float(r["p_i"]), "N_i": int(r["N_i"]),
                "acc": float(r["ensemble_acc"]), "rare_acc": float(r["rare_acc"]),
                "AM": fair["AM"], "FM": fair["FM"], "seed": SEED,
            })
        rec = global_per_class_recall(res, final)
        recall_frames.append(rec)
        summary_rows.append({
            "exp": EXP, "method": method, "pattern": pattern, "seed": SEED,
            "AM": fair["AM"], "FM": fair["FM"], "min_acc": fair["min_acc"],
            "p10": fair["p10"], "cvar10": fair["cvar10"], "CV": fair["CV"],
            "jain": fair["jain"], "rho_p_acc": fair["rho_p_acc"],
            "worst_class": int(rec.loc[rec["recall"].idxmin(), "class_id"]),
            "worst_class_recall": float(rec["recall"].min()),
            "run_id": res["run_id"], "seconds": float(res.get("seconds", np.nan)),
        })
        print(f"[half] {method}/{pattern}: AM={fair['AM']:.4f} FM={fair['FM']:.4f} "
              f"min={fair['min_acc']:.4f} worst-class={rec['recall'].min():.4f}")

HALF_ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)
client_df = pd.DataFrame(all_client_rows)
client_df.to_csv(HALF_ROOT / "results" / "C_half_results.csv", index=False)
summary = pd.DataFrame(summary_rows)
summary.to_csv(HALF_ROOT / "results" / "C_half_summary.csv", index=False)
recall = pd.concat(recall_frames, ignore_index=True)
recall.to_csv(HALF_ROOT / "results" / "C_half_global_recall.csv", index=False)

# Paired-by-client tests on the final per-client vectors.
from scipy.stats import wilcoxon
tests = []
def pair(col, a, b, label, metric="accuracy"):
    x = client_df[client_df[col] == a].set_index("client_id")["acc"]
    y = client_df[client_df[col] == b].set_index("client_id")["acc"]
    z = pd.concat([x, y], axis=1, join="inner").dropna()
    if len(z) < 5:
        return
    stat, p = wilcoxon(z.iloc[:, 0], z.iloc[:, 1], zero_method="wilcox", alternative="two-sided")
    tests.append({"metric": metric, "comparison": label, "n": len(z),
                  "statistic": float(stat), "p_value": float(p),
                  "median_diff": float(np.median((z.iloc[:, 1] - z.iloc[:, 0]).to_numpy()))})
for pat in ["linear", "entropy_correlated"]:
    pair("method", "PHPFL", "Standalone", f"PHPFL vs Standalone [{pat}]")
pair("pattern", "linear", "entropy_correlated", "linear vs entropy [accuracy]")
# dispersion and advantage
php = client_df[client_df.method == "PHPFL"]
z = pd.concat([php[php.pattern == "linear"].set_index("client_id")["acc"],
               php[php.pattern == "entropy_correlated"].set_index("client_id")["acc"]],
              axis=1, join="inner").dropna()
if len(z) >= 5:
    xdev = (z.iloc[:, 0] - z.iloc[:, 0].mean()).abs()
    ydev = (z.iloc[:, 1] - z.iloc[:, 1].mean()).abs()
    stat, p = wilcoxon(xdev, ydev, zero_method="wilcox", alternative="two-sided")
    tests.append({"metric": "|acc - group mean|", "comparison": "linear vs entropy [dispersion]",
                  "n": len(z), "statistic": float(stat), "p_value": float(p),
                  "median_diff": float(np.median((ydev - xdev).to_numpy()))})
st = client_df[(client_df.method == "Standalone") & (client_df.pattern == "linear")].set_index("client_id")["acc"]
adv = pd.concat([php[php.pattern == "linear"].set_index("client_id")["acc"] - st,
                 php[php.pattern == "entropy_correlated"].set_index("client_id")["acc"] - st],
                axis=1, join="inner").dropna()
if len(adv) >= 5:
    stat, p = wilcoxon(adv.iloc[:, 1], adv.iloc[:, 0], zero_method="wilcox", alternative="two-sided")
    tests.append({"metric": "PHPFL - Standalone", "comparison": "linear vs entropy [advantage]",
                  "n": len(adv), "statistic": float(stat), "p_value": float(p),
                  "median_diff": float(np.median((adv.iloc[:, 1] - adv.iloc[:, 0]).to_numpy()))})
stats = pd.DataFrame(tests)
if len(stats):
    stats["holm_p"] = np.nan
    order = np.argsort(stats["p_value"].to_numpy())
    m = len(stats)
    running = 0.0
    for rank, idx in enumerate(order, 1):
        v = (m - rank + 1) * stats.loc[idx, "p_value"]
        running = max(running, v)
        stats.loc[idx, "holm_p"] = min(1.0, running)
    stats["significant_0.05_holm"] = stats["holm_p"] < 0.05
stats.to_csv(HALF_ROOT / "results" / "C_half_stats.csv", index=False)
print(summary.to_string(index=False))
print(stats.to_string(index=False))
print("[half] done")
