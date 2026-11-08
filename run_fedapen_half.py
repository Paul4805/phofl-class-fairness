"""FedAPEN / linear / half-scale, with one OOM retry at batch 32.

This is the single queued run.  It appends its per-client and per-class rows to
the existing C_half CSVs; it does not rerun the four existing runs.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import phofl_lab as pl

HALF = Path("phofl_half")
DATA = HALF / "data_cap"
EXP = "C_half"
SCALE = "half10"
ROUNDS = 60
CLIENTS = 10
SEED = 0
LOCAL_EPOCHS = 1
EVAL_GAP = 10
MODEL_FAMILY = "HtM4"

pl.configure(HALF, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def global_test_loader():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    raw = pl.REPO_DIR / "dataset" / "Cifar10" / "rawdata"
    ds = datasets.CIFAR10(root=str(raw), train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)


def load_model_file(run_id, role, name):
    path = HALF / "run_store" / run_id / "Cifar10" / "PHPFL" / f"{role}_{name}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def global_per_class_recall(result, client_df):
    loader = global_test_loader()
    run_id = result["run_id"]
    n_total = len(loader.dataset)
    probs_sum = np.zeros((n_total, 10), dtype=np.float64)
    lam = {int(r.client_id): float(r.lambda_i) for _, r in client_df.iterrows()}
    for cid in range(CLIENTS):
        local = load_model_file(run_id, f"Client_{cid}", "model").to(pl.DEVICE).eval()
        aux = load_model_file(run_id, f"Client_{cid}", "global_model").to(pl.DEVICE).eval()
        w = lam.get(cid, 0.5)
        start = 0
        for x, y in loader:
            x = x.to(pl.DEVICE)
            out = w * local(x) + (1.0 - w) * aux(x)
            probs_sum[start:start + len(y)] += torch.softmax(out, dim=1).cpu().numpy()
            start += len(y)
        del local, aux
        pl.clear_memory()
    pred = probs_sum.argmax(axis=1)
    target = np.asarray(loader.dataset.targets, dtype=np.int64)
    rows = []
    for k in range(10):
        m = target == k
        rows.append({"run_id": run_id, "method": "FedAPEN", "pattern": "linear",
                     "class_id": k, "recall": float((pred[m] == k).mean()),
                     "support": int(m.sum())})
    return pd.DataFrame(rows)


def append_dedup(path, df, keys):
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=keys, keep="last")
    df.to_csv(path, index=False)


def run_attempt(batch):
    p = pl.participation_probabilities("linear", "Cifar10", CLIENTS, seed=SEED)
    return pl.run_federated(
        exp=EXP, method="FedAPEN", dataset="Cifar10", pattern="linear",
        seed=SEED, rounds=ROUNDS, num_clients=CLIENTS, batch_size=batch,
        local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
        valid_ratio=0.1, agg_mode="plain", aux_arch="googlenet",
        data_root=DATA, scale=SCALE, resume=True, max_hours=4.0,
        p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False)


result = None
last_error = None
for batch in (64, 32):
    try:
        print(f"[fedapen] attempt batch={batch}")
        result = run_attempt(batch)
        break
    except Exception as e:
        last_error = e
        msg = str(e).lower()
        is_oom = "out of memory" in msg or "cuda error" in msg
        print(f"[fedapen] attempt batch={batch} failed: {type(e).__name__}: {e}")
        if is_oom and batch == 64:
            pl.clear_memory()
            continue
        break

if result is None:
    (HALF / "results" / "FedAPEN_FAILED.txt").write_text(
        f"FAILED after 2 attempts: {last_error!r}\n", encoding="utf-8")
    raise SystemExit(2)

if result.get("skipped"):
    pth = pl._result_path(EXP, "FedAPEN", "linear", SEED, scale=SCALE)
    result = json.loads(pth.read_text(encoding="utf-8"))

run_id = result["run_id"]
log = HALF / "logs" / EXP / run_id / "round_metrics.csv"
rm = pd.read_csv(log)
final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").copy()
if len(final) == 0:
    final = rm[rm["round"] == rm["round"].max()].drop_duplicates("client_id", keep="last").copy()
final = final.sort_values("client_id")

fair = pl.compute_fairness(final)
rec = global_per_class_recall(result, final)
worst_i = int(rec["recall"].idxmin())

client_rows = []
for _, r in final.iterrows():
    client_rows.append({
        "exp": EXP, "method": "FedAPEN", "pattern": "linear",
        "client_id": int(r["client_id"]), "arch": r.get("arch", ""),
        "p_i": float(r["p_i"]), "N_i": int(r["N_i"]),
        "acc": float(r["ensemble_acc"]), "rare_acc": float(r["rare_acc"]),
        "AM": fair["AM"], "FM": fair["FM"], "seed": SEED,
    })
append_dedup(HALF / "results" / "C_half_results.csv", pd.DataFrame(client_rows),
             ["run_id", "client_id"] if False else ["method", "pattern", "client_id"])

summary_row = pd.DataFrame([{
    "exp": EXP, "method": "FedAPEN", "pattern": "linear", "seed": SEED,
    "AM": fair["AM"], "FM": fair["FM"], "min_acc": fair["min_acc"],
    "p10": fair["p10"], "cvar10": fair["cvar10"], "CV": fair["CV"],
    "jain": fair["jain"], "rho_p_acc": fair["rho_p_acc"],
    "worst_class": int(rec.loc[worst_i, "class_id"]),
    "worst_class_recall": float(rec["recall"].min()),
    "run_id": run_id, "seconds": float(result.get("seconds", np.nan)),
}])
append_dedup(HALF / "results" / "C_half_summary.csv", summary_row, ["run_id"])
append_dedup(HALF / "results" / "C_half_global_recall.csv", rec,
             ["run_id", "class_id"])

print(summary_row.to_string(index=False))
print(rec.to_string(index=False))
print("[fedapen] complete; rerunning two-axis analysis")
subprocess.run([sys.executable, "analyze_two_axis.py"], check=True)
