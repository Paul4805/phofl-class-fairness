"""Three-seed class-gap sweep: PHPFL, FedAPEN, Standalone x linear/entropy.

Same protocol as the committed half-scale runs: CIFAR-10, 10 clients,
60 rounds, 1 local epoch, HtM4 family, batch 64, eval_gap 10, seed in {0,1,2}.

Existing seed-0 rows were produced on the pinned stack (torch 2.2.2+cu121).
That venv is now blocked by a Windows Application Control policy, so this
sweep runs on the installed stack (torch 2.14.x) and re-runs seed 0 as well
to keep the three seeds on one stack.  Completed runs are skipped via the
harness completion flags, so the script is safe to interrupt and re-run.

Outputs (appended per run, so a crash keeps progress):
  phofl_half/results/C_half_seed_sweep.csv
  phofl_half/results/C_half_seed_sweep_recall.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import phofl_lab as pl

HALF = Path("phofl_half")
DATA = HALF / "data_cap"
EXP = "C_half_s3"
SCALE = "half10"
ROUNDS = 60
CLIENTS = 10
SEEDS = [0, 1, 2]
METHODS = ["PHPFL", "FedAPEN", "Standalone", "FedFV"]
PATTERNS = ["linear", "entropy_correlated"]
BATCH = 64
LOCAL_EPOCHS = 1
EVAL_GAP = 10
MODEL_FAMILY = "HtM4"

OUT_MAIN = HALF / "results" / "C_half_s3_seed_sweep.csv"
OUT_REC = HALF / "results" / "C_half_s3_seed_sweep_recall.csv"
KEYS = ["method", "pattern", "seed"]

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

pl.configure(HALF, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()


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
    """Full CIFAR-10 test set, global ensemble over the 10 clients.

    PHPFL and FedAPEN mix the personalised model with the shared global model
    using the final lambda; Standalone has no server model.
    """
    loader = global_test_loader()
    run_id = result["run_id"]
    method = result["method"]
    pattern = result["pattern"]
    n_total = len(loader.dataset)
    probs_sum = np.zeros((n_total, 10), dtype=np.float64)
    lam = {int(r.client_id): float(r.lambda_i) for _, r in client_df.iterrows()}
    mix = method in ("PHPFL", "FedAPEN", "FedFV")
    for cid in range(CLIENTS):
        local = load_model_file(run_id, f"Client_{cid}", "model").to(pl.DEVICE).eval()
        aux = None
        w = 1.0
        if mix:
            aux = load_model_file(run_id, f"Client_{cid}", "global_model").to(pl.DEVICE).eval()
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
        rows.append({"run_id": run_id, "method": method, "pattern": pattern,
                     "class_id": k, "class": CLASSES[k],
                     "recall": float((pred[m] == k).mean()), "support": int(m.sum())})
    return pd.DataFrame(rows)


def run_torch_version(run_id):
    """Stack that actually produced the run, not the stack reading it."""
    meta = HALF / "logs" / EXP / run_id / "run_meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8"))["packages"].get("torch", "")
        except Exception:
            pass
    return torch.__version__


def append_dedup(path, df, keys):
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=keys, keep="last")
    df.to_csv(path, index=False)


def run_one(method, pattern, seed):
    """Run with one OOM retry at batch 32, mirroring run_fedapen_half.py."""
    last_error = None
    for batch in (BATCH, 32):
        try:
            if method == "Standalone":
                return pl.run_standalone(
                    exp=EXP, dataset="Cifar10", pattern=pattern, seed=seed,
                    rounds=ROUNDS, num_clients=CLIENTS, batch_size=batch,
                    local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP,
                    data_root=DATA, scale=SCALE, resume=True, max_hours=8.0,
                    model_family=MODEL_FAMILY)
            p = pl.participation_probabilities(pattern, "Cifar10", CLIENTS, seed=seed)
            agg = "fedfv" if method == "FedFV" else "plain"
            return pl.run_federated(
                exp=EXP, method=method, dataset="Cifar10", pattern=pattern,
                seed=seed, rounds=ROUNDS, num_clients=CLIENTS, batch_size=batch,
                local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
                valid_ratio=0.1, agg_mode=agg, aux_arch="googlenet",
                data_root=DATA, scale=SCALE, resume=True, max_hours=8.0,
                p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False)
        except Exception as e:  # noqa: BLE001 - keep the sweep alive
            last_error = e
            msg = str(e).lower()
            is_oom = "out of memory" in msg or "cuda error" in msg
            print(f"[sweep] {method}/{pattern}/s{seed} batch={batch} failed: "
                  f"{type(e).__name__}: {e}")
            pl.clear_memory()
            if is_oom and batch == BATCH:
                continue
            raise
    raise RuntimeError(last_error)


def main(methods=None, seeds=None, patterns=None):
    methods = methods or METHODS
    seeds = seeds or SEEDS
    patterns = patterns or PATTERNS
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    total = len(seeds) * len(methods) * len(patterns)
    done = 0
    for seed in seeds:
        for method in methods:
            for pattern in patterns:
                done += 1
                print(f"\n[sweep] {done}/{total} {method}/{pattern}/s{seed}", flush=True)
                res = run_one(method, pattern, seed)
                if res.get("skipped"):
                    pth = pl._result_path(EXP, method, pattern, seed, scale=SCALE)
                    res = json.loads(pth.read_text(encoding="utf-8"))
                run_id = res["run_id"]
                log = HALF / "logs" / EXP / run_id / "round_metrics.csv"
                rm = pd.read_csv(log)
                final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").copy()
                if len(final) == 0:
                    final = (rm[rm["round"] == rm["round"].max()]
                             .drop_duplicates("client_id", keep="last").copy())
                final = final.sort_values("client_id")
                fair = pl.compute_fairness(final)
                rec = global_per_class_recall(res, final)
                append_dedup(OUT_REC, rec.assign(seed=seed),
                             ["method", "pattern", "seed", "class_id"])
                worst = rec.loc[rec["recall"].idxmin()]
                best = rec.loc[rec["recall"].idxmax()]
                row = {
                    "method": method, "pattern": pattern, "seed": seed,
                    "AM": float(fair["AM"]), "FM": float(fair["FM"]),
                    "min_acc": float(fair["min_acc"]),
                    "class_std": float(rec["recall"].std(ddof=0)),
                    "class_gap": float(best["recall"] - worst["recall"]),
                    "worst_class": worst["class"],
                    "worst_class_recall": float(worst["recall"]),
                    "best_class": best["class"],
                    "best_class_recall": float(best["recall"]),
                    "run_id": run_id,
                    "seconds": float(res.get("seconds", np.nan)),
                    "torch": run_torch_version(run_id),
                }
                append_dedup(OUT_MAIN, pd.DataFrame([row]), KEYS)
                print(f"[sweep] done {method}/{pattern}/s{seed}: "
                      f"AM={row['AM']:.4f} FM={row['FM']:.4f} "
                      f"gap={row['class_gap']:.4f} worst={row['worst_class']} "
                      f"{row['worst_class_recall']:.3f}", flush=True)
    print("\n[sweep] all cells complete")
    print(pd.read_csv(OUT_MAIN).to_string(index=False))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--patterns", nargs="+", default=None)
    args = ap.parse_args()
    main(args.methods, args.seeds, args.patterns)
