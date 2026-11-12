"""DermaMNIST class-fairness benchmark: centralized baseline + four FL methods.

Dataset: MedMNIST v2 DermaMNIST, 7 skin-lesion classes, 28x28 RGB resized to 32.
Partition: 10 clients, Dirichlet(0.5) label skew (make_dermamnist_10.py).
Global test: official 2,005-image test split.

Arms
  centralized : the four HtM4 architectures trained independently on the pooled
                client training data and ensembled by mean softmax (no server,
                no client split). This is the non-federated reference.
  PHPFL       : PHP-FL local objective + plain shared-model averaging
  FedAPEN     : FedAPEN client class
  FedFV       : FedFV aggregation on the shared model (alpha=0.1, tau=3)
  Standalone  : local-only training, no aggregation

Usage
  python run_med_sweep.py --part central   # 3 seeds, ~20-30 min each
  python run_med_sweep.py --part fl        # 4 methods x 2 patterns x 3 seeds
  python run_med_sweep.py --part all
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import phofl_lab as pl

ROOT = Path("phofl_med")
DATA = ROOT / "data_cap"
EXP = "MED_s3"
SCALE = "med10"
DATASET = "BloodMNIST"
CLASSES = ["basophil", "eosinophil", "erythroblast", "immature_granulocyte",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
N_CLASSES = len(CLASSES)
SEEDS = [0, 1, 2]
ROUNDS = 40
CLIENTS = 10
BATCH = 64
LOCAL_EPOCHS = 1
EVAL_GAP = 10
LOCAL_LR = 0.01
MODEL_FAMILY = "HtM4"
METHODS = ["PHPFL", "FedAPEN", "FedFV", "Standalone"]
PATTERNS = ["linear", "entropy_correlated"]
OUT_MAIN = ROOT / "results" / "med_sweep.csv"
OUT_REC = ROOT / "results" / "med_sweep_recall.csv"
KEYS = ["arm", "method", "pattern", "seed"]

pl.set_num_classes(N_CLASSES, CLASSES)
pl.configure(ROOT, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()


def build_model(arch: str):
    from torchvision import models as tvm
    from flcore.trainmodel.models import EfficientNet
    if arch == "googlenet":
        return tvm.googlenet(pretrained=False, aux_logits=False, num_classes=N_CLASSES)
    if arch == "densenet121":
        return tvm.densenet121(pretrained=False, num_classes=N_CLASSES)
    if arch == "resnet18":
        return tvm.resnet18(pretrained=False, num_classes=N_CLASSES)
    if arch == "efficientnet-b1":
        return EfficientNet.from_name("efficientnet-b1", num_classes=N_CLASSES)
    raise ValueError(arch)


def pooled_train():
    xs, ys = [], []
    for cid in range(CLIENTS):
        blob = np.load(DATA / DATASET / "train" / f"{cid}.npz", allow_pickle=True)
        d = blob["data"].tolist()
        xs.append(np.asarray(d["x"], dtype=np.float32))
        ys.append(np.asarray(d["y"], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def global_test():
    blob = np.load(DATA / DATASET / "global_test.npz")
    return (torch.tensor(blob["x"], dtype=torch.float32),
            torch.tensor(blob["y"], dtype=torch.int64))


def per_class_recall(pred, target):
    rows = []
    for k in range(N_CLASSES):
        m = target == k
        rows.append({"class_id": k, "class": CLASSES[k], "support": int(m.sum().item()),
                     "recall": float((pred[m] == k).float().mean().item()) if m.any() else float("nan")})
    return rows


def summarize(rows, extra):
    rec = pd.DataFrame(rows)
    gap = float(rec["recall"].max() - rec["recall"].min())
    worst = rec.loc[rec["recall"].idxmin()]
    return {**extra,
            "AM": float(np.average(rec["recall"], weights=rec["support"])),
            "class_gap": gap,
            "class_std": float(rec["recall"].std(ddof=0)),
            "worst_class": worst["class"],
            "worst_class_recall": float(worst["recall"]),
            "best_class": rec.loc[rec["recall"].idxmax(), "class"],
            "best_class_recall": float(rec["recall"].max())}, rec


def append_dedup(path, df, keys):
    if path.exists() and path.stat().st_size > 0:
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df = df.drop_duplicates(subset=keys, keep="last")
    df.to_csv(path, index=False)


def run_centralized_seed(seed: int):
    """Four architectures on pooled data, ensembled. No federation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x_np, y_np = pooled_train()
    x = torch.tensor(x_np)
    y = torch.tensor(y_np)
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                         num_workers=0, pin_memory=True, drop_last=True)
    gx, gy = global_test()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(gx, gy), batch_size=256, shuffle=False)
    probs = torch.zeros((len(gy), N_CLASSES))
    single_gaps = []
    t0 = time.time()
    for arch in ["googlenet", "densenet121", "efficientnet-b1", "resnet18"]:
        model = build_model(arch).to(pl.DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=LOCAL_LR)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        for ep in range(ROUNDS):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(pl.DEVICE), by.to(pl.DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    loss = F.cross_entropy(model(bx), by)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        model.eval()
        with torch.no_grad():
            offset = 0
            single_pred = []
            for bx, by in test_loader:
                out = model(bx.to(pl.DEVICE))
                probs[offset:offset + len(by)] += torch.softmax(out.float(), dim=1).cpu()
                single_pred.append(out.argmax(1).cpu())
                offset += len(by)
        pred_single = torch.cat(single_pred)
        rows = per_class_recall(pred_single, gy)
        single_gaps.append(max(r["recall"] for r in rows) - min(r["recall"] for r in rows))
        print(f"[central] seed={seed} {arch} acc="
              f"{float((pred_single == gy).float().mean()):.4f} "
              f"gap={single_gaps[-1]:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        del model
        pl.clear_memory()
    pred = probs.argmax(dim=1)
    rows = per_class_recall(pred, gy)
    row, rec = summarize(rows, {"arm": "centralized", "method": "Centralized",
                                "pattern": "none", "seed": seed,
                                "seconds": time.time() - t0,
                                "mean_single_gap": float(np.mean(single_gaps))})
    append_dedup(OUT_MAIN, pd.DataFrame([row]), ["method", "seed"])
    append_dedup(OUT_REC, rec.assign(method="Centralized", pattern="none", seed=seed),
                 ["method", "pattern", "seed", "class_id"])
    print(f"[central] done seed={seed} AM={row['AM']:.4f} gap={row['class_gap']:.4f} "
          f"worst={row['worst_class']} {row['worst_class_recall']:.3f}", flush=True)


def global_per_class_recall(result, client_df):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*global_test()), batch_size=256, shuffle=False)
    run_id = result["run_id"]
    method = result["method"]
    pattern = result["pattern"]
    mix = method in ("PHPFL", "FedAPEN", "FedFV")
    lam = {int(r.client_id): float(r.lambda_i) for _, r in client_df.iterrows()}
    probs = torch.zeros((len(loader.dataset), N_CLASSES))
    for cid in range(CLIENTS):
        base = ROOT / "run_store" / run_id / DATASET / "PHPFL"
        local = torch.load(base / f"Client_{cid}_model.pt", map_location="cpu",
                           weights_only=False).to(pl.DEVICE).eval()
        aux = None
        w = 1.0
        if mix:
            aux = torch.load(base / f"Client_{cid}_global_model.pt", map_location="cpu",
                             weights_only=False).to(pl.DEVICE).eval()
            w = lam.get(cid, 0.5)
        offset = 0
        with torch.no_grad():
            for bx, by in loader:
                out = w * local(bx.to(pl.DEVICE))
                if aux is not None:
                    out = out + (1.0 - w) * aux(bx.to(pl.DEVICE))
                probs[offset:offset + len(by)] += torch.softmax(out.float(), dim=1).cpu()
                offset += len(by)
        del local
        if aux is not None:
            del aux
        pl.clear_memory()
    pred = probs.argmax(dim=1)
    gy = global_test()[1]
    rec = pd.DataFrame(per_class_recall(pred, gy))
    rec["run_id"] = run_id
    return rec


def run_fl_cell(method, pattern, seed):
    last_error = None
    for batch in (BATCH, 32):
        try:
            if method == "Standalone":
                return pl.run_standalone(
                    exp=EXP, dataset=DATASET, pattern=pattern, seed=seed,
                    rounds=ROUNDS, num_clients=CLIENTS, batch_size=batch,
                    local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP,
                    data_root=DATA, scale=SCALE, resume=True, max_hours=6.0,
                    model_family=MODEL_FAMILY, local_lr=LOCAL_LR)
            p = pl.participation_probabilities(pattern, DATASET, CLIENTS, seed=seed)
            agg = "fedfv" if method == "FedFV" else "plain"
            return pl.run_federated(
                exp=EXP, method=method, dataset=DATASET, pattern=pattern,
                seed=seed, rounds=ROUNDS, num_clients=CLIENTS, batch_size=batch,
                local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
                valid_ratio=0.1, agg_mode=agg, aux_arch="googlenet",
                data_root=DATA, scale=SCALE, resume=True, max_hours=6.0,
                p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False,
                local_lr=LOCAL_LR)
        except Exception as e:  # noqa: BLE001
            last_error = e
            pl.clear_memory()
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda error" in msg) and batch == BATCH:
                print(f"[med] {method}/{pattern}/s{seed} OOM at batch {batch}, retry 32",
                      flush=True)
                continue
            raise
    raise RuntimeError(last_error)


def run_fl_seed(method, pattern, seed):
    res = run_fl_cell(method, pattern, seed)
    if res.get("skipped"):
        pth = pl._result_path(EXP, method, pattern, seed, scale=SCALE)
        res = json.loads(pth.read_text(encoding="utf-8"))
    run_id = res["run_id"]
    log = ROOT / "logs" / EXP / run_id / "round_metrics.csv"
    rm = pd.read_csv(log)
    final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").copy()
    if len(final) == 0:
        final = rm[rm["round"] == rm["round"].max()].drop_duplicates(
            "client_id", keep="last").copy()
    final = final.sort_values("client_id")
    fair = pl.compute_fairness(final)
    rec = global_per_class_recall(res, final)
    row, rec2 = summarize(rec.to_dict("records"),
                          {"arm": "federated", "method": method, "pattern": pattern,
                           "seed": seed, "run_id": run_id,
                           "seconds": float(res.get("seconds", np.nan)),
                           "FM": float(fair["FM"]), "min_acc": float(fair["min_acc"]),
                           "mean_single_gap": np.nan})
    row["method"] = method
    row["pattern"] = pattern
    row["seed"] = seed
    append_dedup(OUT_MAIN, pd.DataFrame([row]), ["method", "pattern", "seed"])
    append_dedup(OUT_REC, rec2.assign(method=method, pattern=pattern, seed=seed),
                 ["method", "pattern", "seed", "class_id"])
    print(f"[med] done {method}/{pattern}/s{seed}: AM={row['AM']:.4f} "
          f"FM={row['FM']:.4f} gap={row['class_gap']:.4f} "
          f"worst={row['worst_class']} {row['worst_class_recall']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["central", "fl", "all"], default="all")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--patterns", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    methods = args.methods or METHODS
    patterns = args.patterns or PATTERNS
    seeds = args.seeds or SEEDS

    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    if args.part in ("central", "all"):
        done = set()
        if OUT_MAIN.exists():
            prev = pd.read_csv(OUT_MAIN)
            done = set(prev.loc[prev.method == "Centralized", "seed"].astype(int))
        for seed in seeds:
            if seed in done:
                print(f"[central] skip seed {seed}")
                continue
            run_centralized_seed(seed)
    if args.part in ("fl", "all"):
        total = len(seeds) * len(methods) * len(patterns)
        done = 0
        for seed in seeds:
            for method in methods:
                for pattern in patterns:
                    done += 1
                    print(f"\n[med] {done}/{total} {method}/{pattern}/s{seed}", flush=True)
                    run_fl_seed(method, pattern, seed)
    print("\n[med] done")
    if OUT_MAIN.exists():
        print(pd.read_csv(OUT_MAIN).to_string(index=False))


if __name__ == "__main__":
    main()
