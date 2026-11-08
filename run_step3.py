"""Step 3: worst-class-weighted aggregation (gamma=5.0), PHP-FL / linear."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import phofl_lab as pl

HALF = Path("phofl_half")
DATA = HALF / "data_cap"
OUT = Path("phofl_method")
OUT.mkdir(parents=True, exist_ok=True)
EXP = "STEP3"
SCALE = "half10"
SEED = 0
ROUNDS = 60
CLIENTS = 10
BATCH = 64
LOCAL_EPOCHS = 1
EVAL_GAP = 10
MODEL_FAMILY = "HtM4"
GAMMA = 5.0

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
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def global_recall(run_id, exp):
    loader = global_test_loader()
    rm = pd.read_csv(HALF / "logs" / exp / run_id / "round_metrics.csv")
    final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last")
    lam = {int(r.client_id): float(r.lambda_i) for _, r in final.iterrows()}
    probs_sum = np.zeros((len(loader.dataset), 10), dtype=np.float64)
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
    return np.array([float((pred[target == k] == k).mean()) for k in range(10)])


p = pl.participation_probabilities("linear", "Cifar10", CLIENTS, seed=SEED)
res = pl.run_federated(
    exp=EXP, method="PHPFL", dataset="Cifar10", pattern="linear",
    seed=SEED, rounds=ROUNDS, num_clients=CLIENTS, batch_size=BATCH,
    local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
    valid_ratio=0.1, agg_mode="worst_class", aux_arch="googlenet",
    data_root=DATA, scale=SCALE, resume=True, max_hours=4.0,
    p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False)

run_id = res["run_id"]
rm = pd.read_csv(HALF / "logs" / EXP / run_id / "round_metrics.csv")
final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").sort_values("client_id")
fair = pl.compute_fairness(final)
rec = global_recall(run_id, EXP)

base = pd.read_csv(HALF / "results" / "C_half_global_recall.csv")
base = base[(base.method == "PHPFL") & (base.pattern == "linear")].sort_values("class_id")
base_rec = base["recall"].to_numpy(dtype=float)
base_gap = float(base_rec.max() - base_rec.min())
method_gap = float(rec.max() - rec.min())
reduction = (base_gap - method_gap) / base_gap

per_class = pd.DataFrame({
    "class": CLASSES,
    "recall_baseline": base_rec,
    "recall_method": rec,
    "delta": rec - base_rec,
})
per_class.to_csv(OUT / "STEP3_per_class.csv", index=False)

summary = {
    "step": 3, "method": "worst_class_weighted_aggregation", "run_id": run_id,
    "gamma": GAMMA, "AM": fair["AM"], "FM": fair["FM"],
    "class_gap": method_gap, "baseline_class_gap": base_gap,
    "gap_reduction_rel": reduction,
    "dog_recall": float(rec[5]), "frog_recall": float(rec[6]),
    "worst_class": CLASSES[int(np.argmin(rec))],
    "worst_class_recall": float(rec.min()),
    "best_class": CLASSES[int(np.argmax(rec))],
    "best_class_recall": float(rec.max()),
}
pd.DataFrame([summary]).to_csv(OUT / "STEP3_summary.csv", index=False)

closed = reduction > 0.30
with open(OUT / "STEP3_findings.md", "w", encoding="utf-8") as f:
    f.write("# Step 3 - worst-class-weighted aggregation\n\n")
    f.write(
        f"Worst-class-weighted aggregation (gamma={GAMMA}) produced AM={fair['AM']:.4f}, "
        f"FM={fair['FM']:.4f}, and a class gap of {method_gap:.4f} (baseline "
        f"{base_gap:.4f}; relative reduction {reduction*100:.1f}%). Worst-class "
        f"recall is {rec.min():.4f} ({CLASSES[int(np.argmin(rec))]}); dog recall "
        f"is {rec[5]:.4f} and frog recall is {rec[6]:.4f}. "
        + ("The >30% reduction threshold is met; the ladder stops here."
           if closed else
           "The >30% reduction threshold is not met; Step 4 still runs because it "
           "tests a different mechanism.")
    )

print(pd.DataFrame([summary]).to_string(index=False))
print(per_class.to_string(index=False))
print("CLOSED" if closed else "CONTINUE")
