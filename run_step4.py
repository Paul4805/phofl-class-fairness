"""Step 4: per-class distillation KL weighting (T=1.0), PHP-FL / linear."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import phofl_lab as pl

HALF = Path("phofl_half")
DATA = HALF / "data_cap"
OUT = Path("phofl_method")
OUT.mkdir(parents=True, exist_ok=True)
EXP = "STEP4"
SCALE = "half10"
SEED = 0
ROUNDS = 60
CLIENTS = 10
BATCH = 64
LOCAL_EPOCHS = 1
EVAL_GAP = 10
MODEL_FAMILY = "HtM4"
T = 1.0

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

pl.configure(HALF, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()
import flcore.clients.clientbase as cb

client_php_amp, _ = pl.make_amp_client_classes()


class clientDistill(client_php_amp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kl_weights = None

    def _ensure_kl_weights(self):
        if self._kl_weights is not None:
            return
        model = cb.load_item(self.role, "model", self.save_folder_name)
        _, valid = self.load_train_valid_data()
        model.eval()
        correct = np.zeros(10, dtype=np.float64)
        total = np.zeros(10, dtype=np.float64)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=pl.USE_AMP, dtype=torch.float16):
            for x, y in valid:
                x = x.to(pl.DEVICE)
                pred = model(x).argmax(1).cpu().numpy()
                yy = y.cpu().numpy()
                for t, p in zip(yy, pred):
                    if 0 <= int(t) < 10:
                        total[int(t)] += 1
                        if int(p) == int(t):
                            correct[int(t)] += 1
        rec = np.where(total > 0, correct / np.maximum(total, 1), 0.0)
        self._kl_weights = torch.tensor(np.exp(rec / T), dtype=torch.float32, device=pl.DEVICE)
        del model
        pl.clear_memory()

    def weighted_kl(self, logp, p, y):
        if self._kl_weights is None:
            return self.KL_Loss(logp, p)
        w = self._kl_weights.to(logp.device)[y]
        loss = F.kl_div(logp, p, reduction="none").sum(dim=1)
        return (loss * w).mean()

    def train(self, round):
        self._ensure_kl_weights()
        return super().train(round)


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


def best_dog_arch(run_id, exp):
    rm = pd.read_csv(HALF / "logs" / exp / run_id / "round_metrics.csv")
    final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last")
    grouped = final.groupby("arch")["class_5_acc"].mean().sort_values(ascending=False)
    return str(grouped.index[0]), float(grouped.iloc[0]), grouped


p = pl.participation_probabilities("linear", "Cifar10", CLIENTS, seed=SEED)
res = pl.run_federated(
    exp=EXP, method="PHPFL", dataset="Cifar10", pattern="linear",
    seed=SEED, rounds=ROUNDS, num_clients=CLIENTS, batch_size=BATCH,
    local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
    valid_ratio=0.1, agg_mode="plain", aux_arch="googlenet",
    data_root=DATA, scale=SCALE, resume=True, max_hours=4.0,
    p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False,
    client_cls=clientDistill)

run_id = res["run_id"]
rm = pd.read_csv(HALF / "logs" / EXP / run_id / "round_metrics.csv")
final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").sort_values("client_id")
fair = pl.compute_fairness(final)
rec = global_recall(run_id, EXP)

base_id = "C_half__PHPFL__linear__s0__half10__3cf8f53c"
base = pd.read_csv(HALF / "results" / "C_half_global_recall.csv")
base = base[(base.method == "PHPFL") & (base.pattern == "linear")].sort_values("class_id")
base_rec = base["recall"].to_numpy(dtype=float)
base_gap = float(base_rec.max() - base_rec.min())
method_gap = float(rec.max() - rec.min())
reduction = (base_gap - method_gap) / base_gap

base_arch, base_dog, base_group = best_dog_arch(base_id, "C_half")
method_arch, method_dog, method_group = best_dog_arch(run_id, EXP)

per_class = pd.DataFrame({
    "class": CLASSES,
    "recall_baseline": base_rec,
    "recall_method": rec,
    "delta": rec - base_rec,
})
per_class.to_csv(OUT / "STEP4_per_class.csv", index=False)

summary = {
    "step": 4, "method": "per_class_distillation", "run_id": run_id,
    "temperature": T, "AM": fair["AM"], "FM": fair["FM"],
    "class_gap": method_gap, "baseline_class_gap": base_gap,
    "gap_reduction_rel": reduction,
    "dog_recall": float(rec[5]), "frog_recall": float(rec[6]),
    "baseline_best_dog_arch": base_arch, "baseline_best_dog_recall": base_dog,
    "method_best_dog_arch": method_arch, "method_best_dog_recall": method_dog,
    "best_dog_arch_changed": bool(base_arch != method_arch),
    "worst_class": CLASSES[int(np.argmin(rec))],
    "worst_class_recall": float(rec.min()),
    "best_class": CLASSES[int(np.argmax(rec))],
    "best_class_recall": float(rec.max()),
}
pd.DataFrame([summary]).to_csv(OUT / "STEP4_summary.csv", index=False)

closed = reduction > 0.30
with open(OUT / "STEP4_findings.md", "w", encoding="utf-8") as f:
    f.write("# Step 4 - per-class distillation\n\n")
    f.write(
        f"Per-class KL weighting (weight exp(r_i(c)/T), T={T}) produced "
        f"AM={fair['AM']:.4f}, FM={fair['FM']:.4f}, and a class gap of "
        f"{method_gap:.4f} (baseline {base_gap:.4f}; relative reduction "
        f"{reduction*100:.1f}%). Dog recall moved from {base_rec[5]:.4f} to "
        f"{rec[5]:.4f}; frog recall moved from {base_rec[6]:.4f} to {rec[6]:.4f}. "
        f"The best dog architecture was {base_arch} (mean dog recall "
        f"{base_dog:.4f}) before and {method_arch} ({method_dog:.4f}) after; "
        f"the best-dog architecture {'changed' if base_arch != method_arch else 'did not change'}. "
        f"Worst class is {CLASSES[int(np.argmin(rec))]} at {rec.min():.4f}. "
        + ("The >30% reduction threshold is met; run this step with seed 1."
           if closed else
           "The >30% reduction threshold is not met; no seed-1 rerun is triggered.")
    )

print(pd.DataFrame([summary]).to_string(index=False))
print(per_class.to_string(index=False))
print("CLOSED" if closed else "CONTINUE")
