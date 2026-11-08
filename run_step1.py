"""Step 1: class-balanced rehearsal (50 samples/class), PHP-FL / linear."""
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
OUT = Path("phofl_method")
OUT.mkdir(parents=True, exist_ok=True)
EXP = "STEP1"
SCALE = "half10"
SEED = 0
ROUNDS = 60
CLIENTS = 10
BATCH = 64
LOCAL_EPOCHS = 1
EVAL_GAP = 10
MODEL_FAMILY = "HtM4"
MEMORY_PER_CLASS = 50

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

pl.configure(HALF, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()
import flcore.servers.serverphp as sp

client_php_amp, _ = pl.make_amp_client_classes()


class MixedLoader:
    """Yield batches with half fresh samples and half class-balanced memory."""
    def __init__(self, base, client):
        self.base = base
        self.client = client

    def __iter__(self):
        self.client._ensure_buffer(self.base.dataset)
        total = len(self.client._buf_y_all)
        for x, y in self.base:
            if total == 0:
                yield x, y
                continue
            n = x.shape[0]
            n_fresh = max(1, n // 2)
            n_mem = max(1, n - n_fresh)
            idx = torch.randint(0, total, (n_mem,))
            xm = self.client._buf_x_all[idx].clone()
            ym = self.client._buf_y_all[idx].clone()
            yield torch.cat([x[:n_fresh], xm], dim=0), torch.cat([y[:n_fresh], ym], dim=0)


class clientRehearsal(client_php_amp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buf_x_all = None
        self._buf_y_all = None

    def _ensure_buffer(self, dataset):
        if self._buf_x_all is not None:
            return
        buckets_x = [[] for _ in range(10)]
        buckets_y = [[] for _ in range(10)]
        for x, y in dataset:
            c = int(y.item()) if torch.is_tensor(y) else int(y)
            if 0 <= c < 10 and len(buckets_x[c]) < MEMORY_PER_CLASS:
                buckets_x[c].append(x.detach().clone())
                buckets_y[c].append(int(c))
        xs, ys = [], []
        for c in range(10):
            if buckets_x[c]:
                xs.append(torch.stack(buckets_x[c]))
                ys.append(torch.tensor(buckets_y[c], dtype=torch.long))
        if xs:
            self._buf_x_all = torch.cat(xs, dim=0)
            self._buf_y_all = torch.cat(ys, dim=0)
        else:
            self._buf_x_all = torch.empty(0)
            self._buf_y_all = torch.empty(0, dtype=torch.long)

    def load_train_valid_data(self, batch_size=None):
        train, valid = super().load_train_valid_data(batch_size)
        return MixedLoader(train, self), valid


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
def global_per_class_recall(result, client_df):
    loader = global_test_loader()
    run_id = result["run_id"]
    probs_sum = np.zeros((len(loader.dataset), 10), dtype=np.float64)
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
    return np.array([float((pred[target == k] == k).mean()) for k in range(10)])


p = pl.participation_probabilities("linear", "Cifar10", CLIENTS, seed=SEED)
res = pl.run_federated(
    exp=EXP, method="PHPFL", dataset="Cifar10", pattern="linear",
    seed=SEED, rounds=ROUNDS, num_clients=CLIENTS, batch_size=BATCH,
    local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
    valid_ratio=0.1, agg_mode="plain", aux_arch="googlenet",
    data_root=DATA, scale=SCALE, resume=True, max_hours=4.0,
    p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False,
    client_cls=clientRehearsal)

run_id = res["run_id"]
rm = pd.read_csv(HALF / "logs" / EXP / run_id / "round_metrics.csv")
final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").sort_values("client_id")
fair = pl.compute_fairness(final)
rec = global_per_class_recall(res, final)

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
per_class.to_csv(OUT / "STEP1_per_class.csv", index=False)

summary = {
    "step": 1, "method": "class_balanced_rehearsal", "run_id": run_id,
    "AM": fair["AM"], "FM": fair["FM"], "class_gap": method_gap,
    "baseline_class_gap": base_gap,
    "gap_reduction_rel": reduction,
    "dog_recall": float(rec[5]), "frog_recall": float(rec[6]),
    "worst_class": CLASSES[int(np.argmin(rec))],
    "worst_class_recall": float(rec.min()),
    "best_class": CLASSES[int(np.argmax(rec))],
    "best_class_recall": float(rec.max()),
}
pd.DataFrame([summary]).to_csv(OUT / "STEP1_summary.csv", index=False)

closed = reduction > 0.30
with open(OUT / "STEP1_findings.md", "w", encoding="utf-8") as f:
    f.write("# Step 1 - class-balanced rehearsal\n\n")
    f.write(
        f"Class-balanced rehearsal (50 samples/class, mixed 1:1 with fresh batches) "
        f"produced AM={fair['AM']:.4f}, FM={fair['FM']:.4f}, and a class gap of "
        f"{method_gap:.4f} (baseline {base_gap:.4f}; relative reduction "
        f"{reduction*100:.1f}%). Dog recall moved from {base_rec[5]:.4f} to "
        f"{rec[5]:.4f}; frog recall moved from {base_rec[6]:.4f} to {rec[6]:.4f}. "
        f"The worst class is {CLASSES[int(np.argmin(rec))]} at {rec.min():.4f}. "
        + ("The >30% reduction threshold is met; the ladder stops here."
           if closed else
           "The >30% reduction threshold is not met; proceed to Step 2.")
    )

print(pd.DataFrame([summary]).to_string(index=False))
print(per_class.to_string(index=False))
print("CLOSED" if closed else "CONTINUE")
