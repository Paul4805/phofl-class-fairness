"""Step 2: confusion-targeted local loss (lambda=0.1), PHP-FL / linear."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import phofl_lab as pl

HALF = __import__("pathlib").Path("phofl_half")
DATA = HALF / "data_cap"
OUT = __import__("pathlib").Path("phofl_method")
OUT.mkdir(parents=True, exist_ok=True)
EXP = "STEP2"
SCALE = "half10"
SEED = 0
ROUNDS = 60
CLIENTS = 10
BATCH = 64
LOCAL_EPOCHS = 1
EVAL_GAP = 10
MODEL_FAMILY = "HtM4"
LAMBDA = 0.1
TOP_PAIRS = 3
MARGIN = 1.0

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

pl.configure(HALF, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()
import flcore.clients.clientbase as cb

client_php_amp, _ = pl.make_amp_client_classes()


class clientConfusion(client_php_amp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conf_pairs = None

    def _ensure_conf_pairs(self):
        if self._conf_pairs is not None:
            return
        model = cb.load_item(self.role, "model", self.save_folder_name)
        _, valid = self.load_train_valid_data()
        model.eval()
        conf = np.zeros((10, 10), dtype=np.int64)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=pl.USE_AMP, dtype=torch.float16):
            for x, y in valid:
                x = x.to(pl.DEVICE)
                pred = model(x).argmax(1).cpu().numpy()
                yy = y.cpu().numpy()
                for t, p in zip(yy, pred):
                    if 0 <= int(t) < 10 and 0 <= int(p) < 10:
                        conf[int(t), int(p)] += 1
        pairs = []
        for a in range(10):
            for b in range(10):
                if a != b and conf[a, b] > 0:
                    pairs.append((int(conf[a, b]), a, b))
        pairs.sort(reverse=True)
        self._conf_pairs = [(a, b) for _, a, b in pairs[:TOP_PAIRS]]
        del model
        pl.clear_memory()

    def confusion_penalty(self, output, y):
        if not self._conf_pairs:
            return 0.0
        pen = 0.0
        for a, b in self._conf_pairs:
            ma = y == a
            mb = y == b
            if ma.any():
                pen = pen + F.relu(output[ma, b] - output[ma, a] + MARGIN).mean()
            if mb.any():
                pen = pen + F.relu(output[mb, a] - output[mb, b] + MARGIN).mean()
        return LAMBDA * pen / max(1, len(self._conf_pairs) * 2)

    def train(self, round):
        self._ensure_conf_pairs()
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
def global_confusion(run_id, method, pattern):
    loader = global_test_loader()
    logs = HALF / "logs" / ("C_half" if run_id.startswith("C_half") else EXP) / run_id / "round_metrics.csv"
    rm = pd.read_csv(logs)
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
    conf = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(target, pred):
        conf[int(t), int(p)] += 1
    support = conf.sum(axis=1).astype(float)
    recall = np.array([conf[k, k] / support[k] if support[k] else np.nan for k in range(10)])
    return recall, conf, support


p = pl.participation_probabilities("linear", "Cifar10", CLIENTS, seed=SEED)
res = pl.run_federated(
    exp=EXP, method="PHPFL", dataset="Cifar10", pattern="linear",
    seed=SEED, rounds=ROUNDS, num_clients=CLIENTS, batch_size=BATCH,
    local_epochs=LOCAL_EPOCHS, eval_gap=EVAL_GAP, tau=0.2, sig_k=5.0,
    valid_ratio=0.1, agg_mode="plain", aux_arch="googlenet",
    data_root=DATA, scale=SCALE, resume=True, max_hours=4.0,
    p_override=p, model_family=MODEL_FAMILY, log_active_metrics=False,
    client_cls=clientConfusion)

run_id = res["run_id"]
rm = pd.read_csv(HALF / "logs" / EXP / run_id / "round_metrics.csv")
final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last").sort_values("client_id")
fair = pl.compute_fairness(final)
rec, conf, support = global_confusion(run_id, "PHPFL", "linear")

base_id = "C_half__PHPFL__linear__s0__half10__3cf8f53c"
base_rec, base_conf, base_support = global_confusion(base_id, "PHPFL", "linear")
base_gap = float(np.nanmax(base_rec) - np.nanmin(base_rec))
method_gap = float(np.nanmax(rec) - np.nanmin(rec))
reduction = (base_gap - method_gap) / base_gap

dog_to_cat_base = float(base_conf[5, 3] / max(base_support[5], 1))
dog_to_cat_method = float(conf[5, 3] / max(support[5], 1))

per_class = pd.DataFrame({
    "class": CLASSES,
    "recall_baseline": base_rec,
    "recall_method": rec,
    "delta": rec - base_rec,
})
per_class.to_csv(OUT / "STEP2_per_class.csv", index=False)

summary = {
    "step": 2, "method": "confusion_targeted_loss", "run_id": run_id,
    "AM": fair["AM"], "FM": fair["FM"], "class_gap": method_gap,
    "baseline_class_gap": base_gap, "gap_reduction_rel": reduction,
    "dog_recall": float(rec[5]), "frog_recall": float(rec[6]),
    "dog_to_cat_baseline": dog_to_cat_base,
    "dog_to_cat_method": dog_to_cat_method,
    "dog_to_cat_delta": dog_to_cat_method - dog_to_cat_base,
    "worst_class": CLASSES[int(np.nanargmin(rec))],
    "worst_class_recall": float(np.nanmin(rec)),
    "best_class": CLASSES[int(np.nanargmax(rec))],
    "best_class_recall": float(np.nanmax(rec)),
}
pd.DataFrame([summary]).to_csv(OUT / "STEP2_summary.csv", index=False)

closed = reduction > 0.30
with open(OUT / "STEP2_findings.md", "w", encoding="utf-8") as f:
    f.write("# Step 2 - confusion-targeted loss\n\n")
    f.write(
        f"Confusion-targeted local loss (top-3 validation confusion pairs, margin 1.0, "
        f"lambda=0.1) produced AM={fair['AM']:.4f}, FM={fair['FM']:.4f}, and a class "
        f"gap of {method_gap:.4f} (baseline {base_gap:.4f}; relative reduction "
        f"{reduction*100:.1f}%). Dog recall moved from {base_rec[5]:.4f} to {rec[5]:.4f}; "
        f"frog recall moved from {base_rec[6]:.4f} to {rec[6]:.4f}. Dog-to-cat error "
        f"rate moved from {dog_to_cat_base:.4f} to {dog_to_cat_method:.4f} "
        f"({'decreased' if dog_to_cat_method < dog_to_cat_base else 'did not decrease'}). "
        + ("The >30% reduction threshold is met; the ladder stops here."
           if closed else
           "The >30% reduction threshold is not met; proceed to Step 3.")
    )

print(pd.DataFrame([summary]).to_string(index=False))
print(per_class.to_string(index=False))
print("CLOSED" if closed else "CONTINUE")
