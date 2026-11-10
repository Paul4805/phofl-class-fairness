"""Full 10x10 confusion matrices and pair error budgets for baseline + ladder."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import phofl_lab as pl

ROOT = Path(".")
HALF = ROOT / "phofl_half"
OUT = ROOT / "phofl" / "analysis"
CM = OUT / "confusion_matrices"
CM.mkdir(parents=True, exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
PAIRS = [("dog", "cat", 5, 3), ("deer", "horse", 4, 7),
         ("automobile", "truck", 1, 9)]
BASELINE_ID = "C_half__PHPFL__linear__s0__half10__3cf8f53c"
RUNS = {
    "baseline": ("C_half", BASELINE_ID),
    "STEP1_rehearsal": ("STEP1", "STEP1__PHPFL__linear__s0__half10__0669227f"),
    "STEP2_confusion": ("STEP2", "STEP2__PHPFL__linear__s0__half10__130ba15d"),
    "STEP3_worst_agg": ("STEP3", "STEP3__PHPFL__linear__s0__half10__e6b6e177"),
    "STEP4_distill": ("STEP4", "STEP4__PHPFL__linear__s0__half10__682b80ac"),
}

pl.configure(HALF, "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()


def test_loader():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    raw = pl.REPO_DIR / "dataset" / "Cifar10" / "rawdata"
    ds = datasets.CIFAR10(root=str(raw), train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)


def load_model(run_id, role, name):
    path = HALF / "run_store" / run_id / "Cifar10" / "PHPFL" / f"{role}_{name}.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def confusion_for_run(exp, run_id):
    loader = test_loader()
    rm = pd.read_csv(HALF / "logs" / exp / run_id / "round_metrics.csv")
    final = rm[rm["is_final"] == 1].drop_duplicates("client_id", keep="last")
    lam = {int(r.client_id): float(r.lambda_i) for _, r in final.iterrows()}
    probs = np.zeros((len(loader.dataset), 10), dtype=np.float64)
    for cid in range(10):
        local = load_model(run_id, f"Client_{cid}", "model").to(pl.DEVICE).eval()
        aux = load_model(run_id, f"Client_{cid}", "global_model").to(pl.DEVICE).eval()
        w = lam.get(cid, 0.5)
        start = 0
        for x, y in loader:
            x = x.to(pl.DEVICE)
            out = w * local(x) + (1.0 - w) * aux(x)
            probs[start:start + len(y)] += torch.softmax(out, dim=1).cpu().numpy()
            start += len(y)
        del local, aux
        pl.clear_memory()
    pred = probs.argmax(axis=1)
    target = np.asarray(loader.dataset.targets, dtype=np.int64)
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(target, pred):
        cm[int(t), int(p)] += 1
    return cm


rows = []
summary = []
for label, (exp, run_id) in RUNS.items():
    print("confusion", label)
    cm = confusion_for_run(exp, run_id)
    support = cm.sum(axis=1).astype(float)
    recall = np.array([cm[k, k] / support[k] for k in range(10)])
    acc = float(np.trace(cm) / support.sum())
    raw = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    raw.index.name = "true"
    raw.to_csv(CM / f"{label}.csv")
    norm = raw.div(support, axis=0)
    norm.to_csv(CM / f"{label}_normalized.csv")
    summary.append({"run": label, "AM": acc, "worst_class": CLASSES[int(np.argmin(recall))],
                    "worst_recall": float(recall.min()),
                    "dog_recall": float(recall[5]), "cat_recall": float(recall[3])})
    errors = support - np.diag(cm)
    for a_name, b_name, a, b in PAIRS:
        ab = int(cm[a, b])
        ba = int(cm[b, a])
        cross = ab + ba
        pair_support = support[a] + support[b]
        total_class_errors = errors[a] + errors[b]
        rows.append({
            "run": label, "pair": f"{a_name}-{b_name}",
            "a_to_b": ab, "b_to_a": ba,
            "a_to_b_rate": ab / support[a], "b_to_a_rate": ba / support[b],
            "cross_pair_rate": cross / pair_support,
            "class_error_a": errors[a] / support[a],
            "class_error_b": errors[b] / support[b],
            "combined_class_error": (errors[a] / support[a] + errors[b] / support[b]) / 2,
            "cross_share_of_class_errors": cross / max(total_class_errors, 1),
            "direction_asymmetry": abs(ab / support[a] - ba / support[b]),
        })

pd.DataFrame(summary).to_csv(OUT / "confusion_summary.csv", index=False)
pair = pd.DataFrame(rows)
pair.to_csv(OUT / "confusion_pair_budget.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
print(pair.to_string(index=False))
print("wrote", OUT / "confusion_pair_budget.csv")
