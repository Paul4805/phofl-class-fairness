"""Build a 10-client MedMNIST partition for the medical class-fairness run.

Default dataset: BloodMNIST (8 classes, 28x28 RGB). Public, de-identified,
CC BY 4.0; downloaded from Zenodo (MedMNIST v2, record 10519652).

Protocol, matching the CIFAR-10 half-scale runs:
  * pooled training data = official train + val
  * 10 clients via Dirichlet(0.5) label-skew partition
  * each client is split 80/20 into local train / local test
  * the official test split is kept as a global held-out set
  * images are bilinearly resized 28 -> 32 and normalised to [-1, 1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SPECS = {
    "BloodMNIST": {
        "classes": ["basophil", "eosinophil", "erythroblast",
                    "immature_granulocyte", "lymphocyte", "monocyte",
                    "neutrophil", "platelet"],
    },
    "DermaMNIST": {
        "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    },
}
N_CLIENTS = 10
SIZE = 32
ALPHA = 0.5
TRAIN_RATIO = 0.8
SEED = 1


def to_tensor(images: np.ndarray) -> torch.Tensor:
    x = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    x = (x - 0.5) / 0.5
    if x.shape[-1] != SIZE:
        x = F.interpolate(x, size=SIZE, mode="bilinear", align_corners=False)
    return x


def dirichlet_partition(y: np.ndarray, n_clients: int, n_classes: int,
                        alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    stats = np.zeros((n_clients, n_classes), dtype=np.int64)
    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        props = rng.dirichlet(np.repeat(alpha, n_clients))
        props = props / props.sum()
        cuts = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx_k, cuts)):
            parts[cid].extend(part.tolist())
            stats[cid, k] = len(part)
    for cid in range(n_clients):
        if not parts[cid]:
            donor = max(range(n_clients), key=lambda j: len(parts[j]))
            parts[cid].append(parts[donor].pop())
    return [np.asarray(sorted(p), dtype=np.int64) for p in parts], stats


def balanced_partition(y: np.ndarray, n_clients: int, n_classes: int, seed: int):
    """Stratified IID split: every client gets the same class mix and size."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    stats = np.zeros((n_clients, n_classes), dtype=np.int64)
    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        base, rem = divmod(len(idx_k), n_clients)
        start = 0
        for cid in range(n_clients):
            take = base + (1 if cid < rem else 0)
            chunk = idx_k[start:start + take]
            start += take
            parts[cid].extend(chunk.tolist())
            stats[cid, k] = take
    return [np.asarray(sorted(p), dtype=np.int64) for p in parts], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BloodMNIST", choices=sorted(SPECS))
    ap.add_argument("--partition", default="dir", choices=["dir", "balanced"])
    args = ap.parse_args()
    spec = SPECS[args.dataset]
    classes = spec["classes"]
    n_classes = len(classes)
    src = ROOT / "phofl_med" / "medmnist" / f"{args.dataset.lower()}.npz"
    base = ROOT / "phofl_med" / "data_cap"
    out = base / (args.dataset if args.partition == "dir"
                  else f"balanced/{args.dataset}")
    if (out / "train" / "0.npz").exists() and (out / "global_test.npz").exists():
        print("already generated", out)
        return
    if not src.exists():
        raise SystemExit(f"missing {src}; download the MedMNIST npz first")

    blob = np.load(src)
    train_x = np.concatenate([blob["train_images"], blob["val_images"]], axis=0)
    train_y = np.concatenate([blob["train_labels"], blob["val_labels"]], axis=0).ravel()
    pooled_x = to_tensor(train_x).numpy()
    pooled_y = train_y.astype(np.int64)
    global_x = to_tensor(blob["test_images"]).numpy()
    global_y = blob["test_labels"].ravel().astype(np.int64)

    if args.partition == "dir":
        parts, stats = dirichlet_partition(pooled_y, N_CLIENTS, n_classes, ALPHA, SEED)
    else:
        parts, stats = balanced_partition(pooled_y, N_CLIENTS, n_classes, SEED)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    for cid, idx in enumerate(parts):
        xtr, xte, ytr, yte = train_test_split(
            pooled_x[idx], pooled_y[idx], train_size=TRAIN_RATIO, shuffle=True,
            random_state=SEED + cid)
        np.savez_compressed(out / "train" / f"{cid}.npz",
                            data={"x": xtr, "y": ytr.astype(np.int64)})
        np.savez_compressed(out / "test" / f"{cid}.npz",
                            data={"x": xte, "y": yte.astype(np.int64)})

    np.savez_compressed(out / "global_test.npz", x=global_x, y=global_y)
    config = {"dataset": args.dataset, "num_clients": N_CLIENTS,
              "num_classes": n_classes, "non_iid": True, "balance": False,
              "partition": "dir", "alpha": ALPHA, "train_ratio": TRAIN_RATIO,
              "input_size": SIZE, "class_names": classes,
              "pooled_train": int(len(pooled_y)),
              "global_test": int(len(global_y)),
              "per_client_counts": stats.tolist()}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("dataset", args.dataset)
    print("pooled train", len(pooled_y), "global test", len(global_y))
    print("global test class counts",
          np.bincount(global_y, minlength=n_classes).tolist())
    print("per-client train class counts (rows=clients, cols=classes):")
    print(stats)
    print("client train sizes", [int(len(p) * TRAIN_RATIO) for p in parts])
    print("wrote", out)


if __name__ == "__main__":
    main()
