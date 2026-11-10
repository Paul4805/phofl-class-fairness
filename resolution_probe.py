"""Small centralized resolution probe on CIFAR-10 (upsampled) and STL-10 (native 96)."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(".")
OUT = ROOT / "phofl" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print("device", DEVICE, "torch", torch.__version__)


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def make_transform(res):
    ops = []
    if res != 32:
        ops.append(transforms.Resize(res, antialias=True))
    ops += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(ops)


def stratified_indices(targets, per_class, seed=0):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    out = []
    for k in range(10):
        idx = np.where(targets == k)[0]
        rng.shuffle(idx)
        out.extend(idx[:per_class].tolist())
    return np.asarray(sorted(out))


def evaluate(model, loader):
    model.eval()
    cm = np.zeros((10, 10), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[int(t), int(p)] += 1
    return cm


def pair_metrics(cm, a, b):
    support_a = cm[a].sum()
    support_b = cm[b].sum()
    ab = cm[a, b]
    ba = cm[b, a]
    error_a = 1 - cm[a, a] / max(support_a, 1)
    error_b = 1 - cm[b, b] / max(support_b, 1)
    return {
        "a_to_b_rate": ab / max(support_a, 1),
        "b_to_a_rate": ba / max(support_b, 1),
        "cross_error_rate": (ab + ba) / max(support_a + support_b, 1),
        "all_error_rate": (error_a + error_b) / 2,
        "class_error_a": error_a,
        "class_error_b": error_b,
    }


def train_probe(dataset_name, res, train_ds, test_ds, classes, pair_specs, epochs, batch):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0,
                             pin_memory=True)
    model = SmallCNN(10).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total = 0
        correct = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += y.numel()
            correct += int((out.argmax(1) == y).sum().item())
        sched.step()
        print(f"  {dataset_name} res={res} epoch={ep+1}/{epochs} "
              f"train_acc={correct/max(total,1):.4f} elapsed={time.time()-t0:.1f}s")
    cm = evaluate(model, test_loader)
    support = cm.sum(axis=1)
    rec = np.array([cm[k, k] / max(support[k], 1) for k in range(10)])
    row = {
        "dataset": dataset_name, "resolution": res,
        "AM": float(np.trace(cm) / support.sum()),
        "dog_recall": float(rec[classes.index("dog")]),
        "cat_recall": float(rec[classes.index("cat")]),
        "elapsed_s": time.time() - t0,
        "torch": torch.__version__,
    }
    for pair_name, a_name, b_name in pair_specs:
        m = pair_metrics(cm, classes.index(a_name), classes.index(b_name))
        for k, v in m.items():
            row[f"{pair_name}_{k}"] = float(v)
    raw = pd.DataFrame(cm, index=classes, columns=classes)
    raw.index.name = "true"
    raw.to_csv(OUT / f"probe_confusion_{dataset_name}_{res}.csv")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def cifar_probe(res):
    tf = make_transform(res)
    train = datasets.CIFAR10(root=str(ROOT / "third_party/PHP-FL-main/dataset/Cifar10/rawdata"),
                             train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root=str(ROOT / "third_party/PHP-FL-main/dataset/Cifar10/rawdata"),
                            train=False, download=True, transform=tf)
    tr_idx = stratified_indices(train.targets, 2000, seed=0)
    te_idx = stratified_indices(test.targets, 500, seed=0)
    train_ds = Subset(train, tr_idx)
    test_ds = Subset(test, te_idx)
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    pairs = [("dog_cat", "dog", "cat"), ("deer_horse", "deer", "horse"),
             ("auto_truck", "automobile", "truck")]
    return train_probe("cifar10_upsampled", res, train_ds, test_ds, classes, pairs, 15, 128)


def stl_probe(res):
    tf = make_transform(res)
    root = OUT / "stl10"
    train = datasets.STL10(root=str(root), split="train", download=True, transform=tf)
    test = datasets.STL10(root=str(root), split="test", download=True, transform=tf)
    classes = ["airplane", "bird", "car", "cat", "deer",
               "dog", "horse", "monkey", "ship", "truck"]
    pairs = [("dog_cat", "dog", "cat"), ("deer_horse", "deer", "horse"),
             ("auto_truck", "car", "truck")]
    return train_probe("stl10_native96", res, train, test, classes, pairs, 20, 64)


def main():
    rows = []
    for res in [32, 64, 96]:
        rows.append(cifar_probe(res))
        pd.DataFrame(rows).to_csv(OUT / "resolution_probe_cifar.csv", index=False)
    stl_rows = []
    for res in [32, 64, 96]:
        stl_rows.append(stl_probe(res))
        pd.DataFrame(stl_rows).to_csv(OUT / "resolution_probe_stl.csv", index=False)
    allrows = pd.DataFrame(rows + stl_rows)
    allrows.to_csv(OUT / "resolution_probe_all.csv", index=False)
    print(allrows.to_string(index=False))


if __name__ == "__main__":
    main()
