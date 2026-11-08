"""Generate a 10-client CIFAR-10 Dirichlet(0.5) dataset for the C escalation."""
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

repo = Path("third_party/PHP-FL-main")
out_root = Path("phofl_half/data/Cifar10")
if (out_root / "train" / "0.npz").exists() and (out_root / "test" / "0.npz").exists():
    print("already generated", out_root)
    raise SystemExit(0)

spec = importlib.util.spec_from_file_location(
    "phpfl_dataset_utils", repo / "dataset" / "utils" / "dataset_utils.py")
dsu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsu)
dsu.batch_size = 64
dsu.alpha = 0.5
dsu.train_ratio = 0.8

random.seed(1)
np.random.seed(1)
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
raw = repo / "dataset" / "Cifar10" / "rawdata"
tr = datasets.CIFAR10(root=str(raw), train=True, download=True, transform=tf)
te = datasets.CIFAR10(root=str(raw), train=False, download=True, transform=tf)
images = np.concatenate([tr.data, te.data], axis=0).astype(np.float32) / 255.0
images = (images - 0.5) / 0.5
labels = np.concatenate([np.asarray(tr.targets), np.asarray(te.targets)], axis=0)
images = np.transpose(images, (0, 3, 1, 2))

X, y, statistic = dsu.separate_data((images, labels), 10, 10, niid=True,
                                   balance=False, partition="dir", class_per_client=2)
train_data, test_data = dsu.split_data(X, y)
for sub in ["train", "test"]:
    (out_root / sub).mkdir(parents=True, exist_ok=True)
dsu.save_file(str(out_root / "config.json"), str(out_root / "train") + "/",
              str(out_root / "test") + "/", train_data, test_data, 10, 10,
              statistic, niid=True, balance=False, partition="dir")
print("wrote", out_root)
