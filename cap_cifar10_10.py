"""Cap the 10-client CIFAR-10 data for a feasible 6 GB half-scale run."""
import json
import shutil
from pathlib import Path

import numpy as np

src = Path("phofl_half/data/Cifar10")
dst = Path("phofl_half/data_cap/Cifar10")
if (dst / "train" / "0.npz").exists() and (dst / "test" / "0.npz").exists():
    print("already capped", dst)
    raise SystemExit(0)

rng = np.random.default_rng(0)
caps = {"train": 2500, "test": 500}
for split in ["train", "test"]:
    (dst / split).mkdir(parents=True, exist_ok=True)
    for cid in range(10):
        d = np.load(src / split / f"{cid}.npz", allow_pickle=True)["data"].tolist()
        x, y = d["x"], d["y"]
        n = len(y)
        cap = min(caps[split], n)
        idx = rng.choice(n, size=cap, replace=False)
        np.savez_compressed(dst / split / f"{cid}.npz",
                            data={"x": x[idx].astype(np.float32), "y": y[idx].astype(np.int64)})
        print(split, cid, n, "->", cap)
shutil.copy2(src / "config.json", dst / "config.json")
cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
cfg["smoke_cap"] = caps
(dst / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
print("wrote", dst)
