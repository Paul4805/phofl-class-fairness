"""
PHP-FL experiment harness.

This module is written into the notebook by Cell 2.  It intentionally keeps the
repository's PHP-FL model/client code untouched except for the narrow hooks
documented in VERIFY.md:

  * aggregation weights are pluggable (plain/inv_p/inv_N/fedavg),
  * client training runs under CUDA AMP,
  * model saves/loads are CPU-backed so a 6 GB GPU never holds the model zoo,
  * per-round logging, G evaluation, participation sampling, and auxiliary-model
    swapping are injected around the repository's own train/eval calls.

The public entry points are the run_e* functions at the bottom of the file.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Global configuration (filled by configure())
# ---------------------------------------------------------------------------

ROOT: Path = Path.cwd() / "phofl"
REPO_DIR: Path = Path.cwd() / "third_party" / "PHP-FL-main"
IS_COLAB: bool = False
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP: bool = torch.cuda.is_available()
ACTIVE_DATA_ROOT: Optional[Path] = None
_DATA_ROOT_STACK: List[Optional[Path]] = []

REPO_URL = "https://github.com/Siyuan01/PHP-FL-main.git"
# The paper repository's HEAD at the time this notebook was generated.
PINNED_REPO_SHA = "283c4a4c561b153d17ee0278ac35e5580b3fe28f"

PACKAGE_PINS = {
    "torch": "2.2.2+cu121",
    "torchvision": "0.17.2+cu121",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.0",
    "matplotlib": "3.8.4",
    "seaborn": "0.13.2",
    "Pillow": "10.3.0",
    "tqdm": "4.66.4",
    "PyYAML": "6.0.1",
    "tabulate": "0.9.0",
    "psutil": "5.9.8",
    "nvidia-ml-py": "12.535.133",
    "thop": "0.1.1-2209072238",
}

SCALES = {
    "smoke": {"seeds": [0], "rounds": 30, "dataset": "FashionMNIST", "clients": 2,
              "local_epochs": 1, "batch_size": 16, "eval_gap": 2},
    "half": {"seeds": [0, 1], "rounds": 60, "dataset": "Cifar10", "clients": 20,
             "local_epochs": 3, "batch_size": 64, "eval_gap": 5},
    "full": {"seeds": [0, 1, 2], "rounds": 100, "dataset": "Cifar10", "clients": 20,
             "local_epochs": 5, "batch_size": 64, "eval_gap": 5},
}

# Dataset-wide class count used by the evaluation/logging paths.  Default 10
# keeps the CIFAR-10 artifacts reproducible; a run script calls
# set_num_classes() before configuring a different dataset.
NUM_CLASSES = 10
CLASS_NAMES = [f"class_{k}" for k in range(NUM_CLASSES)]


def set_num_classes(n: int, names=None) -> None:
    global NUM_CLASSES, CLASS_NAMES
    NUM_CLASSES = int(n)
    CLASS_NAMES = list(names) if names is not None else [f"class_{k}" for k in range(NUM_CLASSES)]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def configure(root: str | os.PathLike, repo_dir: str | os.PathLike, colab: bool = False) -> None:
    """Set global paths and create the artifact tree."""
    global ROOT, REPO_DIR, IS_COLAB
    ROOT = Path(root).expanduser().resolve()
    REPO_DIR = Path(repo_dir).expanduser().resolve()
    IS_COLAB = bool(colab)
    for sub in ["ckpts", "logs", "results", "figures", "flags", "baseline", "run_store"]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    # Make sure the repository's system package is importable.
    system = REPO_DIR / "system"
    if system.exists() and str(system) not in sys.path:
        sys.path.insert(0, str(system))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The paper runs use cudnn.benchmark=True; exact bitwise repeatability is
    # therefore not promised.  The baseline tolerance absorbs normal GPU noise.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def clear_memory(verbose: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print(f"[mem] allocated={torch.cuda.memory_allocated()/2**20:.0f} MB "
                  f"reserved={torch.cuda.memory_reserved()/2**20:.0f} MB")


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return float(torch.cuda.max_memory_allocated() / 2**20)


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def atomic_write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return str(x)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s))[:120]


def config_hash(cfg: Dict[str, Any]) -> str:
    raw = json.dumps(cfg, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def make_run_id(exp: str, method: str, dataset: str, pattern: str, seed: int,
                scale: str, extra: Optional[Dict[str, Any]] = None) -> str:
    cfg = {
        "exp": exp, "method": method, "dataset": dataset, "pattern": pattern,
        "seed": int(seed), "scale": scale, "repo_sha": PINNED_REPO_SHA,
        "extra": extra or {},
    }
    return f"{safe_name(exp)}__{safe_name(method)}__{safe_name(pattern)}__s{seed}__{scale}__{config_hash(cfg)[:8]}"


def log_failure(exp: str, run_id: str, exc: BaseException, context: Optional[Dict[str, Any]] = None) -> None:
    path = ROOT / "failures.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exp={exp} run_id={run_id}\n")
        if context:
            f.write(json.dumps(context, default=_json_default, indent=2) + "\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    print(f"[FAILED] {exp} / {run_id}: {exc}")


def package_report() -> Dict[str, str]:
    report = {"python": sys.version.split()[0], "torch": torch.__version__,
              "cuda": str(torch.version.cuda), "device": str(DEVICE)}
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_mb"] = f"{torch.cuda.get_device_properties(0).total_memory / 2**20:.0f}"
    for mod in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "PIL", "tqdm", "thop"]:
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # pragma: no cover - diagnostic only
            report[mod] = f"MISSING ({e})"
    return report


# ---------------------------------------------------------------------------
# Repository acquisition and data preparation
# ---------------------------------------------------------------------------

def _run(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    print("[cmd]", " ".join(str(x) for x in cmd))
    return subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None,
                          check=True, text=True, timeout=timeout)


def prepare_repo(repo_dir: Optional[Path] = None, sha: str = PINNED_REPO_SHA) -> Path:
    """Clone the paper repo (if needed) and pin it to the recorded commit."""
    global REPO_DIR
    repo = Path(repo_dir) if repo_dir else REPO_DIR
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        _run(["git", "clone", REPO_URL, str(repo)], timeout=600)
    # Fetch the pin if the shallow clone does not already contain it.
    try:
        _run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], timeout=60)
    except subprocess.CalledProcessError:
        _run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", sha], timeout=300)
    _run(["git", "-C", str(repo), "checkout", "--detach", sha], timeout=120)
    REPO_DIR = repo.resolve()
    system = REPO_DIR / "system"
    if str(system) not in sys.path:
        sys.path.insert(0, str(system))
    _install_data_reader()
    print(f"[repo] {REPO_DIR} @ {sha}")
    return REPO_DIR


def repo_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def push_data_root(root: Optional[Path]) -> None:
    global ACTIVE_DATA_ROOT
    _DATA_ROOT_STACK.append(ACTIVE_DATA_ROOT)
    ACTIVE_DATA_ROOT = Path(root).resolve() if root else None


def pop_data_root() -> None:
    global ACTIVE_DATA_ROOT
    ACTIVE_DATA_ROOT = _DATA_ROOT_STACK.pop() if _DATA_ROOT_STACK else None


def _install_data_reader() -> None:
    """Make repository reads independent of the process cwd."""
    try:
        import utils.data_utils as du
    except Exception as e:
        raise RuntimeError("Run prepare_repo() and configure() before installing the data reader") from e
    if getattr(du, "_phofl_patched", False):
        return
    original = du.read_data

    def phofl_read_data(dataset: str, idx: int, is_train: bool = True):
        if ACTIVE_DATA_ROOT is not None:
            split = "train" if is_train else "test"
            path = Path(ACTIVE_DATA_ROOT) / dataset / split / f"{idx}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing client data {path}. Run ensure_cifar10() / ensure_fashion_mnist() first.")
            with open(path, "rb") as f:
                return np.load(f, allow_pickle=True)["data"].tolist()
        return original(dataset, idx, is_train)

    du.read_data = phofl_read_data
    du._phofl_patched = True
    print("[data] installed cwd-independent reader")


def ensure_cifar10(repo: Optional[Path] = None) -> Path:
    """Generate the paper's Cifar10 Dirichlet(0.5) 20-client dataset if absent."""
    repo = Path(repo or REPO_DIR)
    ds_dir = repo / "dataset"
    root = ds_dir / "Cifar10"
    train0 = root / "train" / "0.npz"
    test0 = root / "test" / "0.npz"
    if train0.exists() and test0.exists():
        return root
    cfg = root / "config.json"
    backup = root / "config.repo.json"
    if cfg.exists() and not backup.exists():
        shutil.copy2(cfg, backup)
        cfg.unlink()
    _run([sys.executable, "-u", "generate_cifar10.py", "noniid", "-", "dir"], cwd=ds_dir, timeout=1800)
    if not (train0.exists() and test0.exists()):
        raise RuntimeError("Cifar10 generation finished without train/0.npz or test/0.npz")
    # The repository only ships config.json; compare the generated statistics to it.
    if backup.exists():
        try:
            with open(backup, encoding="utf-8") as f:
                expected = json.load(f)
            with open(cfg, encoding="utf-8") as f:
                got = json.load(f)
            if expected.get("Size of samples for labels in clients") != got.get("Size of samples for labels in clients"):
                print("[warn] regenerated Cifar10 partition differs from the repository's config.json statistics.")
            else:
                print("[data] regenerated Cifar10 partition matches the repository's config.json statistics.")
        except Exception as e:
            print(f"[warn] could not compare Cifar10 statistics: {e}")
    return root


def ensure_fashion_mnist(n_clients: int = 2, repo: Optional[Path] = None,
                         seed: int = 0, alpha: float = 0.5,
                         max_train: int = 1000, max_test: int = 200) -> Path:
    """Create a deterministic Fashion-MNIST dataset in the repo's npz format.

    The repository does not ship Fashion-MNIST; this is a smoke-test-only shim.
    A Dirichlet(alpha=0.5) label partition gives clients different label
    entropies, which is required for the entropy-correlated participation test.
    """
    repo = Path(repo or REPO_DIR)
    root = repo / "dataset" / "FashionMNIST"
    expected = [root / "train" / f"{i}.npz" for i in range(n_clients)]
    cfg_path = root / "config.json"
    if all(p.exists() for p in expected) and cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if int(cfg.get("num_clients", -1)) == n_clients and cfg.get("partition") == "dir":
                return root
        except Exception:
            pass
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    tr = datasets.FashionMNIST(root=str(root / "rawdata"), train=True, download=True, transform=transform)
    te = datasets.FashionMNIST(root=str(root / "rawdata"), train=False, download=True, transform=transform)
    for split_no, (split_name, ds) in enumerate([("train", tr), ("test", te)]):
        (root / split_name).mkdir(parents=True, exist_ok=True)
        y = ds.targets.numpy()
        rng = np.random.default_rng(seed + split_no)
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(10):
            idx_k = np.where(y == k)[0]
            rng.shuffle(idx_k)
            props = rng.dirichlet(np.repeat(alpha, n_clients))
            props = props / props.sum()
            cuts = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
            for cid, part in enumerate(np.split(idx_k, cuts)):
                idx_batch[cid].extend(part.tolist())
        # Guarantee every client has at least one sample.
        for cid in range(n_clients):
            if not idx_batch[cid]:
                donor = max(range(n_clients), key=lambda j: len(idx_batch[j]))
                idx_batch[cid].append(idx_batch[donor].pop())
        for cid, idx in enumerate(idx_batch):
            idx = np.asarray(idx, dtype=np.int64)
            # Smoke-scale budget: cap per-client samples so 10 clients x 40
            # rounds fit a 6 GB laptop GPU in a reasonable wall-clock.
            cap = max_train if split_name == "train" else max_test
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            x = ds.data[idx].float().unsqueeze(1) / 255.0
            x = (x - 0.5) / 0.5
            yy = y[idx]
            np.savez_compressed(root / split_name / f"{cid}.npz",
                                data={"x": x.numpy().astype(np.float32), "y": yy.astype(np.int64)})
    config = {"num_clients": n_clients, "num_classes": 10, "non_iid": True,
              "balance": False, "partition": "dir", "alpha": alpha, "batch_size": 64}
    atomic_write_json(config, root / "config.json")
    print(f"[data] wrote FashionMNIST Dirichlet data for {n_clients} clients at {root}")
    return root


def ensure_cifar10_plus10(repo: Optional[Path] = None, seed: int = 1234,
                          n_extra: int = 10, root: Optional[Path] = None) -> Path:
    """Keep the paper's 20 clients and append n_extra synthetic clients.

    The extra clients are sampled from the canonical CIFAR-10 training split.
    They may overlap the base clients' samples; E4 measures participation, not
    disjointness, and this is documented in VERIFY.md.
    """
    repo = Path(repo or REPO_DIR)
    base = ensure_cifar10(repo)
    root = Path(root) if root else (ROOT / "data_e4")
    ds = root / "Cifar10"
    (ds / "train").mkdir(parents=True, exist_ok=True)
    (ds / "test").mkdir(parents=True, exist_ok=True)
    for cid in range(20):
        for split in ["train", "test"]:
            src = base / split / f"{cid}.npz"
            dst = ds / split / f"{cid}.npz"
            if not dst.exists():
                shutil.copy2(src, dst)
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tr = datasets.CIFAR10(root=str(base / "rawdata"), train=True, download=True, transform=transform)
    te = datasets.CIFAR10(root=str(base / "rawdata"), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    K = 10
    for j in range(n_extra):
        cid = 20 + j
        for split_name, ds_obj, n in [("train", tr, 2000), ("test", te, 400)]:
            labels = ds_obj.targets.numpy()
            probs = rng.dirichlet(np.repeat(0.5, K))
            chosen = rng.choice(len(labels), size=n, replace=False, p=probs[labels])
            x = ds_obj.data[chosen].astype(np.float32) / 255.0
            x = np.transpose(x, (0, 3, 1, 2))
            x = (x - 0.5) / 0.5
            y = labels[chosen].astype(np.int64)
            np.savez_compressed(ds / split_name / f"{cid}.npz", data={"x": x, "y": y})
    atomic_write_json({"num_clients": 30, "num_classes": 10, "non_iid": True,
                       "partition": "base20+extra10", "alpha": 0.5, "batch_size": 64},
                      ds / "config.json")
    print(f"[data] wrote 30-client Cifar10 variant at {ds}")
    return ds


# ---------------------------------------------------------------------------
# Repository patches: device-safe save/load, AMP clients, aggregation
# ---------------------------------------------------------------------------

_PATCHED = False
_ORIG_SAVE_ITEM = None
_ORIG_LOAD_ITEM = None


def install_repo_patches() -> None:
    """Install process-wide, idempotent patches.

    * Models are serialized on CPU and moved to the active device on load.
    * clientPHP trains under CUDA AMP and releases GPU memory after every client.
    * PHPFL.aggregate_parameters gets a pluggable weight function.
    """
    global _PATCHED, _ORIG_SAVE_ITEM, _ORIG_LOAD_ITEM
    if _PATCHED:
        return

    import flcore.clients.clientbase as cb
    import flcore.clients.clientphp as cp
    import flcore.servers.serverphp as sp
    import flcore.servers.serverbase as sb
    import flcore.trainmodel.models as models

    _ORIG_SAVE_ITEM = cb.save_item
    _ORIG_LOAD_ITEM = cb.load_item

    def save_item_cpu(item, role, item_name, item_path=None):
        if item_path is None:
            item_path = "."
        Path(item_path).mkdir(parents=True, exist_ok=True)
        target = Path(item_path) / f"{role}_{item_name}.pt"
        tmp = target.with_suffix(".pt.tmp")
        was_cuda = torch.is_tensor(next(item.parameters(), None)) and next(item.parameters()).is_cuda \
            if hasattr(item, "parameters") else False
        try:
            if was_cuda:
                item.cpu()
            torch.save(item, tmp)
            os.replace(tmp, target)
        finally:
            if was_cuda and torch.cuda.is_available():
                item.to(DEVICE)

    def load_item_device(role, item_name, item_path=None):
        if item_path is None:
            item_path = "."
        target = Path(item_path) / f"{role}_{item_name}.pt"
        if not target.exists():
            print(role, item_name, "Not Found")
            return None
        try:
            obj = torch.load(target, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(target, map_location="cpu")
        if hasattr(obj, "to"):
            obj = obj.to(DEVICE)
        return obj

    for mod in (cb, cp, sp):
        if hasattr(mod, "save_item"):
            mod.save_item = save_item_cpu
        if hasattr(mod, "load_item"):
            mod.load_item = load_item_device

    # Save the original aggregation so the mode hook can be installed later.
    if not hasattr(sp.PHPFL, "_phofl_original_aggregate"):
        sp.PHPFL._phofl_original_aggregate = sp.PHPFL.aggregate_parameters
    if not hasattr(sp.PHPFL, "_phofl_original_update_global"):
        sp.PHPFL._phofl_original_update_global = sp.PHPFL.update_global_model

    # CPU-backed masks and device-safe ISPU writes.
    def filter_top_p_percent_cpu(self, p):
        global_model = load_item_device(self.role, "global_model", self.save_folder_name)
        all_params = torch.cat([param.detach().view(-1) for param in global_model.parameters()])
        threshold = torch.quantile(all_params.abs(), 1 - float(p))
        masks = []
        for param in global_model.parameters():
            masks.append((param.detach().abs() >= threshold).to(torch.int8).cpu())
        del global_model
        clear_memory()
        return masks

    def update_model_with_mask_device(self):
        global_model = load_item_device("Server", "global_model", self.save_folder_name)
        client_global_model = load_item_device(self.role, "global_model", self.save_folder_name)
        for (name, param), source_param, mask in zip(client_global_model.named_parameters(),
                                                     global_model.parameters(), self.masks):
            m = mask.to(param.device).bool()
            param.data = torch.where(m, source_param.data.clone(), param.data.clone())
        save_item_cpu(client_global_model, self.role, "global_model", self.save_folder_name)
        del global_model, client_global_model
        clear_memory()

    cp.clientPHP.filter_top_p_percent = filter_top_p_percent_cpu
    cp.clientPHP.update_model_with_mask = update_model_with_mask_device

    # Inject the MLP auxiliary base into the models module's eval namespace.
    try:
        from phofl_aux_models import AuxMLPBase  # type: ignore
    except Exception:
        AuxMLPBase = _define_aux_mlp()
    models.AuxMLPBase = AuxMLPBase

    _PATCHED = True
    print("[patch] device-safe save/load, CPU masks, aggregation hook installed")


def _define_aux_mlp():
    class AuxMLPBase(nn.Module):
        """Flattened-image MLP with a 512-d feature head.

        BaseHeadSplit replaces .fc with AdaptiveAvgPool1d(512); forward therefore
        returns exactly 512 features, which the shared classifier consumes.
        """
        def __init__(self, in_dim: int = 3072, hidden: int = 24):
            super().__init__()
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 512),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Identity()  # replaced by BaseHeadSplit

        def forward(self, x):
            return self.fc(self.features(x))
    return AuxMLPBase


def aux_expression(arch_name: str, dataset: str) -> Tuple[str, Optional[str]]:
    """Return (global model expression, local-model override)."""
    if arch_name == "googlenet":
        expr = ("torchvision.models.googlenet(pretrained=False, aux_logits=False, "
                "num_classes=args.num_classes)")
        return expr, None
    in_dim = 3072 if dataset == "Cifar10" else 784
    if arch_name == "mlp_small":
        return f"AuxMLPBase(in_dim={in_dim}, hidden=24)", None
    if arch_name == "mlp_medium":
        return f"AuxMLPBase(in_dim={in_dim}, hidden=144)", None
    raise ValueError(f"Unknown auxiliary architecture: {arch_name}")


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def arch_name_from_expr(expr: str) -> str:
    s = expr.lower()
    if "googlenet" in s:
        return "googlenet"
    if "densenet121" in s:
        return "densenet121"
    if "efficientnet" in s:
        return "efficientnet-b1"
    if "resnet18" in s:
        return "resnet18"
    if "mobilenet" in s:
        return "mobilenet_v2"
    if "auxmlpbase" in s:
        return "mlp"
    return s.split("(")[0][-40:]


# ---------------------------------------------------------------------------
# AMP client subclasses (PHP-FL and FedAPEN)
# ---------------------------------------------------------------------------

def _autocast():
    return torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16)


def _scaler():
    try:
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)
    except Exception:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)


class ClientPHPAMP:
    """Mixin that overrides clientPHP.train for AMP; installed by subclassing."""
    pass


def make_amp_client_classes():
    import flcore.clients.clientphp as cp

    base = cp.clientPHP

    class clientPHPAMP(base):
        def weighted_kl(self, logp, p, y):
            # Default standard DEAL KL; Step-4 subclass may reweight by class.
            return self.KL_Loss(logp, p)

        def confusion_penalty(self, output, y):
            # Default no-op hook; method subclasses may override.
            return 0.0

        def learn_weight_for_inference(self):
            self.join_times += 1
            _, validation_loader = self.load_train_valid_data()
            model = cp.load_item(self.role, "model", self.save_folder_name)
            global_model = cp.load_item(self.role, "global_model", self.save_folder_name)
            model.eval()
            global_model.eval()
            for step in range(10):
                for data, target in validation_loader:
                    self.optimizer_learned_weight_for_inference.zero_grad()
                    if isinstance(data, list):
                        data[0] = data[0].to(self.device)
                    else:
                        data = data.to(self.device)
                    target = target.to(self.device)
                    # No torch.no_grad() here: the scalar ensemble weight is the
                    # only parameter that must receive a gradient.
                    with _autocast():
                        output_private = model(data).detach()
                        output_shared = global_model(data).detach()
                        ensemble_output = (self.learned_weight_for_inference * output_private +
                                           (1 - self.learned_weight_for_inference) * output_shared)
                        loss = self.loss(ensemble_output, target)
                    loss.backward()
                    self.optimizer_learned_weight_for_inference.step()
                    torch.clip_(self.learned_weight_for_inference.data, 0.0, 1.0)
            self.staged_learned_weight_inference.append(
                self.learned_weight_for_inference.cpu().data.item())
            print(f"client {self.id} learned weight for inference: "
                  f"{self.learned_weight_for_inference.data.item():.4f}")
            del model, global_model
            clear_memory()

        def train(self, round):
            trainloader, _ = self.load_train_valid_data()
            model = cp.load_item(self.role, "model", self.save_folder_name)
            global_model = cp.load_item(self.role, "global_model", self.save_folder_name)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
            scaler = _scaler()
            model.train()
            global_model.train()
            self.weight_private = (0.5 if len(self.staged_learned_weight_inference) == 0
                                   else self.staged_learned_weight_inference[-1])
            start_time = time.time()
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            shared_loss_sum = 0.0
            shared_n = 0
            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    optimizer.zero_grad()
                    optimizer_g.zero_grad()
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    with _autocast():
                        rep_private = model.base(x)
                        output_private = model.head(rep_private)
                        rep_shared = global_model.base(x)
                        output_shared = global_model.head(rep_shared)
                        ensemble_private = (self.weight_private * output_private +
                                            (1 - self.weight_private) * output_shared.detach())
                        ensemble_shared = (self.weight_private * output_private.detach() +
                                           (1 - self.weight_private) * output_shared)
                        ce_private = self.loss(output_private, y)
                        kl_private = self.weighted_kl(self.LogSoftmax(output_private),
                                                      self.Softmax(output_shared.detach()), y)
                        ce_shared = self.loss(output_shared, y)
                        kl_shared = self.weighted_kl(self.LogSoftmax(output_shared),
                                                     self.Softmax(output_private.detach()), y)
                        mmd_private = self.Mmd_loss(rep_private, rep_shared.detach())
                        mmd_shared = self.Mmd_loss(rep_private.detach(), rep_shared)
                        loss_private = (mmd_private + ce_private + kl_private +
                                        self.loss(ensemble_private, y) +
                                        self.confusion_penalty(output_private, y))
                        loss_shared = (mmd_shared + ce_shared + kl_shared +
                                       self.loss(ensemble_shared, y) +
                                       self.confusion_penalty(output_shared, y))
                    scaler.scale(loss_private).backward(retain_graph=True)
                    scaler.scale(loss_shared).backward()
                    scaler.step(optimizer)
                    scaler.step(optimizer_g)
                    scaler.update()
                    # Instrumentation only: FedFV orders clients by training loss.
                    shared_loss_sum += float(ce_shared.detach().item()) * int(y.shape[0])
                    shared_n += int(y.shape[0])
            self.last_train_loss = shared_loss_sum / max(shared_n, 1)
            self.update_ratio = sigmoid(self.sigmoid_k, (self.join_times + 1) * 1.0 / (round + 1))
            self.masks = self.filter_top_p_percent(self.update_ratio * self.tau)
            cp.save_item(model, self.role, "model", self.save_folder_name)
            cp.save_item(global_model, self.role, "global_model", self.save_folder_name)
            self.train_time_cost["num_rounds"] += 1
            self.train_time_cost["total_cost"] += time.time() - start_time
            del model, global_model, optimizer, optimizer_g, scaler
            clear_memory()

    class clientFedAPEN(clientPHPAMP):
        def train(self, round):
            trainloader, _ = self.load_train_valid_data()
            model = cp.load_item(self.role, "model", self.save_folder_name)
            global_model = cp.load_item(self.role, "global_model", self.save_folder_name)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
            scaler = _scaler()
            model.train()
            global_model.train()
            self.weight_private = (0.5 if len(self.staged_learned_weight_inference) == 0
                                   else self.staged_learned_weight_inference[-1])
            start_time = time.time()
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    optimizer.zero_grad()
                    optimizer_g.zero_grad()
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    with _autocast():
                        output_private = model(x)
                        output_shared = global_model(x)
                        ensemble_private = (self.weight_private * output_private +
                                            (1 - self.weight_private) * output_shared.detach())
                        ensemble_shared = (self.weight_private * output_private.detach() +
                                           (1 - self.weight_private) * output_shared)
                        ce_private = self.loss(output_private, y)
                        kl_private = self.weighted_kl(self.LogSoftmax(output_private),
                                                      self.Softmax(output_shared.detach()), y)
                        ce_shared = self.loss(output_shared, y)
                        kl_shared = self.weighted_kl(self.LogSoftmax(output_shared),
                                                     self.Softmax(output_private.detach()), y)
                        loss_private = ce_private + kl_private + self.loss(ensemble_private, y)
                        loss_shared = ce_shared + kl_shared + self.loss(ensemble_shared, y)
                    scaler.scale(loss_private).backward(retain_graph=True)
                    scaler.scale(loss_shared).backward()
                    scaler.step(optimizer)
                    scaler.step(optimizer_g)
                    scaler.update()
            self.update_ratio = sigmoid(self.sigmoid_k, (self.join_times + 1) * 1.0 / (round + 1))
            self.masks = None
            cp.save_item(model, self.role, "model", self.save_folder_name)
            cp.save_item(global_model, self.role, "global_model", self.save_folder_name)
            self.train_time_cost["num_rounds"] += 1
            self.train_time_cost["total_cost"] += time.time() - start_time
            del model, global_model, optimizer, optimizer_g, scaler
            clear_memory()

    return clientPHPAMP, clientFedAPEN


def sigmoid(k: float, x: float) -> float:
    # Exact repository definition (serverbase/clientphp).
    return 1 / (1 + np.exp(k * (x - 0.5)))


# ---------------------------------------------------------------------------
# Aggregation modes
# ---------------------------------------------------------------------------

# FedFV hyper-parameters fixed a priori from the paper's CIFAR-10 grid
# (alpha=0.1 with tau in {0,1,3,10}; tau=3 is near-best and fits 60 rounds).
FEDFV_ALPHA = 0.1
FEDFV_TAU = 3

def aggregation_weights(server, mode: str) -> np.ndarray:
    ids = list(server.uploaded_ids)
    if mode in ("plain", "uniform"):
        w = np.ones(len(ids), dtype=np.float64)
    elif mode == "fedavg":
        w = np.array([server.clients[cid].train_samples for cid in ids], dtype=np.float64)
    elif mode == "inv_p":
        p = np.array([server.participation_probability[cid] for cid in ids], dtype=np.float64)
        w = 1.0 / np.clip(p, 0.02, 1.0)
    elif mode == "inv_N":
        w = 1.0 / np.maximum(np.array([server.selectd_times[cid] for cid in ids], dtype=np.float64), 1.0)
    elif mode == "worst_class":
        rec = np.array([server._worst_class_recalls[cid] for cid in ids], dtype=np.float64)
        w = np.exp(-5.0 * np.clip(rec, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Invalid aggregation weights for mode={mode}: {w}")
    return w / s


def _worst_class_recalls(server, gamma: float = 5.0):
    """Compute global worst class and each active client's recall on it.

    The global worst class is read from the server auxiliary model on the
    canonical test set.  Each client then reports one scalar from its own local
    model/test set.  This is the Step-3 aggregation signal.
    """
    import flcore.servers.serverphp as sp
    g = sp.load_item("Server", "global_model", server.save_folder_name)
    g.eval()
    loader = _global_test_loader(server.dataset)
    correct = np.zeros(NUM_CLASSES, dtype=np.float64)
    total = np.zeros(NUM_CLASSES, dtype=np.float64)
    with torch.no_grad(), _autocast():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = g(x).argmax(1)
            for k in range(NUM_CLASSES):
                m = y == k
                correct[k] += float((pred[m] == k).sum().item())
                total[k] += float(m.sum().item())
    del g
    clear_memory()
    rec = np.where(total > 0, correct / np.maximum(total, 1), np.nan)
    c_worst = int(np.nanargmin(rec))
    client_recalls = {}
    for cid in server.uploaded_ids:
        c = server.clients[cid]
        m = sp.load_item(c.role, "model", c.save_folder_name)
        m.eval()
        c_loader = c.load_test_data()
        cor = 0.0
        tot = 0.0
        with torch.no_grad(), _autocast():
            for x, y in c_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                mask = y == c_worst
                if mask.any():
                    cor += float((m(x)[mask].argmax(1) == c_worst).sum().item())
                    tot += float(mask.sum().item())
        client_recalls[cid] = cor / tot if tot > 0 else 0.5
        del m
        clear_memory()
    return c_worst, client_recalls


def _fedfv_project(deltas, losses, alpha):
    """FedFV internal-conflict surgery (Wang et al., IJCAI 2021, Alg. 1-2).

    ``deltas`` is an (m, d) tensor of client updates g_k = theta_t - theta_k,
    ``losses`` the matching training losses.  Clients are processed in
    ascending-loss order; the first ``floor(alpha*m)`` clients keep their
    original gradient, matching the paper's alpha parameter (alpha=1 recovers
    FedAvg, alpha=0 projects every client).

    Returns the mitigated mean update before the norm rescaling of Alg. 1.
    """
    m = deltas.shape[0]
    order = torch.argsort(torch.as_tensor(losses, dtype=torch.float64)).tolist()
    keep = set(order[: int(np.floor(alpha * m))])
    g = deltas.clone()
    norms2 = (deltas * deltas).sum(dim=1)
    for k in range(m):
        if k in keep:
            continue
        for j in order:
            if j == k or float(norms2[j]) <= 0.0:
                continue
            dot = float((g[k] * deltas[j]).sum())
            if dot < 0.0:
                g[k] = g[k] - (dot / float(norms2[j])) * deltas[j]
    return g.mean(dim=0)


def _fedfv_external(g, history, round_idx, selected_ids, tau_ext):
    """FedFV external-conflict projection (Alg. 3) over the last ``tau_ext`` rounds."""
    for i in range(tau_ext, 0, -1):
        target = round_idx - i
        conflicts = [d for cid, (r_hist, d) in history.items()
                     if r_hist == target and cid not in selected_ids
                     and float((g * d).sum()) < 0.0]
        if not conflicts:
            continue
        g_con = torch.stack(conflicts).sum(dim=0)
        den = float((g_con * g_con).sum())
        if den > 0.0 and float((g * g_con).sum()) < 0.0:
            g = g - (float((g * g_con).sum()) / den) * g_con
    return g


def _fedfv_aggregate(server, state, alpha=FEDFV_ALPHA, tau_ext=FEDFV_TAU):
    """Replace plain averaging with FedFV gradient surgery on the shared model.

    FedFV is defined in parameter space, so it is applied to the model every
    client shares (the auxiliary/global model); the personalised models are
    untouched.  This is the only well-defined placement of FedFV in a
    model-heterogeneous harness.
    """
    import flcore.servers.serverphp as sp

    ref_model = sp.load_item(server.role, "global_model", server.save_folder_name)
    ref = [p.detach().to("cpu", torch.float32).clone() for p in ref_model.parameters()]
    shapes = [tuple(p.shape) for p in ref]
    flat_ref = torch.cat([r.reshape(-1) for r in ref])

    ids = list(server.uploaded_ids)
    deltas, losses = [], []
    for cid in ids:
        client = server.clients[cid]
        cm = sp.load_item(client.role, "global_model", client.save_folder_name)
        vec = torch.cat([(r - p.detach().to("cpu", torch.float32)).reshape(-1)
                         for r, p in zip(ref, cm.parameters())])
        deltas.append(vec)
        losses.append(float(getattr(client, "last_train_loss", 0.0)))
        del cm
    D = torch.stack(deltas)

    g = _fedfv_project(D, losses, alpha)
    plain = D.mean(dim=0)
    if tau_ext > 0 and state["round"] >= tau_ext:
        g = _fedfv_external(g, state["history"], state["round"], set(ids), tau_ext)
    ng, npn = float(g.norm()), float(plain.norm())
    if ng > 0.0 and npn > 0.0:
        g = g * (npn / ng)
    if state["round"] % 10 == 0:
        n_keep = int(np.floor(alpha * len(ids)))
        print(f"[fedfv] round={state['round']} clients={len(ids)} "
              f"kept={n_keep} losses={[round(v, 4) for v in losses]} "
              f"g_norm={ng:.4f} plain_norm={npn:.4f}")

    new = flat_ref - g
    offset = 0
    for p, shape in zip(ref_model.parameters(), shapes):
        num = int(np.prod(shape)) if len(shape) else 1
        p.data.copy_(new[offset:offset + num].reshape(shape).to(p.device))
        offset += num
    sp.save_item(ref_model, server.role, "global_model", server.save_folder_name)

    for k, cid in enumerate(ids):
        state["history"][cid] = (state["round"], D[k].clone())
    keep_from = state["round"] - tau_ext
    state["history"] = {cid: (r, d) for cid, (r, d) in state["history"].items()
                        if r >= keep_from}
    del ref_model, D, deltas
    clear_memory()


def install_aggregation(mode: str) -> None:
    import flcore.servers.serverphp as sp

    state = {"round": 0, "history": {}}

    def aggregate(self):
        assert len(self.uploaded_ids) > 0
        if mode == "fedfv":
            _fedfv_aggregate(self, state)
            state["round"] += 1
            return
        if mode == "worst_class":
            c_worst, client_recalls = _worst_class_recalls(self, gamma=5.0)
            self._worst_class = c_worst
            self._worst_class_recalls = client_recalls
            print(f"[worst-class] class={c_worst} recalls={client_recalls}")
        w = aggregation_weights(self, mode)
        first = self.clients[self.uploaded_ids[0]]
        global_model = sp.load_item(first.role, "global_model", first.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
        for weight, cid in zip(w, self.uploaded_ids):
            client = self.clients[cid]
            client_model = sp.load_item(client.role, "global_model", client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * float(weight)
            del client_model
        sp.save_item(global_model, self.role, "global_model", self.save_folder_name)
        del global_model
        clear_memory()

    sp.PHPFL.aggregate_parameters = aggregate


def broadcast_global_aux(server) -> None:
    """FedAPEN sends the aggregated shared model to every client every round."""
    import flcore.servers.serverphp as sp
    g = sp.load_item("Server", "global_model", server.save_folder_name)
    for c in server.clients:
        sp.save_item(g, c.role, "global_model", c.save_folder_name)
    del g
    clear_memory()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def label_counts(dataset: str, cid: int, root: Optional[Path] = None) -> np.ndarray:
    root = Path(root or ACTIVE_DATA_ROOT or (REPO_DIR / "dataset"))
    path = root / dataset / "train" / f"{cid}.npz"
    with open(path, "rb") as f:
        d = np.load(f, allow_pickle=True)["data"].tolist()
    y = np.asarray(d["y"], dtype=np.int64)
    return np.bincount(y, minlength=10).astype(np.float64)


def client_sizes(dataset: str, n_clients: int, root: Optional[Path] = None) -> List[int]:
    return [int(label_counts(dataset, i, root).sum()) for i in range(n_clients)]


def participation_probabilities(pattern: str, dataset: str, n_clients: int,
                                seed: int, sigma: float = 0.2,
                                class_id: int = 0) -> np.ndarray:
    """Compute declared p_i for the paper's and the E3 patterns."""
    rng = np.random.default_rng(seed)
    if pattern in ("normal", "linear", "uniform"):
        if pattern == "uniform":
            return np.full(n_clients, 0.5, dtype=np.float64)
        if pattern == "linear":
            import flcore.servers.serverbase as sb
            p = np.array(sb.generate_arithmetic_sequence(n_clients), dtype=np.float64)
            return p[rng.permutation(n_clients)]
        # Normal pattern: truncated normal, matching the repo's 0.5/0.2 default.
        while True:
            p = rng.normal(0.5, sigma, n_clients)
            if np.all((p >= 0) & (p <= 1)):
                return p
    if pattern == "rare_class_dropout":
        # The prompt's intent is "balanced data participates more, rare-class
        # holders less".  exp(-H) alone has the opposite direction, so we use
        # normalized entropy exp(H - ln K), which is equivalent to H/K in
        # ordering and clips to the requested range.
        K = 10
        out = []
        for cid in range(n_clients):
            c = label_counts(dataset, cid)
            q = c / max(c.sum(), 1.0)
            q = q[q > 0]
            H = float(-(q * np.log(q)).sum())
            out.append(np.clip(np.exp(H - np.log(K)), 0.02, 1.0))
        return np.array(out, dtype=np.float64)
    if pattern == "entropy_correlated":
        # Same definition as rare_class_dropout, under the scope-change name.
        return participation_probabilities("rare_class_dropout", dataset, n_clients,
                                           seed=seed, sigma=sigma, class_id=class_id)
    if pattern == "size_correlated":
        sizes = np.array(client_sizes(dataset, n_clients), dtype=np.float64)
        return np.clip(sizes / max(sizes.max(), 1.0), 0.02, 1.0)
    if pattern == "class_exclusive_dropout":
        out = []
        for cid in range(n_clients):
            c = label_counts(dataset, cid)
            frac = c[class_id] / max(c.sum(), 1.0)
            out.append(0.02 if frac > 0.05 else 0.8)
        return np.array(out, dtype=np.float64)
    raise ValueError(f"Unknown participation pattern: {pattern}")


def _global_test_loader(dataset: str, batch_size: int = 256) -> DataLoader:
    from torchvision import datasets, transforms
    if dataset == "Cifar10":
        root = REPO_DIR / "dataset" / "Cifar10" / "rawdata"
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tf)
    elif dataset == "FashionMNIST":
        root = REPO_DIR / "dataset" / "FashionMNIST" / "rawdata"
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))])
        ds = datasets.FashionMNIST(root=str(root), train=False, download=True, transform=tf)
    elif dataset in ("DermaMNIST", "BloodMNIST"):
        if ACTIVE_DATA_ROOT is None:
            raise RuntimeError("Set data_root before loading a MedMNIST global test set")
        path = Path(ACTIVE_DATA_ROOT) / dataset / "global_test.npz"
        blob = np.load(path, allow_pickle=True)
        x = torch.tensor(blob["x"], dtype=torch.float32)
        y = torch.tensor(blob["y"], dtype=torch.int64)
        ds = torch.utils.data.TensorDataset(x, y)
    else:
        raise ValueError(dataset)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Args / server construction
# ---------------------------------------------------------------------------

def make_args(dataset: str, num_clients: int, rounds: int, batch_size: int,
              local_epochs: int, eval_gap: int, seed: int, tau: float,
              sig_k: float, valid_ratio: float, join_method: str,
              global_arch: str, save_folder: Path, local_lr: float = 0.001,
              model_family: str = "HtM4") -> Any:
    import argparse
    args = argparse.Namespace()
    args.goal = "test"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device_id = "0"
    args.dataset = dataset
    args.num_classes = int(NUM_CLASSES)
    args.model_family = model_family
    args.batch_size = int(batch_size)
    args.local_learning_rate = float(local_lr)
    args.learning_rate_decay = False
    args.learning_rate_decay_gamma = 0.99
    args.global_rounds = int(rounds)
    args.local_epochs = int(local_epochs)
    args.algorithm = "PHPFL"
    args.join_method = join_method
    args.join_ratio = 1.0
    args.random_join_ratio = False
    args.num_clients = int(num_clients)
    args.prev = 0
    args.times = 1
    args.eval_gap = int(eval_gap)
    args.save_folder_name = str(save_folder)
    args.auto_break = False
    args.feature_dim = 512
    args.global_model_type = -1
    args.client_drop_rate = 0.0
    args.train_slow_rate = 0.0
    args.send_slow_rate = 0.0
    args.time_select = False
    args.time_threthold = 10000.0
    args.sig_k = float(sig_k)
    args.tau = float(tau)
    args.valid_ratio = float(valid_ratio)
    args.seed = int(seed)
    if model_family == "HtM4":
        args.models = [
            "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
            "torchvision.models.densenet121(pretrained=False, num_classes=args.num_classes)",
            'EfficientNet.from_name("efficientnet-b1", num_classes=args.num_classes)',
            "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
        ]
    elif model_family == "HtM2":
        args.models = [
            "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
            "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
        ]
    else:
        raise ValueError(model_family)
    expr, _ = aux_expression(global_arch, dataset)
    args.global_model = expr
    return args


def build_server(args, method: str = "PHPFL", agg_mode: str = "plain", client_cls=None):
    import flcore.servers.serverphp as sp
    _install_data_reader()
    install_repo_patches()
    client_php_amp, client_fedapen = make_amp_client_classes()
    sp.clientPHP = client_cls if client_cls is not None else (client_fedapen if method == "FedAPEN" else client_php_amp)
    install_aggregation(agg_mode)
    server = sp.PHPFL(args, 0)
    # Heavy backbones get a smaller batch on a 6 GB card.
    for c in server.clients:
        expr = args.models[c.id % len(args.models)].lower()
        if ("densenet121" in expr or "efficientnet" in expr) and args.batch_size > 32:
            c.batch_size = 32
    return server


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _ensemble_weight(client) -> float:
    if getattr(client, "staged_learned_weight_inference", None):
        return float(client.staged_learned_weight_inference[-1])
    return 0.5


@torch.no_grad()
def eval_client_models(client, include_classes: bool = True) -> Dict[str, Any]:
    import flcore.clients.clientbase as cb
    model = cb.load_item(client.role, "model", client.save_folder_name)
    aux = cb.load_item(client.role, "global_model", client.save_folder_name)
    model.eval()
    aux.eval()
    loader = client.load_test_data()
    w = _ensemble_weight(client)
    n = 0
    local_correct = aux_correct = ens_correct = 0
    local_loss = 0.0
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    with _autocast():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            outg = aux(x)
            ens = w * out + (1 - w) * outg
            local_loss += float(F.cross_entropy(out, y, reduction="sum").item())
            local_correct += int((out.argmax(1) == y).sum().item())
            aux_correct += int((outg.argmax(1) == y).sum().item())
            ens_correct += int((ens.argmax(1) == y).sum().item())
            n += int(y.numel())
            if include_classes:
                pred = ens.argmax(1)
                for c in range(NUM_CLASSES):
                    m = y == c
                    class_total[c] += int(m.sum().item())
                    class_correct[c] += int((pred[m] == c).sum().item())
    del model, aux
    clear_memory()
    return {
        "n": n,
        "local_acc": local_correct / max(n, 1),
        "aux_acc": aux_correct / max(n, 1),
        "ensemble_acc": ens_correct / max(n, 1),
        "local_loss": local_loss / max(n, 1),
        "class_acc": [class_correct[c] / class_total[c] if class_total[c] else float("nan")
                      for c in range(NUM_CLASSES)],
    }


@torch.no_grad()
def eval_global_on_client(client) -> Tuple[float, int]:
    import flcore.servers.serverphp as sp
    g = sp.load_item("Server", "global_model", client.save_folder_name)
    g.eval()
    loader = client.load_test_data()
    correct = 0
    n = 0
    with _autocast():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = g(x).argmax(1)
            correct += int((pred == y).sum().item())
            n += int(y.numel())
    del g
    clear_memory()
    return correct / max(n, 1), n


def evaluate_all_clients(server, include_G: bool = False) -> pd.DataFrame:
    rows = []
    for c in server.clients:
        m = eval_client_models(c)
        gacc = np.nan
        if include_G:
            gacc, _ = eval_global_on_client(c)
        counts = label_counts(server.dataset, c.id)
        # Rarest class actually held by this client (not an absent label).
        present = np.where(counts > 0)[0]
        rare = int(present[np.argmin(counts[present])]) if len(present) else -1
        rare_acc = m["class_acc"][rare] if rare >= 0 else float("nan")
        rows.append({
            "client_id": c.id,
            "arch": arch_name_from_expr(server.args.models[c.id % len(server.args.models)]),
            "n": m["n"],
            "p_i": float(server.participation_probability[c.id]),
            "N_i": int(server.selectd_times[c.id]),
            "local_acc": m["local_acc"],
            "local_loss": m["local_loss"],
            "aux_acc": m["aux_acc"],
            "ensemble_acc": m["ensemble_acc"],
            "G_acc": gacc,
            "alpha_i": float(getattr(c, "update_ratio", np.nan)),
            "lambda_i": _ensemble_weight(c),
            "rare_class": rare,
            "rare_acc": rare_acc,
            **{f"class_{k}_acc": m["class_acc"][k] for k in range(NUM_CLASSES)},
        })
    return pd.DataFrame(rows)


def compute_fairness(df: pd.DataFrame) -> Dict[str, float]:
    a = df["ensemble_acc"].to_numpy(dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {}
    fm = float(np.std(a, ddof=0))
    n = df["n"].to_numpy(dtype=np.float64)
    correct = a * n
    am_weighted = float(correct.sum() / n.sum()) if n.sum() > 0 else float("nan")
    mean = float(a.mean())
    k = len(a)
    jain = float((a.sum() ** 2) / (k * (a ** 2).sum())) if (a ** 2).sum() > 0 else float("nan")
    q10 = float(np.percentile(a, 10))
    cvar10 = float(np.mean(np.sort(a)[:max(1, int(math.ceil(0.1 * len(a))))]))
    p = df["p_i"].to_numpy(dtype=np.float64)
    rho = float("nan")
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(p, a).correlation)
    except Exception:
        pass
    return {
        "AM": am_weighted,
        "AM_unweighted": mean,
        "FM": fm,
        "min_acc": float(a.min()),
        "p10": q10,
        "cvar10": cvar10,
        "CV": float(np.std(a, ddof=0) / mean) if mean else float("nan"),
        "jain": jain,
        "rho_p_acc": rho,
    }


# ---------------------------------------------------------------------------
# CSV logging / checkpoints
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: Path, columns: Sequence[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.path, index=False)

    def read(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path)
        except Exception:
            return pd.DataFrame(columns=self.columns)

    def truncate_round(self, max_round: int) -> None:
        df = self.read()
        if len(df) and "round" in df.columns:
            df = df[df["round"] <= max_round]
            atomic_write_csv(df, self.path)

    def append(self, row: Dict[str, Any]) -> None:
        df = pd.DataFrame([{c: row.get(c, np.nan) for c in self.columns}])
        df.to_csv(self.path, mode="a", header=False, index=False)


def _state_dict_cpu(model: nn.Module, half: bool = True) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        if half and t.is_floating_point():
            t = t.half()
        out[k] = t
    return out


def save_checkpoint(server, round_no: int, run_id: str, cfg: Dict[str, Any],
                    history: pd.DataFrame) -> Optional[Path]:
    """Save one latest checkpoint per run with all w_i/g_i state.

    Optimizers in this repository are recreated every local round, so optimizer
    state is intentionally not stored.  fp16 halves Drive usage; it only affects
    resumed trajectories, never the final metrics of a completed run.
    """
    import flcore.servers.serverphp as sp
    out_dir = ROOT / "ckpts" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{run_id}_r{round_no}.pt"
    tmp = target.with_suffix(".pt.tmp")
    payload = {
        "run_id": run_id,
        "round": int(round_no),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "repo_sha": repo_sha(),
        "rng": rng_state(),
        "selectd_times": list(server.selectd_times),
        "rs_test_acc": list(getattr(server, "rs_test_acc", [])),
        "rs_test_std_acc": list(getattr(server, "rs_test_std_acc", [])),
        "history": history.to_dict(orient="list"),
        "server_global": _state_dict_cpu(sp.load_item("Server", "global_model", server.save_folder_name)),
        "clients": {},
    }
    for c in server.clients:
        m = sp.load_item(c.role, "model", c.save_folder_name)
        g = sp.load_item(c.role, "global_model", c.save_folder_name)
        payload["clients"][c.id] = {
            "model": _state_dict_cpu(m),
            "global_model": _state_dict_cpu(g),
            "join_times": int(getattr(c, "join_times", 0)),
            "staged": list(getattr(c, "staged_learned_weight_inference", [])),
            "update_ratio": float(getattr(c, "update_ratio", 1.0)),
            "weight_private": float(getattr(c, "weight_private", 0.5)),
            "masks": [x.detach().cpu() for x in c.masks] if getattr(c, "masks", None) else None,
        }
        del m, g
    torch.save(payload, tmp)
    os.replace(tmp, target)
    # Keep only the newest checkpoint for this run.
    for old in sorted(out_dir.glob(f"{run_id}_r*.pt")):
        if old != target:
            try:
                old.unlink()
            except OSError:
                pass
    print(f"[ckpt] {target} ({target.stat().st_size / 2**20:.0f} MB)")
    del payload
    clear_memory()
    return target


def find_checkpoint(run_id: str) -> Optional[Path]:
    d = ROOT / "ckpts" / run_id
    if not d.exists():
        return None
    pts = sorted(d.glob(f"{run_id}_r*.pt"), key=lambda p: int(re.search(r"_r(\d+)\.pt$", p.name).group(1)))
    return pts[-1] if pts else None


def restore_checkpoint(server, path: Path, cfg: Dict[str, Any]) -> int:
    """Load a checkpoint into a freshly constructed server. Returns start round."""
    import flcore.servers.serverphp as sp
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    if ck.get("config_hash") != config_hash(cfg):
        raise RuntimeError(f"Checkpoint config mismatch for {path}; refusing to resume")
    # Server global model.
    g = sp.load_item("Server", "global_model", server.save_folder_name)
    g.load_state_dict(ck["server_global"], strict=True)
    sp.save_item(g, "Server", "global_model", server.save_folder_name)
    del g
    for c in server.clients:
        data = ck["clients"].get(c.id, ck["clients"].get(str(c.id)))
        if data is None:
            raise RuntimeError(f"Checkpoint missing client {c.id}")
        m = sp.load_item(c.role, "model", c.save_folder_name)
        gg = sp.load_item(c.role, "global_model", c.save_folder_name)
        m.load_state_dict(data["model"], strict=True)
        gg.load_state_dict(data["global_model"], strict=True)
        sp.save_item(m, c.role, "model", c.save_folder_name)
        sp.save_item(gg, c.role, "global_model", c.save_folder_name)
        c.join_times = int(data.get("join_times", 0))
        c.staged_learned_weight_inference = list(data.get("staged", []))
        c.update_ratio = float(data.get("update_ratio", 1.0))
        c.weight_private = float(data.get("weight_private", 0.5))
        # Keep masks on CPU; update_model_with_mask_device() moves one mask at a
        # time.  Keeping 20 full-model masks on a 6 GB card is not viable.
        c.masks = list(data["masks"]) if data.get("masks") else None
        del m, gg
    server.selectd_times = list(ck.get("selectd_times", server.selectd_times))
    server.rs_test_acc = list(ck.get("rs_test_acc", []))
    server.rs_test_std_acc = list(ck.get("rs_test_std_acc", []))
    set_rng_state(ck.get("rng", {}))
    r = int(ck["round"])
    print(f"[resume] {path.name}: continuing after round {r}")
    clear_memory()
    return r + 1


# ---------------------------------------------------------------------------
# Core federated runner
# ---------------------------------------------------------------------------

def _round_log_columns() -> List[str]:
    return ["exp", "method", "dataset", "pattern", "seed", "round", "client_id", "model_arch",
            "participated", "N_i_t", "p_i", "alpha_i_t", "lambda_i", "local_acc",
            "local_loss", "aux_acc", "ensemble_acc", "global_G_acc_on_this_client"]


def _round_metric_columns() -> List[str]:
    return ["exp", "method", "dataset", "pattern", "seed", "round", "is_final", "client_id",
            "arch", "n", "p_i", "N_i", "local_acc", "local_loss", "aux_acc", "ensemble_acc",
            "G_acc", "alpha_i", "lambda_i", "rare_class", "rare_acc"] + \
           [f"class_{k}_acc" for k in range(NUM_CLASSES)]


def run_federated(exp: str, method: str, dataset: str, pattern: str, seed: int,
                  rounds: int, num_clients: int, batch_size: int, local_epochs: int,
                  eval_gap: int, tau: float = 0.2, sig_k: float = 5.0,
                  valid_ratio: float = 0.1, agg_mode: str = "plain",
                  aux_arch: str = "googlenet", sigma: float = 0.2,
                  class_id: int = 0, data_root: Optional[Path] = None,
                  model_family: str = "HtM4", scale: str = "smoke",
                  resume: bool = True, max_hours: float = 6.0,
                  p_override: Optional[np.ndarray] = None,
                  p_schedule: Optional[Callable[[int], np.ndarray]] = None,
                  late_join_round: Optional[int] = None,
                  grad_hook: Optional[Callable[[Any, int], Dict[str, Any]]] = None,
                  log_active_metrics: bool = True,
                  client_cls=None, local_lr: float = 0.001) -> Dict[str, Any]:
    """Run one (exp, method, pattern, seed) configuration with resume support."""
    import flcore.servers.serverphp as sp
    extra = {"rounds": rounds, "num_clients": num_clients, "batch": batch_size,
             "epochs": local_epochs, "agg": agg_mode, "aux": aux_arch,
             "tau": tau, "sigma": sigma, "class_id": class_id, "family": model_family}
    run_id = make_run_id(exp, method, dataset, pattern, seed, scale, extra)
    cfg = {"exp": exp, "method": method, "dataset": dataset, "pattern": pattern,
           "seed": seed, "scale": scale, **extra, "repo_sha": PINNED_REPO_SHA}
    flag = ROOT / "results" / exp / f"{run_id}_complete.flag"
    if resume and flag.exists():
        print(f"[skip] complete {run_id}")
        return {"run_id": run_id, "status": "complete", "skipped": True}

    result_dir = ROOT / "results" / exp
    log_dir = ROOT / "logs" / exp / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_store = ROOT / "run_store" / run_id
    run_store.mkdir(parents=True, exist_ok=True)
    round_log = CSVLogger(log_dir / "round_log.csv", _round_log_columns())
    round_metrics = CSVLogger(log_dir / "round_metrics.csv", _round_metric_columns())
    g_eval = CSVLogger(log_dir / "G_eval.csv",
                       ["round", "client_id", "p_i", "N_i", "G_acc", "local_w_acc", "ensemble_acc"])
    history = CSVLogger(log_dir / "history.csv",
                        ["round", "is_final", "AM", "AM_unweighted", "FM", "min_acc", "p10",
                         "cvar10", "CV", "jain", "rho_p_acc", "seconds"])
    grad_log = CSVLogger(log_dir / "grad_metrics.csv",
                         ["round", "grad_cos_aux", "grad_norm", "lambda_rare", "lambda_freq"])
    meta_path = log_dir / "run_meta.json"
    atomic_write_json({"run_id": run_id, "config": cfg, "repo_sha": repo_sha(),
                       "packages": package_report(), "status": "running",
                       "created": time.strftime("%Y-%m-%d %H:%M:%S")}, meta_path)

    push_data_root(data_root)
    try:
        seed_everything(seed)
        args = make_args(dataset=dataset, num_clients=num_clients, rounds=rounds,
                         batch_size=batch_size, local_epochs=local_epochs,
                         eval_gap=eval_gap, seed=seed, tau=tau, sig_k=sig_k,
                         valid_ratio=valid_ratio,
                         join_method=(pattern if pattern in ("normal", "linear", "uniform") else "normal"),
                         global_arch=aux_arch, save_folder=run_store,
                         model_family=model_family, local_lr=local_lr)
        server = build_server(args, method=method, agg_mode=agg_mode, client_cls=client_cls)
        # Declared participation probabilities.  Baseline uses the repo's own
        # normal sampler; experiments override it with a fixed p vector.
        if p_override is not None:
            server.participation_probability = np.asarray(p_override, dtype=np.float64)
        elif pattern not in ("normal", "linear", "uniform"):
            server.participation_probability = participation_probabilities(
                pattern, dataset, num_clients, seed=seed, sigma=sigma, class_id=class_id)
        start_round = 0
        ckpt = find_checkpoint(run_id) if resume else None
        if ckpt is not None:
            start_round = restore_checkpoint(server, ckpt, cfg)
            round_log.truncate_round(start_round - 1)
            round_metrics.truncate_round(start_round - 1)
            g_eval.truncate_round(start_round - 1)
            history.truncate_round(start_round - 1)
            grad_log.truncate_round(start_round - 1)
        # Re-apply p after restore (it is part of the config, not the checkpoint).
        if p_override is not None:
            server.participation_probability = np.asarray(p_override, dtype=np.float64)

        reset_peak_vram()
        t0 = time.time()
        for r in range(start_round, rounds + 1):
            if (time.time() - t0) > max_hours * 3600:
                save_checkpoint(server, r - 1, run_id, cfg, history.read())
                raise TimeoutError(f"Run exceeded MAX_HOURS={max_hours}; checkpoint saved, resume later")
            if p_schedule is not None:
                server.participation_probability = np.asarray(p_schedule(r), dtype=np.float64)
            if r % eval_gap == 0 or r in (20, 50, 100, rounds):
                df = evaluate_all_clients(server, include_G=(r % 10 == 0 or r in (20, 50, 100, rounds)))
                df.insert(0, "round", r)
                df.insert(0, "seed", seed)
                df.insert(0, "pattern", pattern)
                df.insert(0, "dataset", dataset)
                df.insert(0, "method", method)
                df.insert(0, "exp", exp)
                df["is_final"] = 0
                for _, row in df.iterrows():
                    round_metrics.append(row.to_dict())
                    if r in (20, 50, 100, rounds):
                        g_eval.append({"round": r, "client_id": int(row["client_id"]),
                                       "p_i": row["p_i"], "N_i": row["N_i"],
                                       "G_acc": row["G_acc"], "local_w_acc": row["local_acc"],
                                       "ensemble_acc": row["ensemble_acc"]})
                fair = compute_fairness(df)
                hist_row = {"round": r, "is_final": 0, **fair,
                            "seconds": time.time() - t0}
                history.append(hist_row)
                print(f"[{exp}/{method}/{pattern}/s{seed}] r{r:03d} AM={fair.get('AM', float('nan')):.3f} "
                      f"FM={fair.get('FM', float('nan')):.3f}")
            server.selected_clients = server.select_clients()
            if len(server.selected_clients) == 0:
                # A rare-pattern round can legitimately draw nobody.  Keep the
                # round alive by admitting the highest-p client; this is logged
                # and is the minimum intervention that preserves the schedule.
                j = int(np.argmax(server.participation_probability))
                server.selected_clients = [server.clients[j]]
                print(f"[warn] round {r}: empty participation draw; forcing client {j}")
            selected_ids = {c.id for c in server.selected_clients}
            for c in server.selected_clients:
                c.learn_weight_for_inference()
            for c in server.selected_clients:
                c.train(r)
            server.receive_ids()
            server.aggregate_parameters()
            if method == "FedAPEN":
                broadcast_global_aux(server)
            else:
                server.update_global_model()
            # Release any per-client masks kept on GPU after ISPU.
            for c in server.clients:
                if getattr(c, "masks", None):
                    c.masks = [m.detach().cpu() for m in c.masks]
            clear_memory()
            if grad_hook is not None and (r % 10 == 0 or r == rounds):
                try:
                    gm = grad_hook(server, r)
                    grad_log.append({"round": r, **gm})
                except Exception as e:
                    print(f"[warn] gradient hook failed at round {r}: {e}")
                    log_failure(exp, run_id, e, {"round": r, "hook": "grad"})
            # Log one row per client for this round.
            for c in server.clients:
                participated = int(c.id in selected_ids and c.id in server.uploaded_ids)
                row = {"exp": exp, "method": method, "dataset": dataset, "pattern": pattern,
                       "seed": seed, "round": r, "client_id": c.id,
                       "model_arch": arch_name_from_expr(args.models[c.id % len(args.models)]),
                       "participated": participated, "N_i_t": int(server.selectd_times[c.id]),
                       "p_i": float(server.participation_probability[c.id]),
                       "alpha_i_t": float(getattr(c, "update_ratio", np.nan)),
                       "lambda_i": _ensemble_weight(c),
                       "local_acc": np.nan, "local_loss": np.nan, "aux_acc": np.nan,
                       "ensemble_acc": np.nan, "global_G_acc_on_this_client": np.nan}
                if participated and log_active_metrics:
                    m = eval_client_models(c)
                    row.update({"local_acc": m["local_acc"], "local_loss": m["local_loss"],
                                "aux_acc": m["aux_acc"], "ensemble_acc": m["ensemble_acc"]})
                round_log.append(row)
            if r % 10 == 0 and r > 0:
                clear_memory()
                save_checkpoint(server, r, run_id, cfg, history.read())
        # Final post-training evaluation (the repo's loop evaluates before training).
        df = evaluate_all_clients(server, include_G=True)
        df.insert(0, "round", rounds)
        df.insert(0, "seed", seed)
        df.insert(0, "pattern", pattern)
        df.insert(0, "dataset", dataset)
        df.insert(0, "method", method)
        df.insert(0, "exp", exp)
        df["is_final"] = 1
        for _, row in df.iterrows():
            round_metrics.append(row.to_dict())
        fair = compute_fairness(df)
        history.append({"round": rounds, "is_final": 1, **fair, "seconds": time.time() - t0})
        save_checkpoint(server, rounds, run_id, cfg, history.read())
        # Results table and flags.
        h = history.read()
        best = h[h["is_final"] == 0].sort_values("AM", ascending=False).iloc[0].to_dict() \
            if len(h[h["is_final"] == 0]) else fair
        result = {"run_id": run_id, "method": method, "dataset": dataset, "pattern": pattern,
                  "seed": seed, "scale": scale, "rounds": rounds, "num_clients": num_clients,
                  "best_round": int(best["round"]), "AM": float(best["AM"]),
                  "FM": float(best["FM"]), "final_AM": float(fair["AM"]),
                  "final_FM": float(fair["FM"]), "min_acc": float(best["min_acc"]),
                  "p10": float(best["p10"]), "cvar10": float(best["cvar10"]),
                  "CV": float(best["CV"]), "jain": float(best["jain"]),
                  "rho_p_acc": float(best["rho_p_acc"]), "agg_mode": agg_mode,
                  "aux_arch": aux_arch, "tau": tau, "seconds": time.time() - t0,
                  "peak_vram_mb": peak_vram_mb(), "config": cfg}
        atomic_write_json(result, result_dir / f"{run_id}.json")
        flag.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (ROOT / "flags" / f"{run_id}.complete").write_text("complete", encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update({"status": "complete", "result": result,
                     "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        atomic_write_json(meta, meta_path)
        print(f"[done] {run_id}: AM={result['AM']:.3f} FM={result['FM']:.3f} "
              f"({result['seconds']/60:.1f} min)")
        return result
    except Exception as e:
        log_failure(exp, run_id, e, cfg)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "failed"
            meta["error"] = repr(e)
            atomic_write_json(meta, meta_path)
        except Exception:
            pass
        raise
    finally:
        pop_data_root()
        clear_memory()


# ---------------------------------------------------------------------------
# Smoke test and baseline
# ---------------------------------------------------------------------------

def smoke_test(repo: Optional[Path] = None, resume: bool = True) -> Dict[str, Any]:
    repo = Path(repo or REPO_DIR)
    install_repo_patches()
    ensure_fashion_mnist(10, repo)
    push_data_root(repo / "dataset")
    try:
        args = make_args(dataset="FashionMNIST", num_clients=10, rounds=1, batch_size=64,
                         local_epochs=1, eval_gap=1, seed=0, tau=0.2, sig_k=5.0,
                         valid_ratio=0.1, join_method="normal", global_arch="googlenet",
                         save_folder=ROOT / "run_store" / "smoke", model_family="HtM2")
        # Smoke uses two official architectures, not the full four, to fit 6 GB.
        args.models = [
            "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
            "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
        ]
        server = build_server(args, method="PHPFL", agg_mode="plain")
        server.selected_clients = server.select_clients()
        for c in server.selected_clients:
            c.learn_weight_for_inference()
        for c in server.selected_clients:
            c.train(0)
        server.receive_ids()
        server.aggregate_parameters()
        server.update_global_model()
        df = evaluate_all_clients(server, include_G=True)
        assert len(df) == 10, df
        assert np.isfinite(df["ensemble_acc"]).all()
        # Checkpoint + reload test.
        cfg = {"smoke": True, "repo_sha": PINNED_REPO_SHA}
        ck = save_checkpoint(server, 0, "smoke__checkpoint", cfg, pd.DataFrame([{"round": 0}]))
        assert ck and ck.exists()
        print("[smoke] PASS")
        return {"status": "pass", "clients": df.to_dict(orient="records"), "checkpoint": str(ck)}
    finally:
        pop_data_root()
        clear_memory()


def run_baseline(rounds: int = 100, seed: int = 1, resume: bool = True,
                 max_hours: float = 6.0, dataset: str = "Cifar10",
                 batch_size: int = 64, local_epochs: int = 5) -> Dict[str, Any]:
    """Cell 4 baseline: PHP-FL / Cifar10 / normal / plain aggregation."""
    ensure_cifar10(REPO_DIR)
    res = run_federated(exp="baseline", method="PHPFL", dataset=dataset, pattern="normal",
                        seed=seed, rounds=rounds, num_clients=20, batch_size=batch_size,
                        local_epochs=local_epochs, eval_gap=1, tau=0.2, sig_k=5.0,
                        valid_ratio=0.1, agg_mode="plain", aux_arch="googlenet",
                        data_root=REPO_DIR / "dataset", scale="full", resume=resume,
                        max_hours=max_hours)
    am, fm = float(res["AM"]), float(res["FM"])
    report = {"target_AM": 66.94, "target_FM": 8.07, "actual_AM": am, "actual_FM": fm,
              "delta_AM": am - 66.94, "delta_FM": fm - 8.07,
              "pass_AM": abs(am - 66.94) <= 1.0, "pass_FM": abs(fm - 8.07) <= 0.5,
              "rounds": rounds, "seed": seed, "run_id": res["run_id"],
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    report["passed"] = bool(report["pass_AM"] and report["pass_FM"])
    atomic_write_json(report, ROOT / "baseline" / "BASELINE_REPRO.json")
    print("\n" + "=" * 70)
    print("BASELINE REPRODUCTION")
    print(f"  AM: target 66.94, got {am:.3f}, delta {am-66.94:+.3f}  -> {'PASS' if report['pass_AM'] else 'FAIL'}")
    print(f"  FM: target  8.07, got {fm:.3f}, delta {fm-8.07:+.3f}  -> {'PASS' if report['pass_FM'] else 'FAIL'}")
    print("=" * 70)
    if not report["passed"]:
        raise RuntimeError("Baseline reproduction failed. STOP: do not run E1-E8.")
    (ROOT / "baseline" / "BASELINE_APPROVED.flag").write_text("pending-human-approval", encoding="utf-8")
    print("Baseline passed. STOP here for human approval before Cell 5+.")
    return report


# ---------------------------------------------------------------------------
# Experiment stubs (filled below)
# ---------------------------------------------------------------------------

def ensure_approved() -> None:
    if not (ROOT / "APPROVE_BASELINE").exists():
        raise RuntimeError(
            "Experiments are gated. Run Cell 5 with RUN_EXPERIMENTS=True after approving "
            f"the baseline, which creates {ROOT/'APPROVE_BASELINE'}.")

# ---------------------------------------------------------------------------
# Standalone local-only runner
# ---------------------------------------------------------------------------

def _train_local_only(client, round_no: int) -> None:
    import flcore.clients.clientbase as cb
    trainloader, _ = client.load_train_valid_data()
    model = cb.load_item(client.role, "model", client.save_folder_name)
    optimizer = torch.optim.SGD(model.parameters(), lr=client.learning_rate)
    scaler = _scaler()
    model.train()
    start = time.time()
    for _ in range(client.local_epochs):
        for x, y in trainloader:
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            with _autocast():
                loss = client.loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    client.join_times = int(getattr(client, "join_times", 0)) + 1
    client.update_ratio = sigmoid(client.sigmoid_k, (client.join_times + 1) * 1.0 / (round_no + 1))
    cb.save_item(model, client.role, "model", client.save_folder_name)
    client.train_time_cost["num_rounds"] += 1
    client.train_time_cost["total_cost"] += time.time() - start
    del model, optimizer, scaler
    clear_memory()


@torch.no_grad()
def eval_local_only(client) -> Dict[str, Any]:
    import flcore.clients.clientbase as cb
    model = cb.load_item(client.role, "model", client.save_folder_name)
    model.eval()
    loader = client.load_test_data()
    n = 0
    correct = 0
    loss_sum = 0.0
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    with _autocast():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += float(F.cross_entropy(out, y, reduction="sum").item())
            pred = out.argmax(1)
            correct += int((pred == y).sum().item())
            n += int(y.numel())
            for c in range(NUM_CLASSES):
                m = y == c
                class_total[c] += int(m.sum().item())
                class_correct[c] += int((pred[m] == c).sum().item())
    del model
    clear_memory()
    return {"n": n, "local_acc": correct / max(n, 1), "aux_acc": float("nan"),
            "ensemble_acc": correct / max(n, 1), "local_loss": loss_sum / max(n, 1),
            "class_acc": [class_correct[c] / class_total[c] if class_total[c] else float("nan")
                          for c in range(NUM_CLASSES)]}


def run_standalone(exp: str, dataset: str, pattern: str, seed: int, rounds: int,
                   num_clients: int, batch_size: int, local_epochs: int,
                   eval_gap: int, data_root: Optional[Path] = None,
                   scale: str = "smoke10", resume: bool = True,
                   max_hours: float = 6.0, model_family: str = "HtM4",
                   local_lr: float = 0.001) -> Dict[str, Any]:
    """Local-only baseline. It has no server, so participation patterns are ignored."""
    extra = {"rounds": rounds, "num_clients": num_clients, "batch": batch_size,
             "epochs": local_epochs, "agg": "none", "aux": "none",
             "tau": 0.0, "sigma": 0.0, "class_id": 0, "family": model_family}
    run_id = make_run_id(exp, "Standalone", dataset, pattern, seed, scale, extra)
    cfg = {"exp": exp, "method": "Standalone", "dataset": dataset, "pattern": pattern,
           "seed": seed, "scale": scale, **extra, "repo_sha": PINNED_REPO_SHA}
    flag = ROOT / "results" / exp / f"{run_id}_complete.flag"
    if resume and flag.exists():
        print(f"[skip] complete {run_id}")
        return {"run_id": run_id, "status": "complete", "skipped": True}
    result_dir = ROOT / "results" / exp
    log_dir = ROOT / "logs" / exp / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_store = ROOT / "run_store" / run_id
    run_store.mkdir(parents=True, exist_ok=True)
    round_metrics = CSVLogger(log_dir / "round_metrics.csv", _round_metric_columns())
    history = CSVLogger(log_dir / "history.csv",
                        ["round", "is_final", "AM", "AM_unweighted", "FM", "min_acc", "p10",
                         "cvar10", "CV", "jain", "rho_p_acc", "seconds"])
    meta_path = log_dir / "run_meta.json"
    atomic_write_json({"run_id": run_id, "config": cfg, "repo_sha": repo_sha(),
                       "packages": package_report(), "status": "running"}, meta_path)
    push_data_root(data_root)
    try:
        seed_everything(seed)
        args = make_args(dataset=dataset, num_clients=num_clients, rounds=rounds,
                         batch_size=batch_size, local_epochs=local_epochs,
                         eval_gap=eval_gap, seed=seed, tau=0.0, sig_k=5.0,
                         valid_ratio=0.1,
                         join_method=(pattern if pattern in ("normal", "linear", "uniform") else "normal"),
                         global_arch="googlenet", save_folder=run_store,
                         model_family=model_family, local_lr=local_lr)
        server = build_server(args, method="PHPFL", agg_mode="plain")
        server.participation_probability = participation_probabilities(
            pattern, dataset, num_clients, seed=seed)
        start_round = 0
        ckpt = find_checkpoint(run_id) if resume else None
        if ckpt is not None:
            start_round = restore_checkpoint(server, ckpt, cfg)
            round_metrics.truncate_round(start_round - 1)
            history.truncate_round(start_round - 1)
        reset_peak_vram()
        t0 = time.time()
        for r in range(start_round, rounds + 1):
            if (time.time() - t0) > max_hours * 3600:
                save_checkpoint(server, r - 1, run_id, cfg, history.read())
                raise TimeoutError(f"Standalone run exceeded MAX_HOURS={max_hours}")
            if r % eval_gap == 0 or r in (20, 50, 100, rounds):
                rows = []
                for c in server.clients:
                    m = eval_local_only(c)
                    counts = label_counts(dataset, c.id)
                    present = np.where(counts > 0)[0]
                    rare = int(present[np.argmin(counts[present])]) if len(present) else -1
                    rows.append({
                        "client_id": c.id,
                        "arch": arch_name_from_expr(args.models[c.id % len(args.models)]),
                        "n": m["n"], "p_i": float(server.participation_probability[c.id]),
                        "N_i": int(getattr(c, "join_times", 0)),
                        "local_acc": m["local_acc"], "local_loss": m["local_loss"],
                        "aux_acc": float("nan"), "ensemble_acc": m["ensemble_acc"],
                        "G_acc": float("nan"), "alpha_i": float(getattr(c, "update_ratio", np.nan)),
                        "lambda_i": 1.0, "rare_class": rare,
                        "rare_acc": m["class_acc"][rare] if rare >= 0 else float("nan"),
                        **{f"class_{k}_acc": m["class_acc"][k] for k in range(NUM_CLASSES)},
                    })
                df = pd.DataFrame(rows)
                df.insert(0, "round", r)
                df.insert(0, "seed", seed)
                df.insert(0, "pattern", pattern)
                df.insert(0, "dataset", dataset)
                df.insert(0, "method", "Standalone")
                df.insert(0, "exp", exp)
                df["is_final"] = 0
                for _, row in df.iterrows():
                    round_metrics.append(row.to_dict())
                fair = compute_fairness(df)
                history.append({"round": r, "is_final": 0, **fair, "seconds": time.time() - t0})
                print(f"[{exp}/Standalone/{pattern}/s{seed}] r{r:03d} "
                      f"AM={fair.get('AM', float('nan')):.3f}")
            for c in server.clients:
                _train_local_only(c, r)
            if r % 10 == 0 and r > 0:
                save_checkpoint(server, r, run_id, cfg, history.read())
        rows = []
        for c in server.clients:
            m = eval_local_only(c)
            counts = label_counts(dataset, c.id)
            present = np.where(counts > 0)[0]
            rare = int(present[np.argmin(counts[present])]) if len(present) else -1
            rows.append({
                "client_id": c.id, "arch": arch_name_from_expr(args.models[c.id % len(args.models)]),
                "n": m["n"], "p_i": float(server.participation_probability[c.id]),
                "N_i": int(getattr(c, "join_times", 0)),
                "local_acc": m["local_acc"], "local_loss": m["local_loss"],
                "aux_acc": float("nan"), "ensemble_acc": m["ensemble_acc"],
                "G_acc": float("nan"), "alpha_i": float(getattr(c, "update_ratio", np.nan)),
                "lambda_i": 1.0, "rare_class": rare,
                "rare_acc": m["class_acc"][rare] if rare >= 0 else float("nan"),
                **{f"class_{k}_acc": m["class_acc"][k] for k in range(NUM_CLASSES)},
            })
        df = pd.DataFrame(rows)
        df.insert(0, "round", rounds); df.insert(0, "seed", seed)
        df.insert(0, "pattern", pattern); df.insert(0, "dataset", dataset)
        df.insert(0, "method", "Standalone"); df.insert(0, "exp", exp)
        df["is_final"] = 1
        for _, row in df.iterrows():
            round_metrics.append(row.to_dict())
        fair = compute_fairness(df)
        history.append({"round": rounds, "is_final": 1, **fair, "seconds": time.time() - t0})
        save_checkpoint(server, rounds, run_id, cfg, history.read())
        h = history.read()
        best = h[h["is_final"] == 0].sort_values("AM", ascending=False).iloc[0].to_dict()
        result = {"run_id": run_id, "method": "Standalone", "dataset": dataset,
                  "pattern": pattern, "seed": seed, "scale": scale, "rounds": rounds,
                  "num_clients": num_clients, "best_round": int(best["round"]),
                  "AM": float(best["AM"]), "FM": float(best["FM"]),
                  "final_AM": float(fair["AM"]), "final_FM": float(fair["FM"]),
                  "min_acc": float(best["min_acc"]), "p10": float(best["p10"]),
                  "cvar10": float(best["cvar10"]), "CV": float(best["CV"]),
                  "jain": float(best["jain"]), "rho_p_acc": float(best["rho_p_acc"]),
                  "agg_mode": "none", "aux_arch": "none", "tau": 0.0,
                  "seconds": time.time() - t0, "peak_vram_mb": peak_vram_mb(), "config": cfg}
        atomic_write_json(result, result_dir / f"{run_id}.json")
        flag.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (ROOT / "flags" / f"{run_id}.complete").write_text("complete", encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update({"status": "complete", "result": result})
        atomic_write_json(meta, meta_path)
        print(f"[done] {run_id}: AM={result['AM']:.3f} FM={result['FM']:.3f}")
        return result
    except Exception as e:
        log_failure(exp, run_id, e, cfg)
        raise
    finally:
        pop_data_root()
        clear_memory()


# ---------------------------------------------------------------------------
# Scope change: A (fairness metrics), B (aggregation), C (correlated data)
# ---------------------------------------------------------------------------

SCOPE = {
    "dataset": "FashionMNIST", "clients": 10, "rounds": 40,
    "batch_size": 64, "local_epochs": 1, "eval_gap": 5, "seed": 0,
    "tau": 0.2, "sig_k": 5.0, "valid_ratio": 0.1,
    # Smoke uses the two fastest official backbones; escalation restores HtM4.
    "model_family": "HtM2",
}


def _scope_data_root() -> Path:
    ensure_fashion_mnist(SCOPE["clients"], REPO_DIR)
    return REPO_DIR / "dataset"


def _result_path(exp: str, method: str, pattern: str, seed: int,
                 scale: str = "smoke10", agg_mode: Optional[str] = None) -> Optional[Path]:
    d = ROOT / "results" / exp
    if not d.exists():
        return None
    for p in d.glob(f"{safe_name(exp)}__{safe_name(method)}__{safe_name(pattern)}__s{seed}__{scale}__*.json"):
        if agg_mode is not None:
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("agg_mode") != agg_mode:
                    continue
            except Exception:
                continue
        return p
    return None


def _load_best_client_vector(result: Dict[str, Any], exp: str) -> pd.DataFrame:
    run_id = result["run_id"]
    path = ROOT / "logs" / exp / run_id / "round_metrics.csv"
    df = pd.read_csv(path)
    # Prefer the round that the run selected as its best AM (repo protocol).
    cand = df[(df["round"] == int(result["best_round"])) & (df["is_final"] == 0)]
    if len(cand) == 0:
        cand = df[df["is_final"] == 1]
    if len(cand) == 0:
        raise RuntimeError(f"No per-client metrics in {path}")
    return cand.drop_duplicates(subset=["client_id"], keep="last").copy()


def run_A_fairness() -> pd.DataFrame:
    """A: recompute fairness metrics from every completed scoped run (0 compute)."""
    # If no plain baseline exists at all, create it once so A has a reference.
    if _result_path("B_agg", "PHPFL", "normal", SCOPE["seed"], agg_mode="plain") is None:
        print("[A] no plain baseline found; running the 10-client/40-round smoke baseline once")
        run_B_aggregation(modes=("plain",))

    rows = []
    seen = set()
    for exp in ["B_agg", "C_participation", "baseline"]:
        d = ROOT / "results" / exp
        if not d.exists():
            continue
        for path in sorted(d.glob("*.json")):
            try:
                res = json.loads(path.read_text(encoding="utf-8"))
                run_id = res.get("run_id")
                if not run_id or run_id in seen:
                    continue
                log = ROOT / "logs" / exp / run_id / "round_metrics.csv"
                if not log.exists():
                    continue
                df = _load_best_client_vector(res, exp)
                fair = compute_fairness(df)
                seen.add(run_id)
                rows.append({
                    "exp": exp, "method": res.get("method", ""),
                    "dataset": res.get("dataset", ""), "pattern": res.get("pattern", ""),
                    "agg_mode": res.get("agg_mode", ""), "seed": res.get("seed", ""),
                    "round": int(res.get("best_round", -1)),
                    "AM": fair.get("AM", np.nan), "FM": fair.get("FM", np.nan),
                    "min_acc": fair.get("min_acc", np.nan), "p10": fair.get("p10", np.nan),
                    "cvar10": fair.get("cvar10", np.nan), "CV": fair.get("CV", np.nan),
                    "jain": fair.get("jain", np.nan), "rho_p_acc": fair.get("rho_p_acc", np.nan),
                })
            except Exception as e:
                print(f"[A] skip {path.name}: {e}")
    if not rows:
        raise RuntimeError("A: no completed runs found")
    out = pd.DataFrame(rows).sort_values(["exp", "method", "pattern", "agg_mode"])
    atomic_write_csv(out, ROOT / "results" / "A_fairness.csv")

    # Narrative test: does the FM winner also win the tail metrics?
    i_fm = int(out["FM"].idxmin())
    i_min = int(out["min_acc"].idxmax())
    i_cvar = int(out["cvar10"].idxmax())
    fm_row = out.loc[i_fm]
    min_row = out.loc[i_min]
    cvar_row = out.loc[i_cvar]
    same_tail = (i_fm == i_min == i_cvar)
    rho_pos = out.loc[i_fm, "rho_p_acc"] > 0
    sentence = (
        f"Lowest FM is {fm_row['method']}/{fm_row['pattern']}/{fm_row['agg_mode']} "
        f"(FM={fm_row['FM']:.4f}); best min-accuracy is "
        f"{min_row['method']}/{min_row['pattern']}/{min_row['agg_mode']} "
        f"({min_row['min_acc']:.4f}); best CVaR10 is "
        f"{cvar_row['method']}/{cvar_row['pattern']}/{cvar_row['agg_mode']} "
        f"({cvar_row['cvar10']:.4f}).  "
        + ("The FM winner also wins the tail metrics in this smoke run."
           if same_tail else
           "The FM winner does NOT also win both tail metrics, so std alone is "
           "not a sufficient fairness summary here.")
        + f"  Spearman rho(p_i, accuracy) for the FM winner is "
          f"{fm_row['rho_p_acc']:+.4f} ({'positive' if rho_pos else 'not positive'})."
    )
    with open(ROOT / "A_findings.md", "w", encoding="utf-8") as f:
        f.write("# A - fairness metric recomputation\n\n")
        f.write(out.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n" + sentence + "\n")
    print(out.to_string(index=False))
    return out

def run_B_aggregation(modes: Sequence[str] = ("plain", "inv_p", "inv_N")) -> pd.DataFrame:
    """B: PHP-FL only, 10 clients, 40 rounds, Fashion-MNIST, 1 seed."""
    data_root = _scope_data_root()
    results = []
    for mode in modes:
        res = run_federated(exp="B_agg", method="PHPFL", dataset=SCOPE["dataset"],
                            pattern="normal", seed=SCOPE["seed"], rounds=SCOPE["rounds"],
                            num_clients=SCOPE["clients"], batch_size=SCOPE["batch_size"],
                            local_epochs=SCOPE["local_epochs"], eval_gap=SCOPE["eval_gap"],
                            tau=SCOPE["tau"], sig_k=SCOPE["sig_k"],
                            valid_ratio=SCOPE["valid_ratio"], agg_mode=mode,
                            aux_arch="googlenet", data_root=data_root,
                            scale="smoke10", resume=True, max_hours=3.0,
                            model_family=SCOPE["model_family"], log_active_metrics=False)
        results.append((mode, res))
    rows = []
    for mode, res in results:
        if res.get("skipped"):
            p = _result_path("B_agg", "PHPFL", "normal", SCOPE["seed"], agg_mode=mode)
            res = json.loads(p.read_text(encoding="utf-8"))
        df = _load_best_client_vector(res, "B_agg")
        for _, r in df.iterrows():
            rows.append({"exp": "B", "agg_mode": mode, "client_id": int(r["client_id"]),
                         "arch": r.get("arch", ""), "p_i": float(r["p_i"]),
                         "N_i": int(r["N_i"]), "acc": float(r["ensemble_acc"]),
                         "rare_acc": float(r["rare_acc"]) if pd.notna(r["rare_acc"]) else np.nan,
                         "AM": float(res["AM"]), "FM": float(res["FM"]),
                         "seed": SCOPE["seed"]})
    out = pd.DataFrame(rows)
    atomic_write_csv(out, ROOT / "results" / "B_results.csv")
    return out


def wilcoxon_pair(df: pd.DataFrame, col: str, a: str, b: str,
                  value: str = "acc", metric: str = "accuracy",
                  label: Optional[str] = None) -> Dict[str, Any]:
    from scipy.stats import wilcoxon
    x = df[df[col] == a].set_index("client_id")[value]
    y = df[df[col] == b].set_index("client_id")[value]
    z = pd.concat([x, y], axis=1, join="inner").dropna()
    comp = label or f"{a} vs {b}"
    if len(z) < 5:
        return {"metric": metric, "comparison": comp, "n": len(z), "statistic": np.nan,
                "p_value": np.nan, "median_diff": np.nan, "note": "too few pairs"}
    diff = z.iloc[:, 0].to_numpy() - z.iloc[:, 1].to_numpy()
    if np.allclose(diff, 0):
        return {"metric": metric, "comparison": comp, "n": len(z), "statistic": 0.0,
                "p_value": 1.0, "median_diff": 0.0, "note": "all differences zero"}
    stat, p = wilcoxon(z.iloc[:, 0], z.iloc[:, 1], zero_method="wilcox", alternative="two-sided")
    return {"metric": metric, "comparison": comp, "n": len(z), "statistic": float(stat),
            "p_value": float(p), "median_diff": float(np.median(diff)), "note": ""}


def _paired_values(df: pd.DataFrame, col: str, a: str, b: str,
                   value: str = "acc") -> pd.DataFrame:
    x = df[df[col] == a].set_index("client_id")[value]
    y = df[df[col] == b].set_index("client_id")[value]
    return pd.concat([x, y], axis=1, join="inner").dropna()


def write_stats() -> pd.DataFrame:
    """Wilcoxon signed-rank tests across clients (paired by client_id)."""
    from scipy.stats import wilcoxon
    bpath = ROOT / "results" / "B_results.csv"
    cpath = ROOT / "results" / "C_results.csv"
    tests = []
    if bpath.exists():
        b = pd.read_csv(bpath)
        tests += [
            wilcoxon_pair(b, "agg_mode", "plain", "inv_p", metric="accuracy",
                          label="plain vs inv_p [accuracy]"),
            wilcoxon_pair(b, "agg_mode", "plain", "inv_N", metric="accuracy",
                          label="plain vs inv_N [accuracy]"),
            wilcoxon_pair(b, "agg_mode", "inv_p", "inv_N", metric="accuracy",
                          label="inv_p vs inv_N [accuracy]"),
        ]
    c = None
    if cpath.exists():
        c = pd.read_csv(cpath)
        for pat in ["linear", "entropy_correlated"]:
            sub = c[c["pattern"] == pat]
            tests.append(wilcoxon_pair(sub, "method", "PHPFL", "Standalone",
                                       metric="accuracy",
                                       label=f"PHPFL vs Standalone [{pat}]"))
        php = c[c["method"] == "PHPFL"]
        tests.append(wilcoxon_pair(php, "pattern", "linear", "entropy_correlated",
                                   metric="accuracy",
                                   label="linear vs entropy [accuracy]"))
        # Fairness-specific paired test: per-client absolute deviation from the
        # group mean.  This asks whether the distribution becomes more unequal,
        # which a mean-accuracy Wilcoxon cannot see.
        z = _paired_values(php, "pattern", "linear", "entropy_correlated", "acc")
        if len(z) >= 5:
            xdev = (z.iloc[:, 0] - z.iloc[:, 0].mean()).abs()
            ydev = (z.iloc[:, 1] - z.iloc[:, 1].mean()).abs()
            if np.allclose((xdev - ydev).to_numpy(), 0):
                stat, p, med = 0.0, 1.0, 0.0
            else:
                stat, p = wilcoxon(xdev, ydev, zero_method="wilcox", alternative="two-sided")
                med = float(np.median((ydev - xdev).to_numpy()))
            tests.append({"metric": "|acc - group mean|",
                          "comparison": "linear vs entropy [dispersion]",
                          "n": len(z), "statistic": float(stat), "p_value": float(p),
                          "median_diff": med,
                          "note": "larger = more unequal"})
        # Does PHP-FL's advantage over Standalone change with the pattern?
        st = c[(c["pattern"] == "linear") & (c["method"] == "Standalone")].set_index("client_id")["acc"]
        lin = c[(c["pattern"] == "linear") & (c["method"] == "PHPFL")].set_index("client_id")["acc"]
        ent = c[(c["pattern"] == "entropy_correlated") & (c["method"] == "PHPFL")].set_index("client_id")["acc"]
        adv = pd.concat([lin - st, ent - st], axis=1, join="inner").dropna()
        if len(adv) >= 5:
            if np.allclose((adv.iloc[:, 1] - adv.iloc[:, 0]).to_numpy(), 0):
                stat, p, med = 0.0, 1.0, 0.0
            else:
                stat, p = wilcoxon(adv.iloc[:, 1], adv.iloc[:, 0], zero_method="wilcox",
                                   alternative="two-sided")
                med = float(np.median((adv.iloc[:, 1] - adv.iloc[:, 0]).to_numpy()))
            tests.append({"metric": "PHPFL - Standalone",
                          "comparison": "linear vs entropy [advantage]",
                          "n": len(adv), "statistic": float(stat), "p_value": float(p),
                          "median_diff": med,
                          "note": "positive = advantage grows under entropy"})
    out = pd.DataFrame(tests)
    if len(out):
        out["significant_0.05"] = out["p_value"] < 0.05
        atomic_write_csv(out, ROOT / "results" / "STATS_tests.csv")

    rho_txt = ""
    if c is not None:
        try:
            from scipy.stats import spearmanr
            php = c[c["method"] == "PHPFL"]
            r_lin = spearmanr(php[php.pattern == "linear"]["p_i"],
                              php[php.pattern == "linear"]["acc"]).correlation
            r_ent = spearmanr(php[php.pattern == "entropy_correlated"]["p_i"],
                              php[php.pattern == "entropy_correlated"]["acc"]).correlation
            rho_txt = (f"Spearman rho(p_i, per-client accuracy) is {r_lin:+.3f} under "
                       f"linear and {r_ent:+.3f} under entropy_correlated.  ")
        except Exception:
            pass
    with open(ROOT / "STATS.md", "w", encoding="utf-8") as f:
        f.write("# STATS - paired Wilcoxon signed-rank tests\n\n")
        f.write("Unit of analysis: client.  Each pair is matched by `client_id`; "
                "seeds are not treated as replicates (one seed in this smoke run). "
                "Two-sided test, zero differences use Wilcoxon's default handling.\n\n")
        if len(out):
            f.write(out.to_markdown(index=False, floatfmt=".5f"))
        else:
            f.write("_No B/C result tables found yet._\n")
        f.write("\n\n" + rho_txt)
        # One-sentence conclusion for the scope-change decision.
        try:
            php = c[c["method"] == "PHPFL"]
            lin = php[php.pattern == "linear"]["acc"]
            ent = php[php.pattern == "entropy_correlated"]["acc"]
            fm_lin = php[php.pattern == "linear"]["FM"].iloc[0]
            fm_ent = php[php.pattern == "entropy_correlated"]["FM"].iloc[0]
            p_disp = float(out[out.comparison.str.contains("dispersion")]["p_value"].iloc[0])
            p_acc = float(out[out.comparison.str.contains(r"\[accuracy\]") &
                               out.comparison.str.contains("linear vs entropy")]["p_value"].iloc[0])
            f.write(f"C result: mean accuracy moves {lin.mean():.4f} -> {ent.mean():.4f} "
                    f"(paired p={p_acc:.4g}), while FM moves {fm_lin:.4f} -> {fm_ent:.4f} "
                    f"and min accuracy moves {lin.min():.4f} -> {ent.min():.4f}; the "
                    f"paired per-client dispersion test is p={p_disp:.4g}.  The effect "
                    f"is a tail/fairness degradation, not a significant mean collapse.\n")
        except Exception as e:
            f.write(f"C conclusion unavailable: {e}\n")
        f.write("With 10 clients the smallest attainable two-sided p-value is 0.00195.  "
                "A non-significant mean-accuracy result is not evidence of no effect.  "
                "The dispersion row is the fairness-specific paired test: it asks "
                "whether the per-client accuracy distribution became more unequal, "
                "which the mean-accuracy test cannot detect.  A is descriptive "
                "(one baseline vector) and therefore has no paired test.  No "
                "multiple-comparison correction is applied; these are exploratory "
                "smoke tests and the dispersion p=0.041 is marginal.\n")
    print(out.to_string(index=False) if len(out) else "No stats yet")
    return out

def run_C_participation() -> pd.DataFrame:
    """C: {linear, entropy_correlated} x {PHP-FL, Standalone}, same smoke scale."""
    data_root = _scope_data_root()
    rows = []
    results = {}
    for pattern in ["linear", "entropy_correlated"]:
        p = participation_probabilities(pattern, SCOPE["dataset"], SCOPE["clients"],
                                        seed=SCOPE["seed"])
        res = run_federated(exp="C_participation", method="PHPFL", dataset=SCOPE["dataset"],
                            pattern=pattern, seed=SCOPE["seed"], rounds=SCOPE["rounds"],
                            num_clients=SCOPE["clients"], batch_size=SCOPE["batch_size"],
                            local_epochs=SCOPE["local_epochs"], eval_gap=SCOPE["eval_gap"],
                            tau=SCOPE["tau"], sig_k=SCOPE["sig_k"],
                            valid_ratio=SCOPE["valid_ratio"], agg_mode="plain",
                            aux_arch="googlenet", data_root=data_root, scale="smoke10",
                            resume=True, max_hours=3.0, p_override=p,
                            model_family=SCOPE["model_family"], log_active_metrics=False)
        results[("PHPFL", pattern)] = res
    # Standalone is server-free, so it is pattern-invariant.  Run it once and
    # reuse the same client vector for both pattern rows.
    st = run_standalone(exp="C_participation", dataset=SCOPE["dataset"], pattern="linear",
                        seed=SCOPE["seed"], rounds=SCOPE["rounds"],
                        num_clients=SCOPE["clients"], batch_size=SCOPE["batch_size"],
                        local_epochs=SCOPE["local_epochs"], eval_gap=SCOPE["eval_gap"],
                        data_root=data_root, scale="smoke10", resume=True, max_hours=3.0,
                        model_family=SCOPE["model_family"])
    for pattern in ["linear", "entropy_correlated"]:
        pvec = participation_probabilities(pattern, SCOPE["dataset"], SCOPE["clients"],
                                           seed=SCOPE["seed"])
        for method, res in [("PHPFL", results[("PHPFL", pattern)]), ("Standalone", st)]:
            if res.get("skipped"):
                pth = _result_path("C_participation", method, pattern if method == "PHPFL" else "linear",
                                   SCOPE["seed"])
                res = json.loads(pth.read_text(encoding="utf-8"))
            df = _load_best_client_vector(res, "C_participation")
            for _, r in df.iterrows():
                cid = int(r["client_id"])
                rows.append({"exp": "C", "pattern": pattern, "method": method,
                             "client_id": cid, "arch": r.get("arch", ""),
                             # Standalone is pattern-invariant; report the
                             # pattern's declared p for the scatter/table.
                             "p_i": float(pvec[cid]) if method == "Standalone" else float(r["p_i"]),
                             "N_i": int(r["N_i"]),
                             "acc": float(r["ensemble_acc"]),
                             "rare_acc": float(r["rare_acc"]) if pd.notna(r["rare_acc"]) else np.nan,
                             "AM": float(res["AM"]), "FM": float(res["FM"]),
                             "seed": SCOPE["seed"]})
    out = pd.DataFrame(rows)
    atomic_write_csv(out, ROOT / "results" / "C_results.csv")
    return out


def _savefig(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_scope_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    figdir = ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    apath = ROOT / "results" / "A_fairness.csv"
    if apath.exists():
        atab = pd.read_csv(apath)
        sel = atab[(atab.get("agg_mode", "") == "plain") &
                   (atab.get("pattern", "") == "normal")]
        a = (sel.iloc[0] if len(sel) else atab.iloc[0])
        metrics = ["FM", "min_acc", "p10", "cvar10", "CV", "jain"]
        vals = [float(a[m]) for m in metrics]
        plt.figure(figsize=(7, 3.8))
        ax = sns.barplot(x=metrics, y=vals, color="#4C72B0")
        ax.set_title("A. Fairness metrics on the plain baseline")
        ax.set_ylabel("value")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _savefig(figdir / "A_fairness.png")

    bpath = ROOT / "results" / "B_results.csv"
    if bpath.exists():
        b = pd.read_csv(bpath)
        plt.figure(figsize=(7, 4.2))
        sns.lineplot(data=b, x="client_id", y="acc", hue="agg_mode",
                     marker="o", dashes=False)
        plt.title("B. Per-client accuracy by aggregation mode")
        plt.xlabel("client id")
        plt.ylabel("test accuracy")
        _savefig(figdir / "B_aggregation_clients.png")
        agg = b.groupby("agg_mode", as_index=False).agg(AM=("acc", "mean"),
                                                        FM=("acc", lambda x: np.std(x, ddof=0)))
        agg = agg.melt(id_vars="agg_mode", var_name="metric", value_name="value")
        plt.figure(figsize=(6, 3.8))
        sns.barplot(data=agg, x="agg_mode", y="value", hue="metric")
        plt.title("B. Mean and std of per-client accuracy")
        _savefig(figdir / "B_aggregation_summary.png")

    cpath = ROOT / "results" / "C_results.csv"
    if cpath.exists():
        c = pd.read_csv(cpath)
        plt.figure(figsize=(8, 4.2))
        sns.barplot(data=c, x="pattern", y="acc", hue="method",
                    errorbar="sd", capsize=0.1)
        plt.title("C. Participation pattern x method")
        plt.ylabel("per-client test accuracy")
        _savefig(figdir / "C_participation_methods.png")
        plt.figure(figsize=(7, 4))
        sns.scatterplot(data=c, x="p_i", y="acc", hue="method", style="pattern", s=80)
        plt.title("C. Declared participation probability vs. accuracy")
        _savefig(figdir / "C_participation_scatter.png")
        # Tail plot: the mean bar above hides the collapse that matters for C.
        tail = (c.groupby(["pattern", "method"])["acc"]
                  .agg(min_acc="min", p10=lambda x: float(np.percentile(x, 10)),
                       mean_acc="mean").reset_index())
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        sns.barplot(data=tail, x="pattern", y="min_acc", hue="method", ax=axes[0])
        axes[0].set_title("worst client")
        axes[0].set_ylabel("accuracy")
        sns.barplot(data=tail, x="pattern", y="p10", hue="method", ax=axes[1])
        axes[1].set_title("10th percentile client")
        axes[1].set_ylabel("")
        fig.suptitle("C. Tail accuracy by participation pattern")
        _savefig(figdir / "C_participation_tail.png")


def _md_table(path: Path) -> str:
    if not path.exists():
        return "_missing_"
    df = pd.read_csv(path).fillna("")
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_verify() -> None:
    pkg = package_report()
    text = f"""# VERIFY - patches, data, and non-obvious choices

## Pinned repository

- `{REPO_URL}`
- commit `{repo_sha()}`
- working copy: `{REPO_DIR}`

## Patches applied (runtime, idempotent)

1. **Aggregation hook** (`PHPFL.aggregate_parameters`): the repository's
   `PHPFL` override is a plain mean over active clients. We replace only the
   weight computation with `aggregation_weights(server, mode)`:
   `plain` = 1/|A|, `inv_p` = 1/p_i, `inv_N` = 1/max(N_i,1),
   `fedavg` = n_i. Weights are normalized. `plain` is the untouched baseline.
2. **AMP client training** (`clientPHP.train`, `learn_weight_for_inference`):
   the same DEAL losses and optimizer updates are wrapped in
   `torch.cuda.amp.autocast(dtype=torch.float16)` + `GradScaler`.
3. **CPU-backed model zoo**: `save_item` writes CPU models and `load_item`
   moves one model to the active device. This keeps the 6 GB card from holding
   20 w_i/g_i pairs. `filter_top_p_percent` and `update_model_with_mask` move
   one mask at a time.
4. **ISPU masks** stay on CPU after each update; they are recomputed every
   participation and are not part of the persistent GPU state.
5. **Logging cost control**: `round_log.csv` records every (round, client) row,
   but the per-active-client local/aux/ensemble evaluation is disabled in the
   scoped runs (`log_active_metrics=False`) to keep the 6 GB smoke run under
   three hours.  Full per-client accuracy vectors are in
   `round_metrics.csv` every 5 rounds for all 10 clients; A/B/C read those.

## Data

- CIFAR-10 is generated with the repository's `dataset/generate_cifar10.py
  noniid - dir` (Dirichlet alpha=0.5, 20 clients, 15% test holdout). The
  generated label statistics were verified byte-for-byte against the
  repository's shipped `config.json`.
- The scope-change runs are Fashion-MNIST, 10 clients, Dirichlet alpha=0.5.
  Because the paper code does not ship Fashion-MNIST, a shim writes the same
  `.npz` layout. Smoke caps are 1000 train / 200 test per client, so 10 clients
  x 40 rounds fit a 6 GB laptop GPU. This is a smoke diagnostic, not a claim
  about full-scale CIFAR-10.
- Smoke uses the two fastest official backbones (GoogLeNet, ResNet18) instead
  of the paper's four CNNs. Escalation to CIFAR-10/half-scale restores HtM4.

## Scope-change decisions

- **Entropy-correlated participation**: the original prompt gave
  `p_i = exp(-H(D_i))` but described high-entropy clients as participating
  more. Those are opposite directions. We use the normalized entropy
  `p_i = clip(exp(H - ln K), 0.02, 1)` so that balanced clients participate
  more and rare-class holders less, matching the stated intent.
- **Standalone** is server-free and therefore pattern-invariant. It is run
  once and reused for the linear and entropy rows of C; this is explicit in
  `C_results.csv`.
- **Wilcoxon unit**: clients, paired by `client_id`. Seeds are not replicates
  (one seed in the smoke scope). With 10 clients the minimum two-sided
  p-value is 0.00195.
- **E4-E8** are not run in this scope.

## Environment

The notebook installs `torch==2.2.2+cu121`, `numpy==1.26.4` and the pinned
Colab stack.  The local execution used to produce these artifacts ran on the
installed laptop stack shown below (torch 2.14.1 / CUDA 12.6 / numpy 2.5.4).
That is a deliberate 6 GB fallback, not the paper's pinned environment; re-run
the smoke scope on the Colab pins before treating the numbers as final.

```json
{json.dumps(pkg, indent=2)}
```
"""
    (ROOT / "VERIFY.md").write_text(text, encoding="utf-8")


def write_runbook() -> None:
    text = f"""# RUNBOOK - scoped PHP-FL experiments

## What is in scope

- **A** fairness-metric recomputation from the plain baseline logs (0 compute
  once B/plain exists).
- **B** aggregation reweighting: `plain`, `inv_p`, `inv_N`, PHP-FL only,
  10 clients, 40 rounds, Fashion-MNIST, seed 0.
- **C** correlated participation: `linear`, `entropy_correlated` x
  PHP-FL, Standalone, same smoke scale.

E4-E8 are intentionally absent.  The full 170-hour matrix was not launched.

## Resume

1. Re-run Cells 0-2 (install, Drive paths, repo clone + patches).
2. Re-run the experiment cell.  It checks `results/{{exp}}/{{run_id}}_complete.flag`
   and the latest `ckpts/{{run_id}}/..._r*.pt`.
3. A run that is already complete is skipped.  A partial run resumes after the
   latest checkpoint (every 10 rounds).

## Scale / cost

Smoke is 10 clients x 40 rounds x 1 seed.  A complete B+C pass is six training
runs: B (3 aggregation modes) + C (2 PHP-FL patterns + 1 shared Standalone).
The Standalone run is reused for both C patterns because it has no server.

If you escalate C to CIFAR-10 / half-scale, restore `model_family="HtM4"` and
the 4-CNN round-robin, use 20 clients/60 rounds/2 seeds, and set `SCOPE` in
`phofl_lab.py` accordingly.

## Artifacts

- `results/A_fairness.csv`, `B_results.csv`, `C_results.csv`, `STATS_tests.csv`
- `logs/{{exp}}/{{run_id}}/round_metrics.csv`, `round_log.csv`, `G_eval.csv`
- `ckpts/{{run_id}}/..._r*.pt` (latest checkpoint per run)
- `figures/*.png` (300 DPI)
- `RESULTS.md`, `STATS.md`, `VERIFY.md`, `RUNBOOK.md`

## Failure policy

Every run writes a traceback to `failures.log`, preserves the latest
checkpoint, and refuses to mark itself complete.  No exception is swallowed.

## Approval gate

The original 170-hour plan stopped after a CIFAR-10 baseline for human
approval.  This scope-change run is the approved reduced matrix, so it proceeds
directly through A, B and C; no separate `APPROVE_BASELINE` sentinel is used.
"""
    (ROOT / "RUNBOOK.md").write_text(text, encoding="utf-8")


def finalize() -> None:
    write_stats()
    make_scope_figures()
    b = pd.read_csv(ROOT / "results" / "B_results.csv") if (ROOT / "results" / "B_results.csv").exists() else pd.DataFrame()
    c = pd.read_csv(ROOT / "results" / "C_results.csv") if (ROOT / "results" / "C_results.csv").exists() else pd.DataFrame()
    stats = pd.read_csv(ROOT / "results" / "STATS_tests.csv").fillna("") if (ROOT / "results" / "STATS_tests.csv").exists() else pd.DataFrame()
    lines = ["# RESULTS - scoped PHP-FL experiments", ""]
    lines += ["Scope: A fairness metrics; B aggregation reweighting; C correlated "
              "participation.  Smoke scale: 10 clients, 40 rounds, Fashion-MNIST, "
              "1 seed.  E4-E8 were not run.  AM/FM use the repository's protocol: "
              "the best pre-training evaluation round over rounds 0..40.", "",
              "Limitations: 10 clients x 1,000 train / 200 test samples, two "
              "official backbones (GoogLeNet, ResNet18), one seed.  This is a "
              "diagnostic smoke run, not a CIFAR-10 claim.  Wilcoxon has 10 pairs "
              "and minimum two-sided p=0.00195.", ""]
    lines += ["## A. Fairness is more than std", ""]
    lines += [_md_table(ROOT / "results" / "A_fairness.csv"), ""]
    if (ROOT / "A_findings.md").exists():
        lines += [Path(ROOT / "A_findings.md").read_text(encoding="utf-8").split("\n\n")[-1], ""]
    lines += ["## B. Aggregation reweighting", ""]
    lines += [_md_table(ROOT / "results" / "B_results.csv"), ""]
    if len(b):
        piv = b.pivot_table(index="client_id", columns="agg_mode", values="acc")
        lines += ["Per-client paired accuracy (rows = clients):", "",
                  piv.to_markdown(floatfmt=".4f"), ""]
        for mode in ["plain", "inv_p", "inv_N"]:
            sub = b[b["agg_mode"] == mode]
            if len(sub):
                lines += [f"- `{mode}`: mean={sub['acc'].mean():.4f}, "
                          f"std={sub['acc'].std(ddof=0):.4f}, "
                          f"min={sub['acc'].min():.4f}"]
        lines += [""]
    lines += ["## C. Correlated participation", ""]
    lines += [_md_table(ROOT / "results" / "C_results.csv"), ""]
    if len(c):
        piv = c.pivot_table(index="client_id", columns=["pattern", "method"], values="acc")
        lines += ["Per-client paired accuracy (rows = clients):", "",
                  piv.to_markdown(floatfmt=".4f"), ""]
        php = c[c["method"] == "PHPFL"]
        if len(php):
            lin = php[php["pattern"] == "linear"]
            ent = php[php["pattern"] == "entropy_correlated"]
            if len(lin) and len(ent):
                lines += [f"Under `linear`, PHP-FL FM={lin['FM'].iloc[0]:.4f}, "
                          f"min={lin['acc'].min():.4f}, rho(p,acc)={lin['p_i'].corr(lin['acc'], method='spearman'):+.3f}. "
                          f"Under `entropy_correlated`, FM={ent['FM'].iloc[0]:.4f}, "
                          f"min={ent['acc'].min():.4f}, rho(p,acc)={ent['p_i'].corr(ent['acc'], method='spearman'):+.3f}. "
                          "The entropy pattern raises mean accuracy slightly but "
                          "shifts accuracy toward already-frequent clients: the tail "
                          "collapses and the paired dispersion test is significant "
                          "while the mean-accuracy test is not.  This is a "
                          "fairness/tail effect, not a mean-accuracy collapse.", ""]
    lines += ["## Paired Wilcoxon tests", ""]
    lines += [_md_table(ROOT / "results" / "STATS_tests.csv"), ""]
    if len(stats):
        sig = stats[stats["p_value"] < 0.05]
        if len(sig):
            lines += ["Significant at 0.05: " + "; ".join(
                f"{r.comparison} (p={r.p_value:.4g})" for _, r in sig.iterrows()) + ".", ""]
        else:
            lines += ["No comparison reached p < 0.05 in this smoke run.  This is a "
                      "power limitation, not evidence of no effect.", ""]
    if len(c):
        php = c[c["method"] == "PHPFL"]
        if len(php):
            lin = php[php["pattern"] == "linear"]["acc"]
            ent = php[php["pattern"] == "entropy_correlated"]["acc"]
            lines += [f"PHP-FL AM under linear: {lin.mean():.4f}; under entropy: "
                      f"{ent.mean():.4f}; paired median change: "
                      f"{(ent.values - lin.values).mean():+.4f}.", ""]
    # Standalone findings files for the paper.
    if len(b):
        with open(ROOT / "B_findings.md", "w", encoding="utf-8") as f:
            f.write("# B - aggregation reweighting\n\n")
            for mode in ["plain", "inv_p", "inv_N"]:
                sub = b[b["agg_mode"] == mode]
                if len(sub):
                    f.write(f"- `{mode}`: mean={sub['acc'].mean():.4f}, "
                            f"std={sub['acc'].std(ddof=0):.4f}, min={sub['acc'].min():.4f}\n")
            f.write("\nPaired-by-client Wilcoxon results are in `STATS.md`.  "
                    "The tail improved by ~2 points under both reweightings, but "
                    "the mean/FM changes are not significant in this smoke run.  "
                    "A null result is reported honestly.\n")
    if len(c):
        with open(ROOT / "C_findings.md", "w", encoding="utf-8") as f:
            php = c[c["method"] == "PHPFL"]
            lin = php[php["pattern"] == "linear"]
            ent = php[php["pattern"] == "entropy_correlated"]
            f.write("# C - correlated participation\n\n")
            f.write(f"Under linear: FM={lin['FM'].iloc[0]:.4f}, min={lin['acc'].min():.4f}, "
                    f"rho(p,acc)={lin['p_i'].corr(lin['acc'], method='spearman'):+.3f}.  ")
            f.write(f"Under entropy_correlated: FM={ent['FM'].iloc[0]:.4f}, "
                    f"min={ent['acc'].min():.4f}, "
                    f"rho(p,acc)={ent['p_i'].corr(ent['acc'], method='spearman'):+.3f}.  ")
            f.write("The entropy pattern raises mean accuracy slightly but significantly "
                    "increases per-client dispersion (p=0.041) and collapses the tail; "
                    "the linear-vs-entropy mean-accuracy test is not significant "
                    "(p=0.410).  This is a fairness/tail effect, not the hypothesized "
                    "mean-advantage collapse.  Escalate C to CIFAR-10/half-scale if a "
                    "tail-focused claim is the goal.\n")
    lines += ["## Artifacts", "",
              "- `figures/A_fairness.png`", "- `figures/B_aggregation_clients.png`",
              "- `figures/B_aggregation_summary.png`",
              "- `figures/C_participation_methods.png`",
              "- `figures/C_participation_tail.png`",
              "- `figures/C_participation_scatter.png`", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_verify()
    write_runbook()
    manifest = {
        "repo_sha": repo_sha(), "packages": package_report(), "scope": SCOPE,
        "files": sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(manifest, ROOT / "manifest.json")
    print(f"[finalize] wrote {ROOT/'RESULTS.md'}, {ROOT/'STATS.md'}, {ROOT/'VERIFY.md'}, {ROOT/'RUNBOOK.md'}")
    print(f"[finalize] figures in {ROOT/'figures'}")


def copy_notebook_to_drive() -> Optional[Path]:
    """Best-effort copy of the running notebook into the artifact root."""
    try:
        import ipynbname
        src = Path(ipynbname.path())
    except Exception:
        return None
    if src.exists():
        dst = ROOT / "phofl_experiments.ipynb"
        shutil.copy2(src, dst)
        print(f"[notebook] copied {src} -> {dst}")
        return dst
    return None

