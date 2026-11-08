import json
from pathlib import Path

root = Path(__file__).parent
helper = (root / "phofl_lab.py").read_text(encoding="utf-8")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text})


md(r"""# PHP-FL scoped experiments

**Scope after the change request:** A (fairness-metric recomputation), B
(aggregation reweighting: plain / inv_p / inv_N, PHP-FL only), and C
(correlated participation: linear / entropy_correlated, PHP-FL vs Standalone).

Smoke scale: **10 clients, 40 rounds, Fashion-MNIST, 1 seed**.  E4-E8 are not
run.  All tests are paired Wilcoxon signed-rank across clients, not seeds.

**Resume:** re-run Cells 0-2, then the experiment cell.  Every run checks
`results/{exp}/{run_id}_complete.flag` and the latest checkpoint.
""")

code(r"""# Cell 0 - install pinned dependencies and check the runtime.
import os, sys, warnings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning)

# Colab / CUDA 12.1 pins.  If torch is already imported by the runtime, restart
# the runtime after this cell so the newly installed wheel is the one loaded.
%pip install -q torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
%pip install -q numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.0 matplotlib==3.8.4 seaborn==0.13.2 Pillow==10.3.0 tqdm==4.66.4 PyYAML==6.0.1 tabulate==0.9.0 psutil==5.9.8 nvidia-ml-py==12.535.133 ipynbname==2023.8.8.0 efficientnet-pytorch==0.7.1 h5py==3.10.0 ujson==5.10.0 wandb==0.19.6

import torch, torchvision, numpy, pandas, scipy, sklearn, matplotlib, seaborn
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__, "| cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("numpy      ", numpy.__version__)
print("pandas     ", pandas.__version__)
if torch.cuda.is_available():
    print("gpu        ", torch.cuda.get_device_name(0),
          f"({torch.cuda.get_device_properties(0).total_memory/2**20:.0f} MB)")
else:
    print("gpu        ", "CPU only")
if not torch.__version__.startswith("2.2.2"):
    print("\nWARNING: torch is not 2.2.2.  Runtime > Restart and run all before "
          "trusting the pinned environment.\n")
""")

code(r"""# Cell 1 - Drive mount, artifact paths, seeds.
import os, sys, json, random, shutil
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount failed:", e)
    ROOT = Path("/content/drive/MyDrive/phofl")
    REPO_DIR = Path("/content/PHP-FL-main")
else:
    ROOT = Path.cwd() / "phofl"
    REPO_DIR = Path.cwd() / "third_party" / "PHP-FL-main"

for sub in ["ckpts", "logs", "results", "figures", "flags", "baseline", "run_store"]:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

SEEDS = [0]  # smoke scope: one seed; Wilcoxon pairs clients, not seeds
SCALE = "smoke"
print("ROOT     ", ROOT)
print("REPO_DIR ", REPO_DIR)
print("colab    ", IN_COLAB)
""")

code("%%writefile phofl_lab.py\n" + helper)

code(r"""# Cell 2b - import the harness, clone/pin the repo, install runtime patches.
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import phofl_lab as pl
importlib.reload(pl)

pl.configure(ROOT, REPO_DIR, colab=IN_COLAB)
pl.prepare_repo(pl.REPO_DIR)
pl.install_repo_patches()
pl.ensure_fashion_mnist(pl.SCOPE["clients"], pl.REPO_DIR)

print("\nRuntime patches are applied in memory; the repository source is clean.")
print(subprocess_output := __import__("subprocess").check_output(
    ["git", "-C", str(pl.REPO_DIR), "status", "--short"], text=True) or "(clean)")
print("\nPackages:")
for k, v in pl.package_report().items():
    print(f"  {k:14s} {v}")
print("\nSCOPE:", pl.SCOPE)
""")

code(r"""# Cell 3 - smoke test (10 clients, 1 round, 6 GB safe).
result = pl.smoke_test()
print("smoke status:", result["status"])
print("clients:", len(result["clients"]), "| checkpoint:", result["checkpoint"])
""")

md(r"""## A. Fairness-metric recomputation

If a completed `B_agg / PHPFL / normal / plain` run exists, this cell is
0 compute and only reads its `round_metrics.csv`.  If not, it creates that one
baseline run (10 clients / 40 rounds / Fashion-MNIST) so A has data.
""")

code(r"""# Cell 4 - A: fairness metrics from the plain baseline.
from IPython.display import display
plain = pl._result_path("B_agg", "PHPFL", "normal", pl.SCOPE["seed"], agg_mode="plain")
if plain is None:
    print("No plain baseline yet; creating it once (40 rounds).")
    pl.run_B_aggregation(modes=("plain",))
A = pl.run_A_fairness()
display(A)
""")

md(r"""## B. Aggregation reweighting (PHP-FL only)

`plain` is the repository's untouched uniform mean over active clients.
`inv_p` uses `1/p_i`; `inv_N` uses `1/max(N_i(t),1)`.  All weights are
normalized.  Same 10 clients / 40 rounds / seed 0.
""")

code(r"""# Cell 5 - B: plain, inv_p, inv_N.
B = pl.run_B_aggregation()
display(B.pivot_table(index="client_id", columns="agg_mode", values="acc"))
display(B.groupby("agg_mode")[["acc"]].agg(["mean", "std", "min"]))
""")

md(r"""## C. Correlated participation (PHP-FL vs Standalone)

`linear` is the paper's shuffled arithmetic pattern.  `entropy_correlated`
uses `p_i = clip(exp(H_i - ln K), 0.02, 1.0)`: balanced-label clients
participate more, rare-class-heavy clients less.  Standalone is server-free and
therefore pattern-invariant; it is run once and reused for both pattern rows.
""")

code(r"""# Cell 6 - C: linear / entropy_correlated x PHP-FL / Standalone.
C = pl.run_C_participation()
display(C.pivot_table(index="client_id", columns=["pattern", "method"], values="acc"))
display(C.groupby(["pattern", "method"])[["acc"]].agg(["mean", "std", "min"]))
""")

code(r"""# Cell 7 - paired Wilcoxon tests + RESULTS.md + STATS.md + figures.
S = pl.write_stats()
display(S)
pl.finalize()
pl.copy_notebook_to_drive()
print("\nSTATS.md:\n")
print((pl.ROOT / "STATS.md").read_text(encoding="utf-8"))
print("\nRESULTS.md:\n")
print((pl.ROOT / "RESULTS.md").read_text(encoding="utf-8")[:6000])
""")

md(r"""## Escalation rule

If C shows a significant paired effect, escalate **only C** to CIFAR-10,
half-scale: 20 clients, 60 rounds, 2 seeds, `model_family="HtM4"` (the paper's
four-CNN round-robin), and 20-client CIFAR-10 Dirichlet data.  B stays at the
smoke scale unless a reviewer asks for the aggregation sweep at scale.

`STATS.md` reports the Wilcoxon statistic and two-sided p-value for:
B: plain vs inv_p, plain vs inv_N, inv_p vs inv_N;
C: PHP-FL vs Standalone within each pattern, and linear vs entropy within
PHP-FL, paired by client_id.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = root / "phofl_experiments.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/2**10:.1f} KiB, {len(cells)} cells)")
