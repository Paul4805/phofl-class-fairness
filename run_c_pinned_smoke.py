"""Pinned-stack repeat of only the C smoke experiment (no A/B)."""
import phofl_lab as pl

pl.configure("phofl_pinned", "third_party/PHP-FL-main", colab=False)
pl.prepare_repo(pl.REPO_DIR)
pl.ensure_fashion_mnist(pl.SCOPE["clients"], pl.REPO_DIR)
print("[pinned] running smoke C only")
c = pl.run_C_participation()
print(c.groupby(["pattern", "method"])[["acc"]].agg(["mean", "std", "min"]).to_string())
print("[pinned] stats")
pl.write_stats()
print("[pinned] done")
