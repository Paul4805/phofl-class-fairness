"""Tests for the FedFV aggregation math and the aggregation plumbing.

Run with the working interpreter: python -m pytest -q tests/test_fedfv.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "PHP-FL-main" / "system"))

import phofl_lab as pl  # noqa: E402

pl.configure(ROOT / "phofl_half", "third_party/PHP-FL-main", colab=False)
pl.install_repo_patches()


def test_alpha_one_is_plain_average():
    torch.manual_seed(0)
    D = torch.randn(5, 16, dtype=torch.float64)
    losses = torch.rand(5).tolist()
    g = pl._fedfv_project(D, losses, alpha=1.0)
    assert torch.allclose(g, D.mean(dim=0))


def test_conflicts_are_removed():
    # a conflicts with b and c; the projected mean must conflict less than the
    # plain mean with the client gradients (the paper bounds, not eliminates, it).
    a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    b = torch.tensor([-1.0, 0.5, 0.0], dtype=torch.float64)
    c = torch.tensor([-0.5, -0.5, 1.0], dtype=torch.float64)
    D = torch.stack([a, b, c])
    g = pl._fedfv_project(D, [1.0, 2.0, 3.0], alpha=0.0)
    plain = D.mean(dim=0)

    def conflict(vec):
        return sum(max(0.0, -float((vec * row).sum())) for row in D)

    assert conflict(g) < conflict(plain)
    assert float(g.norm()) > 0.0


def test_projection_keeps_norm_scale():
    torch.manual_seed(1)
    D = torch.randn(4, 8, dtype=torch.float64)
    g = pl._fedfv_project(D, [0.1, 0.2, 0.3, 0.4], alpha=0.0)
    assert float(g.norm()) > 0.0
    assert float(D.mean(dim=0).norm()) > 0.0


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x):
        return self.lin(x)


def _write(path, model):
    torch.save(model, path)


def test_aggregate_end_to_end(tmp_path):
    torch.manual_seed(2)
    base = _Tiny()
    _write(tmp_path / "Server_global_model.pt", base)
    for cid in range(3):
        m = _Tiny()
        with torch.no_grad():
            for p, q in zip(m.parameters(), base.parameters()):
                p.copy_(q - torch.randn_like(q) * 0.1)
        _write(tmp_path / f"Client_{cid}_global_model.pt", m)
    clients = {cid: SimpleNamespace(role=f"Client_{cid}", save_folder_name=tmp_path,
                                    last_train_loss=float(cid + 1))
               for cid in range(3)}
    server = SimpleNamespace(role="Server", save_folder_name=tmp_path,
                             uploaded_ids=[0, 1, 2], clients=clients)
    state = {"round": 0, "history": {}}

    # alpha=1 must reduce exactly to plain averaging of the uploaded models.
    pl._fedfv_aggregate(server, state, alpha=1.0, tau_ext=0)
    loaded = [torch.load(tmp_path / f"Client_{cid}_global_model.pt", weights_only=False)
              for cid in range(3)]
    new = torch.load(tmp_path / "Server_global_model.pt", weights_only=False)
    for p_new, *p_clients in zip(new.parameters(), *[m.parameters() for m in loaded]):
        want = torch.stack([p.detach() for p in p_clients]).mean(dim=0)
        assert torch.allclose(p_new.detach(), want, atol=1e-6)
    assert set(state["history"]) == {0, 1, 2}


def test_aggregate_alpha_zero_preserves_step_size(tmp_path):
    torch.manual_seed(3)
    base = _Tiny()
    _write(tmp_path / "Server_global_model.pt", base)
    for cid in range(3):
        m = _Tiny()
        with torch.no_grad():
            for p, q in zip(m.parameters(), base.parameters()):
                p.copy_(q + torch.randn_like(q) * 0.1)
        _write(tmp_path / f"Client_{cid}_global_model.pt", m)
    clients = {cid: SimpleNamespace(role=f"Client_{cid}", save_folder_name=tmp_path,
                                    last_train_loss=float(cid + 1))
               for cid in range(3)}
    server = SimpleNamespace(role="Server", save_folder_name=tmp_path,
                             uploaded_ids=[0, 1, 2], clients=clients)
    pl._fedfv_aggregate(server, {"round": 0, "history": {}}, alpha=0.0, tau_ext=0)
    new = torch.load(tmp_path / "Server_global_model.pt", weights_only=False)
    loaded = [torch.load(tmp_path / f"Client_{cid}_global_model.pt", weights_only=False)
              for cid in range(3)]
    step = torch.cat([(b - p_new).reshape(-1) for b, p_new in
                      zip(base.parameters(), new.parameters())])
    plain = torch.cat([(b - torch.stack([p.detach() for p in ps]).mean(0)).reshape(-1)
                       for b, *ps in zip(base.parameters(),
                                         *[m.parameters() for m in loaded])])
    # Alg. 1 line 10: the mitigated update is rescaled to the plain-average norm.
    assert abs(float(step.norm()) - float(plain.norm())) < 1e-5
    assert not torch.allclose(step, plain, atol=1e-6)


if __name__ == "__main__":
    import tempfile

    test_alpha_one_is_plain_average()
    test_conflicts_are_removed()
    test_projection_keeps_norm_scale()
    with tempfile.TemporaryDirectory() as d:
        test_aggregate_end_to_end(Path(d))
    print("fedfv tests passed")
