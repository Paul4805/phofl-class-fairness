"""Lightweight reproducibility checks for the public repository."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_valid_json():
    notebook = json.loads((ROOT / "phofl_experiments.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])


def test_ladder_tables_exist_and_have_expected_schema():
    expected = ["class", "recall_baseline", "recall_method", "delta"]
    for step in range(1, 5):
        path = ROOT / "phofl_method" / f"STEP{step}_per_class.csv"
        assert path.exists(), path
        df = pd.read_csv(path)
        assert list(df.columns) == expected
        assert len(df) == 10
        assert df["class"].is_unique


def test_two_axis_summary_is_five_runs():
    path = ROOT / "phofl/analysis/two_axis_run_summary.csv"
    df = pd.read_csv(path)
    assert len(df) == 5
    for col in ["AM", "AM_std", "class_std", "class_gap", "gap_over_am",
                "gap_over_ceiling", "supported_class_gap", "gap_over_am_supported"]:
        assert col in df.columns
    assert set(df["method"]) == {"PHPFL", "Standalone", "FedAPEN"}


def test_two_axis_support_filtering_is_recorded():
    path = ROOT / "phofl/analysis/two_axis_client_class.csv"
    df = pd.read_csv(path)
    for col in ["recall", "n_samples", "supported_ge_10"]:
        assert col in df.columns
    assert df["n_samples"].isna().sum() == 0
    assert int((~df["supported_ge_10"]).sum()) == 170


def test_ladder_summary_negative_result():
    text = (ROOT / "phofl_method/LADDER_SUMMARY.md").read_text(encoding="utf-8")
    assert "7.9%" in text
    assert "No method contribution is claimed" in text
    assert "0.657" in text and "0.683" in text


def test_paper_sections_exist():
    required = [
        "00_abstract.md", "01_intro.md", "02_related_work.md", "03_method.md",
        "04_results.md", "04_fedapen.md", "05_ladder.md", "06_discussion.md",
        "07_conclusion.md",
    ]
    for name in required:
        assert (ROOT / "phofl/paper" / name).exists(), name


def test_public_metadata_exists():
    for name in ["LICENSE", "CITATION.cff", "README.md", "CONTRIBUTING.md",
                 "CODE_OF_CONDUCT.md", "SECURITY.md", "DATA.md"]:
        assert (ROOT / name).exists(), name


def test_generated_artifacts_are_not_tracked():
    import subprocess
    try:
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    except Exception:
        import pytest
        pytest.skip("git metadata unavailable")
    forbidden_prefixes = [
        "third_party/",
        "phofl_half/data/",
        "phofl_half/data_cap/",
        "phofl_half/run_store/",
        "phofl_half/ckpts/",
        "phofl/ckpts/",
        "phofl_pinned/",
        ".venv-pinned/",
    ]
    for prefix in forbidden_prefixes:
        assert not any(path.startswith(prefix) for path in tracked), prefix
