"""Regression tests for the helper-generated notebook export.

These tests cover the current notebook-backed workflow without editing the
notebook itself.
They focus on regression protection for exported notebook behavior and helper
integration, not on replacing the notebook as the canonical analysis record.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "blankTemplate.ipynb"
HELPER_SCRIPT = REPO_ROOT / "scripts" / "extract_notebook_code.py"

EXPECTED_CLEAN_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
    "LUNG_CANCER",
]

EXPECTED_CORRELATION_COLUMNS = [
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]

BINARY_COLUMNS = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]


@pytest.fixture(scope="session")
def exported_notebook_code(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("notebook_export")
    output_path = output_dir / "notebook_code.py"

    subprocess.run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "--notebook",
            str(NOTEBOOK_PATH),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return output_path


@pytest.fixture(scope="session")
def notebook_module(exported_notebook_code: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pyplot = importlib.import_module("matplotlib.pyplot")
    pytest.importorskip("seaborn")
    pytest.importorskip("sklearn")

    spec = importlib.util.spec_from_file_location(
        "generated_notebook_code",
        exported_notebook_code,
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    original_cwd = Path.cwd()
    original_show = pyplot.show

    try:
        os.chdir(REPO_ROOT)
        pyplot.show = lambda *args, **kwargs: None
        spec.loader.exec_module(module)
    finally:
        pyplot.show = original_show
        os.chdir(original_cwd)

    return module


def test_helper_exports_notebook_code(exported_notebook_code: Path) -> None:
    exported_text = exported_notebook_code.read_text(encoding="utf-8")

    assert 'Source notebook: blankTemplate.ipynb' in exported_text
    assert "# %% Notebook cell" in exported_text
    assert 'dataset = pd.read_csv("datasets/givenData.csv", thousands=",")' in exported_text


def test_notebook_loads_expected_raw_dataset_shape(notebook_module) -> None:
    assert notebook_module.dataset.shape == (309, 16)
    assert int(notebook_module.dataset.duplicated().sum()) == 33


def test_current_dedup_step_matches_notebook_tail_drop_logic(notebook_module) -> None:
    expected_frame = notebook_module.dataset.iloc[:284].copy()
    expected_frame.columns = expected_frame.columns.str.strip()

    assert notebook_module.dataset_dedup.shape == (284, 16)
    assert notebook_module.dataset_dedup.reset_index(drop=True).equals(
        expected_frame.reset_index(drop=True)
    )
    assert int(notebook_module.dataset_dedup.duplicated().sum()) == 8


def test_clean_dataset_columns_and_binary_recoding(notebook_module) -> None:
    dataset_clean = notebook_module.dataset_clean

    assert dataset_clean.shape == (284, 16)
    assert dataset_clean.columns.tolist() == EXPECTED_CLEAN_COLUMNS

    for column_name in [
        "CHRONIC DISEASE",
        "ALCOHOL CONSUMING",
        "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY",
        "CHEST PAIN",
    ]:
        assert column_name not in dataset_clean.columns

    for column_name in BINARY_COLUMNS + ["GENDER", "LUNG_CANCER"]:
        assert set(dataset_clean[column_name].dropna().unique()) <= {0, 1}


def test_clean_dataset_class_and_gender_counts(notebook_module) -> None:
    dataset_clean = notebook_module.dataset_clean

    assert dataset_clean["LUNG_CANCER"].value_counts().sort_index().to_dict() == {
        0: 38,
        1: 246,
    }
    assert dataset_clean["GENDER"].value_counts().sort_index().to_dict() == {
        0: 148,
        1: 136,
    }


def test_correlation_matrix_tracks_current_numeric_columns(notebook_module) -> None:
    corr_matrix = notebook_module.corr_matrix

    assert corr_matrix.shape == (14, 14)
    assert corr_matrix.columns.tolist() == EXPECTED_CORRELATION_COLUMNS
    assert corr_matrix.index.tolist() == EXPECTED_CORRELATION_COLUMNS
