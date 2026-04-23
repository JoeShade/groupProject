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
    "BATCH_1",
    "BATCH_2",
    "BATCH_3",
    "AGE_BINS",
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

EXPECTED_MODEL_ORDER = [
    "RandomForest",
    "GradientBoosting",
    "LogisticRegression",
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
    pytest.importorskip("scipy")
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


def test_row_trimming_step_excludes_only_final_raw_block(notebook_module) -> None:
    expected_frame = notebook_module.dataset.iloc[:-30].reset_index(drop=True).copy()
    expected_frame.columns = expected_frame.columns.str.strip()

    assert notebook_module.dataset_dedup.shape == (279, 16)
    assert notebook_module.dataset_dedup.equals(expected_frame)
    assert int(notebook_module.dataset_dedup.duplicated().sum()) == 8


def test_clean_dataset_columns_and_binary_recoding(notebook_module) -> None:
    dataset_clean = notebook_module.dataset_clean

    assert dataset_clean.shape == (279, 20)
    assert dataset_clean.columns.tolist() == EXPECTED_CLEAN_COLUMNS

    for column_name in [
        "CHRONIC DISEASE",
        "ALCOHOL CONSUMING",
        "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY",
        "CHEST PAIN",
        "LIFESTYLE_RISK",
        "HEAVY_SMOKER",
        "RESPIRATORY_DISTRESS",
    ]:
        assert column_name not in dataset_clean.columns

    for column_name in BINARY_COLUMNS + ["GENDER", "LUNG_CANCER"]:
        assert set(dataset_clean[column_name].dropna().unique()) <= {0, 1}

    assert set(dataset_clean["BATCH_1"].dropna().unique()) <= {0, 1, 2, 3, 4}
    assert set(dataset_clean["BATCH_2"].dropna().unique()) <= {0, 1, 2, 3}
    assert set(dataset_clean["BATCH_3"].dropna().unique()) <= {0, 1, 2, 3}

    assert set(dataset_clean["AGE_BINS"].dropna().astype(str).unique()) <= {
        "0",
        "1",
        "2",
        "3",
        "4",
    }


def test_clean_dataset_class_and_gender_counts(notebook_module) -> None:
    dataset_clean = notebook_module.dataset_clean

    assert dataset_clean["LUNG_CANCER"].value_counts().sort_index().to_dict() == {
        0: 35,
        1: 244,
    }
    assert dataset_clean["GENDER"].value_counts().sort_index().to_dict() == {
        0: 145,
        1: 134,
    }


def test_preprocess_normalises_age_to_zero_one(notebook_module) -> None:
    assert notebook_module.age_features == ["AGE"]
    assert "AGE" not in notebook_module.other_numeric_features
    assert notebook_module.categorical_features == ["AGE_BINS"]

    age_transformer = notebook_module.preprocess.transformers[0][1]
    numeric_transformer = notebook_module.preprocess.transformers[1][1]

    assert notebook_module.preprocess.transformers[0][0] == "age"
    assert notebook_module.preprocess.transformers[1][0] == "numeric"
    assert age_transformer.named_steps["scaler"].__class__.__name__ == "MinMaxScaler"
    assert numeric_transformer.named_steps["scaler"].__class__.__name__ == "StandardScaler"


def test_correlation_matrix_tracks_current_numeric_columns(notebook_module) -> None:
    corr_matrix = notebook_module.corr_matrix

    assert corr_matrix.shape == (14, 14)
    assert corr_matrix.columns.tolist() == EXPECTED_CORRELATION_COLUMNS
    assert corr_matrix.index.tolist() == EXPECTED_CORRELATION_COLUMNS


def test_model_comparison_prefers_random_forest(notebook_module) -> None:
    model_comparison_df = notebook_module.model_comparison_df

    assert model_comparison_df["Model"].tolist() == EXPECTED_MODEL_ORDER
    assert notebook_module.final_model_name == "RandomForest"

    top_row = model_comparison_df.iloc[0].to_dict()
    assert top_row["Mean CV Accuracy"] == pytest.approx(0.9174, abs=1e-4)
    assert top_row["Mean CV Precision"] == pytest.approx(0.9624, abs=1e-4)
    assert top_row["Mean CV Recall"] == pytest.approx(0.9425, abs=1e-4)
    assert top_row["Mean CV F1"] == pytest.approx(0.9523, abs=1e-4)
    assert top_row["Mean CV ROC AUC"] == pytest.approx(0.9213, abs=1e-4)


def test_final_model_holdout_metrics_and_threshold_summary(notebook_module) -> None:
    metrics_row = notebook_module.final_metrics_df.iloc[0].to_dict()

    assert metrics_row["Accuracy"] == pytest.approx(0.8750, abs=1e-4)
    assert metrics_row["Precision"] == pytest.approx(0.9773, abs=1e-4)
    assert metrics_row["Recall"] == pytest.approx(0.8776, abs=1e-4)
    assert metrics_row["F1"] == pytest.approx(0.9247, abs=1e-4)
    assert metrics_row["ROC AUC"] == pytest.approx(0.9184, abs=1e-4)

    assert notebook_module.conf_matrix.tolist() == [[6, 1], [6, 43]]

    threshold_row = notebook_module.high_recall_threshold_df.iloc[0].to_dict()
    assert threshold_row["Threshold"] == pytest.approx(0.2321, abs=1e-4)
    assert threshold_row["Precision"] == pytest.approx(0.9216, abs=1e-4)
    assert threshold_row["Recall"] == pytest.approx(0.9592, abs=1e-4)
    assert threshold_row["F1"] == pytest.approx(0.9400, abs=1e-4)


def test_feature_importance_and_subgroup_error_tables(notebook_module) -> None:
    feature_importance_df = notebook_module.feature_importance_df
    subgroup_error_df = notebook_module.subgroup_error_df

    assert feature_importance_df.head(3)["Feature"].tolist() == [
        "ALCOHOL_CONSUMING",
        "BATCH_1",
        "GENDER",
    ]

    assert subgroup_error_df.to_dict(orient="records") == [
        {
            "Gender": "Male",
            "Sample Size": 31,
            "False Positives": 0,
            "False Negatives": 4,
            "Recall": 0.8519,
        },
        {
            "Gender": "Female",
            "Sample Size": 25,
            "False Positives": 1,
            "False Negatives": 2,
            "Recall": 0.9091,
        },
    ]
