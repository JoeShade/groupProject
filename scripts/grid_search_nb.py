"""GPU-accelerated GaussianNB grid search aligned with the project notebook."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from cuda_runtime import configure_cuda_runtime

configure_cuda_runtime()

import cupy as cp  # noqa: E402


GRID_TEST_SIZE = 0.2
GRID_RANDOM_SEEDS = list(range(100))
VAR_SMOOTHING_GRID = np.logspace(-14, 2, 4001, dtype=np.float32)
MANUAL_POSITIVE_PRIOR_GRID = np.round(
    np.linspace(0.50, 0.98, 241, dtype=np.float32),
    3,
)
VAR_SMOOTHING_CHUNK_SIZE = 256
PRIOR_CHUNK_SIZE = 128
RANKED_RESULTS_PATH = Path("artifacts/gaussian_nb_grid_search_ranked.csv")
BEST_RESULTS_PATH = Path("artifacts/gaussian_nb_grid_search_best.csv")


def calculate_f2_score(precision_values: np.ndarray, recall_values: np.ndarray) -> np.ndarray:
    numerator = 5.0 * precision_values * recall_values
    denominator = (4.0 * precision_values) + recall_values
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )


def calculate_f2_weighted_composite_quality_score(metrics_df: pd.DataFrame) -> pd.Series:
    return (
        metrics_df["Accuracy"]
        + metrics_df["Precision"]
        + metrics_df["Recall"]
        + metrics_df["ROC-AUC"]
        + (2 * metrics_df["F2-score"])
    ) / 6.0


def load_clean_dataset(repo_root: Path) -> pd.DataFrame:
    dataset = pd.read_csv(repo_root / "datasets" / "givenData.csv", thousands=",")
    dataset.columns = dataset.columns.str.strip()

    dataset_dedup = dataset.drop_duplicates().reset_index(drop=True).copy()
    dataset_clean = dataset_dedup.copy()
    dataset_clean.columns = dataset_clean.columns.str.strip()
    dataset_clean.rename(
        columns={
            "CHRONIC DISEASE": "CHRONIC_DISEASE",
            "ALCOHOL CONSUMING": "ALCOHOL_CONSUMING",
            "SHORTNESS OF BREATH": "SHORTNESS_OF_BREATH",
            "SWALLOWING DIFFICULTY": "SWALLOWING_DIFFICULTY",
            "CHEST PAIN": "CHEST_PAIN",
        },
        inplace=True,
    )

    binary_yes_no_columns = [
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
    dataset_clean[binary_yes_no_columns] = dataset_clean[binary_yes_no_columns] - 1
    dataset_clean["GENDER"] = dataset_clean["GENDER"].map({"M": 0, "F": 1})
    dataset_clean["LUNG_CANCER"] = dataset_clean["LUNG_CANCER"].map({"NO": 0, "YES": 1})

    return dataset_clean


def evaluate_gpu_grid_search(
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    feature_names = [name for name in feature_matrix.dtype.names] if feature_matrix.dtype.names else None
    if feature_names is not None:
        raise RuntimeError("The feature matrix must be a dense numeric array.")

    num_var_values = len(VAR_SMOOTHING_GRID)
    num_prior_values = 1 + len(MANUAL_POSITIVE_PRIOR_GRID)
    total_configurations = num_var_values * num_prior_values

    manual_priors = np.column_stack(
        [
            1.0 - MANUAL_POSITIVE_PRIOR_GRID,
            MANUAL_POSITIVE_PRIOR_GRID,
        ]
    ).astype(np.float32)
    all_priors = np.vstack(
        [
            np.array([[np.nan, np.nan]], dtype=np.float32),
            manual_priors,
        ]
    )

    accuracy_sums = np.zeros((num_var_values, num_prior_values), dtype=np.float64)
    precision_sums = np.zeros((num_var_values, num_prior_values), dtype=np.float64)
    recall_sums = np.zeros((num_var_values, num_prior_values), dtype=np.float64)
    f2_sums = np.zeros((num_var_values, num_prior_values), dtype=np.float64)
    roc_auc_sums = np.zeros((num_var_values, num_prior_values), dtype=np.float64)

    search_start_time = time.perf_counter()

    for random_seed in GRID_RANDOM_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            target_vector,
            test_size=GRID_TEST_SIZE,
            random_state=random_seed,
            stratify=target_vector,
        )

        X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
        X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
        y_test_gpu = cp.asarray(y_test, dtype=cp.int32)

        class_zero_mask = y_train_gpu == 0
        class_one_mask = y_train_gpu == 1
        X_train_zero = X_train_gpu[class_zero_mask]
        X_train_one = X_train_gpu[class_one_mask]

        theta = cp.stack(
            [
                X_train_zero.mean(axis=0),
                X_train_one.mean(axis=0),
            ],
            axis=0,
        )
        variance = cp.stack(
            [
                X_train_zero.var(axis=0),
                X_train_one.var(axis=0),
            ],
            axis=0,
        )

        empirical_priors = np.array(
            [
                float((y_train == 0).mean()),
                float((y_train == 1).mean()),
            ],
            dtype=np.float32,
        )
        resolved_priors = all_priors.copy()
        resolved_priors[0] = empirical_priors

        base_variance_max = X_train_gpu.var(axis=0).max()
        expanded_test = X_test_gpu[None, None, :, :]
        expanded_theta = theta[None, :, None, :]
        y_test_bool = y_test_gpu.astype(cp.bool_)
        positive_mask = y_test_bool == 1
        negative_mask = ~positive_mask

        for var_start in range(0, num_var_values, VAR_SMOOTHING_CHUNK_SIZE):
            var_end = min(var_start + VAR_SMOOTHING_CHUNK_SIZE, num_var_values)
            var_chunk = VAR_SMOOTHING_GRID[var_start:var_end]
            epsilon_values = cp.asarray(var_chunk) * base_variance_max
            smoothed_variance = variance[None, :, :] + epsilon_values[:, None, None]
            expanded_variance = smoothed_variance[:, :, None, :]

            log_likelihood = -0.5 * (
                cp.log(2.0 * cp.pi * expanded_variance)
                + ((expanded_test - expanded_theta) ** 2) / expanded_variance
            ).sum(axis=-1)

            for prior_start in range(0, num_prior_values, PRIOR_CHUNK_SIZE):
                prior_end = min(prior_start + PRIOR_CHUNK_SIZE, num_prior_values)
                prior_chunk = cp.asarray(
                    resolved_priors[prior_start:prior_end],
                    dtype=cp.float32,
                )
                log_priors = cp.log(prior_chunk)
                joint_log_probabilities = (
                    log_likelihood[:, None, :, :] + log_priors[None, :, :, None]
                )

                negative_joint = joint_log_probabilities[:, :, 0, :]
                positive_joint = joint_log_probabilities[:, :, 1, :]
                joint_maximum = cp.maximum(negative_joint, positive_joint)
                positive_probability = cp.exp(positive_joint - joint_maximum) / (
                    cp.exp(negative_joint - joint_maximum)
                    + cp.exp(positive_joint - joint_maximum)
                )

                flat_positive_probability = positive_probability.reshape(
                    -1,
                    X_test.shape[0],
                )
                flat_predictions = flat_positive_probability >= 0.5

                predicted_positive = flat_predictions
                predicted_negative = ~predicted_positive
                expanded_y_test = y_test_bool[None, :]

                true_positive = cp.logical_and(
                    predicted_positive,
                    expanded_y_test,
                ).sum(axis=1).astype(cp.float32)
                false_positive = cp.logical_and(
                    predicted_positive,
                    ~expanded_y_test,
                ).sum(axis=1).astype(cp.float32)
                true_negative = cp.logical_and(
                    predicted_negative,
                    ~expanded_y_test,
                ).sum(axis=1).astype(cp.float32)
                false_negative = cp.logical_and(
                    predicted_negative,
                    expanded_y_test,
                ).sum(axis=1).astype(cp.float32)

                accuracy = (true_positive + true_negative) / y_test.shape[0]
                precision_denominator = true_positive + false_positive
                recall_denominator = true_positive + false_negative
                precision = cp.where(
                    precision_denominator != 0,
                    true_positive / precision_denominator,
                    cp.zeros_like(true_positive),
                )
                recall = cp.where(
                    recall_denominator != 0,
                    true_positive / recall_denominator,
                    cp.zeros_like(true_positive),
                )
                f2_denominator = (4.0 * precision) + recall
                f2_score = cp.where(
                    f2_denominator != 0,
                    (5.0 * precision * recall) / f2_denominator,
                    cp.zeros_like(precision),
                )

                positive_scores = flat_positive_probability[:, positive_mask]
                negative_scores = flat_positive_probability[:, negative_mask]
                roc_auc = (
                    (positive_scores[:, :, None] > negative_scores[:, None, :]).sum(axis=(1, 2))
                    + 0.5
                    * (positive_scores[:, :, None] == negative_scores[:, None, :]).sum(axis=(1, 2))
                ) / (positive_scores.shape[1] * negative_scores.shape[1])
                roc_auc = roc_auc.astype(cp.float32)

                reshaped_accuracy = cp.asnumpy(accuracy).reshape(var_end - var_start, prior_end - prior_start)
                reshaped_precision = cp.asnumpy(precision).reshape(var_end - var_start, prior_end - prior_start)
                reshaped_recall = cp.asnumpy(recall).reshape(var_end - var_start, prior_end - prior_start)
                reshaped_f2 = cp.asnumpy(f2_score).reshape(var_end - var_start, prior_end - prior_start)
                reshaped_roc_auc = cp.asnumpy(roc_auc).reshape(var_end - var_start, prior_end - prior_start)

                accuracy_sums[var_start:var_end, prior_start:prior_end] += reshaped_accuracy
                precision_sums[var_start:var_end, prior_start:prior_end] += reshaped_precision
                recall_sums[var_start:var_end, prior_start:prior_end] += reshaped_recall
                f2_sums[var_start:var_end, prior_start:prior_end] += reshaped_f2
                roc_auc_sums[var_start:var_end, prior_start:prior_end] += reshaped_roc_auc

    cp.cuda.Stream.null.synchronize()
    search_duration_seconds = time.perf_counter() - search_start_time

    averaged_metrics_df = pd.DataFrame(
        {
            "var_smoothing": np.repeat(VAR_SMOOTHING_GRID, num_prior_values),
            "prior_mode": np.tile(
                np.array(
                    ["empirical"] + ["manual"] * len(MANUAL_POSITIVE_PRIOR_GRID),
                    dtype=object,
                ),
                num_var_values,
            ),
            "prior_negative": np.tile(
                np.concatenate(([np.nan], 1.0 - MANUAL_POSITIVE_PRIOR_GRID)),
                num_var_values,
            ),
            "prior_positive": np.tile(
                np.concatenate(([np.nan], MANUAL_POSITIVE_PRIOR_GRID)),
                num_var_values,
            ),
            "Accuracy": accuracy_sums.reshape(-1) / len(GRID_RANDOM_SEEDS),
            "Precision": precision_sums.reshape(-1) / len(GRID_RANDOM_SEEDS),
            "Recall": recall_sums.reshape(-1) / len(GRID_RANDOM_SEEDS),
            "F2-score": f2_sums.reshape(-1) / len(GRID_RANDOM_SEEDS),
            "ROC-AUC": roc_auc_sums.reshape(-1) / len(GRID_RANDOM_SEEDS),
        }
    )
    averaged_metrics_df["F2-CQS"] = calculate_f2_weighted_composite_quality_score(
        averaged_metrics_df
    )

    averaged_metrics_df.sort_values(
        by=["F2-CQS", "F2-score", "ROC-AUC", "Recall", "Precision", "Accuracy"],
        ascending=[False, False, False, False, False, False],
        inplace=True,
        ignore_index=True,
    )
    averaged_metrics_df.insert(0, "rank", np.arange(1, len(averaged_metrics_df) + 1))

    return averaged_metrics_df, search_duration_seconds


def verify_best_configuration(
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
    best_configuration: pd.Series,
) -> pd.DataFrame:
    verified_run_metrics = []
    best_priors = None
    if best_configuration["prior_mode"] == "manual":
        best_priors = [
            float(best_configuration["prior_negative"]),
            float(best_configuration["prior_positive"]),
        ]

    for random_seed in GRID_RANDOM_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            target_vector,
            test_size=GRID_TEST_SIZE,
            random_state=random_seed,
            stratify=target_vector,
        )

        model = GaussianNB(
            priors=best_priors,
            var_smoothing=float(best_configuration["var_smoothing"]),
        )

        fit_start_time = time.perf_counter()
        model.fit(X_train, y_train)
        fit_duration_seconds = time.perf_counter() - fit_start_time

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        verified_run_metrics.append(
            {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision,
                "Recall": recall,
                "F2-score": calculate_f2_score(
                    np.array([precision], dtype=np.float32),
                    np.array([recall], dtype=np.float32),
                )[0],
                "ROC-AUC": roc_auc_score(y_test, probabilities),
                "Training time (seconds)": fit_duration_seconds,
            }
        )

    verified_metrics_df = pd.DataFrame(verified_run_metrics)
    return pd.DataFrame(
        {
            "Accuracy": [verified_metrics_df["Accuracy"].mean()],
            "Precision": [verified_metrics_df["Precision"].mean()],
            "Recall": [verified_metrics_df["Recall"].mean()],
            "F2-score": [verified_metrics_df["F2-score"].mean()],
            "ROC-AUC": [verified_metrics_df["ROC-AUC"].mean()],
            "Training time (seconds)": [verified_metrics_df["Training time (seconds)"].mean()],
        }
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    dataset_clean = load_clean_dataset(repo_root)
    feature_columns = [column for column in dataset_clean.columns if column != "LUNG_CANCER"]
    feature_matrix = dataset_clean[feature_columns].to_numpy(dtype=np.float32)
    target_vector = dataset_clean["LUNG_CANCER"].to_numpy(dtype=np.int32)

    ranked_results_df, search_duration_seconds = evaluate_gpu_grid_search(
        feature_matrix=feature_matrix,
        target_vector=target_vector,
    )

    best_configuration = ranked_results_df.iloc[0]
    verified_best_metrics_df = verify_best_configuration(
        feature_matrix=feature_matrix,
        target_vector=target_vector,
        best_configuration=best_configuration,
    )

    best_result_df = pd.DataFrame(
        {
            "var_smoothing": [best_configuration["var_smoothing"]],
            "prior_mode": [best_configuration["prior_mode"]],
            "prior_negative": [best_configuration["prior_negative"]],
            "prior_positive": [best_configuration["prior_positive"]],
            "Accuracy": [verified_best_metrics_df.at[0, "Accuracy"]],
            "Precision": [verified_best_metrics_df.at[0, "Precision"]],
            "Recall": [verified_best_metrics_df.at[0, "Recall"]],
            "F2-score": [verified_best_metrics_df.at[0, "F2-score"]],
            "ROC-AUC": [verified_best_metrics_df.at[0, "ROC-AUC"]],
            "F2-CQS": [
                calculate_f2_weighted_composite_quality_score(
                    pd.DataFrame(
                        {
                            "Accuracy": [verified_best_metrics_df.at[0, "Accuracy"]],
                            "Precision": [verified_best_metrics_df.at[0, "Precision"]],
                            "Recall": [verified_best_metrics_df.at[0, "Recall"]],
                            "F2-score": [verified_best_metrics_df.at[0, "F2-score"]],
                            "ROC-AUC": [verified_best_metrics_df.at[0, "ROC-AUC"]],
                        }
                    )
                ).iat[0]
            ],
            "Training time (seconds)": [
                verified_best_metrics_df.at[0, "Training time (seconds)"]
            ],
            "grid_size": [len(VAR_SMOOTHING_GRID) * (1 + len(MANUAL_POSITIVE_PRIOR_GRID))],
            "num_splits": [len(GRID_RANDOM_SEEDS)],
            "test_size": [GRID_TEST_SIZE],
            "feature_set": ["Original predictors only"],
            "search_wall_time_seconds": [search_duration_seconds],
        }
    )

    ranked_output_path = repo_root / RANKED_RESULTS_PATH
    best_output_path = repo_root / BEST_RESULTS_PATH
    ranked_results_df.to_csv(ranked_output_path, index=False)
    best_result_df.to_csv(best_output_path, index=False)

    print(f"Saved ranked grid-search results to: {ranked_output_path}")
    print(f"Saved best GaussianNB hyperparameters to: {best_output_path}")
    print(best_result_df.to_string(index=False))


if __name__ == "__main__":
    main()
