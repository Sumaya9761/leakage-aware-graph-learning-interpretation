#!/usr/bin/env python3
"""Leakage-safe post-training validation for the diagnosis classifier.

The script uses a fixed participant-level train/validation/test split. A
logistic-regression comparator is fitted on training rows only. Temperature,
an MCI log-probability offset, and an optional GNN/logistic blending weight are
selected on validation rows only and then locked before test evaluation.

Participant identifiers and per-visit probabilities are read locally but are
never written to the aggregate output files.
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_VARS = [
    "clinical_entry_age",
    "clinical_GENDER",
    "clinical_EDUCAT",
    "clinical_APOE4_count",
    "clinical_MMSCORE",
    "clinical_FAQTOTAL",
    "clinical_LDELTOTAL",
    "clinical_TRABSCOR",
    "mri_hippocampus_vol_mean",
    "mri_entorhinal_vol_mean",
    "mri_amygdala_vol_mean",
    "mri_inferior_temporal_vol_mean",
    "mri_middle_temporal_vol_mean",
    "mri_lateral_ventricle_vol_mean",
    "mri_inf_lat_vent_vol_mean",
    "clinical_ABETA42",
    "pet_PTAU",
    "pet_TAU",
]
DERIVED_VARS = ["visit_month", "visit_age"]
PROBABILITY_COLUMNS = ["prob_NC", "prob_MCI", "prob_AD"]
CLASS_NAMES = ["NC", "MCI", "AD"]


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have a positive sum")
    return probabilities / row_sums


def parse_visit_month(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"sc", "scr", "screening", "bl", "baseline"}:
        return 0.0
    match = re.match(r"^m(\d+)$", text)
    return float(match.group(1)) if match else np.nan


def prepare_dataset(data_csv, split_csv):
    df = pd.read_csv(data_csv)
    cdr_columns = [c for c in df.columns if "CDR" in c.upper() or "CDGLOBAL" in c.upper()]
    if cdr_columns:
        raise ValueError(f"CDR columns are forbidden in the primary analysis: {cdr_columns}")

    required = {"subject_id", "clinical_session_id", "DIAGNOSIS", *SOURCE_VARS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")

    df = df.copy()
    df["visit_month"] = df["clinical_session_id"].map(parse_visit_month)
    df["visit_age"] = df["clinical_entry_age"] + df["visit_month"] / 12.0

    diagnosis = pd.to_numeric(df["DIAGNOSIS"], errors="raise")
    unique = set(diagnosis.dropna().astype(float).unique())
    if unique.issubset({1.0, 2.0, 3.0}):
        diagnosis = diagnosis.map({1.0: 0, 2.0: 1, 3.0: 2})
    elif not unique.issubset({0.0, 1.0, 2.0}):
        raise ValueError(f"Unexpected DIAGNOSIS values: {sorted(unique)}")
    df["target"] = diagnosis.astype(int)

    assignments = pd.read_csv(split_csv, dtype={"subject_id": str})
    if not {"subject_id", "split"}.issubset(assignments.columns):
        raise KeyError("Split CSV must contain subject_id and split columns")
    assignments = assignments[["subject_id", "split"]].drop_duplicates()
    if assignments["subject_id"].duplicated().any():
        raise ValueError("Each subject must have exactly one split assignment")
    split_map = assignments.set_index("subject_id")["split"].str.lower()
    df["subject_id"] = df["subject_id"].astype(str)
    df["split"] = df["subject_id"].map(split_map)
    if df["split"].isna().any():
        raise ValueError("The split file does not assign every analyzed subject")
    if not set(df["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError(f"Unexpected split labels: {sorted(df['split'].unique())}")

    subject_splits = df[["subject_id", "split"]].drop_duplicates()
    if subject_splits["subject_id"].duplicated().any():
        raise AssertionError("A subject appears in more than one split")
    return df, SOURCE_VARS + DERIVED_VARS


def load_split_probabilities(probability_csv, df, split_name):
    table = pd.read_csv(probability_csv)
    required = {"node_idx", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Probability file is missing columns: {missing}")
    table["node_idx"] = pd.to_numeric(table["node_idx"], errors="raise").astype(int)
    if table["node_idx"].duplicated().any():
        raise ValueError("Probability file contains duplicate node_idx values")

    expected = np.flatnonzero(df["split"].eq(split_name).to_numpy())
    observed = np.sort(table["node_idx"].to_numpy())
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"{split_name} probabilities do not match the fixed split: "
            f"expected {len(expected)} rows, observed {len(observed)}"
        )
    table = table.set_index("node_idx").loc[expected].reset_index()

    if "subject_id" in table.columns:
        expected_subjects = df.iloc[expected]["subject_id"].astype(str).to_numpy()
        if not np.array_equal(table["subject_id"].astype(str).to_numpy(), expected_subjects):
            raise ValueError("Probability subject IDs do not align with the dataset")
    probs = table[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    return expected, probs


def load_seed_probabilities(probability_csv, df, split_name):
    table = pd.read_csv(probability_csv)
    required = {"seed", "node_idx", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Per-seed probability file is missing columns: {missing}")
    table["seed"] = pd.to_numeric(table["seed"], errors="raise").astype(int)
    table["node_idx"] = pd.to_numeric(table["node_idx"], errors="raise").astype(int)
    if table[["seed", "node_idx"]].duplicated().any():
        raise ValueError("Per-seed probability file contains duplicate seed/node rows")

    expected = np.flatnonzero(df["split"].eq(split_name).to_numpy())
    output = {}
    for seed, seed_table in table.groupby("seed", sort=True):
        seed_table = seed_table.sort_values("node_idx")
        if not np.array_equal(seed_table["node_idx"].to_numpy(), expected):
            raise ValueError(f"Seed {seed} does not contain the complete {split_name} split")
        output[int(seed)] = normalize_probabilities(
            seed_table[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        )
    if not output:
        raise ValueError("No per-seed probabilities were found")
    return expected, output


def select_best_validation_seed(seed_probabilities, labels):
    scores = {
        int(seed): float(
            balanced_accuracy_score(labels, probabilities.argmax(axis=1))
        )
        for seed, probabilities in seed_probabilities.items()
    }
    best_score = max(scores.values())
    best_seed = min(seed for seed, score in scores.items() if np.isclose(score, best_score))
    return int(best_seed), float(best_score), scores


def expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = normalize_probabilities(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        include = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            include = (confidence >= lower) & (confidence <= upper)
        if include.any():
            ece += include.mean() * abs(correct[include].mean() - confidence[include].mean())
    return float(ece)


def multiclass_brier_score(y_true, probabilities):
    one_hot = np.eye(probabilities.shape[1], dtype=float)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = normalize_probabilities(probabilities)
    predictions = probabilities.argmax(axis=1)
    recalls = recall_score(y_true, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    try:
        macro_auc = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_auc": macro_auc,
        "recall": {name: float(value) for name, value in zip(CLASS_NAMES, recalls)},
        "negative_log_likelihood": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "ece_10_bin": expected_calibration_error(y_true, probabilities, n_bins=10),
    }


def apply_temperature(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def select_temperature(validation_probabilities, validation_labels):
    objective = lambda value: log_loss(
        validation_labels,
        apply_temperature(validation_probabilities, value),
        labels=[0, 1, 2],
    )
    result = minimize_scalar(objective, bounds=(0.25, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(result.x)


def apply_mci_offset(probabilities, offset):
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    logits[:, 1] += float(offset)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def select_mci_offset(validation_probabilities, validation_labels, offsets=None):
    if offsets is None:
        offsets = np.linspace(-1.0, 1.0, 51)
    best_offset = 0.0
    best_score = -math.inf
    for offset in offsets:
        adjusted = apply_mci_offset(validation_probabilities, offset)
        score = balanced_accuracy_score(validation_labels, adjusted.argmax(axis=1))
        if score > best_score:
            best_offset = float(offset)
            best_score = float(score)
    return best_offset, best_score


def select_fusion_weight(gnn_validation, lr_validation, validation_labels, weights=None):
    """Select the GNN weight from validation data only."""
    if weights is None:
        weights = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    candidates = []
    for weight in weights:
        probabilities = weight * gnn_validation + (1.0 - weight) * lr_validation
        score = balanced_accuracy_score(validation_labels, probabilities.argmax(axis=1))
        candidates.append((float(score), float(weight)))
    best_score = max(score for score, _ in candidates)
    tied = [weight for score, weight in candidates if np.isclose(score, best_score)]
    return max(tied), float(best_score), candidates


def cluster_bootstrap(
    y_true,
    subject_ids,
    probability_sets,
    reference_name,
    n_bootstrap=5000,
    seed=42,
):
    """Participant-cluster percentile intervals and paired BA differences."""
    y_true = np.asarray(y_true, dtype=int)
    subject_ids = np.asarray(subject_ids).astype(str)
    unique_subjects = np.unique(subject_ids)
    subject_rows = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in probability_sets}
    differences = {name: [] for name in probability_sets if name != reference_name}
    rejected = 0

    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([subject_rows[subject] for subject in sampled])
        labels = y_true[indices]
        if len(np.unique(labels)) < 3:
            rejected += 1
            continue
        replicate = {}
        for name, probabilities in probability_sets.items():
            score = balanced_accuracy_score(labels, probabilities[indices].argmax(axis=1))
            draws[name].append(float(score))
            replicate[name] = float(score)
        for name in differences:
            differences[name].append(replicate[name] - replicate[reference_name])

    if not draws[reference_name]:
        raise RuntimeError("No valid cluster-bootstrap replicates were produced")

    output = {
        "unit": "participant",
        "requested_replicates": int(n_bootstrap),
        "accepted_replicates": int(len(draws[reference_name])),
        "rejected_missing_class": int(rejected),
        "seed": int(seed),
        "balanced_accuracy": {},
        "paired_difference_vs_" + reference_name: {},
    }
    for name, values in draws.items():
        values = np.asarray(values)
        output["balanced_accuracy"][name] = {
            "mean": float(values.mean()),
            "ci95": [float(v) for v in np.percentile(values, [2.5, 97.5])],
        }
    difference_key = "paired_difference_vs_" + reference_name
    for name, values in differences.items():
        values = np.asarray(values)
        output[difference_key][name] = {
            "mean": float(values.mean()),
            "ci95": [float(v) for v in np.percentile(values, [2.5, 97.5])],
            "probability_difference_gt_0": float(np.mean(values > 0)),
        }
    return output


def plot_reliability(y_true, probability_sets, out_path, n_bins=10):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    for name, probabilities in probability_sets.items():
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == y_true
        x_values, y_values = [], []
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            include = (confidence > lower) & (confidence <= upper)
            if lower == 0.0:
                include = (confidence >= lower) & (confidence <= upper)
            if include.any():
                x_values.append(float(confidence[include].mean()))
                y_values.append(float(correct[include].mean()))
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=name)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Ideal")
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Test-set reliability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--gnn_validation_probabilities", required=True)
    parser.add_argument("--gnn_test_probabilities", required=True)
    parser.add_argument("--gnn_validation_seed_probabilities")
    parser.add_argument("--gnn_test_seed_probabilities")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, feature_columns = prepare_dataset(args.data_csv, args.split_csv)
    val_idx, gnn_val = load_split_probabilities(
        args.gnn_validation_probabilities, df, "val"
    )
    test_idx, gnn_test = load_split_probabilities(args.gnn_test_probabilities, df, "test")

    train_mask = df["split"].eq("train").to_numpy()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipeline.fit(df.loc[train_mask, feature_columns], df.loc[train_mask, "target"])
    lr_val = pipeline.predict_proba(df.iloc[val_idx][feature_columns])
    lr_test = pipeline.predict_proba(df.iloc[test_idx][feature_columns])
    y_val = df.iloc[val_idx]["target"].to_numpy(dtype=int)
    y_test = df.iloc[test_idx]["target"].to_numpy(dtype=int)

    temperature = select_temperature(gnn_val, y_val)
    gnn_temperature_test = apply_temperature(gnn_test, temperature)
    mci_offset, mci_val_score = select_mci_offset(gnn_val, y_val)
    gnn_mci_test = apply_mci_offset(gnn_test, mci_offset)
    fusion_weight, fusion_val_score, fusion_candidates = select_fusion_weight(
        gnn_val, lr_val, y_val
    )
    fusion_test = fusion_weight * gnn_test + (1.0 - fusion_weight) * lr_test

    probability_sets = {
        "GNN ensemble": gnn_test,
        "GNN temperature-scaled": gnn_temperature_test,
        "GNN MCI-adjusted": gnn_mci_test,
        "Logistic regression": lr_test,
        "Validation-selected fusion": fusion_test,
    }
    validation_metrics = {
        "GNN ensemble": classification_metrics(y_val, gnn_val),
        "Logistic regression": classification_metrics(y_val, lr_val),
        "GNN MCI-adjusted": classification_metrics(y_val, apply_mci_offset(gnn_val, mci_offset)),
        "Validation-selected fusion": classification_metrics(
            y_val, fusion_weight * gnn_val + (1.0 - fusion_weight) * lr_val
        ),
    }

    seed_selection = None
    seed_paths = [
        args.gnn_validation_seed_probabilities,
        args.gnn_test_seed_probabilities,
    ]
    if any(seed_paths) and not all(seed_paths):
        raise ValueError("Both validation and test per-seed probability files are required")
    if all(seed_paths):
        seed_val_idx, validation_by_seed = load_seed_probabilities(
            args.gnn_validation_seed_probabilities, df, "val"
        )
        seed_test_idx, test_by_seed = load_seed_probabilities(
            args.gnn_test_seed_probabilities, df, "test"
        )
        if not np.array_equal(seed_val_idx, val_idx) or not np.array_equal(seed_test_idx, test_idx):
            raise ValueError("Per-seed and ensemble probability files do not align")
        if set(validation_by_seed) != set(test_by_seed):
            raise ValueError("Validation and test files contain different seed sets")
        best_seed, best_seed_score, seed_scores = select_best_validation_seed(
            validation_by_seed, y_val
        )
        selected_name = "Validation-selected GNN seed"
        probability_sets[selected_name] = test_by_seed[best_seed]
        validation_metrics[selected_name] = classification_metrics(
            y_val, validation_by_seed[best_seed]
        )
        seed_selection = {
            "status": "exploratory deployment-model selection",
            "selected_seed": best_seed,
            "validation_balanced_accuracy": best_seed_score,
            "candidate_validation_balanced_accuracy": seed_scores,
        }

    metrics = {
        name: classification_metrics(y_test, probs)
        for name, probs in probability_sets.items()
    }
    bootstrap = cluster_bootstrap(
        y_test,
        df.iloc[test_idx]["subject_id"].to_numpy(),
        probability_sets,
        reference_name="Logistic regression",
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )

    selection = {
        "selection_partition": "validation only",
        "temperature": temperature,
        "mci_log_probability_offset": mci_offset,
        "mci_adjusted_validation_balanced_accuracy": mci_val_score,
        "fusion_gnn_weight": fusion_weight,
        "fusion_validation_balanced_accuracy": fusion_val_score,
        "fusion_candidates": [
            {"balanced_accuracy": score, "gnn_weight": weight}
            for score, weight in fusion_candidates
        ],
        "seed_selection": seed_selection,
    }
    output = {
        "schema_version": 1,
        "design": {
            "split": "fixed subject-disjoint train/validation/test",
            "model_selection": "training and validation rows only",
            "test_rows": int(len(test_idx)),
            "test_participants": int(df.iloc[test_idx]["subject_id"].nunique()),
            "features": feature_columns,
            "cdr_used": False,
        },
        "selection": selection,
        "validation_metrics": validation_metrics,
        "test_metrics": metrics,
        "participant_cluster_bootstrap": bootstrap,
    }
    with open(out_dir / "publication_validation.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(output), handle, indent=2)

    rows = []
    for partition, collection in [("validation", validation_metrics), ("test", metrics)]:
        for model_name, values in collection.items():
            rows.append(
                {
                    "partition": partition,
                    "model": model_name,
                    "balanced_accuracy": values["balanced_accuracy"],
                    "accuracy": values["accuracy"],
                    "macro_f1": values["macro_f1"],
                    "macro_auc": values["macro_auc"],
                    "brier_score": values["brier_score"],
                    "ece_10_bin": values["ece_10_bin"],
                    "nc_recall": values["recall"]["NC"],
                    "mci_recall": values["recall"]["MCI"],
                    "ad_recall": values["recall"]["AD"],
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)
    plot_reliability(y_test, probability_sets, out_dir / "reliability_diagram.png")

    print(json.dumps(json_ready({"selection": selection, "test_metrics": metrics}), indent=2))
    print(f"Aggregate outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
