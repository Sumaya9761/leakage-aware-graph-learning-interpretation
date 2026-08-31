"""Paired participant-bootstrap inference for fixed-split model ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


CLASS_LABELS = ("0", "0.5", "1+")
PROBABILITY_COLUMNS = ("prob_NC", "prob_MCI", "prob_AD")
KEY_COLUMNS = ("subject_id", "clinical_session_id")


def load_probabilities(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"subject_id": str, "clinical_session_id": str, "true_class": str},
    )
    required = {*KEY_COLUMNS, "true_class", *PROBABILITY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{path} contains duplicate participant-session keys")
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{path} contains invalid probabilities")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"{path} probability rows do not sum to one")
    frame = frame.copy()
    frame["true_class"] = frame["true_class"].replace({"NC": "0", "MCI": "0.5", "AD": "1"})
    frame["true_class"] = frame["true_class"].replace({"1": "1+", "1.0": "1+"})
    unknown = sorted(set(frame["true_class"]) - set(CLASS_LABELS))
    if unknown:
        raise ValueError(f"{path} contains unknown true-class labels: {unknown}")
    frame["predicted_class"] = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def align_candidate(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> pd.DataFrame:
    reference_keys = reference.loc[:, KEY_COLUMNS]
    candidate_keys = candidate.loc[:, KEY_COLUMNS]
    if not reference_keys.equals(candidate_keys):
        raise ValueError(f"{name} does not contain the same ordered participant-session rows")
    if not reference["true_class"].equals(candidate["true_class"]):
        raise ValueError(f"{name} true labels do not match the reference model")
    return candidate


def classification_metrics(frame: pd.DataFrame, row_indices: np.ndarray | None = None) -> dict[str, object]:
    selected = frame if row_indices is None else frame.iloc[row_indices]
    y_true = selected["true_class"].to_numpy()
    y_pred = selected["predicted_class"].to_numpy()
    recalls = recall_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "recall": {label: float(value) for label, value in zip(CLASS_LABELS, recalls)},
    }


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def two_sided_bootstrap_probability(values: np.ndarray) -> float:
    lower_tail = (np.count_nonzero(values <= 0) + 1) / (values.size + 1)
    upper_tail = (np.count_nonzero(values >= 0) + 1) / (values.size + 1)
    return float(min(1.0, 2.0 * min(lower_tail, upper_tail)))


def paired_participant_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    participants = reference["subject_id"].drop_duplicates().to_numpy()
    participant_rows = {
        participant: np.flatnonzero(reference["subject_id"].to_numpy() == participant)
        for participant in participants
    }
    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = {
        "balanced_accuracy": [],
        **{f"recall_{label}": [] for label in CLASS_LABELS},
    }
    rejected = 0
    for _ in range(replicates):
        sampled = rng.choice(participants, size=participants.size, replace=True)
        rows = np.concatenate([participant_rows[participant] for participant in sampled])
        if set(reference.iloc[rows]["true_class"]) != set(CLASS_LABELS):
            rejected += 1
            continue
        reference_metrics = classification_metrics(reference, rows)
        candidate_metrics = classification_metrics(candidate, rows)
        differences["balanced_accuracy"].append(
            reference_metrics["balanced_accuracy"] - candidate_metrics["balanced_accuracy"]
        )
        for label in CLASS_LABELS:
            differences[f"recall_{label}"].append(
                reference_metrics["recall"][label] - candidate_metrics["recall"][label]
            )

    if not differences["balanced_accuracy"]:
        raise RuntimeError("No bootstrap replicate retained all three outcome classes")

    summary: dict[str, object] = {}
    for metric, samples in differences.items():
        values = np.asarray(samples, dtype=float)
        summary[metric] = {
            "mean_difference": float(values.mean()),
            "ci95": percentile_interval(values),
            "probability_reference_gt_candidate": float(np.mean(values > 0)),
            "two_sided_bootstrap_probability": two_sided_bootstrap_probability(values),
        }
    return {
        "requested_replicates": replicates,
        "accepted_replicates": len(differences["balanced_accuracy"]),
        "rejected_missing_class": rejected,
        "seed": seed,
        "unit": "participant",
        "difference_direction": "full model minus ablation",
        "differences": summary,
    }


def parse_ablation(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("ablation must be formatted NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("ablation name cannot be empty")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_csv", required=True, type=Path)
    parser.add_argument("--ablation", required=True, action="append", type=parse_ablation)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument("--bootstrap_replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap_replicates must be positive")

    reference = load_probabilities(args.reference_csv)
    comparisons: dict[str, object] = {}
    seen_names: set[str] = set()
    for name, path in args.ablation:
        if name in seen_names:
            parser.error(f"duplicate ablation name: {name}")
        seen_names.add(name)
        candidate = align_candidate(reference, load_probabilities(path), name)
        comparisons[name] = {
            "ablation_metrics": classification_metrics(candidate),
            "paired_participant_bootstrap": paired_participant_bootstrap(
                reference,
                candidate,
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            ),
        }

    output = {
        "schema_version": 1,
        "design": {
            "evaluation": "fixed subject-disjoint test partition",
            "reference_model": "full hybrid GNN probability ensemble",
            "test_visits": int(len(reference)),
            "test_participants": int(reference["subject_id"].nunique()),
            "selection_use": "inference only; ablations were not used to reselect the primary model",
        },
        "reference_metrics": classification_metrics(reference),
        "comparisons": comparisons,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
