"""Create manuscript result figures from a completed held-out GNN run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CLASS_ORDER = ["0", "0.5", "1+"]
CLASS_LABELS = ["NC", "MCI", "AD"]
COLORS = ["#2A9D8F", "#E9A23B", "#4E79A7"]


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ensemble_confusion(results_dir: Path, out_dir: Path) -> dict:
    frame = pd.read_csv(
        results_dir / "ensemble_probabilities.csv",
        dtype={"true_class": str, "pred_class": str},
    )
    matrix = confusion_matrix(
        frame["true_class"], frame["pred_class"], labels=CLASS_ORDER
    )
    row_fraction = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            ax.text(
                column,
                row,
                f"{matrix[row, column]}\n({row_fraction[row, column]:.1%})",
                ha="center",
                va="center",
                fontsize=11,
                color=color,
            )
    ax.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        xlabel="Predicted class",
        ylabel="Reference class",
        title="Five-model Ensemble Confusion Matrix",
    )
    ax.tick_params(labelsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, out_dir / "confusion_matrix.png")
    return {
        "matrix": matrix.tolist(),
        "row_recall": dict(zip(CLASS_LABELS, row_fraction.diagonal().tolist())),
    }


def plot_counterfactuals(results_dir: Path, out_dir: Path) -> dict:
    frame = pd.read_csv(
        results_dir / "counterfactual_explanations.csv",
        dtype={"pred_class": str, "cf_class": str},
    )
    rows = []
    for class_value, label in zip(CLASS_ORDER, CLASS_LABELS):
        group = frame.loc[frame["pred_class"] == class_value]
        found = group["cf_class"].notna()
        rows.append(
            {
                "class": label,
                "n": int(len(group)),
                "found": int(found.sum()),
                "one": int((found & group["n_changes"].eq(1)).sum()),
                "two": int((found & group["n_changes"].eq(2)).sum()),
            }
        )

    x = np.arange(len(rows))
    width = 0.23
    denominators = np.array([row["n"] for row in rows], dtype=float)
    values = [
        100 * np.array([row[key] for row in rows], dtype=float) / denominators
        for key in ("found", "one", "two")
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = ["Counterfactual found", "One-feature change", "Two-feature change"]
    for index, (series, label, color) in enumerate(zip(values, labels, COLORS)):
        bars = ax.bar(x + (index - 1) * width, series, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, [row["class"] for row in rows])
    for position, row in zip(x, rows):
        ax.text(
            position,
            -0.14,
            f"n={row['n']}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Predictions (%)")
    ax.set_title("Counterfactuals by Predicted Class")
    ax.set_ylim(0, max(max(series) for series in values) + 12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, out_dir / "counterfactual_examples.png")
    return {row["class"]: row for row in rows}


def plot_cdr_discordance(results_dir: Path, cdr_csv: Path, out_dir: Path) -> dict:
    uncertainty = pd.read_csv(
        results_dir / "uncertainty_estimates.csv",
        dtype={
            "subject_id": str,
            "clinical_session_id": str,
            "true_class": str,
            "pred_class": str,
        },
    )
    cdr = pd.read_csv(
        cdr_csv,
        dtype={"subject_id": str, "clinical_session_id": str},
    )
    joined = uncertainty.merge(
        cdr[
            [
                "subject_id",
                "clinical_session_id",
                "DIAGNOSIS_CDR_STAGE",
                "DIAGNOSIS",
            ]
        ],
        on=["subject_id", "clinical_session_id"],
        how="inner",
        validate="one_to_one",
    )
    joined["concordant"] = joined["DIAGNOSIS_CDR_STAGE"].eq(joined["DIAGNOSIS"])
    joined["correct"] = joined["true_class"].eq(joined["pred_class"])
    joined["is_high_uncertainty"] = joined["is_high_uncertainty"].astype(str).str.lower().eq("true")

    summary = []
    for status, label in ((True, "Concordant"), (False, "Discordant")):
        group = joined.loc[joined["concordant"] == status]
        summary.append(
            {
                "group": label,
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "high_uncertainty": float(group["is_high_uncertainty"].mean()),
                "mean_entropy": float(group["predictive_entropy"].mean()),
            }
        )

    x = np.arange(2)
    width = 0.32
    accuracy = 100 * np.array([row["accuracy"] for row in summary])
    uncertainty_rate = 100 * np.array([row["high_uncertainty"] for row in summary])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bars_a = ax.bar(x - width / 2, accuracy, width, color="#4E79A7", label="Accuracy")
    bars_u = ax.bar(
        x + width / 2,
        uncertainty_rate,
        width,
        color="#E15759",
        label="High uncertainty",
    )
    ax.bar_label(bars_a, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars_u, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, [f"{row['group']}\n(n={row['n']})" for row in summary])
    ax.set_ylabel("Visits (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Performance by Diagnosis-CDR Concordance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir / "cdr_discordance.png")
    return {row["group"].lower(): row for row in summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=Path)
    parser.add_argument("--cdr_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "ensemble_confusion": plot_ensemble_confusion(args.results_dir, args.out_dir),
        "counterfactuals": plot_counterfactuals(args.results_dir, args.out_dir),
        "diagnosis_cdr": plot_cdr_discordance(
            args.results_dir, args.cdr_csv, args.out_dir
        ),
    }
    with (args.out_dir / "figure_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
