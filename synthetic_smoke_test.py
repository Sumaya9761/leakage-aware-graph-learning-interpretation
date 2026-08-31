"""Run the selected hybrid-GNN pipeline on a generated, non-ADNI cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FEATURES = (
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
)


def generate_synthetic_cohort(
    seed: int = 2026,
    subjects_per_class: int = 12,
) -> pd.DataFrame:
    """Create a deterministic three-class longitudinal cohort for software testing."""
    if subjects_per_class < 4:
        raise ValueError("subjects_per_class must be at least four")
    rng = np.random.default_rng(seed)
    sessions = (("bl", 0), ("m06", 6), ("m12", 12))
    rows: list[dict[str, object]] = []
    for class_index, diagnosis_code in enumerate((1, 2, 3)):
        for subject_index in range(subjects_per_class):
            subject_id = f"SYN_C{class_index}_{subject_index:03d}"
            entry_age = 68.0 + 3.0 * class_index + rng.normal(0, 2.0)
            gender = "Female" if subject_index % 2 == 0 else "Male"
            education = 16.0 - 0.5 * class_index + rng.normal(0, 1.0)
            apoe4 = int(rng.random() < (0.15 + 0.25 * class_index))
            for session_name, month in sessions:
                progression = month / 12.0
                noise = lambda scale: float(rng.normal(0, scale))
                rows.append(
                    {
                        "subject_id": subject_id,
                        "clinical_session_id": session_name,
                        "clinical_entry_age": entry_age,
                        "clinical_GENDER": gender,
                        "clinical_EDUCAT": education,
                        "clinical_APOE4_count": apoe4,
                        "clinical_MMSCORE": 29.0 - 4.5 * class_index - progression + noise(0.7),
                        "clinical_FAQTOTAL": 1.0 + 7.0 * class_index + 1.5 * progression + noise(1.0),
                        "clinical_LDELTOTAL": 13.0 - 5.0 * class_index - progression + noise(0.8),
                        "clinical_TRABSCOR": 65.0 + 48.0 * class_index + 8.0 * progression + noise(8.0),
                        "mri_hippocampus_vol_mean": 3700.0 - 430.0 * class_index - 70.0 * progression + noise(90.0),
                        "mri_entorhinal_vol_mean": 1900.0 - 230.0 * class_index - 45.0 * progression + noise(65.0),
                        "mri_amygdala_vol_mean": 1450.0 - 170.0 * class_index - 35.0 * progression + noise(55.0),
                        "mri_inferior_temporal_vol_mean": 9800.0 - 650.0 * class_index - 90.0 * progression + noise(180.0),
                        "mri_middle_temporal_vol_mean": 9900.0 - 620.0 * class_index - 80.0 * progression + noise(180.0),
                        "mri_lateral_ventricle_vol_mean": 17000.0 + 6200.0 * class_index + 850.0 * progression + noise(900.0),
                        "mri_inf_lat_vent_vol_mean": 850.0 + 470.0 * class_index + 70.0 * progression + noise(90.0),
                        "clinical_ABETA42": 1200.0 - 250.0 * class_index - 35.0 * progression + noise(80.0),
                        "pet_PTAU": 24.0 + 9.0 * class_index + 1.5 * progression + noise(3.0),
                        "pet_TAU": 235.0 + 55.0 * class_index + 8.0 * progression + noise(15.0),
                        "DIAGNOSIS": diagnosis_code,
                    }
                )
    frame = pd.DataFrame(rows)
    # Exercise training-only imputation without making graph-defining cognition sparse.
    frame.loc[frame.index[::29], "mri_amygdala_vol_mean"] = np.nan
    frame.loc[frame.index[::31], "clinical_ABETA42"] = np.nan
    return frame


def validate_smoke_outputs(out_dir: Path) -> dict[str, object]:
    required = (
        "metrics.json",
        "label_mapping.json",
        "ensemble_probabilities.csv",
        "validation_ensemble_probabilities.csv",
        "split_subject_ids.csv",
    )
    missing = [name for name in required if not (out_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"smoke run did not create required outputs: {missing}")

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    split = metrics.get("split", {})
    if any(int(split.get(partition, 0)) <= 0 for partition in ("train", "val", "test")):
        raise RuntimeError(f"smoke run produced an empty partition: {split}")
    if metrics.get("classes") != ["0", "0.5", "1+"]:
        raise RuntimeError(f"unexpected class mapping: {metrics.get('classes')}")

    probabilities = pd.read_csv(out_dir / "ensemble_probabilities.csv")
    if probabilities.empty or probabilities["true_class"].nunique() != 3:
        raise RuntimeError("smoke-test probabilities do not cover all three classes")
    probability_columns = ["prob_NC", "prob_MCI", "prob_AD"]
    if not np.allclose(probabilities[probability_columns].sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("smoke-test probability rows do not sum to one")
    if probabilities["subject_id"].astype(str).str.match(r"\d{3}_S_\d{4}").any():
        raise RuntimeError("smoke-test output unexpectedly resembles an ADNI identifier")

    return {
        "status": "passed",
        "rows": int(sum(int(value) for value in split.values())),
        "split": {key: int(value) for key, value in split.items()},
        "test_rows": int(len(probabilities)),
        "test_balanced_accuracy": float(metrics["ensemble_test"]["balanced_accuracy"]),
    }


def run_smoke_test(work_dir: Path) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    data_csv = work_dir / "synthetic_longitudinal_cohort.csv"
    out_dir = work_dir / "results"
    cohort = generate_synthetic_cohort()
    if tuple(column for column in SOURCE_FEATURES if column not in cohort.columns):
        raise RuntimeError("synthetic cohort is missing a required source feature")
    cohort.to_csv(data_csv, index=False)

    command = [
        sys.executable,
        str(Path(__file__).with_name("hybrid_gnn.py")),
        "--single",
        str(data_csv),
        "--out_dir",
        str(out_dir),
        "--target",
        "diagnosis",
        "--epochs",
        "2",
        "--patience",
        "2",
        "--warmup_epochs",
        "0",
        "--num_seeds",
        "1",
        "--hidden",
        "16",
        "--num_gcn_layers",
        "1",
        "--temporal_branch_layers",
        "1",
        "--temporal_branch_hidden",
        "8",
        "--topk",
        "5",
        "--dropout",
        "0.1",
        "--feat_mask_rate",
        "0",
        "--normalization",
        "masked_batch",
        "--temporal_direction",
        "causal",
        "--drop_edge_rate",
        "0",
        "--uncertainty_reference",
        "validation",
        "--no_explain",
        "--no_counterfactual",
    ]
    environment = os.environ.copy()
    environment.update({"MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
    completed = subprocess.run(
        command,
        cwd=Path(__file__).parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "synthetic pipeline failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    summary = validate_smoke_outputs(out_dir)
    summary["command"] = command
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Optional directory in which to retain the synthetic input and outputs.",
    )
    args = parser.parse_args()

    context = nullcontext(args.work_dir) if args.work_dir else tempfile.TemporaryDirectory()
    with context as raw_directory:
        work_dir = Path(raw_directory)
        summary = run_smoke_test(work_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
