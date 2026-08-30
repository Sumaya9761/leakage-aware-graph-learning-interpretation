# Leakage-aware hybrid graph learning for Alzheimer's disease classification

This repository contains the analysis code for the accompanying manuscript on
subject-disjoint ADNI diagnosis classification and verifier-controlled
patient-level interpretation. ADNI data are not distributed here.

## Repository contents

- `hybrid_gnn.py`: hybrid population GCN, temporal GCN, and MLP model
- `baseline_comparison.py`: subject-disjoint conventional ML baselines
- `bio_leaflet.py`: structured patient-level interpretation and verification
- `ADNIT5.py`: FLAN-T5 refinement and evaluation
- `run_bioleaflet_rag_prototype.py`: evidence-grounded follow-up evaluation
- `experiment_commands.json`: commands used for all reported experiments
- `tests/`: data-free tests for fixed-split and target-label behavior

## Environment

Use Python 3.12. Package versions are pinned in `requirements.txt`; some of
the pinned packages do not support Python 3.11 or earlier.

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Linux or macOS, create the environment with `python3.12 -m venv .venv`
and activate it with `source .venv/bin/activate`. The matched diagnosis/CDR
verification runs were tested on Windows with Python 3.12.14 and
PyTorch 2.11.0+cpu. A compatible CUDA build of PyTorch may be substituted for
faster training; record the software and device versions because small
numerical differences can occur across platforms.

## Required input schema

The model expects one row per ADNI visit. Identifier and target columns are:

- `subject_id`
- `clinical_session_id`
- `DIAGNOSIS` (ADNI diagnosis codes 1, 2, and 3)
- `clinical_CDGLOBAL` only for the CDR sensitivity analyses

The 18 source predictors are:

```text
clinical_entry_age, clinical_GENDER, clinical_EDUCAT,
clinical_APOE4_count, clinical_MMSCORE, clinical_FAQTOTAL,
clinical_LDELTOTAL, clinical_TRABSCOR,
mri_hippocampus_vol_mean, mri_entorhinal_vol_mean,
mri_amygdala_vol_mean, mri_inferior_temporal_vol_mean,
mri_middle_temporal_vol_mean, mri_lateral_ventricle_vol_mean,
mri_inf_lat_vent_vol_mean, clinical_ABETA42, pet_PTAU, pet_TAU
```

`visit_month` and `visit_age` are derived by the scripts, giving 20 model
inputs. Numeric imputation and scaling are fitted on training rows only.
Do not commit ADNI CSV files, subject-level exports, or trained checkpoints.

## Main analysis

```bash
python hybrid_gnn.py \
  --single study_data_no_cdr.csv \
  --out_dir results/main_model \
  --target diagnosis \
  --verbose
```

The run exports `split_subject_ids.csv`. This file is a restricted local
artifact because it contains ADNI subject identifiers.

## Fully matched diagnosis versus CDR-stage analysis

Both targets must use the same CDR-complete visits and the exact same saved
subject assignments. The diagnosis run filters to CDR-complete rows and then
removes CDR from the predictor set. The CDR-stage run uses CDR only as its
target. Both retain the original cognition-only graph.

```bash
python hybrid_gnn.py \
  --single study_data_with_cdr.csv \
  --out_dir results/cdr_matched_diagnosis \
  --target diagnosis \
  --require_cdr_complete \
  --exclude_cdr_feature \
  --split_file results/main_model/split_subject_ids.csv \
  --no_explain --verbose

python hybrid_gnn.py \
  --single study_data_with_cdr.csv \
  --out_dir results/cdr_matched_target \
  --target cdr \
  --require_cdr_complete \
  --split_file results/main_model/split_subject_ids.csv \
  --no_explain --verbose
```

The split loader allows the source split file to contain subjects absent from
the complete-case cohort, but every analyzed subject must have exactly one
train, validation, or test assignment.

## Conventional baselines

```bash
python baseline_comparison.py \
  --data_csv study_data_no_cdr.csv \
  --out_dir results/baselines
```

Additional nested-CV, inductive, ablation, interpretation, FLAN-T5, and RAG
commands are recorded in `experiment_commands.json`.

Non-identifying aggregate metrics from the verified runs are recorded in
`aggregate_results/classification_summary.json`. Per-visit probabilities,
participant assignments, model checkpoints, and ADNI records are deliberately
excluded.

## Data availability

The study data were obtained from the Alzheimer's Disease Neuroimaging
Initiative (ADNI) and remain subject to ADNI data-use requirements. This
repository intentionally excludes data rows and direct participant
identifiers. Only code and non-identifying aggregate summaries should be made
public.
