# Leakage-aware hybrid graph learning for Alzheimer's disease classification

This repository contains the analysis code for the accompanying manuscript on
subject-disjoint ADNI diagnosis classification and verifier-controlled
patient-level interpretation. ADNI data are not distributed here.

## Repository contents

- `hybrid_gnn.py`: hybrid population GCN, temporal GCN, and MLP model
- `baseline_comparison.py`: subject-disjoint conventional ML baselines
- `bio_leaflet.py`: structured patient-level interpretation and verification
- `generate_and_evaluate_bioleaflets.py`: visit-matched BioLeaflet assembly,
  deterministic field audit, and ROUGE evaluation
- `ADNIT5.py`: FLAN-T5 refinement and evaluation
- `flan_t5_oof.py`: five-fold participant-disjoint FLAN-T5 evaluation with
  semantic verification and deterministic fallback
- `run_bioleaflet_rag_prototype.py`: evidence-grounded follow-up evaluation
- `publication_validation.py`: calibration, matched logistic regression, and
  participant-cluster bootstrap analysis
- `verifier_stress_test.py`: clean-control and deterministic corruption tests
- `experiment_commands.json`: commands used for all reported experiments
- `requirements_flan_t5_oof.txt`: exact direct dependencies for the reported
  out-of-fold language-model run
- `tests/`: data-free tests for splitting, targets, verification, and metrics

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

## Participant-disjoint FLAN-T5 evaluation

The publication analysis evaluates every diagnostic-model test visit out of
fold. The 34 participants are partitioned into five outer folds, the base
FLAN-T5 model is reinitialized for each fold, and no participant contributes
text to both model training and that fold's test set. Paraphrased visit targets
and participant summaries are training-only augmentations; validation and test
sets contain one original target per visit.

The reported run used Python 3.9.19 and the versions in
`requirements_flan_t5_oof.txt`. Install a CUDA-enabled PyTorch 2.3.1 build
appropriate for the local device when GPU training is required.

Run each fold separately so that GPU memory is released between folds, then
aggregate and reverify the saved outputs:

```bash
python flan_t5_oof.py --results_dir results/main_model \
  --data_csv study_data_no_cdr.csv --out_dir results/flan_t5_oof --fold 1
python flan_t5_oof.py --results_dir results/main_model \
  --data_csv study_data_no_cdr.csv --out_dir results/flan_t5_oof --fold 2
python flan_t5_oof.py --results_dir results/main_model \
  --data_csv study_data_no_cdr.csv --out_dir results/flan_t5_oof --fold 3
python flan_t5_oof.py --results_dir results/main_model \
  --data_csv study_data_no_cdr.csv --out_dir results/flan_t5_oof --fold 4
python flan_t5_oof.py --results_dir results/main_model \
  --data_csv study_data_no_cdr.csv --out_dir results/flan_t5_oof --fold 5
python flan_t5_oof.py --results_dir results/main_model \
  --data_csv study_data_no_cdr.csv --out_dir results/flan_t5_oof \
  --reverify_saved
```

The aggregate summary is safe to share and is included as
`aggregate_results/flan_t5_oof_summary.json`. Fold predictions, generated
text, participant assignments, training corpora, and checkpoints are
restricted local artifacts. The older `ADNIT5.py` train/evaluate commands are
retained for exploratory use but are not the reported participant-disjoint
publication evaluation.

## Publication validation analyses

The nested-CV command reports the raw GNN result and, separately, a post-hoc
MCI decision sensitivity analysis. Within each outer fold, an additive MCI
log-probability offset is selected using only the internal validation rows,
locked, and then applied to the untouched outer-test rows. This is a decision
rule adjustment, not probability calibration, and the raw result remains the
primary estimate.

`publication_validation.py` uses the fixed subject-disjoint split to fit a
matched logistic-regression comparator on training rows only, select
temperature scaling and any candidate fusion on validation rows only, and
compute calibration metrics plus 10,000-replicate participant-cluster
bootstrap intervals. If validation selects a GNN fusion weight of 1.0, no
fusion improvement should be claimed.

GNNExplainer produces a local ranking for every held-out visit, targeting the
ensemble-predicted class. Aggregate class profiles are computed separately
from correctly classified visits. BioLeaflet reports label each ranking as
patient-specific and explicitly label the predicted-class aggregate fallback
when a legacy result directory has no local evidence for a visit.

Explanation faithfulness is evaluated by setting each correctly classified
visit's three highest-ranked preprocessed inputs to zero while holding both
graphs fixed. The resulting predicted-class probability drop is compared with
100 matched random feature deletions. Because the analyzed set is restricted
to correctly classified visits, this is a local ranking-faithfulness test
rather than a causal analysis.

`verifier_stress_test.py` first checks clean BioLeaflet and RAG outputs, then
applies deterministic single-error and contradiction mutations. The reported
detection rates characterize those predefined challenges; they are not
estimates of safety for unconstrained language-model output or clinical use.

Non-identifying aggregate metrics from the verified runs are recorded under
`aggregate_results/`, including classification, calibration, explanation
faithfulness, BioLeaflet, out-of-fold FLAN-T5, RAG, and verifier-stress
summaries. Per-visit
probabilities, participant assignments, model checkpoints, and ADNI records
are deliberately excluded.

## Data availability

The study data were obtained from the Alzheimer's Disease Neuroimaging
Initiative (ADNI) and remain subject to ADNI data-use requirements. This
repository intentionally excludes data rows and direct participant
identifiers. Only code and non-identifying aggregate summaries should be made
public.
