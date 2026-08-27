# Leakage-aware hybrid graph learning for multiclass Alzheimer’s disease classification with verifier-controlled patient-level interpretation

This repository contains the code used for the experiments reported in the accompanying manuscript. The study uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

## Repository structure

- `hybrid_gnn.py` – main hybrid GNN, nested CV, inductive analysis, and ablations
- `baseline_comparison.py` – conventional machine-learning baselines
- `bio_leaflet.py` – patient-level interpretation and verification
- `ADNIT5.py` – FLAN-T5 training and evaluation
- `run_bioleaflet_rag_prototype.py` – RAG follow-up evaluation
- `experiment_commands.json` – commands and saved arguments for all experiments
- `requirements.txt` – package versions used for the experiments

## Setup

```bash
pip install -r requirements.txt
```

## Experiments and outputs

Exact commands for all experiments are provided in `experiment_commands.json`.

| Experiment | Expected output | Manuscript support |
| --- | --- | --- |
| Main hybrid GNN | `metrics.json` | Table 3 |
| Nested cross-validation | `nested_cv_results.json` | Table 3 |
| Inductive sensitivity analysis | `inductive/inductive_results.json` | Table 3 |
| Machine-learning baselines | `baseline_summary.csv` | Table 4 |
| Architectural ablations | `metrics.json` | Table 5 |
| CDR leakage analyses | experiment summary outputs | Table 6 |
| Patient-level interpretation and verification | `verification_summary.csv` | Table 7 |
| FLAN-T5 evaluation | evaluation summary outputs | Table 7 |
| RAG follow-up evaluation | `rag_followup_summary.json` | Section 4.5 |

## Data availability

The data used in this study were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) and are not distributed through this repository due to ADNI data-use restrictions.
