"""Participant-disjoint out-of-fold evaluation for FLAN-T5 refinement.

The script regenerates safe narrative targets from the corrected BioLeaflet
templates, trains one fresh FLAN-T5 model per outer fold, and produces exactly
one held-out narrative for every visit. Participant identifiers, generated
text, and checkpoints remain in the local output directory. Only the
non-identifying aggregate JSON is suitable for public release.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_MODEL = "google/flan-t5-base"
DEFAULT_PREFIX = (
    "Generate a clinical explanation for an Alzheimer's disease prediction.\n\n"
)
REQUIRED_HEADERS = ("CURRENT STATUS:", "SUMMARY:")
REQUIRED_GROUNDING_SNIPPETS = (
    "this assignment reflects the trained model's use of the supplied multimodal inputs",
    "it does not independently establish cognitive status, biomarker pathology, or a clinical diagnosis",
    "the accompanying sections summarize model attribution, graph context, longitudinal observations, counterfactual sensitivity, and uncertainty",
    "this is an automated research summary, not a clinical diagnosis",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _original_nodes(corpus: pd.DataFrame) -> pd.DataFrame:
    required = {"input", "target", "pred_class", "subject_id", "node_idx", "mode"}
    missing = sorted(required - set(corpus.columns))
    if missing:
        raise ValueError(f"Training corpus is missing columns: {missing}")

    nodes = corpus.loc[corpus["mode"].eq("node")].copy()
    nodes["subject_id"] = nodes["subject_id"].astype(str).str.strip()
    if nodes.empty:
        raise ValueError("Training corpus contains no original node rows")
    if nodes.duplicated(["subject_id", "node_idx"]).any():
        raise ValueError("Original node rows are not unique by participant and node")
    return nodes.reset_index(drop=True)


def make_outer_splits(
    corpus: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> list[dict[str, object]]:
    """Return participant-disjoint outer splits covering each node once."""
    nodes = _original_nodes(corpus)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    splits: list[dict[str, object]] = []
    seen_test_subjects: set[str] = set()
    seen_test_nodes: set[int] = set()

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(nodes, nodes["pred_class"], groups=nodes["subject_id"]),
        start=1,
    ):
        train_nodes = nodes.iloc[train_idx]
        test_nodes = nodes.iloc[test_idx]
        train_subjects = set(train_nodes["subject_id"])
        test_subjects = set(test_nodes["subject_id"])
        if train_subjects & test_subjects:
            raise AssertionError(f"Participant overlap in outer fold {fold}")
        if seen_test_subjects & test_subjects:
            raise AssertionError("A participant appears in more than one outer test fold")

        test_node_ids = set(test_nodes["node_idx"].astype(int))
        if seen_test_nodes & test_node_ids:
            raise AssertionError("A node appears in more than one outer test fold")
        seen_test_subjects.update(test_subjects)
        seen_test_nodes.update(test_node_ids)

        splits.append(
            {
                "fold": fold,
                "train_subjects": sorted(train_subjects),
                "test_subjects": sorted(test_subjects),
                "test_node_indices": test_nodes.index.to_numpy(),
            }
        )

    if seen_test_subjects != set(nodes["subject_id"]):
        raise AssertionError("Outer folds do not cover every participant")
    if seen_test_nodes != set(nodes["node_idx"].astype(int)):
        raise AssertionError("Outer folds do not cover every original node")
    return splits


def _inner_train_validation_subjects(
    outer_train_nodes: pd.DataFrame, fold: int, seed: int
) -> tuple[set[str], set[str]]:
    """Choose a subject-disjoint validation fold with full class coverage."""
    n_subjects = outer_train_nodes["subject_id"].nunique()
    n_splits = min(5, n_subjects)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed + fold
    )
    classes = set(outer_train_nodes["pred_class"].astype(str))
    candidates = []
    global_dist = outer_train_nodes["pred_class"].value_counts(normalize=True)

    for candidate, (train_idx, val_idx) in enumerate(
        splitter.split(
            outer_train_nodes,
            outer_train_nodes["pred_class"],
            groups=outer_train_nodes["subject_id"],
        )
    ):
        train_part = outer_train_nodes.iloc[train_idx]
        val_part = outer_train_nodes.iloc[val_idx]
        val_classes = set(val_part["pred_class"].astype(str))
        val_dist = val_part["pred_class"].value_counts(normalize=True)
        imbalance = sum(
            abs(float(val_dist.get(cls, 0.0)) - float(global_dist.get(cls, 0.0)))
            for cls in classes
        )
        candidates.append(
            (
                0 if val_classes == classes else 1,
                imbalance,
                abs(candidate - ((fold - 1) % n_splits)),
                train_part,
                val_part,
            )
        )

    _, _, _, train_part, val_part = min(candidates, key=lambda item: item[:3])
    train_subjects = set(train_part["subject_id"].astype(str))
    val_subjects = set(val_part["subject_id"].astype(str))
    if train_subjects & val_subjects:
        raise AssertionError(f"Participant overlap in inner fold {fold}")
    return train_subjects, val_subjects


def _tokenize_dataset(frame, tokenizer, max_input_length, max_target_length):
    from datasets import Dataset

    dataset = Dataset.from_pandas(frame[["input", "target"]], preserve_index=False)

    def preprocess(examples):
        model_inputs = tokenizer(
            [DEFAULT_PREFIX + text for text in examples["input"]],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def _generate_batchwise(
    model,
    tokenizer,
    inputs: list[str],
    device: torch.device,
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
) -> list[str]:
    predictions: list[str] = []
    model.eval()
    for start in range(0, len(inputs), batch_size):
        batch = [DEFAULT_PREFIX + text for text in inputs[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_target_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return predictions


def _rouge_rows(predictions: list[str], references: list[str]) -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rows = []
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        rows.append({name: float(score.fmeasure) for name, score in scores.items()})
    return pd.DataFrame(rows)


def _cluster_bootstrap_interval(
    values: np.ndarray,
    subjects: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    unique_subjects = np.unique(subjects)
    row_lookup = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        sampled_rows = np.concatenate([row_lookup[subject] for subject in sampled])
        estimates[index] = float(np.mean(values[sampled_rows]))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _build_context_lookup(results_dir: Path, data_csv: Path):
    from ADNIT5 import get_patient_context, load_gnn_outputs, load_patient_data

    gnn = load_gnn_outputs(results_dir)
    data = load_patient_data(data_csv)
    uncertainty = gnn["uncertainty"].copy()
    uncertainty["subject_id"] = uncertainty["subject_id"].astype(str).str.strip()
    uncertainty = uncertainty.set_index("node_idx", drop=False)

    def lookup(subject_id: str, node_idx: int):
        row = uncertainty.loc[int(node_idx)]
        session_id = row.get("clinical_session_id")
        if pd.isna(session_id):
            session_id = None
        patient = data.loc[data["subject_id"].eq(str(subject_id).strip())]
        if patient.empty:
            raise ValueError(f"No source rows found for participant at node {node_idx}")
        return get_patient_context(
            str(subject_id).strip(),
            int(node_idx),
            gnn,
            patient,
            session_id=None if session_id is None else str(session_id).strip(),
        )

    return lookup


def _guardrail_predictions(
    frame: pd.DataFrame,
    raw_predictions: list[str],
    context_lookup,
) -> tuple[list[str], list[dict[str, object]]]:
    from ADNIT5 import (
        _parse_t5_sections,
        _post_process_narrative,
        _validate_training_target,
    )
    from bio_leaflet import (
        build_stage_interpretation,
        build_summary,
        render_leaflet_with_t5,
        verify_leaflet,
    )

    processed_predictions: list[str] = []
    audits: list[dict[str, object]] = []
    for (_, row), raw_prediction in zip(frame.iterrows(), raw_predictions):
        context = context_lookup(str(row["subject_id"]), int(row["node_idx"]))
        processed = _post_process_narrative(raw_prediction, context)
        stage_display = str(context["pred_class_display"])
        processed = re.sub(
            r"model output:\s*this visit was assigned to the .*? class\.",
            f"Model output: this visit was assigned to the {stage_display} class.",
            processed,
            flags=re.IGNORECASE,
        )
        processed = re.sub(
            r"model output:(?!\s*this visit was assigned)\s*[^.]*\.",
            f"Model output: {stage_display}.",
            processed,
            flags=re.IGNORECASE,
        )
        headers_present = all(header in processed for header in REQUIRED_HEADERS)
        target_ok, target_issues = _validate_training_target(
            str(row["input"]), processed, pred_class=str(row["pred_class"])
        )
        processed_lower = processed.lower()
        semantic_issues = [
            f"MISSING_GROUNDING_STATEMENT_{index}"
            for index, snippet in enumerate(REQUIRED_GROUNDING_SNIPPETS, start=1)
            if snippet not in processed_lower
        ]
        canonical_assignment = (
            f"model output: this visit was assigned to the {stage_display.lower()} class."
        )
        canonical_summary = f"model output: {stage_display.lower()}."
        if canonical_assignment not in processed_lower:
            semantic_issues.append("NONCANONICAL_STAGE_ASSIGNMENT")
        if canonical_summary not in processed_lower:
            semantic_issues.append("NONCANONICAL_STAGE_SUMMARY")
        has_uncertainty_warning = "elevated uncertainty" in processed_lower
        if bool(context.get("is_high_uncertainty")) != has_uncertainty_warning:
            semantic_issues.append("UNCERTAINTY_LANGUAGE_MISMATCH")
        semantic_ok = not semantic_issues
        interpretation, summary = _parse_t5_sections(processed)
        candidate = render_leaflet_with_t5(
            context.copy(),
            t5_stage_interpretation=interpretation,
            t5_summary=summary,
        )
        report_status, report_issues, report_scores = verify_leaflet(candidate, context)
        accepted = bool(
            headers_present and target_ok and semantic_ok and report_status == "PASS"
        )

        if accepted:
            final_report = candidate
            fell_back = False
        else:
            final_report = render_leaflet_with_t5(
                context.copy(),
                t5_stage_interpretation=build_stage_interpretation(context),
                t5_summary=build_summary(context),
            )
            fell_back = True
        final_status, final_issues, _ = verify_leaflet(final_report, context)

        categories = Counter()
        for issue in [*target_issues, *semantic_issues, *report_issues]:
            categories[str(issue).split(":", 1)[0]] += 1
        if not headers_present:
            categories["MISSING_REQUIRED_HEADER"] += 1

        processed_predictions.append(processed)
        audits.append(
            {
                "headers_present": headers_present,
                "target_guardrail_pass": bool(target_ok),
                "semantic_guardrail_pass": semantic_ok,
                "hybrid_report_pass": report_status == "PASS",
                "candidate_accepted": accepted,
                "fell_back_to_template": fell_back,
                "final_report_pass": final_status == "PASS",
                "candidate_fields_checked": int(report_scores.get("fields_checked", 0)),
                "issue_categories": ";".join(
                    f"{key}:{value}" for key, value in sorted(categories.items())
                ),
                "final_issue_count": len(final_issues),
            }
        )
    return processed_predictions, audits


def summarize_oof(
    predictions: pd.DataFrame,
    bootstrap_replicates: int = 10000,
    seed: int = 42,
) -> dict[str, object]:
    required = {
        "subject_id",
        "fold",
        "pred_class",
        "raw_prediction",
        "prediction",
        "target",
        "candidate_accepted",
        "fell_back_to_template",
        "final_report_pass",
        "headers_present",
        "target_guardrail_pass",
        "semantic_guardrail_pass",
        "hybrid_report_pass",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"OOF prediction table is missing columns: {missing}")

    processed_rouge_rows = _rouge_rows(
        predictions["prediction"].astype(str).tolist(),
        predictions["target"].astype(str).tolist(),
    )
    raw_rouge_rows = _rouge_rows(
        predictions["raw_prediction"].astype(str).tolist(),
        predictions["target"].astype(str).tolist(),
    )
    subjects = predictions["subject_id"].astype(str).to_numpy()
    rouge_summary = {}
    for source_index, (name, score_rows) in enumerate(
        (
            ("raw_generation", raw_rouge_rows),
            ("postprocessed_candidate", processed_rouge_rows),
        )
    ):
        rouge_summary[name] = {}
        for metric_index, metric in enumerate(("rouge1", "rouge2", "rougeL")):
            values = score_rows[metric].to_numpy()
            low, high = _cluster_bootstrap_interval(
                values,
                subjects,
                bootstrap_replicates,
                seed + source_index * 10 + metric_index,
            )
            rouge_summary[name][metric] = {
                "mean": round(float(values.mean()), 4),
                "participant_cluster_bootstrap_95_ci": [
                    round(low, 4),
                    round(high, 4),
                ],
            }

    fold_rows = []
    for fold, group in predictions.groupby("fold", sort=True):
        fold_raw_rouge = raw_rouge_rows.loc[group.index]
        fold_processed_rouge = processed_rouge_rows.loc[group.index]
        fold_rows.append(
            {
                "fold": int(fold),
                "participants": int(group["subject_id"].nunique()),
                "visits": int(len(group)),
                "raw_rouge1": round(float(fold_raw_rouge["rouge1"].mean()), 4),
                "raw_rouge2": round(float(fold_raw_rouge["rouge2"].mean()), 4),
                "raw_rougeL": round(float(fold_raw_rouge["rougeL"].mean()), 4),
                "postprocessed_rouge1": round(
                    float(fold_processed_rouge["rouge1"].mean()), 4
                ),
                "postprocessed_rouge2": round(
                    float(fold_processed_rouge["rouge2"].mean()), 4
                ),
                "postprocessed_rougeL": round(
                    float(fold_processed_rouge["rougeL"].mean()), 4
                ),
                "candidate_acceptance_rate": round(
                    float(group["candidate_accepted"].mean()), 4
                ),
            }
        )

    count = len(predictions)
    issue_counts = Counter()
    if "issue_categories" in predictions.columns:
        for encoded in predictions["issue_categories"].fillna("").astype(str):
            for item in encoded.split(";"):
                if not item:
                    continue
                category, count_text = item.rsplit(":", 1)
                issue_counts[category] += int(count_text)
    rejected = predictions.loc[~predictions["candidate_accepted"].astype(bool)]
    rejections_by_class = {
        str(label): int(count)
        for label, count in rejected["pred_class"].value_counts().items()
    }

    return {
        "design": "five-fold participant-disjoint out-of-fold evaluation",
        "n_participants": int(predictions["subject_id"].nunique()),
        "n_visit_reports": int(count),
        "n_folds": int(predictions["fold"].nunique()),
        "all_participants_evaluated_out_of_fold": True,
        "rouge": rouge_summary,
        "guardrails": {
            "required_headers_present": int(predictions["headers_present"].sum()),
            "target_guardrail_pass": int(predictions["target_guardrail_pass"].sum()),
            "semantic_guardrail_pass": int(
                predictions["semantic_guardrail_pass"].sum()
            ),
            "hybrid_candidate_report_pass": int(predictions["hybrid_report_pass"].sum()),
            "candidate_accepted": int(predictions["candidate_accepted"].sum()),
            "candidate_acceptance_rate": round(
                float(predictions["candidate_accepted"].mean()), 4
            ),
            "template_fallbacks": int(predictions["fell_back_to_template"].sum()),
            "template_fallback_rate": round(
                float(predictions["fell_back_to_template"].mean()), 4
            ),
            "final_reports_passing_verification": int(
                predictions["final_report_pass"].sum()
            ),
            "final_report_verification_rate": round(
                float(predictions["final_report_pass"].mean()), 4
            ),
            "candidate_rejection_reasons": dict(sorted(issue_counts.items())),
            "candidate_rejections_by_predicted_class": rejections_by_class,
        },
        "folds": fold_rows,
        "bootstrap_replicates": int(bootstrap_replicates),
        "unit_of_resampling": "participant",
    }


def reverify_saved_outputs(args: argparse.Namespace) -> dict[str, object]:
    """Reapply current guardrails to saved raw OOF generations."""
    out_dir = Path(args.out_dir)
    fold_paths = [
        out_dir / f"restricted_fold_{fold}_predictions.csv" for fold in range(1, 6)
    ]
    missing = [str(path) for path in fold_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Saved OOF fold files are missing: {missing}")

    context_lookup = _build_context_lookup(Path(args.results_dir), Path(args.data_csv))
    refreshed_frames = []
    for path in fold_paths:
        frame = pd.read_csv(path)
        processed, audits = _guardrail_predictions(
            frame, frame["raw_prediction"].astype(str).tolist(), context_lookup
        )
        frame["prediction"] = processed
        audit_frame = pd.DataFrame(audits)
        for column in audit_frame.columns:
            frame[column] = audit_frame[column].to_numpy()
        frame.to_csv(path, index=False)
        refreshed_frames.append(frame)

    predictions = pd.concat(refreshed_frames, ignore_index=True)
    if len(predictions) != 148 or predictions["subject_id"].nunique() != 34:
        raise AssertionError("Saved OOF files do not contain the expected 148 visits")
    if predictions.duplicated(["subject_id", "node_idx"]).any():
        raise AssertionError("Saved OOF files contain duplicate visits")
    predictions.to_csv(out_dir / "restricted_oof_predictions.csv", index=False)

    summary = summarize_oof(
        predictions,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    prior_path = out_dir / "flan_t5_oof_summary.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    for key in ("training", "runtime", "privacy"):
        if key in prior:
            summary[key] = prior[key]
    if "training" in summary:
        fold_details = summary["training"].get("fold_details", [])
        if "runtime" not in summary:
            summary["runtime"] = {}
        summary["runtime"]["total_fold_runtime_seconds"] = round(
            sum(float(detail["runtime_seconds"]) for detail in fold_details), 1
        )
        summary["runtime"].pop("total_seconds", None)
    summary["postprocessing_and_verification"] = {
        "canonical_stage_label_normalization": True,
        "required_grounding_statements_checked": len(REQUIRED_GROUNDING_SNIPPETS),
        "uncertainty_language_consistency_checked": True,
        "failed_candidates_use_deterministic_template_fallback": True,
    }
    prior_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Reverified {len(predictions)} saved OOF visit narratives.")
    print(json.dumps(summary["rouge"], indent=2))
    print(json.dumps(summary["guardrails"], indent=2))
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reverify_saved:
        return reverify_saved_outputs(args)

    from ADNIT5 import generate_training_data
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        __version__ as transformers_version,
    )

    corpus_dir = out_dir / "restricted_training_corpus"
    if args.corpus_csv:
        corpus_path = Path(args.corpus_csv)
    else:
        generate_training_data(
            args.results_dir,
            args.data_csv,
            corpus_dir,
            verbose=False,
            legacy=False,
        )
        corpus_path = corpus_dir / "adni_t5_all.csv"

    corpus = pd.read_csv(corpus_path)
    corpus["subject_id"] = corpus["subject_id"].astype(str).str.strip()
    nodes = _original_nodes(corpus)
    outer_splits = make_outer_splits(corpus, n_splits=5, seed=args.seed)
    context_lookup = _build_context_lookup(Path(args.results_dir), Path(args.data_csv))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )

    selected_splits = outer_splits
    if args.fold is not None:
        selected_splits = [split for split in outer_splits if int(split["fold"]) == args.fold]
        if not selected_splits:
            raise ValueError(f"Requested fold {args.fold} is unavailable")

    all_prediction_frames = []
    training_summaries = []
    run_start = time.time()
    for split in selected_splits:
        fold = int(split["fold"])
        fold_seed = args.seed + fold
        seed_everything(fold_seed)
        outer_test_subjects = set(split["test_subjects"])
        outer_train_nodes = nodes.loc[~nodes["subject_id"].isin(outer_test_subjects)]
        train_subjects, val_subjects = _inner_train_validation_subjects(
            outer_train_nodes, fold, args.seed
        )
        if (train_subjects | val_subjects) & outer_test_subjects:
            raise AssertionError(f"Outer-test leakage in fold {fold}")

        train_frame = corpus.loc[corpus["subject_id"].isin(train_subjects)].copy()
        val_frame = nodes.loc[nodes["subject_id"].isin(val_subjects)].copy()
        test_frame = nodes.loc[nodes["subject_id"].isin(outer_test_subjects)].copy()
        test_frame = test_frame.sort_values(["subject_id", "node_idx"]).reset_index(drop=True)
        if set(train_frame["subject_id"]) & set(test_frame["subject_id"]):
            raise AssertionError(f"Training/test participant leakage in fold {fold}")

        print(
            f"\nFold {fold}/5: train {len(train_subjects)} participants/"
            f"{len(train_frame)} pairs, validation {len(val_subjects)} participants/"
            f"{len(val_frame)} visits, test {len(outer_test_subjects)} participants/"
            f"{len(test_frame)} visits",
            flush=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name, local_files_only=args.local_files_only
        )
        train_dataset = _tokenize_dataset(
            train_frame, tokenizer, args.max_input_length, args.max_target_length
        )
        val_dataset = _tokenize_dataset(
            val_frame, tokenizer, args.max_input_length, args.max_target_length
        )
        fold_dir = out_dir / "restricted_checkpoints" / f"fold_{fold}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(fold_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            label_smoothing_factor=0.1,
            warmup_ratio=0.10,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            bf16=use_bf16,
            fp16=bool(torch.cuda.is_available() and not use_bf16),
            logging_strategy="epoch",
            report_to="none",
            push_to_hub=False,
            seed=fold_seed,
            data_seed=fold_seed,
            disable_tqdm=False,
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        fold_start = time.time()
        train_output = trainer.train()
        best_eval = trainer.evaluate()["eval_loss"]
        raw_predictions = _generate_batchwise(
            trainer.model,
            tokenizer,
            test_frame["input"].astype(str).tolist(),
            trainer.model.device,
            args.generation_batch_size,
            args.max_input_length,
            args.max_target_length,
        )
        processed, audits = _guardrail_predictions(
            test_frame, raw_predictions, context_lookup
        )

        fold_predictions = test_frame[
            ["subject_id", "node_idx", "pred_class", "input", "target"]
        ].copy()
        fold_predictions.insert(0, "fold", fold)
        fold_predictions["raw_prediction"] = raw_predictions
        fold_predictions["prediction"] = processed
        audit_frame = pd.DataFrame(audits)
        fold_predictions = pd.concat(
            [fold_predictions.reset_index(drop=True), audit_frame], axis=1
        )
        all_prediction_frames.append(fold_predictions)
        fold_training_summary = {
            "fold": fold,
            "train_participants": len(train_subjects),
            "validation_participants": len(val_subjects),
            "test_participants": len(outer_test_subjects),
            "train_pairs": len(train_frame),
            "validation_visits": len(val_frame),
            "test_visits": len(test_frame),
            "best_validation_loss": round(float(best_eval), 6),
            "training_loss": round(float(train_output.training_loss), 6),
            "runtime_seconds": round(time.time() - fold_start, 1),
        }
        training_summaries.append(fold_training_summary)
        fold_predictions.to_csv(
            out_dir / f"restricted_fold_{fold}_predictions.csv", index=False
        )
        (out_dir / f"fold_{fold}_training_summary.json").write_text(
            json.dumps(fold_training_summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Fold {fold} complete: validation loss {best_eval:.4f}; "
            f"accepted {int(audit_frame['candidate_accepted'].sum())}/"
            f"{len(audit_frame)} generated candidates",
            flush=True,
        )

        del (
            trainer,
            model,
            collator,
            training_args,
            train_output,
            train_dataset,
            val_dataset,
            raw_predictions,
            processed,
            audits,
            audit_frame,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.keep_checkpoints and fold_dir.exists():
            shutil.rmtree(fold_dir)

    fold_prediction_paths = [
        out_dir / f"restricted_fold_{fold}_predictions.csv" for fold in range(1, 6)
    ]
    available_paths = [path for path in fold_prediction_paths if path.exists()]
    if len(available_paths) < 5:
        print(
            f"Saved {len(available_paths)}/5 completed fold files. "
            "Run the remaining folds to create the aggregate summary."
        )
        return {
            "status": "partial",
            "completed_folds": [
                int(path.stem.split("_")[2]) for path in available_paths
            ],
        }

    predictions = pd.concat(
        [pd.read_csv(path) for path in fold_prediction_paths], ignore_index=True
    )
    if len(predictions) != len(nodes):
        raise AssertionError("OOF predictions do not cover all original visits")
    if predictions.duplicated(["subject_id", "node_idx"]).any():
        raise AssertionError("OOF prediction table contains duplicate visits")

    predictions_path = out_dir / "restricted_oof_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    summary = summarize_oof(
        predictions,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    training_summary_paths = [
        out_dir / f"fold_{fold}_training_summary.json" for fold in range(1, 6)
    ]
    completed_training_summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in training_summary_paths
    ]
    summary["training"] = {
        "model": args.model_name,
        "base_model_reinitialized_each_fold": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 2,
        "learning_rate": args.learning_rate,
        "max_input_tokens": args.max_input_length,
        "max_target_tokens": args.max_target_length,
        "training_augmentation": "node paraphrases plus threefold subject summaries",
        "validation_and_test_modes": "original visit targets only",
        "fold_details": completed_training_summaries,
    }
    summary["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_fold_runtime_seconds": round(
            sum(float(detail["runtime_seconds"]) for detail in completed_training_summaries),
            1,
        ),
    }
    summary["privacy"] = {
        "aggregate_contains_participant_identifiers": False,
        "predictions_and_checkpoints_are_restricted_local_artifacts": True,
    }
    summary_path = out_dir / "flan_t5_oof_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nOOF predictions: {predictions_path}")
    print(f"Aggregate summary: {summary_path}")
    print(json.dumps(summary["rouge"], indent=2))
    print(json.dumps(summary["guardrails"], indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--corpus_csv", default=None)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_input_length", type=int, default=416)
    parser.add_argument("--max_target_length", type=int, default=224)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--reverify_saved", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--keep_checkpoints", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
