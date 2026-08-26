#!/usr/bin/env python3
"""
FLAN-T5 module for verifier-controlled patient-level interpretation.

Fine-tunes google/flan-t5-base to refine the Interpretation and Summary
sections from structured GNN outputs. Quantitative and model-derived
information remains template-controlled.

Usage:
    # Fine-tune FLAN-T5
    python ADNIT5.py train \
        --results_dir <gnn_results> \
        --data_csv <authorized_data.csv> \
        --output_dir <model_output>

    # Generate patient-level interpretations
    python ADNIT5.py generate \
        --results_dir <gnn_results> \
        --data_csv <authorized_data.csv> \
        --model_path <model_checkpoint> \
        --out_dir <output_dir>

    # Evaluate the fine-tuned model
    python ADNIT5.py evaluate \
        --results_dir <gnn_results> \
        --data_csv <authorized_data.csv> \
        --model_path <model_checkpoint> \
        --out_dir <evaluation_output>
"""

import argparse
import json
import re
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

def _ensure_hf_login():
    """Log in to HuggingFace Hub (only when model access is needed)."""
    from huggingface_hub import login as _hf_login
    _hf_token = os.environ.get("HF_TOKEN", None)
    if _hf_token:
        _hf_login(token=_hf_token)
    else:
        _hf_login()

# Import shared utilities from the Jinja2 template system
sys.path.insert(0, str(Path(__file__).parent))
from bio_leaflet import (
    load_gnn_outputs,
    load_patient_data,
    get_patient_context,
    get_top_features,
    compute_deltas,
    generate_leaflet,
    render_leaflet,
    render_leaflet_with_t5,
    render_key_biomarker_values,
    extract_narrative_sections,
    build_stage_interpretation,
    build_longitudinal_text,
    build_risk_context,
    build_summary,
    FEATURE_DISPLAY,
    STAGE_DISPLAY,
    FIXED_FEATURES,
    safe_fmt,
    display_name,
    sex_label,
    apoe4_label,
    session_to_month,
    certainty_label,
    format_delta,
    build_neighborhood_interpretation,
    build_biomarker_interpretation,
    build_counterfactual_interpretation,
    build_uncertainty_interpretation,
    build_progression_narrative,
    confidence_qualifier,
    clinical_magnitude,
)

# Constants

PREFIX = "Generate a clinical explanation for an Alzheimer's disease prediction.\n\n"

MODEL_NAME = "google/flan-t5-base"

GENERATION_MODES = {
    "deterministic": {
        "num_beams": 4,
        "do_sample": False,
        "early_stopping": True,
    },
    "conservative": {
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40,
        "num_beams": 1,
    },
    "balanced": {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 50,
        "num_beams": 1,
    },
    "creative": {
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 60,
        "num_beams": 1,
    },
}

# Short display names for the structured input (ADNI column → readable tag)
INPUT_FIELD_TAGS = {
    "clinical_MMSCORE": "MMSE",
    "clinical_FAQTOTAL": "FAQ",
    "clinical_LDELTOTAL": "LDEL",
    "clinical_TRABSCOR": "TrailB",
    "mri_hippocampus_vol_mean": "Hippocampus",
    "mri_entorhinal_vol_mean": "Entorhinal",
    "mri_amygdala_vol_mean": "Amygdala",
    "mri_inferior_temporal_vol_mean": "InfTemporal",
    "mri_middle_temporal_vol_mean": "MidTemporal",
    "mri_lateral_ventricle_vol_mean": "Ventricles",
    "mri_inf_lat_vent_vol_mean": "InfLatVent",
    "clinical_ABETA42": "CSF-Abeta42",
    "pet_PTAU": "CSF-PTau",       # CSF biomarker despite pet_ prefix
    "pet_TAU": "CSF-Tau",         # CSF biomarker despite pet_ prefix
}

# Class ordering for progression analysis (CN < MCI < AD)
_CLASS_ORDER = {"0": 0, "0.5": 1, "1+": 2}
_CLASS_SHORT = {"0": "CN", "0.5": "MCI", "1+": "AD"}


# SUBJECT-LEVEL AGGREGATION HELPERS

def gather_subject_nodes(subject_id, gnn, data_df):
    """Collect all nodes for a subject, build per-node contexts, sort by visit.

    Prefers explicit session matching via clinical_session_id in the
    uncertainty CSV; falls back to node_idx as a row index into data_df
    for older result files without that column.

    Returns ctx dicts sorted chronologically, each with added visit_month
    and visit_session keys.
    """
    unc_df = gnn["uncertainty"]
    rows = unc_df[unc_df["subject_id"].str.strip() == subject_id]
    if rows.empty:
        return []

    patient_df = data_df[data_df["subject_id"] == subject_id]
    if patient_df.empty:
        return []

    has_session_col = "clinical_session_id" in unc_df.columns

    # Build (node_idx, session_id) pairs with explicit session resolution
    node_session_pairs = []
    for _, unc_row in rows.iterrows():
        node_idx = int(unc_row["node_idx"])
        if has_session_col and pd.notna(unc_row.get("clinical_session_id")):
            sess = str(unc_row["clinical_session_id"]).strip()
        else:
            # Fallback: use node_idx as row index into the original DataFrame
            if node_idx < len(data_df) and "clinical_session_id" in data_df.columns:
                sess = str(data_df.iloc[node_idx]["clinical_session_id"]).strip()
            else:
                sess = None
        node_session_pairs.append((node_idx, sess))

    # Match each node to a visit row in patient_df and resolve month
    node_contexts = []
    pdf = patient_df.copy()
    pdf["_vm"] = pdf["clinical_session_id"].apply(session_to_month)

    for node_idx, sess in node_session_pairs:
        # Determine the visit month for sorting
        if sess is not None:
            match = pdf[pdf["clinical_session_id"].str.strip() == sess]
            visit_month = int(match.iloc[0]["_vm"]) if not match.empty else 0
        else:
            visit_month = 0

        try:
            ctx = get_patient_context(
                subject_id, node_idx, gnn, patient_df, session_id=sess,
            )
        except Exception:
            continue
        ctx["node_idx"] = node_idx
        ctx["visit_month"] = visit_month
        ctx["visit_session"] = sess if sess is not None else "unknown"
        node_contexts.append(ctx)

    node_contexts.sort(key=lambda c: c["visit_month"])

    return node_contexts


def compute_prediction_stability(node_contexts):
    """Analyse whether predicted class stayed stable or shifted across visits.

    Returns dict with keys: stability_label, stability_detail,
    first_class, last_class, transitions, unique_classes.
    """
    classes = [ctx["pred_class"] for ctx in node_contexts]
    months = [ctx["visit_month"] for ctx in node_contexts]
    unique = set(classes)

    first_cls, last_cls = classes[0], classes[-1]
    n = len(classes)

    transitions = []
    for i in range(1, n):
        if classes[i] != classes[i - 1]:
            transitions.append((
                classes[i - 1], classes[i], months[i - 1], months[i],
            ))

    if len(unique) == 1:
        label = "Stable"
        detail = (
            f"Stable {_CLASS_SHORT.get(first_cls, first_cls)} across "
            f"{n} visits ({months[-1] - months[0]} months)"
        )
    else:
        orders = [_CLASS_ORDER.get(c, -1) for c in classes]
        if all(orders[i] <= orders[i + 1] for i in range(n - 1)):
            label = "Progressed"
        elif all(orders[i] >= orders[i + 1] for i in range(n - 1)):
            label = "Improved"
        else:
            label = "Fluctuating"

        parts = []
        for fc, tc, fm, tm in transitions:
            parts.append(
                f"{_CLASS_SHORT.get(fc, fc)} -> "
                f"{_CLASS_SHORT.get(tc, tc)} (m{fm}->m{tm})"
            )
        detail = f"{label}: {' | '.join(parts)}"

    return {
        "stability_label": label,
        "stability_detail": detail,
        "first_class": first_cls,
        "last_class": last_cls,
        "transitions": transitions,
        "unique_classes": unique,
    }


def build_progression_timeline(node_contexts):
    """Render a compact visit-by-visit table of predictions."""
    header = (
        f"  {'Visit':<8}| {'Month':>5} | {'Prediction':<32}| "
        f"{'Confidence':>10} | Certainty"
    )
    sep = "  " + "-" * 8 + "|" + "-" * 7 + "|" + "-" * 32 + "|" + "-" * 12 + "|" + "-" * 12
    rows = [header, sep]
    for ctx in node_contexts:
        sess = ctx.get("visit_session", "?")
        month = ctx.get("visit_month", "?")
        pred = ctx.get("pred_class_display", "?")
        conf = ctx.get("confidence_pct", "?")
        cert = ctx.get("certainty_label", "?")
        rows.append(
            f"  {sess:<8}| {str(month):>5} | {pred:<32}| "
            f"{str(conf) + '%':>10} | {cert}"
        )
    return "\n".join(rows)


def build_subject_context(subject_id, gnn, data_df):
    """Aggregate all nodes for a subject into a single subject-level context.

    Returns the latest node's ctx enriched with progression, timeline,
    and stability information, or None if no nodes found.
    """
    node_contexts = gather_subject_nodes(subject_id, gnn, data_df)
    if not node_contexts:
        return None

    # Latest node is the base context
    subj_ctx = node_contexts[-1].copy()

    # Aggregated fields
    subj_ctx["node_contexts"] = node_contexts
    subj_ctx["n_nodes"] = len(node_contexts)
    subj_ctx["all_node_indices"] = [ctx["node_idx"] for ctx in node_contexts]
    subj_ctx["latest_node_idx"] = node_contexts[-1]["node_idx"]
    subj_ctx["progression_timeline"] = build_progression_timeline(node_contexts)
    subj_ctx["stability"] = compute_prediction_stability(node_contexts)

    confidences = []
    n_high = 0
    for ctx in node_contexts:
        try:
            confidences.append(float(ctx["confidence_pct"]))
        except (ValueError, TypeError):
            pass
        if ctx.get("is_high_uncertainty"):
            n_high += 1

    subj_ctx["avg_confidence"] = (
        round(sum(confidences) / len(confidences), 1) if confidences else "N/A"
    )
    subj_ctx["min_confidence"] = (
        round(min(confidences), 1) if confidences else "N/A"
    )
    subj_ctx["max_confidence"] = (
        round(max(confidences), 1) if confidences else "N/A"
    )
    subj_ctx["any_high_uncertainty"] = n_high > 0
    subj_ctx["n_high_uncertainty_visits"] = n_high

    return subj_ctx


#  BUILD STRUCTURED INPUT (the T5 input representation)

def build_structured_input(ctx, subject_mode=False):
    """Convert a patient context dict into the compact structured text
    format that T5 will be trained on.

    Parameters
    ----------
    ctx : dict
        Patient context (single-node) or subject-level context (from
        build_subject_context) when subject_mode=True.
    subject_mode : bool
        When True, includes STABILITY, TIMELINE, and multi-visit FLAG
        lines derived from subject-level aggregation.

    Returns
    -------
    str
        Structured input text for T5.
    """
    lines = []

    # --- DEMOGRAPHICS ---
    demo_parts = [ctx["patient_id"]]
    if ctx["age"] != "N/A":
        demo_parts.append(f"{ctx['age']}y")
    demo_parts.append(ctx["sex"])
    demo_parts.append(f"APOE4:{ctx['apoe4_status']}")
    if ctx["education_years"] != "N/A":
        demo_parts.append(f"Education:{ctx['education_years']}y")
    demo_parts.append(f"Visits:{ctx['n_visits']}")
    lines.append("DEMOGRAPHICS: " + " | ".join(demo_parts))
    lines.append("")

    # --- COGNITIVE ---
    cog_parts = []
    if ctx["mmse"] != "N/A":
        cog_parts.append(f"MMSE:{ctx['mmse']}/30")
    if ctx["faq"] != "N/A":
        cog_parts.append(f"FAQ:{ctx['faq']}")
    if ctx["ldel"] != "N/A":
        cog_parts.append(f"LDEL:{ctx['ldel']}")
    if ctx["trails_b"] != "N/A":
        cog_parts.append(f"TrailB:{ctx['trails_b']}s")
    if cog_parts:
        cdr_val = ctx.get("pred_class", "?")
        lines.append(f"COGNITIVE (Dx {cdr_val}): " + " | ".join(cog_parts))
        lines.append("")

    # --- MRI ---
    mri_parts = []
    for field, tag in [
        ("hippo_vol", "Hippocampus"),
        ("entorhinal_vol", "Entorhinal"),
        ("amygdala_vol", "Amygdala"),
        ("ventricle_vol", "Ventricles"),
    ]:
        val = ctx.get(field, "N/A")
        if val != "N/A":
            mri_parts.append(f"{tag}:{val} mm³")
    if mri_parts:
        lines.append("MRI: " + " | ".join(mri_parts))
        lines.append("")

    # --- CSF biomarkers (pet_PTAU/pet_TAU are CSF despite prefix) ---
    csf_parts = []
    for field, tag in [("abeta42", "Abeta42"), ("ptau", "PTau"), ("tau", "Tau")]:
        val = ctx.get(field, "N/A")
        if val != "N/A":
            csf_parts.append(f"{tag}:{val}")
    if csf_parts:
        lines.append("CSF: " + " | ".join(csf_parts))
        lines.append("")

    # --- PREDICTION ---
    if subject_mode:
        lines.append(
            f"PREDICTION (latest): {ctx['pred_class_display']} | "
            f"confidence:{ctx['confidence_pct']}% | "
            f"certainty:{ctx['certainty_label']}"
        )
    else:
        lines.append(
            f"PREDICTION: {ctx['pred_class_display']} | "
            f"confidence:{ctx['confidence_pct']}% | "
            f"certainty:{ctx['certainty_label']}"
        )

    # --- SUBJECT-LEVEL: stability + timeline ---
    if subject_mode and "stability" in ctx:
        stab = ctx["stability"]
        lines.append("")
        lines.append(f"STABILITY: {stab['stability_detail']}")

        # Compact timeline: m0:CN(69%) | m6:MCI(61%) | ...
        tl_parts = []
        for nc in ctx.get("node_contexts", []):
            sess = nc.get("visit_session", "?")
            cls_short = _CLASS_SHORT.get(nc["pred_class"], nc["pred_class"])
            conf = nc.get("confidence_pct", "?")
            tl_parts.append(f"{sess}:{cls_short}({conf}%)")
        lines.append("TIMELINE: " + " | ".join(tl_parts))

        # Flag for high-uncertainty visits
        n_high = ctx.get("n_high_uncertainty_visits", 0)
        n_nodes = ctx.get("n_nodes", 0)
        if n_high > 0:
            lines.append(f"FLAG: {n_high}/{n_nodes} visits had HIGH_UNCERTAINTY")
    else:
        if ctx["is_high_uncertainty"]:
            lines.append("FLAG: HIGH_UNCERTAINTY")

    lines.append("")

    # --- LONGITUDINAL CHANGES (key clinical features only) ---
    deltas = ctx.get("deltas", {})
    if deltas:
        change_parts = []
        # Prioritize key clinical features
        priority = [
            "clinical_MMSCORE", "clinical_FAQTOTAL", "clinical_LDELTOTAL",
            "clinical_TRABSCOR", "mri_hippocampus_vol_mean",
            "mri_entorhinal_vol_mean", "mri_lateral_ventricle_vol_mean",
        ]
        for feat in priority:
            if feat in deltas:
                tag = INPUT_FIELD_TAGS.get(feat, feat)
                change_parts.append(f"{tag}:{format_delta(deltas[feat])}")
        if change_parts:
            lines.append("CHANGES: " + " | ".join(change_parts))

    return "\n".join(lines).strip()


#  BUILD NARRATIVE-ONLY TARGET (what T5 is trained to generate)

def _sanitize_narrative_numerics(text):
    """Strip biomarker and demographic numbers from narrative text.

    T5 should only produce qualitative language — exact numbers belong in
    the template-rendered sections. This removes any numeric values that
    leaked through the narrative builders.
    """
    # --- Cognitive / clinical scores ---
    # MMSE scores like "27/30" or "27 / 30"
    text = re.sub(r"\b\d{1,2}\s*/\s*30\b", "", text)
    # FAQ scores like "FAQ: 15" or "FAQ of 12"
    text = re.sub(r"FAQ[\s:]+\d+\.?\d*", "FAQ", text)
    # LDEL / logical memory scores
    text = re.sub(r"LDEL(?:TOTAL)?[\s:]+\d+\.?\d*", "LDEL", text)
    # Trail A / B times like "102s", "102 s", "102 seconds"
    text = re.sub(r"\b\d{2,4}\s*(?:seconds?|s)\b", "", text)

    # --- Imaging / biomarker values ---
    # MRI volumes like "3200 mm³" or "3200.5 mm³"
    text = re.sub(r"\b\d{3,}\.?\d*\s*mm[³3]?\b", "", text)
    # CSF values like "210 pg/mL"
    text = re.sub(r"\b\d+\.?\d*\s*pg/mL\b", "", text)
    # PET SUVR values like "1.23" near "SUVR" or "uptake"
    # NOTE: ADNI dataset currently has no PET SUVR features (pet_PTAU/TAU
    # are CSF biomarkers despite the prefix). Kept for future-proofing.
    text = re.sub(r"SUVR[\s:]*\d+\.\d+", "SUVR", text)

    # --- Demographics ---
    # Age like "72.3 years" or "72 year"
    text = re.sub(r"\b\d{2,3}\.?\d*\s*years?\b", "", text)
    # Education years near "education" or "cognitive reserve"
    text = re.sub(
        r"(?i)(education|cognitive reserve)[:\s]*\d{1,2}\s*years?",
        r"\1", text,
    )

    # --- Confidence / percentages ---
    # Standalone percentages like "78.5%" or "(65.2% model confidence)"
    text = re.sub(r"\b\d{1,3}\.?\d*%", "", text)
    # Parenthetical percentage phrases "(78% model confidence)"
    text = re.sub(r"\([^)]*\d+\.?\d*%[^)]*\)", "", text)

    # --- Visit counts / time spans ---
    # "from 4 visits spanning 36 months"
    text = re.sub(
        r"(?:from\s+)?\d+\s+visits?\s+spanning\s+\d+\s+months?",
        "across multiple visits", text,
    )
    # Standalone "N visits" or "N months"
    text = re.sub(r"\b\d+\s+visits?\b", "multiple visits", text)
    text = re.sub(r"\b\d+\s+months?\b", "the follow-up period", text)

    # --- Cleanup ---
    text = re.sub(r"  +", " ", text)          # double spaces
    text = re.sub(r"\(\s*\)", "", text)        # empty parentheticals
    text = re.sub(r"\s+([.,;:])", r"\1", text) # space before punctuation
    text = re.sub(r":\s*,", ":", text)         # ": ," artefact
    return text.strip()


# Biomarker-format patterns that should never appear in a training target.
_RESIDUAL_NUMERIC_PATTERNS = [
    re.compile(r"\b\d{1,2}\s*/\s*30\b"),          # MMSE scores
    re.compile(r"\b\d{3,}\.?\d*\s*mm[³3]?\b"),    # MRI volumes
    re.compile(r"\b\d+\.?\d*\s*pg/mL\b"),          # CSF values
    re.compile(r"\b\d{2,3}\.?\d*\s*years?\b"),     # Age values
    re.compile(r"\b\d{1,3}\.?\d*%"),               # Percentages
    re.compile(r"SUVR[\s:]*\d+\.\d+"),             # PET SUVR
]

# Phrases that should never appear in a research-grade narrative.
_FORBIDDEN_PHRASES = [
    "diagnosed with alzheimer", "confirmed dementia",
    "definitive diagnosis", "alzheimer's confirmed",
    "clinical diagnosis of", "we diagnose",
    "patient has ad", "patient has alzheimer",
    "we recommend", "you should", "prescribe",
    "treatment plan", "medication", "100% confidence",
    "certain diagnosis", "definitely has",
]

# Stage-inappropriate language maps: pred_class → forbidden terms
_STAGE_FORBIDDEN_LANGUAGE = {
    "0": [
        "dementia-stage", "alzheimer's disease", "ad-level",
        "severe impairment", "severe atrophy", "advanced neurodegeneration",
        "progressive impairment", "marked deterioration",
    ],
    "1+": [
        "normal aging", "cognitively intact", "no impairment",
        "healthy brain", "cognitively healthy", "normal range",
        "fully preserved",
    ],
    "0.5": [
        "end-stage", "terminal decline",
    ],
}


def _validate_training_target(input_text, target_text, pred_class=None):
    """Validate that a training target is free of leaked numerics and
    forbidden language before it reaches the model.

    pred_class ("0", "0.5", "1+") drives the stage-language check.
    Returns (is_valid, issues).
    """
    issues = []
    target_lower = target_text.lower()

    #  Check for residual biomarker numerics
    for pat in _RESIDUAL_NUMERIC_PATTERNS:
        match = pat.search(target_text)
        if match:
            issues.append(f"RESIDUAL_NUMERIC: '{match.group()}' in target")

    #  Check for forbidden diagnostic phrases
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in target_lower:
            issues.append(f"FORBIDDEN_PHRASE: '{phrase}'")

    # Stage-language consistency
    if pred_class and pred_class in _STAGE_FORBIDDEN_LANGUAGE:
        for term in _STAGE_FORBIDDEN_LANGUAGE[pred_class]:
            if term in target_lower:
                issues.append(
                    f"STAGE_MISMATCH: '{term}' inappropriate for class {pred_class}"
                )

    #  Minimum length sanity check (narrative should be substantive)
    if len(target_text.split()) < 30:
        issues.append("TOO_SHORT: target has fewer than 30 words")

    return (len(issues) == 0, issues)


def build_narrative_target(ctx, subject_mode=False, tone=None):
    """Build the training target: CURRENT STATUS (+ PROGRESSION if
    subject_mode) and SUMMARY sections, numeric-sanitized so T5 learns
    qualitative language only. tone="accessible" applies jargon
    simplification via _apply_tone() for extra training diversity.
    """
    # Reuse the same narrative builders from bio_leaflet.py so that
    # the training target is identical to what we expect at inference.
    stage_interp = build_stage_interpretation(ctx)

    sections = [
        "CURRENT STATUS:",
        stage_interp,
        "",
    ]

    if subject_mode and "stability" in ctx:
        sections.extend([
            "PROGRESSION:",
            build_progression_narrative(ctx),
            "",
        ])

    # SUMMARY section — T5 learns to generate this too
    sections.extend([
        "SUMMARY:",
        build_summary(ctx),
        "",
    ])

    raw = "\n".join(sections)
    sanitized = _sanitize_narrative_numerics(raw)

    if tone == "accessible":
        from bio_leaflet import _apply_tone
        sanitized = _apply_tone(sanitized)

    return sanitized


# Deterministic paraphrase variants for training-data diversity

# Synonym pairs: (original, replacement). Applied in order.
_SYNONYM_PAIRS = [
    ("observed", "noted"),
    ("consistent with", "compatible with"),
    ("warrant monitoring", "merit continued follow-up"),
    ("suggestive of", "indicative of"),
    ("decline in", "reduction in"),
    ("relatively preserved", "largely maintained"),
    ("significant impairment", "notable impairment"),
    ("cognitive profile", "cognitive pattern"),
    ("brain regions", "neural structures"),
    ("atrophy", "volume loss"),
    ("elevated levels", "increased levels"),
    ("within normal limits", "in the expected range"),
    ("trajectory", "trend"),
    ("preserved", "intact"),
    ("impairment", "deficit"),
    ("monitoring", "follow-up"),
    ("pathology", "disease process"),
    ("neurodegeneration", "neural decline"),
    ("biomarkers", "biological markers"),
    ("findings", "results"),
]

# Reverse synonyms (swap direction)
_SYNONYM_PAIRS_REV = [(b, a) for a, b in _SYNONYM_PAIRS]


def _paraphrase_target(text, variant_idx):
    """Create a deterministic paraphrase variant of a training target.

    variant_idx: 0 = identity, 1 = forward synonym substitution,
    2 = reverse synonyms + within-section sentence reorder.
    """
    if variant_idx == 0:
        return text

    # --- Variant 1: forward synonym swaps ---
    if variant_idx == 1:
        result = text
        for orig, repl in _SYNONYM_PAIRS:
            result = re.sub(re.escape(orig), repl, result, flags=re.IGNORECASE)
        return result

    # --- Variant 2: reverse synonyms + sentence reorder within sections ---
    if variant_idx == 2:
        result = text
        for orig, repl in _SYNONYM_PAIRS_REV:
            result = re.sub(re.escape(orig), repl, result, flags=re.IGNORECASE)

        # Reorder sentences within each section (between section headers)
        section_headers = [
            "CURRENT STATUS:", "PROGRESSION:", "LONGITUDINAL CONTEXT:",
            "RISK CONTEXT:", "SUMMARY:",
        ]
        # Split into (header, body) blocks
        blocks = re.split(
            r"^(" + "|".join(re.escape(h) for h in section_headers) + r")\s*$",
            result, flags=re.MULTILINE,
        )
        rebuilt = []
        i = 0
        while i < len(blocks):
            chunk = blocks[i]
            if chunk.strip() in section_headers:
                # This is a header; next block is the body
                rebuilt.append(chunk)
                if i + 1 < len(blocks):
                    body = blocks[i + 1]
                    # Split body into sentences, reverse order
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', body.strip()) if s.strip()]
                    if len(sentences) > 1:
                        sentences.reverse()
                    rebuilt.append("\n" + " ".join(sentences) + "\n")
                    i += 2
                else:
                    i += 1
            else:
                rebuilt.append(chunk)
                i += 1
        return "".join(rebuilt)

    return text  # fallback


# Full-leaflet training target 

def _strip_decorators(text):
    text = re.sub(r'^[=━]{10,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Generated by Bio-Leaflet System.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_full_leaflet_target(ctx):  
    """Deprecated: use build_narrative_target() instead.
    """
    from bio_leaflet import render_leaflet
    leaflet = render_leaflet(ctx)
    return _strip_decorators(leaflet)


#  RENDER KEY BIOMARKER VALUES + GNN TEMPLATE SECTIONS

def render_key_biomarker_values(ctx):  # DEPRECATED — now in bio_leaflet.py
    """Deprecated: use render_key_biomarker_values() from bio_leaflet.py.
    """
    lines = ["KEY BIOMARKER VALUES :"]

    # --- Cognitive ---
    if ctx.get("mmse") != "N/A":
        lines.append(f"  Global cognition (MMSE):           {ctx['mmse']} / 30")
    if ctx.get("faq") != "N/A":
        lines.append(f"  Functional assessment (FAQ):        {ctx['faq']}")
    if ctx.get("ldel") != "N/A":
        lines.append(f"  Logical memory (delayed recall):    {ctx['ldel']}")
    if ctx.get("trails_b") != "N/A":
        lines.append(f"  Executive function (Trail Making B): {ctx['trails_b']} s")

    # --- Structural MRI ---
    if ctx.get("hippo_vol") != "N/A":
        lines.append(f"  Hippocampal volume (mean):          {ctx['hippo_vol']} mm\u00b3")
    if ctx.get("entorhinal_vol") != "N/A":
        lines.append(f"  Entorhinal cortex volume:           {ctx['entorhinal_vol']} mm\u00b3")
    if ctx.get("amygdala_vol") != "N/A":
        lines.append(f"  Amygdala volume:                    {ctx['amygdala_vol']} mm\u00b3")
    if ctx.get("ventricle_vol") != "N/A":
        lines.append(f"  Lateral ventricle volume:           {ctx['ventricle_vol']} mm\u00b3")

    # --- CSF biomarkers ---
    if ctx.get("abeta42") != "N/A":
        lines.append(f"  CSF Abeta42:                        {ctx['abeta42']} pg/mL")
    if ctx.get("ptau") != "N/A":
        lines.append(f"  CSF Phosphorylated-Tau:             {ctx['ptau']} pg/mL")
    if ctx.get("tau") != "N/A":
        lines.append(f"  CSF Total-Tau:                      {ctx['tau']} pg/mL")

    # Only return block if we have at least one value
    if len(lines) <= 1:
        return ""

    lines.append("")
    return "\n".join(lines)


def render_gnn_sections(ctx):  # DEPRECATED — use render_leaflet_with_t5() from bio_leaflet.py
    """Render the GNN-specific sections as clinically readable text blocks.

    Deprecated — replaced by render_leaflet_with_t5() in bio_leaflet.py.
    """
    is_subject = "progression_timeline" in ctx
    sections = []

    # Prediction (with confidence folded in)
    sections.append("PREDICTION :")
    if is_subject:
        sections.append(f"  Predicted stage (latest):  {ctx['pred_class_display']}")
        sections.append(f"  Model confidence (latest): {ctx['confidence_pct']}%")
    else:
        sections.append(f"  Predicted stage:       {ctx['pred_class_display']}")
        sections.append(f"  Model confidence:      {ctx['confidence_pct']}%")
    sections.append(f"  Prediction reliability: {ctx['certainty_label']}")
    qual = ctx.get("confidence_qualifier", confidence_qualifier(ctx))
    sections.append(f"  Overall confidence:    {qual}")
    if is_subject:
        stab = ctx["stability"]
        sections.append(f"  Prediction stability:  {stab['stability_detail']}")
    if ctx["is_high_uncertainty"]:
        sections.append(
            "  [!] This prediction carries elevated uncertainty. The model\n"
            "  shows notable disagreement — additional clinical evaluation\n"
            "  is recommended before acting on this classification."
        )
    interp = build_uncertainty_interpretation(ctx)
    if interp:
        sections.append(f"  {interp}")
    if is_subject and ctx.get("n_high_uncertainty_visits", 0) > 0:
        sections.append(
            f"  {ctx['n_high_uncertainty_visits']} of {ctx['n_nodes']} "
            f"visits flagged with elevated uncertainty."
        )
    sections.append("")

    # Visit Progression (subject-level only)
    if is_subject:
        sections.append("VISIT PROGRESSION :")
        sections.append(ctx["progression_timeline"])
        sections.append("")

    # Key Clinical Features (ranked, no importance scores)
    sections.append("KEY CLINICAL FEATURES :")
    for i, feat in enumerate(ctx["top_features"], 1):
        sections.append(f"  {i}. {feat['display_name']}: {feat['value']}")
    interp = build_biomarker_interpretation(ctx)
    if interp:
        sections.append(f"  {interp}")
    sections.append("")

    # Graph Context (qualitative only)
    sections.append("GRAPH CONTEXT :")
    sections.append(
        f"  Among patients with the most similar clinical profiles, the "
        f"dominant classification is {ctx['dominant_neighbor_class']}."
    )
    sections.append(f"  Agreement with this patient's prediction: {ctx['agreement_label']}.")
    interp = build_neighborhood_interpretation(ctx)
    if interp:
        sections.append(f"  {interp}")
    sections.append("")

    # What-If Analysis (clinical magnitude, not SD)
    if ctx.get("has_counterfactual"):
        sections.append("WHAT-IF ANALYSIS :")
        sections.append(
            f"  To shift the prediction from {ctx['pred_class_display']} "
            f"toward {ctx['cf_class_display']}:"
        )
        for ch in ctx["cf_changes"]:
            mag_label = ch.get("clinical_magnitude", clinical_magnitude(ch.get("magnitude")))
            fixed_note = " (non-modifiable)" if ch.get("is_fixed") else ""
            sections.append(
                f"  • {ch['feature_display']}: {ch['direction']} by "
                f"{mag_label}{fixed_note}"
            )
        interp = build_counterfactual_interpretation(ctx)
        if interp:
            sections.append(f"  {interp}")
        sections.append("")

    return "\n".join(sections)


#  TRAINING DATA GENERATION

def generate_training_data(results_dir, data_csv, out_dir,
                           verbose=False, legacy=False):
    """Generate (structured_input, bio_leaflet_target) pairs for T5 training.

    Default mode (subject-level): generates one pair per unique subject
    plus per-node pairs as augmentation. Subject-level pairs are upsampled
    3x to balance against the more numerous per-node pairs.

    Legacy mode: generates one pair per node 

   
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gnn = load_gnn_outputs(results_dir)
    data_df = load_patient_data(data_csv)
    unc_df = gnn["uncertainty"]

    rows = []
    skipped = 0

    has_session_col = "clinical_session_id" in unc_df.columns

    # --- Per-node pairs (used in both modes) ---
    print(f"Building per-node training pairs from {len(unc_df)} nodes...")
    for _, unc_row in unc_df.iterrows():
        node_idx = int(unc_row["node_idx"])
        subject_id = str(unc_row["subject_id"]).strip()

        # Resolve session_id for this node
        if has_session_col and pd.notna(unc_row.get("clinical_session_id")):
            sess = str(unc_row["clinical_session_id"]).strip()
        elif node_idx < len(data_df) and "clinical_session_id" in data_df.columns:
            sess = str(data_df.iloc[node_idx]["clinical_session_id"]).strip()
        else:
            sess = None

        patient_df = data_df[data_df["subject_id"] == subject_id]
        if patient_df.empty:
            skipped += 1
            continue

        try:
            ctx = get_patient_context(subject_id, node_idx, gnn, patient_df,
                                      session_id=sess)
        except Exception as e:
            if verbose:
                print(f"  SKIP {subject_id} node {node_idx}: {e}")
            skipped += 1
            continue

        inp = build_structured_input(ctx)
        tgt = build_narrative_target(ctx)

        is_valid, val_issues = _validate_training_target(
            inp, tgt, pred_class=str(ctx["pred_class"])
        )
        if not is_valid:
            if verbose:
                print(f"  SKIP {subject_id} node {node_idx}: invalid target — {val_issues}")
            skipped += 1
            continue

        rows.append({
            "input": inp,
            "target": tgt,
            "pred_class": ctx["pred_class"],
            "subject_id": subject_id,
            "node_idx": node_idx,
            "mode": "node",
        })

    if skipped:
        print(f"  Skipped {skipped} nodes (missing patient data).")
    print(f"  Per-node pairs: {len(rows)}")

    # --- Paraphrase variants for per-node pairs ---
    n_base = len(rows)
    variant_rows = []
    for base_row in rows[:n_base]:
        if base_row["mode"] != "node":
            continue
        for v_idx in [1, 2]:
            v_tgt = _paraphrase_target(base_row["target"], v_idx)
            if v_idx == 2:
                from bio_leaflet import _apply_tone
                v_tgt = _apply_tone(v_tgt)
            is_valid, _ = _validate_training_target(
                base_row["input"], v_tgt, pred_class=str(base_row["pred_class"])
            )
            if is_valid:
                variant_rows.append({
                    "input": base_row["input"],
                    "target": v_tgt,
                    "pred_class": base_row["pred_class"],
                    "subject_id": base_row["subject_id"],
                    "node_idx": base_row["node_idx"],
                    "mode": f"node_v{v_idx}",
                })
    rows.extend(variant_rows)
    print(f"  Paraphrase variants added: {len(variant_rows)}")

    # --- Subject-level pairs (new default mode) ---
    if not legacy:
        unique_subjects = unc_df["subject_id"].str.strip().unique()
        n_subj = 0
        n_subj_skip = 0
        subj_rows = []

        print(f"Building subject-level pairs from {len(unique_subjects)} subjects...")
        for subject_id in unique_subjects:
            try:
                subj_ctx = build_subject_context(subject_id, gnn, data_df)
            except Exception as e:
                if verbose:
                    print(f"  SKIP subject {subject_id}: {e}")
                n_subj_skip += 1
                continue
            if subj_ctx is None:
                n_subj_skip += 1
                continue

            subj_inp = build_structured_input(subj_ctx, subject_mode=True)
            subj_tgt = build_narrative_target(subj_ctx, subject_mode=True)

            is_valid, val_issues = _validate_training_target(
                subj_inp, subj_tgt, pred_class=str(subj_ctx["pred_class"])
            )
            if not is_valid:
                if verbose:
                    print(f"  SKIP subject {subject_id}: invalid target — {val_issues}")
                n_subj_skip += 1
                continue

            subj_rows.append({
                "input": subj_inp,
                "target": subj_tgt,
                "pred_class": subj_ctx["pred_class"],
                "subject_id": subject_id,
                "node_idx": subj_ctx["latest_node_idx"],
                "mode": "subject",
            })
            n_subj += 1

        if n_subj_skip:
            print(f"  Skipped {n_subj_skip} subjects.")
        print(f"  Subject-level pairs: {n_subj}")

        # Upsample subject-level pairs 3x to balance against per-node pairs
        rows.extend(subj_rows * 3)
        print(f"  Total pairs (with 3x subject upsampling): {len(rows)}")

    df = pd.DataFrame(rows)

    # Stratified subject-level split 80/10/10 — prevents data leakage
    # and ensures all pred_classes appear in val and test splits.
    subj_class = df.groupby("subject_id")["pred_class"].first()
    val_subjects, test_subjects = [], []
    for cls in sorted(subj_class.unique(), key=str):
        cls_subjs = subj_class[subj_class == cls].index.tolist()
        rng = np.random.RandomState(42)
        rng.shuffle(cls_subjs)
        n = len(cls_subjs)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.1))
        test_subjects.extend(cls_subjs[:n_test])
        val_subjects.extend(cls_subjs[n_test:n_test + n_val])

    train_df = df[~df["subject_id"].isin(val_subjects + test_subjects)]
    val_df = df[df["subject_id"].isin(val_subjects)]
    test_df = df[df["subject_id"].isin(test_subjects)]

    # Keep only original (non-paraphrased) targets in val/test for
    # consistent ROUGE evaluation — paraphrase variants stay in training.
    original_modes = {"node", "subject"}
    val_df = val_df[val_df["mode"].isin(original_modes)].reset_index(drop=True)
    test_df = test_df[test_df["mode"].isin(original_modes)].reset_index(drop=True)

    # Save (only input + target columns for HuggingFace)
    train_path = out_path / "adni_t5_train.csv"
    val_path = out_path / "adni_t5_val.csv"
    test_path = out_path / "adni_t5_test.csv"

    train_df[["input", "target"]].to_csv(train_path, index=False)
    val_df[["input", "target"]].to_csv(val_path, index=False)
    test_df[["input", "target"]].to_csv(test_path, index=False)

    # Also save full metadata version
    df.to_csv(out_path / "adni_t5_all.csv", index=False)

    print(f"\n  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Class distribution (train):")
    for cls, count in train_df["pred_class"].value_counts().items():
        print(f"    {STAGE_DISPLAY.get(str(cls), cls)}: {count}")
    if not legacy:
        mode_counts = train_df["mode"].value_counts()
        print(f"  Mode distribution (train): {dict(mode_counts)}")
    print(f"\n  Saved to: {out_path}")
    print(f"    {train_path.name}")
    print(f"    {val_path.name}")
    print(f"    {test_path.name}")

    # Print example
    if verbose and len(rows) > 0:
        # Show a subject-level example if available
        example = next((r for r in rows if r["mode"] == "subject"), rows[0])
        print("\n" + "=" * 70)
        print(f"EXAMPLE STRUCTURED INPUT ({example['mode']} mode):")
        print("=" * 70)
        print(example["input"])
        print("\n" + "=" * 70)
        print("EXAMPLE TARGET (narrative only):")
        print("=" * 70)
        print(example["target"])

    return train_path, val_path, test_path


#  FINE-TUNING

def _build_rouge_compute_metrics(tokenizer):
    """Create a compute_metrics callback for ROUGE evaluation during training.

   
    """
    import nltk
    from evaluate import load as eval_load

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    metric = eval_load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # HuggingFace sometimes wraps predictions in a tuple
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.asarray(predictions, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)

        # If logits (3-D), argmax to token IDs
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)

        vocab_size = tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE expects newline-separated sentences
        decoded_preds = [
            "\n".join(nltk.sent_tokenize(p.strip())) for p in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(l.strip())) for l in decoded_labels
        ]

        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}

        # Mean generated length
        pred_lens = [
            np.count_nonzero(p != tokenizer.pad_token_id)
            for p in predictions
        ]
        result["gen_len"] = round(np.mean(pred_lens), 1)

        return result

    return compute_metrics


def fine_tune(train_csv, val_csv, output_dir, epochs=10, batch_size=4, lr=5e-5):
    """Fine-tune google/flan-t5-base on ADNI training data.

    train_csv/val_csv need 'input' and 'target' columns; output_dir is
    where the fine-tuned model and tokenizer get saved.
    """
    _ensure_hf_login()
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from datasets import Dataset

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.generation_config.no_repeat_ngram_size = 3

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"  Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    train_dataset = Dataset.from_pandas(train_df[["input", "target"]])
    val_dataset = Dataset.from_pandas(val_df[["input", "target"]])

    # Tokenization — sized for structured inputs (~150-300 tokens)
    # and narrative-only targets (~150-200 tokens)
    max_input_length = 384
    max_target_length = 512

    def preprocess(examples):
        inputs = [PREFIX + doc for doc in examples["input"]]
        model_inputs = tokenizer(
            inputs,
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

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        preprocess, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        preprocess, batched=True, remove_columns=val_dataset.column_names
    )

    compute_metrics = _build_rouge_compute_metrics(tokenizer)

    # select best checkpoint by ROUGE-2
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(Path(output_dir) / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rouge2",
        greater_is_better=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        predict_with_generate=True,
        generation_max_length=512,
        generation_num_beams=6,
        fp16=False,
        logging_dir=str(Path(output_dir) / "logs"),
        logging_steps=10,
        report_to="none",
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting fine-tuning for {epochs} epochs...")
    print(f"  Batch size: {batch_size} (effective: {batch_size * 2} with grad accum)")
    print(f"  Learning rate: {lr}")
    print(f"  Best model selected by: ROUGE-2")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    trainer.train()

    final_path = Path(output_dir)
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"\nModel saved to: {final_path}")

    val_results = trainer.evaluate()
    print("\nFinal validation scores:")
    for key, val in sorted(val_results.items()):
        if "rouge" in key or "loss" in key:
            print(f"  {key}: {val}")
    rouge2 = val_results.get("eval_rouge2", 0)
    if rouge2 < 85:
        print(f"\n  WARNING: ROUGE-2 ({rouge2:.1f}) is below 85 — "
              f"model may be generating off-template language. "
              f"Consider more training epochs or checking training data quality.")

    # Auto-run ROUGE on test set and print val-vs-test comparison
    test_csv_path = Path(output_dir) / "adni_t5_test.csv"
    if not test_csv_path.exists():
        test_csv_path = Path(output_dir).parent / "adni_t5_test.csv"
    if test_csv_path.exists():
        print("\nEvaluating ROUGE on test set...")
        test_results = evaluate_rouge_on_test(str(final_path), str(test_csv_path))

        print("\nROUGE Comparison (Validation vs Test):")
        print("-" * 55)
        print(f"  {'Metric':>12}  {'Val':>10}  {'Test':>10}  {'Diff':>10}")
        print("-" * 55)
        for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            v = val_results.get(f"eval_{key}", 0)
            t = test_results.get(f"eval_{key}", 0)
            diff = t - v
            print(f"  {key:>12}  {v:>9.2f}%  {t:>9.2f}%  {diff:>+9.2f}%")
        print("-" * 55)
    else:
        print(f"\n  (Test CSV not found — skipping test ROUGE evaluation)")

    return str(final_path)


def evaluate_rouge_on_test(model_path, test_csv):
    """Load a saved model and compute ROUGE scores on a test CSV.

    test_csv needs 'input' and 'target' columns. Returns a dict of ROUGE scores.
    """
    _ensure_hf_login()
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_df = pd.read_csv(test_csv)
    test_dataset = Dataset.from_pandas(test_df[["input", "target"]])

    def preprocess(examples):
        inputs = [PREFIX + doc for doc in examples["input"]]
        model_inputs = tokenizer(
            inputs, max_length=384, truncation=True, padding=False,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=512, truncation=True, padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_test = test_dataset.map(
        preprocess, batched=True, remove_columns=test_dataset.column_names,
    )

    compute_metrics = _build_rouge_compute_metrics(tokenizer)

    args = Seq2SeqTrainingArguments(
        output_dir="/tmp/rouge_eval",
        predict_with_generate=True,
        generation_max_length=512,
        generation_num_beams=4,
        per_device_eval_batch_size=4,
        fp16=False,
        report_to="none",
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100,
    )
    trainer = Seq2SeqTrainer(
        model, args,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate(eval_dataset=tokenized_test)
    print("\nTest ROUGE scores:")
    for key, val in sorted(results.items()):
        if "rouge" in key:
            print(f"  {key}: {val}")
    return results


def evaluate_rouge_all_subjects(results_dir, data_csv, model_path, out_dir):
    """Compute ROUGE across every subject, not just the held-out test split.

    NOTE: most subjects were seen during fine-tuning, so this is a
    whole-dataset fidelity check, not a generalization metric.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gnn = load_gnn_outputs(results_dir)
    data_df = load_patient_data(data_csv)
    unc_df = gnn["uncertainty"]
    unique_subjects = unc_df["subject_id"].str.strip().unique()

    rows = []
    for subject_id in unique_subjects:
        subj_ctx = build_subject_context(subject_id, gnn, data_df)
        if subj_ctx is None:
            continue
        inp = build_structured_input(subj_ctx, subject_mode=True)
        tgt = build_narrative_target(subj_ctx, subject_mode=True)
        rows.append({"input": inp, "target": tgt})

    csv_path = out_path / "adni_t5_all_subjects.csv"
    pd.DataFrame(rows)[["input", "target"]].to_csv(csv_path, index=False)

    print(f"\nEvaluating ROUGE on all {len(rows)} subjects...")
    return evaluate_rouge_on_test(model_path, str(csv_path))


# INFERENCE — Generate T5 Leaflet

def load_t5_model(model_path):
    """Load fine-tuned T5 model and tokenizer."""
    _ensure_hf_login()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path} (device: {device})")
    return model, tokenizer, device


def generate_t5_text(model, tokenizer, device, input_text,
                     mode="balanced", max_new_tokens=512):
    """Run T5 inference on a single structured input, generating the
    clinical narrative sections (CURRENT STATUS, LONGITUDINAL, RISK, SUMMARY).

    mode: deterministic, conservative, balanced, or creative.
    """
    full_prompt = PREFIX + input_text

    encoded = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    gen_kwargs = GENERATION_MODES[mode].copy()

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_length=50,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )

    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return summary


# POST-GENERATION GUARDRAILS

# MMSE qualitative bands (standard clinical interpretation)
def _mmse_qualitative(score):
    """Convert MMSE score to qualitative descriptor."""
    try:
        s = int(float(score))
    except (ValueError, TypeError):
        return None
    if s >= 27:
        return "within normal range"
    elif s >= 24:
        return "mildly below expected range"
    elif s >= 18:
        return "moderately impaired"
    else:
        return "significantly impaired"


# Template section headers that T5 should NOT generate (they're template-rendered)
_TEMPLATE_SECTION_HEADERS = [
    "PREDICTION :", "KEY CLINICAL FEATURES :",
    "GRAPH CONTEXT :", "WHAT-IF ANALYSIS :",
    "KEY BIOMARKER VALUES :", "VISIT PROGRESSION :",
    "LONGITUDINAL CONTEXT :", "UNCERTAINTY ANALYSIS :",
    "RISK CONTEXT :", "DISEASE PROGRESSION :",
    # Backward-compat with older model outputs
    "SIMILAR PATIENT COMPARISON :", "SENSITIVITY ANALYSIS :",
    "PREDICTION CONFIDENCE :",
]

# Stage-appropriate replacement maps
_STAGE_REPLACEMENTS = {
    "0": {  # CN — replace AD-severity language
        "dementia-stage": "age-related",
        "alzheimer's disease": "healthy aging",
        "ad-level": "age-appropriate",
        "severe impairment": "preserved function",
        "severe atrophy": "age-related structural change",
        "advanced neurodegeneration": "age-related structural change",
        "progressive impairment": "stable function",
        "marked deterioration": "age-related change",
    },
    "1+": {  # AD — replace CN language
        "normal aging": "neurodegenerative change",
        "cognitively intact": "cognitively impaired",
        "no impairment": "multi-domain impairment",
        "healthy brain": "structural changes consistent with neurodegeneration",
        "cognitively healthy": "cognitively impaired",
        "fully preserved": "impaired",
    },
    # MCI gets no replacements — mixed language is appropriate
}


def _post_process_narrative(raw_t5_text, ctx):
    """Apply deterministic, ctx-grounded corrections to T5 output before
    assembly — catches hallucinated content that slipped past training-time
    sanitisation.
    """
    text = raw_t5_text

    # 1. Remove any leaked biomarker numerics
    # MMSE scores → qualitative band
    mmse_qual = _mmse_qualitative(ctx.get("mmse"))
    if mmse_qual:
        text = re.sub(
            r"\b\d{1,2}\s*/\s*30\b",
            mmse_qual, text,
        )
    else:
        text = re.sub(r"\b\d{1,2}\s*/\s*30\b", "", text)

    # MRI volumes
    text = re.sub(
        r"\b\d{3,}\.?\d*\s*mm[³3]?\b",
        "", text,
    )
    # CSF values
    text = re.sub(r"\b\d+\.?\d*\s*pg/mL\b", "", text)
    # Confidence percentages
    text = re.sub(r"\b\d{1,3}\.?\d*%", "", text)
    # Age values
    text = re.sub(r"\b\d{2,3}\.?\d*\s*years?\b", "", text)
    # Trail times
    text = re.sub(r"\b\d{2,4}\s*(?:seconds?|s)\b", "", text)

    # 2. Stage-language enforcement
    pred_class = ctx.get("pred_class", "")
    replacements = _STAGE_REPLACEMENTS.get(pred_class, {})
    for bad_phrase, good_phrase in replacements.items():
        text = re.sub(re.escape(bad_phrase), good_phrase, text, flags=re.IGNORECASE)

    # 3. Longitudinal claim validation
    n_visits = ctx.get("n_visits", 1)
    try:
        n_visits = int(n_visits)
    except (ValueError, TypeError):
        n_visits = 1

    if n_visits < 2:
        # Single visit — no longitudinal claims allowed
        longitudinal_claims = [
            r"decline\s+(?:is\s+)?observed",
            r"improvement\s+(?:is\s+)?noted",
            r"worsened\s+over\s+time",
            r"progressed\s+(?:from|over)",
            r"longitudinal\s+(?:decline|change|improvement)",
        ]
        for pat in longitudinal_claims:
            text = re.sub(
                pat,
                "longitudinal change cannot be evaluated from a single visit",
                text, flags=re.IGNORECASE, count=1,
            )

    # 4. Confidence/uncertainty consistency
    if ctx.get("is_high_uncertainty"):
        confidence_claims = [
            (r"high\s+confidence", "elevated uncertainty"),
            (r"confident\s+prediction", "uncertain prediction"),
            (r"reliable\s+classification", "classification with elevated uncertainty"),
        ]
        for pat, replacement in confidence_claims:
            text = re.sub(pat, replacement, text, flags=re.IGNORECASE)

    # 5. Remove leaked template section headers
    for header in _TEMPLATE_SECTION_HEADERS:
        text = text.replace(header, "")

    # 6. Repetition removal — drop duplicate sentences
    sentences = text.split(".")
    seen = set()
    deduped = []
    for sent in sentences:
        normalised = sent.strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            deduped.append(sent)
        elif not normalised:
            deduped.append(sent)  # preserve empty splits for formatting
    text = ".".join(deduped)

    # 7. Cleanup
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Mode fallback chain for retry: more constrained each attempt
_MODE_FALLBACK = {
    "creative": "conservative",
    "balanced": "conservative",
    "conservative": "deterministic",
    "deterministic": "deterministic",
}


def _parse_t5_sections(t5_text):
    """Split T5 output into interpretation and summary sections.

    T5 output has headers like "CURRENT STATUS: ... SUMMARY: ...". Returns
    (interpretation, summary); summary is None if no SUMMARY: header found.
    """
    marker = "SUMMARY:"
    idx = t5_text.find(marker)
    if idx == -1:
        return t5_text.strip(), None
    interpretation = t5_text[:idx].strip()
    summary = t5_text[idx + len(marker):].strip()
    return interpretation, summary if summary else None


def _reject_and_retry(ctx, model, tokenizer, device, structured_input,
                      mode="balanced", hybrid=True, max_retries=2,
                      verbose=False):
    """Generate and verify the T5 narrative with retry and template fallback.

If verification fails, generation is retried with a more constrained mode.
If all retries fail, the template-based interpretation is used instead.

Returns the generated interpretation, verification status, scores, and metadata.
    """
    attempt = 0
    current_mode = mode
    last_issues = []

    while attempt <= max_retries:
        t5_text = generate_t5_text(
            model, tokenizer, device, structured_input, current_mode,
        )

        # Post-process to fix hallucinations
        t5_text = _post_process_narrative(t5_text, ctx)

        t5_interpretation, t5_summary = _parse_t5_sections(t5_text)

        # Assemble leaflet using the full Jinja2 template, with T5
        # providing INTERPRETATION and SUMMARY sections
        if hybrid:
            leaflet = render_leaflet_with_t5(
                ctx,
                t5_stage_interpretation=t5_interpretation,
                t5_summary=t5_summary,
            )
        else:
            leaflet = t5_text

        status, issues, scores = verify_t5_leaflet(leaflet, ctx)

        if status == "PASS":
            return leaflet, status, scores, {
                "n_retries": attempt,
                "final_mode": current_mode,
                "fell_back_to_template": False,
            }

        if verbose:
            print(f"    Attempt {attempt + 1} ({current_mode}): FAIL — "
                  f"{len(issues)} issues")
            for iss in issues[:3]:
                print(f"      ! {iss}")

        last_issues = issues
        current_mode = _MODE_FALLBACK.get(current_mode, "deterministic")
        attempt += 1

    # All retries exhausted — fall back to pure template rendering
    if verbose:
        print(f"    All retries exhausted. Falling back to template rendering.")

    try:
        patient_df_ctx = ctx  # ctx already contains what render_leaflet needs
        template_leaflet = render_leaflet(ctx)
        status, issues, scores = verify_t5_leaflet(template_leaflet, ctx)
        template_leaflet += (
            "\n\n[Note: T5 narrative generation was replaced with template "
            "rendering due to verification failures.]"
        )
        return template_leaflet, status, scores, {
            "n_retries": max_retries + 1,
            "final_mode": "template_fallback",
            "fell_back_to_template": True,
        }
    except Exception as e:
        # If even template rendering fails, return the last T5 attempt with warnings
        if hybrid:
            leaflet = render_leaflet_with_t5(
                ctx,
                t5_stage_interpretation=t5_interpretation,
                t5_summary=t5_summary,
            )
        else:
            leaflet = t5_text
        leaflet += f"\n\n[VERIFICATION WARNING: {len(last_issues)} issues; "
        leaflet += f"template fallback also failed: {e}]\n"
        for iss in last_issues:
            leaflet += f"  - {iss}\n"
        return leaflet, "FAIL", scores, {
            "n_retries": max_retries + 1,
            "final_mode": "failed",
            "fell_back_to_template": False,
        }


def assemble_hybrid_leaflet(t5_narrative, gnn_sections, ctx):  # DEPRECATED
    """Combine T5-generated narrative with template-rendered GNN sections.

    Deprecated — replaced by render_leaflet_with_t5() in bio_leaflet.py.
    """
    is_subject = "progression_timeline" in ctx
    lines = []
    lines.append("=" * 80)
    lines.append("                        BRAIN HEALTH BIO-LEAFLET")
    lines.append("=" * 80)
    lines.append("")

    # Patient header (from ctx — guaranteed accurate)
    lines.append(f"Patient ID:       {ctx['patient_id']}")
    lines.append(f"Clinical Status:  {ctx['clinical_status']}")
    lines.append(f"Age:              {ctx['age']} years")
    lines.append(f"Sex:              {ctx['sex']}")
    lines.append(f"APOE4 Status:     {ctx['apoe4_status']}")
    lines.append(f"Education:        {ctx['education_years']} years")
    if is_subject:
        lines.append(f"Number of visits: {ctx['n_nodes']}")
        first_m = ctx["node_contexts"][0].get("visit_month", 0)
        last_m = ctx["node_contexts"][-1].get("visit_month", 0)
        lines.append(f"Time span:        {last_m - first_m} months")
    lines.append("")

    # Key biomarker values (template-rendered, exact numbers)
    biomarker_block = render_key_biomarker_values(ctx)
    if biomarker_block:
        lines.append(biomarker_block)

    # GNN sections (template-rendered, exact numbers)
    lines.append(gnn_sections)

    # T5 narrative (already has section headers: CURRENT STATUS,
    # PROGRESSION (subject-level), LONGITUDINAL CONTEXT, RISK CONTEXT, SUMMARY)
    lines.append("INTERPRETATION :")
    lines.append(t5_narrative)
    lines.append("")

    lines.append("=" * 80)
    lines.append("  Generated by Bio-Leaflet System | ADNI Spectral GCN Pipeline")
    lines.append("  This automated research summary is non-diagnostic and should not")
    lines.append("  replace clinical evaluation.")
    lines.append("=" * 80)

    return "\n".join(lines)


# FACTUAL VERIFICATION

# Field-to-category mapping for error distribution reporting (Table III style)
FACT_FIELD_CATEGORIES = {
    "patient_id": "Demographics",
    "age": "Demographics",
    "sex": "Demographics",
    "apoe4_status": "Demographics",
    "education_years": "Demographics",
    "mmse": "Cognitive Scores",
    "faq": "Cognitive Scores",
    "ldel": "Cognitive Scores",
    "trails_b": "Cognitive Scores",
    "hippo_vol": "MRI Biomarkers",
    "entorhinal_vol": "MRI Biomarkers",
    "amygdala_vol": "MRI Biomarkers",
    "ventricle_vol": "MRI Biomarkers",
    "abeta42": "CSF Biomarkers",
    "ptau": "CSF Biomarkers",
    "tau": "CSF Biomarkers",
    "confidence_pct": "GNN Prediction",
    "pred_class": "GNN Prediction",
    "predictive_entropy": "GNN Prediction",
    "graph_agreement_pct": "GNN Prediction",
    "n_visits": "Longitudinal",
    "stability_label": "Longitudinal",
}

# Tolerance definitions for numeric field matching
FACT_TOLERANCES = {
    "age": 1.0,
    "confidence_pct": 0.5,
    "predictive_entropy": 0.01,
    "graph_agreement_pct": 0.5,
    "hippo_vol": 1.0,
    "entorhinal_vol": 1.0,
    "amygdala_vol": 1.0,
    "ventricle_vol": 1.0,
    "abeta42": 0.1,
    "ptau": 0.1,
    "tau": 0.1,
}

# Fields that use exact string matching (all others use numeric tolerance)
FACT_EXACT_FIELDS = {
    "patient_id", "sex", "mmse", "n_visits", "stability_label",
    "apoe4_status", "education_years", "pred_class",
    "faq", "ldel", "trails_b",
}


def extract_t5_facts(text):
    """Regex-extract factual claims from a T5 generated/assembled leaflet."""
    facts = {}

    # --- Demographics ---
    m = re.search(r"Patient ID:\s*(\S+)", text)
    if m:
        facts["patient_id"] = m.group(1)

    m = re.search(r"Age:\s*([\d.]+)", text)
    if m:
        facts["age"] = float(m.group(1))

    m = re.search(r"Sex:\s*(\w+)", text)
    if m:
        facts["sex"] = m.group(1)

    m = re.search(r"APOE.*?:\s*(.+)", text)
    if m:
        facts["apoe4_status"] = m.group(1).strip()

    m = re.search(r"Education:\s*([\d.]+)\s*y", text)
    if m:
        facts["education_years"] = m.group(1).strip()

    # --- Cognitive Scores ---
    m = re.search(r"MMSE.*?:\s*(\d+)\s*/\s*30", text)
    if m:
        facts["mmse"] = int(m.group(1))

    m = re.search(r"Functional assessment \(FAQ\):\s*(\d+)", text)
    if m:
        facts["faq"] = int(m.group(1))

    m = re.search(r"Logical memory.*?:\s*(\d+)", text)
    if m:
        facts["ldel"] = int(m.group(1))

    m = re.search(r"Trail Making B\):\s*([\d.]+)\s*s", text)
    if m:
        facts["trails_b"] = float(m.group(1))

    # --- MRI Biomarkers ---
    m = re.search(r"Hippocampal volume.*?:\s*([\d.]+)\s*mm", text)
    if m:
        facts["hippo_vol"] = float(m.group(1))

    m = re.search(r"Entorhinal cortex volume:\s*([\d.]+)\s*mm", text)
    if m:
        facts["entorhinal_vol"] = float(m.group(1))

    m = re.search(r"Amygdala volume:\s*([\d.]+)\s*mm", text)
    if m:
        facts["amygdala_vol"] = float(m.group(1))

    m = re.search(r"Lateral ventricle volume:\s*([\d.]+)\s*mm", text)
    if m:
        facts["ventricle_vol"] = float(m.group(1))

    # --- CSF Biomarkers ---
    m = re.search(r"CSF Abeta42:\s*([\d.]+)", text)
    if m:
        facts["abeta42"] = float(m.group(1))

    m = re.search(r"CSF Phosphorylated-Tau:\s*([\d.]+)", text)
    if m:
        facts["ptau"] = float(m.group(1))

    m = re.search(r"CSF Total-Tau:\s*([\d.]+)", text)
    if m:
        facts["tau"] = float(m.group(1))

    # --- GNN Prediction ---
    m = re.search(r"(?:Model|Prediction) confidence[^:]*:\s*([\d.]+)%", text)
    if m:
        facts["confidence_pct"] = float(m.group(1))

    m = re.search(r"Predicted stage:\s*(.+)", text)
    if m:
        facts["pred_class"] = m.group(1).strip()

    m = re.search(r"Predictive entropy:\s*([\d.]+)", text)
    if m:
        facts["predictive_entropy"] = float(m.group(1))

    m = re.search(r"Graph agreement:\s*([\d.]+)%", text)
    if m:
        facts["graph_agreement_pct"] = float(m.group(1))

    # --- Longitudinal (subject-level) ---
    m = re.search(r"Number of visits:\s*(\d+)", text)
    if m:
        facts["n_visits"] = int(m.group(1))

    m = re.search(r"Prediction stability:\s*(\w+)", text)
    if m:
        facts["stability_label"] = m.group(1)

    return facts


def compute_factual_metrics(gt_facts, gen_facts):
    """Compute TP/FP/FN, precision, recall, F1 between ground-truth and
    generated facts.

    """
    tp = fp = fn = 0
    field_results = {}

    for key in FACT_FIELD_CATEGORIES:
        if key not in gt_facts:
            field_results[key] = "N/A"
            continue

        gt_val = gt_facts[key]
        if key not in gen_facts:
            fn += 1
            field_results[key] = "FN"
            continue

        gen_val = gen_facts[key]

        if key in FACT_EXACT_FIELDS:
            match = str(gt_val).strip().lower() == str(gen_val).strip().lower()
        else:
            tol = FACT_TOLERANCES.get(key, 0)
            try:
                match = abs(float(gt_val) - float(gen_val)) <= tol
            except (ValueError, TypeError):
                match = str(gt_val).strip().lower() == str(gen_val).strip().lower()

        if match:
            tp += 1
            field_results[key] = "TP"
        else:
            fp += 1
            field_results[key] = "FP"

    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "field_results": field_results,
    }


def verify_t5_leaflet(leaflet_text, ctx):
    """Verify factual accuracy of a T5-generated leaflet against source data.

    Returns (status, issues, scores).
    """
    gen_facts = extract_t5_facts(leaflet_text)
    issues = []
    n_checked = 0
    n_correct = 0

    # Ground truth from ctx — demographics
    checks = {
        "patient_id": (ctx["patient_id"], "exact"),
        "sex": (ctx["sex"], "exact"),
    }
    try:
        checks["age"] = (float(ctx["age"]), "tolerant_1.0")
    except (ValueError, TypeError):
        pass
    if ctx.get("apoe4_status") and ctx["apoe4_status"] != "N/A":
        checks["apoe4_status"] = (str(ctx["apoe4_status"]), "exact")
    if ctx.get("education_years") and ctx["education_years"] != "N/A":
        checks["education_years"] = (str(ctx["education_years"]), "exact")

    # Cognitive scores
    try:
        checks["mmse"] = (int(float(ctx["mmse"])), "exact")
    except (ValueError, TypeError):
        pass
    if ctx.get("faq") and ctx["faq"] != "N/A":
        try:
            checks["faq"] = (int(float(ctx["faq"])), "exact")
        except (ValueError, TypeError):
            pass
    if ctx.get("ldel") and ctx["ldel"] != "N/A":
        try:
            checks["ldel"] = (int(float(ctx["ldel"])), "exact")
        except (ValueError, TypeError):
            pass
    if ctx.get("trails_b") and ctx["trails_b"] != "N/A":
        try:
            checks["trails_b"] = (float(ctx["trails_b"]), "exact")
        except (ValueError, TypeError):
            pass

    # MRI biomarkers
    for mri_key, tol in [("hippo_vol", 1.0), ("entorhinal_vol", 1.0),
                          ("amygdala_vol", 1.0), ("ventricle_vol", 1.0)]:
        if ctx.get(mri_key) and ctx[mri_key] != "N/A":
            try:
                checks[mri_key] = (float(ctx[mri_key]), f"tolerant_{tol}")
            except (ValueError, TypeError):
                pass

    # CSF biomarkers
    for csf_key, tol in [("abeta42", 0.1), ("ptau", 0.1), ("tau", 0.1)]:
        if ctx.get(csf_key) and ctx[csf_key] != "N/A":
            try:
                checks[csf_key] = (float(ctx[csf_key]), f"tolerant_{tol}")
            except (ValueError, TypeError):
                pass

    # GNN prediction
    try:
        checks["confidence_pct"] = (float(ctx["confidence_pct"]), "tolerant_0.5")
    except (ValueError, TypeError):
        pass
    if ctx.get("pred_class_display"):
        checks["pred_class"] = (str(ctx["pred_class_display"]), "exact")
    try:
        checks["predictive_entropy"] = (float(ctx["predictive_entropy"]), "tolerant_0.01")
    except (ValueError, TypeError):
        pass
    try:
        checks["graph_agreement_pct"] = (float(ctx["graph_agreement_pct"]), "tolerant_0.5")
    except (ValueError, TypeError):
        pass

    for key, (gt_val, match_type) in checks.items():
        if key not in gen_facts:
            continue
        n_checked += 1
        gen_val = gen_facts[key]

        if match_type == "exact":
            if str(gt_val).strip().lower() == str(gen_val).strip().lower():
                n_correct += 1
            else:
                issues.append(f"{key}: expected '{gt_val}', got '{gen_val}'")
        else:
            tol = float(match_type.split("_")[1])
            try:
                if abs(float(gt_val) - float(gen_val)) <= tol:
                    n_correct += 1
                else:
                    issues.append(
                        f"{key}: expected {gt_val}, got {gen_val} (tolerance ±{tol})"
                    )
            except (ValueError, TypeError):
                issues.append(f"{key}: cannot compare '{gt_val}' vs '{gen_val}'")

    # Hallucination check — forbidden diagnostic / prescriptive phrases
    text_lower = leaflet_text.lower()
    forbidden = [
        # Diagnostic claims
        "diagnosed with alzheimer", "confirmed dementia",
        "definitive diagnosis", "alzheimer's confirmed",
        "clinical diagnosis of", "we diagnose", "patient has ad",
        "patient has alzheimer",
        # Prescriptive / clinical-advice language
        "we recommend", "you should", "prescribe",
        "treatment plan", "100% confidence", "certain diagnosis",
        "definitely has",
    ]
    for term in forbidden:
        if term in text_lower:
            issues.append(f"HALLUCINATION: contains '{term}'")

    narrative_lower = extract_narrative_sections(leaflet_text).lower()
    if not narrative_lower:
        narrative_lower = text_lower
        issues.append("NOTE: narrative sections not found; stage check ran on full text")

    # Stage consistency check — expanded
    pred = ctx["pred_class"]
    if pred == "0":
        cn_bad = [
            "alzheimer's disease", "ad-level", "dementia-stage",
            "severe impairment", "severe atrophy", "advanced neurodegeneration",
            "progressive impairment", "marked deterioration",
        ]
        for term in cn_bad:
            if term in narrative_lower:
                # Allow negated context ("not alzheimer's disease")
                idx = narrative_lower.find(term)
                prefix = narrative_lower[max(0, idx - 20):idx]
                if "not" not in prefix and "no " not in prefix and "without" not in prefix:
                    issues.append(f"STAGE_MISMATCH: CN patient — '{term}'")
    elif pred == "1+":
        ad_bad = [
            "normal aging", "cognitively intact", "no impairment",
            "healthy brain", "cognitively healthy", "normal range",
            "fully preserved",
        ]
        for term in ad_bad:
            if term in narrative_lower:
                issues.append(f"STAGE_MISMATCH: AD patient — '{term}'")
    elif pred == "0.5":
        mci_bad = ["end-stage", "terminal decline"]
        for term in mci_bad:
            if term in narrative_lower:
                issues.append(f"STAGE_MISMATCH: MCI patient — '{term}'")

    # Confidence consistency: T5 says "high confidence" but patient is high uncertainty
    if ctx.get("is_high_uncertainty"):
        confidence_phrases = ["high confidence", "confident prediction",
                              "reliable classification"]
        for phrase in confidence_phrases:
            if phrase in narrative_lower:
                idx = narrative_lower.find(phrase)
                prefix = narrative_lower[max(0, idx - 15):idx]
                if "not" not in prefix and "no " not in prefix:
                    issues.append(
                        f"CONSISTENCY: '{phrase}' but patient is HIGH UNCERTAINTY"
                    )

    # Numeric claim scanner — flag ungrounded numbers in T5 narrative
    # Extract ONLY the INTERPRETATION section (T5-generated), not the full leaflet
    _NEXT_SECTION_HEADERS = [
        "GRAPH CONTEXT :", "WHAT-IF ANALYSIS :", "LONGITUDINAL CONTEXT :",
        "UNCERTAINTY ANALYSIS :", "RISK CONTEXT :", "SUMMARY :",
        "PREDICTION :", "KEY BIOMARKER VALUES :", "DISEASE PROGRESSION :",
    ]
    interp_idx = leaflet_text.find("INTERPRETATION :")
    if interp_idx >= 0:
        after_interp = leaflet_text[interp_idx:]
        # Find the nearest next template section header
        next_section_idx = len(after_interp)
        for header in _NEXT_SECTION_HEADERS:
            idx = after_interp.find(header)
            if idx > 0 and idx < next_section_idx:
                next_section_idx = idx
        narrative_section = after_interp[:next_section_idx]
    else:
        narrative_section = ""

    # Also extract the T5-generated SUMMARY section
    summary_idx = leaflet_text.find("SUMMARY :")
    if summary_idx >= 0:
        after_summary = leaflet_text[summary_idx + len("SUMMARY :"):]
        footer_idx = after_summary.find("Generated by Bio-Leaflet System")
        summary_section = after_summary[:footer_idx] if footer_idx >= 0 else after_summary
    else:
        summary_section = ""

    # Combine T5-generated sections for checking
    t5_sections = narrative_section + "\n" + summary_section

    if t5_sections.strip():
        # MMSE scores in narrative
        for m in re.finditer(r"(\d{1,2})\s*/\s*30", t5_sections):
            try:
                claimed = int(m.group(1))
                actual = int(float(ctx.get("mmse", -1)))
                if actual >= 0 and abs(claimed - actual) > 1:
                    issues.append(
                        f"UNGROUNDED_NUMERIC: MMSE '{m.group(0)}' in narrative "
                        f"(source: {actual}/30)"
                    )
            except (ValueError, TypeError):
                pass
        # Volume claims in narrative
        for m in re.finditer(r"(\d{3,})\s*mm", t5_sections):
            issues.append(
                f"UNGROUNDED_NUMERIC: volume '{m.group(0)}' found in narrative "
                f"section (numbers should be in template sections only)"
            )
        # CSF values in narrative
        for m in re.finditer(r"\d+\.?\d*\s*pg/mL", t5_sections):
            issues.append(
                f"UNGROUNDED_NUMERIC: CSF '{m.group(0)}' found in narrative"
            )
        # Confidence percentages in narrative
        for m in re.finditer(r"\d{1,3}\.?\d*%", t5_sections):
            issues.append(
                f"UNGROUNDED_NUMERIC: percentage '{m.group(0)}' in narrative"
            )

    # Section leak detection — T5 generated template section headers
    for header in _TEMPLATE_SECTION_HEADERS:
        if header in t5_sections:
            issues.append(
                f"SECTION_LEAK: T5 generated template header '{header}'"
            )

    # Repetition detection — same sentence appearing 2+ times in narrative
    if t5_sections.strip():
        sents = [s.strip().lower() for s in t5_sections.split(".")
                 if s.strip() and len(s.strip()) > 20]
        seen_sents = set()
        for sent in sents:
            if sent in seen_sents:
                issues.append(
                    f"REPETITION: duplicate sentence in narrative"
                )
                break  # one flag is enough
            seen_sents.add(sent)

    # Longitudinal grounding — check directional claims vs deltas
    deltas = ctx.get("deltas", {})
    n_visits = ctx.get("n_visits", 1)
    try:
        n_visits = int(n_visits)
    except (ValueError, TypeError):
        n_visits = 1
    if narrative_section and n_visits < 2:
        decline_pats = [
            r"decline\s+(?:is\s+)?observed", r"worsened\s+over",
            r"progressive\s+decline", r"deteriorat",
        ]
        for pat in decline_pats:
            if re.search(pat, narrative_section, re.IGNORECASE):
                issues.append(
                    "UNGROUNDED_LONGITUDINAL: longitudinal claim with single visit"
                )

    # Subject-level checks
    if "n_visits" in gen_facts and "n_nodes" in ctx:
        n_checked += 1
        if gen_facts["n_visits"] == ctx["n_nodes"]:
            n_correct += 1
        else:
            issues.append(
                f"n_visits: expected {ctx['n_nodes']}, got {gen_facts['n_visits']}"
            )

    if "stability_label" in gen_facts and "stability" in ctx:
        n_checked += 1
        expected = ctx["stability"]["stability_label"]
        if gen_facts["stability_label"].lower() == expected.lower():
            n_correct += 1
        else:
            issues.append(
                f"stability: expected '{expected}', got '{gen_facts['stability_label']}'"
            )

    precision = n_correct / n_checked if n_checked > 0 else 0.0

    # Compute TP/FP/FN metrics using the structured metrics function
    gt_facts_for_metrics = {key: val for key, (val, _) in checks.items()}
    fact_metrics = compute_factual_metrics(gt_facts_for_metrics, gen_facts)

    scores = {
        "fields_checked": n_checked,
        "fields_correct": n_correct,
        "precision": round(precision, 4),
        "tp": fact_metrics["tp"],
        "fp": fact_metrics["fp"],
        "fn": fact_metrics["fn"],
        "recall": fact_metrics["recall"],
        "f1": fact_metrics["f1"],
        "field_results": fact_metrics["field_results"],
    }
    has_hallucination = any("HALLUCINATION" in i for i in issues)
    has_ungrounded = any("UNGROUNDED_NUMERIC" in i for i in issues)
    has_stage_mismatch = any("STAGE_MISMATCH" in i for i in issues)
    has_section_leak = any("SECTION_LEAK" in i for i in issues)
    has_ungrounded_long = any("UNGROUNDED_LONGITUDINAL" in i for i in issues)
    # FAIL on any critical issue or low precision
    is_fail = (
        precision < 0.95
        or has_hallucination
        or has_ungrounded
        or has_stage_mismatch
        or has_section_leak
        or has_ungrounded_long
    )
    status = "FAIL" if is_fail else "PASS"
    return status, issues, scores


#  BATCH FACTUAL ACCURACY EVALUATION

def _extract_subject_ids_from_csv(csv_path):
    """Extract unique subject IDs from a training/val/test CSV.

    Parses subject_id from the 'input' column via the DEMOGRAPHICS line.
    """
    df = pd.read_csv(csv_path)
    ids = set()
    for inp in df["input"]:
        m = re.search(r"DEMOGRAPHICS:\s*(\S+)", str(inp))
        if m:
            ids.add(m.group(1))
    return ids


def _aggregate_split_results(sample_results):
    """Aggregate per-sample factual accuracy results into summary metrics.

    sample_results: list of dicts with subject_id, pred_class, status, scores
    (from verify_t5_leaflet). Returns n, pass_rate, field_accuracy,
    micro/macro precision/recall/F1, by_category, by_cdr.
    """
    if not sample_results:
        return {
            "n": 0, "pass_rate": 0, "fields_checked": 0, "fields_correct": 0,
            "field_accuracy": 0, "micro_p": 0, "micro_r": 0, "micro_f1": 0,
            "macro_p": 0, "macro_r": 0, "macro_f1": 0,
            "by_category": {}, "by_cdr": {},
        }

    n = len(sample_results)
    n_pass = sum(1 for r in sample_results if r["status"] == "PASS")

    total_checked = sum(r["scores"]["fields_checked"] for r in sample_results)
    total_correct = sum(r["scores"]["fields_correct"] for r in sample_results)

    # Micro-averaged P/R/F1
    total_tp = sum(r["scores"]["tp"] for r in sample_results)
    total_fp = sum(r["scores"]["fp"] for r in sample_results)
    total_fn = sum(r["scores"]["fn"] for r in sample_results)

    micro_p = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    micro_r = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # Macro-averaged P/R/F1
    sample_ps = [r["scores"].get("recall", 0) for r in sample_results]  # noqa — see below
    sample_ps = [r["scores"].get("precision", 0) for r in sample_results]
    # precision from compute_factual_metrics is 0-100 scale; the legacy
    # "precision" in scores is 0-1 scale.  Use the tp/fp/fn based values.
    per_sample_p = []
    per_sample_r = []
    per_sample_f1 = []
    for r in sample_results:
        s = r["scores"]
        tp, fp_, fn_ = s["tp"], s["fp"], s["fn"]
        p = (tp / (tp + fp_) * 100) if (tp + fp_) > 0 else 0.0
        rec = (tp / (tp + fn_) * 100) if (tp + fn_) > 0 else 0.0
        f = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
        per_sample_p.append(p)
        per_sample_r.append(rec)
        per_sample_f1.append(f)

    macro_p = sum(per_sample_p) / n if n else 0.0
    macro_r = sum(per_sample_r) / n if n else 0.0
    macro_f1 = sum(per_sample_f1) / n if n else 0.0

    # Error distribution by field category (with per-category TP/FP/FN for Table V)
    category_stats = {}  # {cat: {"total", "errors", "tp", "fp", "fn"}}
    for r in sample_results:
        field_results = r["scores"].get("field_results", {})
        for field, result in field_results.items():
            cat = FACT_FIELD_CATEGORIES.get(field, "Other")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "errors": 0,
                                       "tp": 0, "fp": 0, "fn": 0}
            if result != "N/A":
                category_stats[cat]["total"] += 1
                if result == "TP":
                    category_stats[cat]["tp"] += 1
                elif result == "FP":
                    category_stats[cat]["fp"] += 1
                    category_stats[cat]["errors"] += 1
                elif result == "FN":
                    category_stats[cat]["fn"] += 1
                    category_stats[cat]["errors"] += 1

    # Performance by CDR/pred_class
    cdr_groups = {}
    for r, p_, rec_, f_ in zip(sample_results, per_sample_p, per_sample_r, per_sample_f1):
        cls = r["pred_class"]
        if cls not in cdr_groups:
            cdr_groups[cls] = {
                "n": 0, "n_pass": 0,
                "total_checked": 0, "total_correct": 0,
                "f1_scores": [],
            }
        g = cdr_groups[cls]
        g["n"] += 1
        if r["status"] == "PASS":
            g["n_pass"] += 1
        g["total_checked"] += r["scores"]["fields_checked"]
        g["total_correct"] += r["scores"]["fields_correct"]
        g["f1_scores"].append(f_)

    by_cdr = {}
    for cls, g in cdr_groups.items():
        by_cdr[cls] = {
            "n": g["n"],
            "field_accuracy": round(g["total_correct"] / max(g["total_checked"], 1) * 100, 2),
            "avg_f1": round(sum(g["f1_scores"]) / max(len(g["f1_scores"]), 1), 2),
            "pass_rate": round(g["n_pass"] / max(g["n"], 1) * 100, 1),
        }

    return {
        "n": n,
        "pass_rate": round(n_pass / n * 100, 1),
        "fields_checked": total_checked,
        "fields_correct": total_correct,
        "field_accuracy": round(total_correct / max(total_checked, 1) * 100, 2),
        "micro_tp": total_tp,
        "micro_fp": total_fp,
        "micro_fn": total_fn,
        "micro_p": round(micro_p, 2),
        "micro_r": round(micro_r, 2),
        "micro_f1": round(micro_f1, 2),
        "macro_p": round(macro_p, 2),
        "macro_r": round(macro_r, 2),
        "macro_f1": round(macro_f1, 2),
        "by_category": category_stats,
        "by_cdr": by_cdr,
    }


def test_factual_accuracy(results_dir, data_csv, model_path,
                          out_dir="./eval_results", mode="deterministic"):
    """Run factual accuracy evaluation on val and test splits
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gnn = load_gnn_outputs(results_dir)
    data_df = load_patient_data(data_csv)
    model, tokenizer, device = load_t5_model(model_path)

    # Identify val/test subjects from training CSVs
    model_dir = Path(model_path)
    val_csv = model_dir / "adni_t5_val.csv"
    test_csv = model_dir / "adni_t5_test.csv"
    if not val_csv.exists():
        val_csv = model_dir.parent / "adni_t5_val.csv"
    if not test_csv.exists():
        test_csv = model_dir.parent / "adni_t5_test.csv"

    val_subjects = _extract_subject_ids_from_csv(str(val_csv)) if val_csv.exists() else set()
    test_subjects = _extract_subject_ids_from_csv(str(test_csv)) if test_csv.exists() else set()

    if not val_subjects and not test_subjects:
        print("ERROR: Could not find adni_t5_val.csv or adni_t5_test.csv.")
        print(f"  Searched: {model_dir} and {model_dir.parent}")
        return {}

    # All available subjects from GNN outputs
    unc_df = gnn["uncertainty"]
    all_subjects = set(unc_df["subject_id"].str.strip().unique())

    val_subjects = val_subjects & all_subjects
    test_subjects = test_subjects & all_subjects

    print(f"Factual accuracy evaluation (mode: {mode})")
    print(f"  Val subjects:  {len(val_subjects)}")
    print(f"  Test subjects: {len(test_subjects)}")
    print()

    def _eval_split(subject_ids, split_name):
        results = []
        for i, subject_id in enumerate(sorted(subject_ids)):
            result = generate_one_subject_leaflet(
                subject_id, gnn, data_df,
                model, tokenizer, device,
                mode=mode, hybrid=True, verbose=False,
            )
            if result is None:
                continue

            leaflet, status, scores, meta = result
            subj_ctx = build_subject_context(subject_id, gnn, data_df)
            pred_class = subj_ctx["pred_class"] if subj_ctx else "?"

            results.append({
                "subject_id": subject_id,
                "pred_class": pred_class,
                "status": status,
                "scores": scores,
            })

            if (i + 1) % 20 == 0:
                print(f"  [{split_name}] {i+1}/{len(subject_ids)} evaluated...")

        print(f"  [{split_name}] {len(results)}/{len(subject_ids)} completed.")
        return results

    val_results = _eval_split(val_subjects, "Val") if val_subjects else []
    test_results = _eval_split(test_subjects, "Test") if test_subjects else []

    val_agg = _aggregate_split_results(val_results)
    test_agg = _aggregate_split_results(test_results)
    combined_agg = _aggregate_split_results(val_results + test_results)

    all_rows = []
    for split_name, results in [("val", val_results), ("test", test_results)]:
        for r in results:
            all_rows.append({
                "split": split_name,
                "subject_id": r["subject_id"],
                "pred_class": r["pred_class"],
                "status": r["status"],
                "fields_checked": r["scores"]["fields_checked"],
                "fields_correct": r["scores"]["fields_correct"],
                "precision": r["scores"]["precision"],
                "tp": r["scores"]["tp"],
                "fp": r["scores"]["fp"],
                "fn": r["scores"]["fn"],
                "recall": r["scores"]["recall"],
                "f1": r["scores"]["f1"],
            })
    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            out_path / "factual_accuracy_details.csv", index=False
        )

    return {"val": val_agg, "test": test_agg, "combined": combined_agg}


# 7. FULL GENERATION PIPELINE

def generate_one_leaflet(subject_id, node_idx, gnn, data_df,
                         model, tokenizer, device,
                         mode="balanced", verbose=False,
                         session_id=None, hybrid=True, **kwargs):
    """Generate a single T5 bio-leaflet for one patient node.

    Hybrid mode (default): T5 generates narrative, template renders data sections.
    """
    patient_df = data_df[data_df["subject_id"] == subject_id]
    if patient_df.empty:
        warnings.warn(f"Subject {subject_id} not found in data CSV.")
        return None

    ctx = get_patient_context(subject_id, node_idx, gnn, patient_df,
                              session_id=session_id)
    structured_input = build_structured_input(ctx)

    if verbose:
        print(f"\n--- Structured Input for {subject_id} node {node_idx} ---")
        print(structured_input)
        print("---")

    if hybrid:
        leaflet, status, scores, meta = _reject_and_retry(
            ctx, model, tokenizer, device, structured_input,
            mode=mode, hybrid=True, verbose=verbose,
        )
    else:
        leaflet = generate_t5_text(model, tokenizer, device, structured_input, mode)
        status, _issues, scores = verify_t5_leaflet(leaflet, ctx)
        meta = {"n_retries": 0, "final_mode": mode, "fell_back_to_template": False}

    if verbose:
        print(f"  [{subject_id}] verification: {status} "
              f"({scores['fields_correct']}/{scores['fields_checked']} correct)")

    return leaflet, status, scores, meta


def generate_one_subject_leaflet(subject_id, gnn, data_df,
                                  model, tokenizer, device,
                                  mode="balanced", verbose=False,
                                  hybrid=True, **kwargs):
    """Generate a single subject-level T5 bio-leaflet aggregating all nodes.

    Hybrid mode (default): T5 generates narrative, template renders data sections.
    """
    subj_ctx = build_subject_context(subject_id, gnn, data_df)
    if subj_ctx is None:
        warnings.warn(f"Subject {subject_id} not found or has no nodes.")
        return None

    structured_input = build_structured_input(subj_ctx, subject_mode=True)

    if verbose:
        print(f"\n--- Subject-Level Input for {subject_id} "
              f"({subj_ctx['n_nodes']} nodes) ---")
        print(structured_input)
        print("---")

    if hybrid:
        leaflet, status, scores, meta = _reject_and_retry(
            subj_ctx, model, tokenizer, device, structured_input,
            mode=mode, hybrid=True, verbose=verbose,
        )
    else:
        leaflet = generate_t5_text(model, tokenizer, device, structured_input, mode)
        status, _issues, scores = verify_t5_leaflet(leaflet, subj_ctx)
        meta = {"n_retries": 0, "final_mode": mode, "fell_back_to_template": False}

    if verbose:
        print(f"  [{subject_id}] verification: {status} "
              f"({scores['fields_correct']}/{scores['fields_checked']} correct)")

    return leaflet, status, scores, meta


def generate_all_t5_leaflets(results_dir, data_csv, model_path, out_dir,
                              mode="balanced", verbose=False,
                              legacy=False, hybrid=True, **kwargs):
    """Batch-generate T5 leaflets.

    Default: one leaflet per unique subject (subject-level aggregation).
    Legacy: one leaflet per node (original behaviour).
    Hybrid mode (default): T5 generates narrative, template renders data sections.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gnn = load_gnn_outputs(results_dir)
    data_df = load_patient_data(data_csv)
    model, tokenizer, device = load_t5_model(model_path)

    unc_df = gnn["uncertainty"]
    n_generated = 0
    n_pass = 0
    n_high_unc = 0
    verification_rows = []

    if legacy:
        # --- Legacy per-node mode ---
        n_total = len(unc_df)
        print(f"Generating T5 bio-leaflets for {n_total} nodes (legacy mode)...")
        print(f"  Mode: {mode}")
        print()

        has_session_col = "clinical_session_id" in unc_df.columns
        for _, row in unc_df.iterrows():
            node_idx = int(row["node_idx"])
            subject_id = str(row["subject_id"]).strip()
            if has_session_col and pd.notna(row.get("clinical_session_id")):
                sess = str(row["clinical_session_id"]).strip()
            elif node_idx < len(data_df) and "clinical_session_id" in data_df.columns:
                sess = str(data_df.iloc[node_idx]["clinical_session_id"]).strip()
            else:
                sess = None

            if verbose:
                print(f"Processing node {node_idx} ({subject_id})...")

            result = generate_one_leaflet(
                subject_id, node_idx, gnn, data_df,
                model, tokenizer, device,
                mode=mode, verbose=verbose,
                session_id=sess, hybrid=hybrid,
            )
            if result is None:
                continue

            leaflet, status, scores, meta = result
            n_generated += 1
            if status == "PASS":
                n_pass += 1

            is_high = str(row["is_high_uncertainty"]).strip().lower() == "true"
            if is_high:
                n_high_unc += 1

            safe_id = re.sub(r"[^\w\-.]", "_", subject_id)
            fname = f"{safe_id}_node{node_idx}_t5_leaflet.txt"
            with open(out_path / fname, "w", encoding="utf-8") as f:
                f.write(leaflet)

            verification_rows.append({
                "subject_id": subject_id,
                "node_idx": node_idx,
                "pred_class": str(row["pred_class"]),
                "verification_status": status,
                "fields_checked": scores["fields_checked"],
                "fields_correct": scores["fields_correct"],
                "precision": scores["precision"],
                "tp": scores.get("tp", 0),
                "fp": scores.get("fp", 0),
                "fn": scores.get("fn", 0),
                "recall": scores.get("recall", 0),
                "f1": scores.get("f1", 0),
                "is_high_uncertainty": is_high,
                "n_retries": meta["n_retries"],
                "final_mode": meta["final_mode"],
                "fell_back_to_template": meta["fell_back_to_template"],
            })
    else:
        # --- Subject-level mode (default) ---
        unique_subjects = unc_df["subject_id"].str.strip().unique()
        n_total = len(unique_subjects)
        print(f"Generating subject-level T5 bio-leaflets for {n_total} subjects...")
        print(f"  Mode: {mode}")
        print()

        sample_printed = set()  # print one sample leaflet per class (CN/MCI/AD)

        for subject_id in unique_subjects:
            if verbose:
                print(f"Processing subject {subject_id}...")

            result = generate_one_subject_leaflet(
                subject_id, gnn, data_df,
                model, tokenizer, device,
                mode=mode, verbose=verbose, hybrid=hybrid,
            )
            if result is None:
                continue

            leaflet, status, scores, meta = result
            n_generated += 1
            if status == "PASS":
                n_pass += 1

            subj_ctx = build_subject_context(subject_id, gnn, data_df)
            if subj_ctx and subj_ctx.get("any_high_uncertainty"):
                n_high_unc += 1

            safe_id = re.sub(r"[^\w\-.]", "_", subject_id)
            fname = f"{safe_id}_summary_leaflet.txt"
            with open(out_path / fname, "w", encoding="utf-8") as f:
                f.write(leaflet)

            # Print one sample leaflet per class to console
            if verbose and subj_ctx and subj_ctx.get("pred_class") not in sample_printed:
                cls = subj_ctx["pred_class"]
                sample_printed.add(cls)
                label = {"0": "CN", "0.5": "MCI", "1+": "AD"}.get(cls, cls)
                print(f"\n{'='*60}")
                print(f"  SAMPLE LEAFLET -- {label} ({subject_id})")
                print(f"{'='*60}")
                print(leaflet)
                print(f"{'='*60}\n")

            verification_rows.append({
                "subject_id": subject_id,
                "n_nodes": subj_ctx["n_nodes"] if subj_ctx else 0,
                "pred_class": subj_ctx["pred_class"] if subj_ctx else "?",
                "stability": subj_ctx["stability"]["stability_label"] if subj_ctx else "?",
                "verification_status": status,
                "fields_checked": scores["fields_checked"],
                "fields_correct": scores["fields_correct"],
                "precision": scores["precision"],
                "tp": scores.get("tp", 0),
                "fp": scores.get("fp", 0),
                "fn": scores.get("fn", 0),
                "recall": scores.get("recall", 0),
                "f1": scores.get("f1", 0),
                "any_high_uncertainty": subj_ctx.get("any_high_uncertainty", False) if subj_ctx else False,
                "n_retries": meta["n_retries"],
                "final_mode": meta["final_mode"],
                "fell_back_to_template": meta["fell_back_to_template"],
            })

    ver_df = pd.DataFrame(verification_rows)
    ver_df.to_csv(out_path / "verification_summary.csv", index=False)

    unit = "nodes" if legacy else "subjects"
    n_retried = sum(1 for r in verification_rows if r.get("n_retries", 0) > 0)
    n_fallback = sum(1 for r in verification_rows if r.get("fell_back_to_template"))
    print()
    print("=" * 60)
    print("T5 GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Total {unit}:           {n_total}")
    print(f"  Leaflets generated:      {n_generated}")
    print(f"  Verification PASS:       {n_pass}")
    print(f"  Verification FAIL:       {n_generated - n_pass}")
    print(f"  Required retries:        {n_retried}")
    print(f"  Template fallbacks:      {n_fallback}")
    print(f"  High-uncertainty flagged: {n_high_unc}")
    print(f"  Output directory:        {out_dir}")
    print("=" * 60)


# 8. AUDIT — comprehensive hallucination analysis

def run_audit(results_dir, data_csv, model_path, out_dir="./audit_results",
              mode="deterministic"):
    """Run a comprehensive hallucination audit on all test subjects.

    Generates leaflets for every subject, categorises all verification
    failures by type, and saves a detailed audit report.

    Also evaluates ROUGE on the test set if training CSVs are found.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gnn = load_gnn_outputs(results_dir)
    data_df = load_patient_data(data_csv)
    model, tokenizer, device = load_t5_model(model_path)

    unc_df = gnn["uncertainty"]
    unique_subjects = unc_df["subject_id"].str.strip().unique()

    audit_rows = []
    all_issues = []
    n_pass = 0
    n_fail = 0
    n_fallback = 0

    print(f"Auditing {len(unique_subjects)} subjects (mode: {mode})...")
    print()

    for subject_id in unique_subjects:
        result = generate_one_subject_leaflet(
            subject_id, gnn, data_df,
            model, tokenizer, device,
            mode=mode, hybrid=True, verbose=False,
        )
        if result is None:
            continue

        leaflet, status, scores, meta = result

        # Re-run verification to get issues list directly
        subj_ctx = build_subject_context(subject_id, gnn, data_df)
        if subj_ctx is None:
            continue
        _, issues, _ = verify_t5_leaflet(leaflet, subj_ctx)

        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1
        if meta["fell_back_to_template"]:
            n_fallback += 1

        # Categorise issues
        issue_cats = {}
        for iss in issues:
            cat = iss.split(":")[0] if ":" in iss else "OTHER"
            issue_cats[cat] = issue_cats.get(cat, 0) + 1
            all_issues.append({
                "subject_id": subject_id,
                "category": cat,
                "detail": iss,
                "pred_class": subj_ctx["pred_class"],
            })

        audit_rows.append({
            "subject_id": subject_id,
            "pred_class": subj_ctx["pred_class"],
            "status": status,
            "n_issues": len(issues),
            "issue_categories": "; ".join(
                f"{k}({v})" for k, v in sorted(issue_cats.items())
            ),
            "n_retries": meta["n_retries"],
            "final_mode": meta["final_mode"],
            "fell_back_to_template": meta["fell_back_to_template"],
            "precision": scores["precision"],
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out_path / "audit_summary.csv", index=False)

    issues_df = pd.DataFrame(all_issues)
    if not issues_df.empty:
        issues_df.to_csv(out_path / "audit_issues_detail.csv", index=False)

    n_total = n_pass + n_fail
    print()
    print("=" * 70)
    print("HALLUCINATION AUDIT REPORT")
    print("=" * 70)
    print(f"  Subjects audited:     {n_total}")
    print(f"  PASS:                 {n_pass} ({100*n_pass/max(n_total,1):.1f}%)")
    print(f"  FAIL:                 {n_fail} ({100*n_fail/max(n_total,1):.1f}%)")
    print(f"  Template fallbacks:   {n_fallback}")
    print()

    if not issues_df.empty:
        print("Issue breakdown by category:")
        cat_counts = issues_df["category"].value_counts()
        for cat, count in cat_counts.items():
            n_subjects = issues_df[issues_df["category"] == cat]["subject_id"].nunique()
            print(f"  {cat:<30} {count:>4} occurrences  ({n_subjects} subjects)")
        print()

        # Per-stage breakdown
        print("Issues by predicted stage:")
        for stage in ["0", "0.5", "1+"]:
            stage_issues = issues_df[issues_df["pred_class"] == stage]
            if not stage_issues.empty:
                print(f"  {STAGE_DISPLAY.get(stage, stage)}: "
                      f"{len(stage_issues)} issues across "
                      f"{stage_issues['subject_id'].nunique()} subjects")
    else:
        print("  No issues detected — all leaflets passed verification.")

    print()
    print(f"  Detailed results saved to: {out_path}")
    print("=" * 70)

    # Try ROUGE evaluation on test set if available
    model_dir = Path(model_path)
    test_csv = model_dir / "adni_t5_test.csv"
    if not test_csv.exists():
        test_csv = model_dir.parent / "adni_t5_test.csv"
    if test_csv.exists():
        print("\nRunning ROUGE evaluation on test set...")
        evaluate_rouge_on_test(str(model_path), str(test_csv))
    else:
        print(f"\n  (Test CSV not found at {test_csv} — skipping ROUGE evaluation)")

    print("\nRunning ROUGE evaluation on all subjects...")
    evaluate_rouge_all_subjects(results_dir, data_csv, model_path, out_dir)

    return audit_df


# FULL EVALUATION — paper-style tables

def _print_table(title, headers, rows, col_widths=None):
    """Print a formatted ASCII table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    print(f"\n{title}")
    print(sep)
    print("|" + "|".join(str(h).center(w) for h, w in zip(headers, col_widths)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(str(row[i]).center(w) for i, w in enumerate(col_widths)) + "|")
    print(sep)


def run_full_evaluation(results_dir, data_csv, model_path,
                        output_dir="./evaluation_results", mode="deterministic"):
    """Evaluate FLAN-T5 on the validation and test sets.

Reports ROUGE, factual accuracy, error distribution, precision/recall/F1,
and category- and CDR-level results.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

   
    print("COMPREHENSIVE EVALUATION REPORT")


 
    # TABLE ROUGE scores on val and test
  
    model_dir = Path(model_path)
    val_csv = model_dir / "adni_t5_val.csv"
    test_csv = model_dir / "adni_t5_test.csv"
    if not val_csv.exists():
        val_csv = model_dir.parent / "adni_t5_val.csv"
    if not test_csv.exists():
        test_csv = model_dir.parent / "adni_t5_test.csv"

    val_rouge = {}
    test_rouge = {}
    if val_csv.exists():
        print("\nComputing ROUGE on validation set...")
        val_rouge = evaluate_rouge_on_test(str(model_path), str(val_csv))
    else:
        print(f"\n  (Val CSV not found — skipping val ROUGE)")

    if test_csv.exists():
        print("\nComputing ROUGE on test set...")
        test_rouge = evaluate_rouge_on_test(str(model_path), str(test_csv))
    else:
        print(f"\n  (Test CSV not found — skipping test ROUGE)")

    all_rouge = evaluate_rouge_all_subjects(results_dir, data_csv, model_path, str(out_path))

    if val_rouge or test_rouge or all_rouge:
        rouge_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_rows = []
        for key in rouge_keys:
            v = val_rouge.get(f"eval_{key}", 0)
            t = test_rouge.get(f"eval_{key}", 0)
            a = all_rouge.get(f"eval_{key}", 0)
            diff = t - v if (v and t) else 0
            rouge_rows.append([
                key.upper().replace("ROUGE", "ROUGE-"),
                f"{v:.2f}%" if v else "N/A",
                f"{t:.2f}%" if t else "N/A",
                f"{diff:+.2f}%" if (v and t) else "N/A",
                f"{a:.2f}%" if a else "N/A",
            ])
        _print_table(
            "TABLE I: ROUGE SCORES ON VALIDATION, TEST, AND ALL SUBJECTS",
            ["Metric", "Validation", "Test", "Difference", "All Subjects*"],
            rouge_rows,
            col_widths=[14, 14, 14, 14, 16],
        )
        print("  * All Subjects includes subjects seen during training — "
              "not a generalization metric.")

    
    # TABLES Factual accuracy
    
    print("\nRunning factual accuracy evaluation...")
    fact_results = test_factual_accuracy(
        results_dir, data_csv, model_path,
        out_dir=str(out_path), mode=mode,
    )

    if not fact_results:
        print("  Factual accuracy evaluation failed — no results.")
        return

    # TABLE  Factual Accuracy
    fa_rows = []
    for split_name, key in [("Validation", "val"), ("Test", "test"), ("Combined", "combined")]:
        agg = fact_results[key]
        if agg["n"] == 0:
            continue
        fa_rows.append([
            split_name, agg["n"], f"{agg['pass_rate']}%",
            agg["fields_checked"], agg["fields_correct"],
            f"{agg['field_accuracy']}%",
        ])
    _print_table(
        "TABLE II: FACTUAL ACCURACY RESULTS",
        ["Dataset", "N", "Pass Rate", "Fields Checked", "Correct Fields", "Accuracy"],
        fa_rows,
        col_widths=[12, 6, 12, 16, 16, 10],
    )

    # TABLE Error Distribution by Field Category
    combined = fact_results["combined"]
    cat_rows = []
    for cat, stats in sorted(combined["by_category"].items()):
        total = stats["total"]
        errors = stats["errors"]
        rate = round(errors / max(total, 1) * 100, 2)
        cat_rows.append([cat, total, errors, f"{rate}%"])
    if cat_rows:
        _print_table(
            "TABLE III: ERROR DISTRIBUTION BY FIELD CATEGORY",
            ["Category", "Total Instances", "Errors", "Error Rate"],
            cat_rows,
            col_widths=[18, 18, 10, 12],
        )

    # TABLE IV — Precision, Recall, F1
    prf_rows = []
    for split_name, key in [("Validation", "val"), ("Test", "test"), ("Combined", "combined")]:
        agg = fact_results[key]
        if agg["n"] == 0:
            continue
        prf_rows.append([f"{split_name} (N={agg['n']})", "", "", ""])
        prf_rows.append(["  Micro-averaged", "", "", ""])
        prf_rows.append(["    True Positives", str(agg["micro_tp"]), "", ""])
        prf_rows.append(["    False Positives", str(agg["micro_fp"]), "", ""])
        prf_rows.append(["    False Negatives", str(agg["micro_fn"]), "", ""])
        prf_rows.append(["    Precision", f"{agg['micro_p']:.2f}%", "", ""])
        prf_rows.append(["    Recall", f"{agg['micro_r']:.2f}%", "", ""])
        prf_rows.append(["    F1 Score", f"{agg['micro_f1']:.2f}%", "", ""])
        prf_rows.append(["  Macro-averaged", "", "", ""])
        prf_rows.append(["    Avg Precision", f"{agg['macro_p']:.2f}%", "", ""])
        prf_rows.append(["    Avg Recall", f"{agg['macro_r']:.2f}%", "", ""])
        prf_rows.append(["    Avg F1 Score", f"{agg['macro_f1']:.2f}%", "", ""])

    print("\nTABLE IV: PRECISION, RECALL, AND F1 SCORES")
    print("=" * 50)
    for row in prf_rows:
        if row[1]:
            print(f"  {row[0]:<30} {row[1]}")
        else:
            print(f"  {row[0]}")
    print("=" * 50)

    # TABLE V — Precision and Recall by Clinical Category
    cat_prf_rows = []
    overall_tp = overall_fp = overall_fn = 0
    cat_order = ["Demographics", "Cognitive Scores", "MRI Biomarkers",
                 "CSF Biomarkers", "GNN Prediction", "Longitudinal"]
    for cat in cat_order:
        if cat not in combined["by_category"]:
            continue
        s = combined["by_category"][cat]
        c_tp, c_fp, c_fn = s.get("tp", 0), s.get("fp", 0), s.get("fn", 0)
        overall_tp += c_tp
        overall_fp += c_fp
        overall_fn += c_fn
        c_p = (c_tp / (c_tp + c_fp) * 100) if (c_tp + c_fp) > 0 else 0.0
        c_r = (c_tp / (c_tp + c_fn) * 100) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
        cat_prf_rows.append([
            cat, c_tp, c_fp, c_fn,
            f"{c_p:.2f}%", f"{c_r:.2f}%", f"{c_f1:.2f}%",
        ])
    # Overall row
    o_p = (overall_tp / (overall_tp + overall_fp) * 100) if (overall_tp + overall_fp) > 0 else 0.0
    o_r = (overall_tp / (overall_tp + overall_fn) * 100) if (overall_tp + overall_fn) > 0 else 0.0
    o_f1 = (2 * o_p * o_r / (o_p + o_r)) if (o_p + o_r) > 0 else 0.0
    cat_prf_rows.append([
        "Overall", overall_tp, overall_fp, overall_fn,
        f"{o_p:.2f}%", f"{o_r:.2f}%", f"{o_f1:.2f}%",
    ])
    if cat_prf_rows:
        _print_table(
            "TABLE V: PRECISION AND RECALL BY CLINICAL CATEGORY",
            ["Category", "TP", "FP", "FN", "Precision", "Recall", "F1"],
            cat_prf_rows,
            col_widths=[18, 6, 6, 6, 12, 12, 12],
        )

    # TABLE VI — Performance by Diagnosis Category
    cdr_display = {"0": "0 (Normal/CN)", "0.5": "0.5 (MCI)", "1+": "1+ (AD)"}
    cdr_rows = []
    for cls in ["0", "0.5", "1+"]:
        if cls in combined["by_cdr"]:
            c = combined["by_cdr"][cls]
            cdr_rows.append([
                cdr_display.get(cls, cls),
                c["n"],
                f"{c['field_accuracy']}%",
                f"{c['avg_f1']}%",
                f"{c['pass_rate']}%",
            ])
    if cdr_rows:
        _print_table(
            "TABLE VI: PERFORMANCE BY DIAGNOSIS CATEGORY",
            ["Dx Category", "N", "Field Accuracy", "F1 Score", "Pass Rate"],
            cdr_rows,
            col_widths=[18, 6, 16, 12, 12],
        )

    summary_rows = []
    for split_name, key in [("val", "val"), ("test", "test"), ("combined", "combined")]:
        agg = fact_results[key]
        if agg["n"] == 0:
            continue
        row = {"split": split_name, "n": agg["n"]}
        row["pass_rate"] = agg["pass_rate"]
        row["field_accuracy"] = agg["field_accuracy"]
        row["micro_p"] = agg["micro_p"]
        row["micro_r"] = agg["micro_r"]
        row["micro_f1"] = agg["micro_f1"]
        row["macro_p"] = agg["macro_p"]
        row["macro_r"] = agg["macro_r"]
        row["macro_f1"] = agg["macro_f1"]
        # Add ROUGE if available
        rouge_data = val_rouge if key == "val" else test_rouge if key == "test" else {}
        for rk in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            row[rk] = rouge_data.get(f"eval_{rk}", "")
            row[f"all_{rk}"] = all_rouge.get(f"eval_{rk}", "")
        summary_rows.append(row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            out_path / "evaluation_report.csv", index=False
        )
        print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return fact_results


#  CLI

def main():
    parser = argparse.ArgumentParser(
        description="FLAN-T5 bio-leaflet generation for ADNI Spectral GCN pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # --- train ---
    train_p = subparsers.add_parser(
        "train", help="Generate training data and fine-tune T5"
    )
    train_p.add_argument("--results_dir", required=True,
                         help="Directory with paper8_1.py output files")
    train_p.add_argument("--data_csv", required=True,
                         help="Path to ADNI_version1.csv")
    train_p.add_argument("--output_dir", default="./flan-t5-adni-final",
                         help="Output dir for model + training CSVs")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch_size", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--legacy", action="store_true",
                         help="Use per-node training pairs only (no subject-level)")
    train_p.add_argument("--verbose", action="store_true")

    # --- generate ---
    gen_p = subparsers.add_parser(
        "generate", help="Generate leaflets using fine-tuned model"
    )
    gen_p.add_argument("--results_dir", required=True)
    gen_p.add_argument("--data_csv", required=True)
    gen_p.add_argument("--model_path", required=True,
                       help="Path to fine-tuned T5 model directory")
    gen_p.add_argument("--out_dir", default="./results_t5")
    gen_p.add_argument("--mode", default="balanced",
                       choices=["deterministic", "conservative", "balanced", "creative"])
    gen_p.add_argument("--hybrid", action="store_true", default=True,
                       help="Use template-rendered GNN sections (default)")
    gen_p.add_argument("--no_hybrid", dest="hybrid", action="store_false",
                       help="Let T5 generate everything (no template GNN sections)")
    gen_p.add_argument("--patient", type=str, default=None,
                       help="Generate for a single patient subject_id")
    gen_p.add_argument("--legacy", action="store_true",
                       help="Generate per-node leaflets (no subject-level aggregation)")
    gen_p.add_argument("--verbose", action="store_true")

    # --- preview (quick test: show structured input without training) ---
    prev_p = subparsers.add_parser(
        "preview", help="Preview structured input for a patient (no model needed)"
    )
    prev_p.add_argument("--results_dir", required=True)
    prev_p.add_argument("--data_csv", required=True)
    prev_p.add_argument("--patient", type=str, default=None,
                        help="Subject ID (default: first subject)")
    prev_p.add_argument("--legacy", action="store_true",
                        help="Show per-node preview instead of subject-level")

    # --- audit ---
    audit_p = subparsers.add_parser(
        "audit", help="Run hallucination audit on all subjects"
    )
    audit_p.add_argument("--results_dir", required=True)
    audit_p.add_argument("--data_csv", required=True)
    audit_p.add_argument("--model_path", required=True,
                         help="Path to fine-tuned T5 model directory")
    audit_p.add_argument("--out_dir", default="./audit_results")
    audit_p.add_argument("--mode", default="deterministic",
                         choices=["deterministic", "conservative", "balanced", "creative"])

    # --- evaluate ---
    eval_p = subparsers.add_parser(
        "evaluate", help="Run full evaluation (ROUGE + factual accuracy on val & test)"
    )
    eval_p.add_argument("--results_dir", required=True,
                        help="Directory with paper8_1.py output files")
    eval_p.add_argument("--data_csv", required=True,
                        help="Path to ADNI_version1.csv")
    eval_p.add_argument("--model_path", required=True,
                        help="Path to fine-tuned T5 model directory")
    eval_p.add_argument("--out_dir", default="./evaluation_results",
                        help="Output dir for evaluation reports")
    eval_p.add_argument("--mode", default="deterministic",
                        choices=["deterministic", "conservative", "balanced", "creative"],
                        help="Generation mode for evaluation (deterministic recommended)")

    args = parser.parse_args()

    if args.command == "train":
        train_csv, val_csv, test_csv = generate_training_data(
            args.results_dir, args.data_csv, args.output_dir,
            verbose=args.verbose, legacy=args.legacy,
        )
        fine_tune(
            train_csv, val_csv, args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

    elif args.command == "generate":
        if args.patient:
            gnn = load_gnn_outputs(args.results_dir)
            data_df = load_patient_data(args.data_csv)
            model, tokenizer, device = load_t5_model(args.model_path)

            if args.legacy:
                # Legacy: generate per-node leaflets for this patient
                unc_df = gnn["uncertainty"]
                matches = unc_df[unc_df["subject_id"].str.strip() == args.patient]
                if matches.empty:
                    print(f"ERROR: Subject '{args.patient}' not found.")
                    return
                has_sess = "clinical_session_id" in unc_df.columns
                for _, row in matches.iterrows():
                    nidx = int(row["node_idx"])
                    if has_sess and pd.notna(row.get("clinical_session_id")):
                        sess = str(row["clinical_session_id"]).strip()
                    elif nidx < len(data_df) and "clinical_session_id" in data_df.columns:
                        sess = str(data_df.iloc[nidx]["clinical_session_id"]).strip()
                    else:
                        sess = None
                    result = generate_one_leaflet(
                        args.patient, nidx, gnn, data_df,
                        model, tokenizer, device,
                        mode=args.mode, hybrid=args.hybrid, verbose=True,
                        session_id=sess,
                    )
                    if result:
                        print(result[0])
            else:
                # Default: one subject-level leaflet
                result = generate_one_subject_leaflet(
                    args.patient, gnn, data_df,
                    model, tokenizer, device,
                    mode=args.mode, hybrid=args.hybrid, verbose=True,
                )
                if result:
                    print(result[0])
                else:
                    print(f"ERROR: Subject '{args.patient}' not found or has no nodes.")
        else:
            generate_all_t5_leaflets(
                args.results_dir, args.data_csv, args.model_path,
                args.out_dir, mode=args.mode, hybrid=args.hybrid,
                verbose=args.verbose, legacy=args.legacy,
            )

    elif args.command == "preview":
        gnn = load_gnn_outputs(args.results_dir)
        data_df = load_patient_data(args.data_csv)
        unc_df = gnn["uncertainty"]

        if args.legacy:
            # Legacy per-node preview
            if args.patient:
                matches = unc_df[unc_df["subject_id"].str.strip() == args.patient]
            else:
                matches = unc_df.head(1)

            if matches.empty:
                print("No matching patient found.")
                return

            has_sess = "clinical_session_id" in unc_df.columns
            for _, row in matches.iterrows():
                subject_id = str(row["subject_id"]).strip()
                node_idx = int(row["node_idx"])
                if has_sess and pd.notna(row.get("clinical_session_id")):
                    sess = str(row["clinical_session_id"]).strip()
                elif node_idx < len(data_df) and "clinical_session_id" in data_df.columns:
                    sess = str(data_df.iloc[node_idx]["clinical_session_id"]).strip()
                else:
                    sess = None
                patient_df = data_df[data_df["subject_id"] == subject_id]
                if patient_df.empty:
                    continue
                ctx = get_patient_context(subject_id, node_idx, gnn, patient_df,
                                          session_id=sess)
                print("=" * 70)
                print(f"STRUCTURED INPUT — {subject_id} (node {node_idx})")
                print("=" * 70)
                print(PREFIX + build_structured_input(ctx))
                print()
                print("=" * 70)
                print("NARRATIVE TARGET (what T5 learns to generate)")
                print("=" * 70)
                print(build_narrative_target(ctx))
        else:
            # Subject-level preview (default)
            if args.patient:
                subjects = [args.patient]
            else:
                subjects = [unc_df["subject_id"].str.strip().iloc[0]]

            for subject_id in subjects:
                subj_ctx = build_subject_context(subject_id, gnn, data_df)
                if subj_ctx is None:
                    print(f"Subject '{subject_id}' not found.")
                    continue

                print("=" * 70)
                print(f"STRUCTURED INPUT — {subject_id} "
                      f"({subj_ctx['n_nodes']} nodes)")
                print("=" * 70)
                print(PREFIX + build_structured_input(subj_ctx, subject_mode=True))
                print()
                print("=" * 70)
                print("NARRATIVE TARGET (what T5 learns to generate)")
                print("=" * 70)
                print(build_narrative_target(subj_ctx, subject_mode=True))

    elif args.command == "audit":
        run_audit(
            args.results_dir, args.data_csv, args.model_path,
            out_dir=args.out_dir, mode=args.mode,
        )

    elif args.command == "evaluate":
        run_full_evaluation(
            args.results_dir, args.data_csv, args.model_path,
            output_dir=args.out_dir, mode=args.mode,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
