#!/usr/bin/env python3
"""
bio_leaflet.py — Patient-level interpretation module for the hybrid GNN pipeline.

Reads model outputs from hybrid_gnn.py, including feature attribution,
uncertainty, graph-neighborhood information, and counterfactual explanations,
and renders structured patient-level interpretations using Jinja2 templates.
"""

import argparse
import json
import re
import os
import warnings
from collections import defaultdict
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
from jinja2 import Template

# Constants

STAGE_DISPLAY = {
    "0":   "Cognitively Normal (CN)",
    "0.5": "Mild Cognitive Impairment (MCI)",
    "1+":  "Alzheimer's Disease (AD)",
}

STAGE_SHORT = {
    "Cognitively Normal (CN)": "CN",
    "Mild Cognitive Impairment (MCI)": "MCI",
    "Alzheimer's Disease (AD)": "AD",
}

FEATURE_DISPLAY = {
    "clinical_MMSCORE":                "Global cognition (MMSE)",
    "clinical_FAQTOTAL":               "Functional assessment (FAQ)",
    "clinical_LDELTOTAL":              "Logical memory — delayed recall",
    "clinical_TRABSCOR":               "Executive function (Trail Making B)",
    "clinical_APOE4_count":            "APOE4 carrier status",
    "clinical_EDUCAT":                 "Years of education",
    "clinical_entry_age":              "Age at entry",
    "clinical_GENDER":                 "Sex",
    "clinical_ABETA42":                "CSF Amyloid-beta 42",
    "mri_hippocampus_vol_mean":        "Hippocampal volume (mean L/R)",
    "mri_entorhinal_vol_mean":         "Entorhinal cortex volume",
    "mri_amygdala_vol_mean":           "Amygdala volume",
    "mri_inferior_temporal_vol_mean":  "Inferior temporal volume",
    "mri_middle_temporal_vol_mean":    "Middle temporal volume",
    "mri_lateral_ventricle_vol_mean":  "Lateral ventricle volume",
    "mri_inf_lat_vent_vol_mean":       "Inferior lateral ventricle volume",
    "pet_PTAU":                        "CSF Phosphorylated-Tau",
    "pet_TAU":                         "CSF Total-Tau",
}

# Non-modifiable features — excluded from counterfactual suggestions
FIXED_FEATURES = {"clinical_APOE4_count", "clinical_entry_age", "clinical_EDUCAT"}

FEATURE_CATEGORIES = {
    "clinical_MMSCORE": "cognitive",
    "clinical_FAQTOTAL": "functional",
    "clinical_LDELTOTAL": "cognitive",
    "clinical_TRABSCOR": "cognitive",
    "clinical_APOE4_count": "genetic",
    "clinical_EDUCAT": "demographic",
    "clinical_entry_age": "demographic",
    "clinical_GENDER": "demographic",
    "clinical_ABETA42": "csf_biomarker",
    "pet_PTAU": "csf_biomarker",
    "pet_TAU": "csf_biomarker",
}

_CAT_LABELS = {
    "cognitive": "cognitive test scores",
    "functional": "functional assessment measures",
    "structural_mri": "structural brain volumes",
    "csf_biomarker": "CSF biomarkers",
    "pet_imaging": "PET imaging biomarkers",
    "genetic": "genetic factors",
    "demographic": "demographic factors",
    "clinical": "clinical measures",
}

_DRIVER_PHRASES = {
    "cognitive": "cognitive function",
    "functional": "functional status",
    "structural_mri": "brain structure",
    "csf_biomarker": "molecular pathology",
    "pet_imaging": "amyloid/tau PET imaging",
    "genetic": "genetic risk",
    "demographic": "demographic profile",
    "clinical": "clinical presentation",
}


def _categorize_feature(fname):
    """Map a feature name to its category via exact match then prefix fallback."""
    if fname in FEATURE_CATEGORIES:
        return FEATURE_CATEGORIES[fname]
    if fname.startswith("mri_"):
        return "structural_mri"
    if fname.startswith("pet_"):
        # pet_PTAU/pet_TAU are actually CSF biomarkers — the pipeline just
        # kept the pet_ prefix. Real PET imaging would carry an SUVR/uptake
        # suffix (e.g., pet_precuneus_suvr).
        if "_suvr" in fname.lower() or "_uptake" in fname.lower():
            return "pet_imaging"
        return "csf_biomarker"
    if fname.startswith("clinical_"):
        return "clinical"
    return "clinical"


# Utility helpers

def safe_fmt(value, decimals=2):
    """Format a numeric value safely; return 'N/A' for missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_delta(value, feature_name=""):
    """Format a longitudinal delta value with +/- prefix."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    v = float(value)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"


def auto_display_name(col_name):
    """Fallback human-readable name: strip prefix, replace underscores."""
    for prefix in ("clinical_", "mri_", "pet_", "csf_"):
        if col_name.startswith(prefix):
            col_name = col_name[len(prefix):]
            break
    return col_name.replace("_", " ").title()


def display_name(feature):
    """Return best available display name for a feature."""
    return FEATURE_DISPLAY.get(feature, auto_display_name(feature))


def sex_label(val):
    """Convert numeric sex code to string."""
    mapping = {1: "Male", 2: "Female", "1": "Male", "2": "Female",
               "M": "Male", "F": "Female", "Male": "Male", "Female": "Female"}
    return mapping.get(val, str(val))


def apoe4_label(count):
    """Human-readable APOE4 status."""
    try:
        c = int(float(count))
    except (ValueError, TypeError):
        return "N/A"
    return {0: "Non-carrier (0 copies)", 1: "Heterozygous (1 copy)",
            2: "Homozygous (2 copies)"}.get(c, f"{c} copies")


def session_to_month(session_id):
    """Convert ADNI session_id (sc, m06, m12, …) to numeric month."""
    s = str(session_id).strip().lower()
    if s in ("sc", "scmri", "bl"):
        return 0
    m = re.match(r"m?(\d+)", s)
    return int(m.group(1)) if m else 0


def certainty_label(is_high_unc):
    """Map boolean high-uncertainty flag to a readable label."""
    if isinstance(is_high_unc, str):
        is_high_unc = is_high_unc.strip().lower() == "true"
    return "limited — interpret with caution" if is_high_unc else "adequate"


def confidence_qualifier(ctx):
    """Map predictive entropy to a plain-language confidence level."""
    try:
        entropy = float(ctx["predictive_entropy"])
    except (ValueError, TypeError):
        return "unknown"
    if entropy < 0.3:
        return "high"
    elif entropy < 0.7:
        return "moderate"
    return "low"


def clinical_magnitude(sd_value):
    """Map a standard-deviation magnitude to plain clinical language."""
    try:
        v = abs(float(sd_value))
    except (ValueError, TypeError):
        return "an uncertain amount"
    if v < 0.5:
        return "a small change"
    elif v < 1.5:
        return "a moderate change"
    return "a large change"


# Accessible-tone jargon simplification

# (clinical term, accessible replacement)
_ACCESSIBLE_SUBSTITUTIONS = [
    ("neurodegenerative change", "brain changes associated with aging and disease"),
    ("neurodegenerative processes", "brain changes associated with aging and disease"),
    ("medial temporal structures", "memory-related brain areas"),
    ("medial temporal regions", "memory-related brain areas"),
    ("hippocampal volume", "memory center size"),
    ("entorhinal cortex", "memory gateway area"),
    ("Alzheimer's-type pathology", "Alzheimer's-related protein changes"),
    ("amyloid pathology", "Alzheimer's-related protein buildup"),
    ("tau pathology", "protein tangles linked to brain cell damage"),
    ("cerebrospinal fluid", "spinal fluid"),
    ("cognitive reserve", "brain resilience built through education and activity"),
    ("executive function", "planning and decision-making ability"),
    ("cortical atrophy", "brain tissue thinning"),
    ("mild cognitive impairment", "early memory and thinking changes"),
    ("neuropsychological assessment", "memory and thinking tests"),
    ("biomarker profile", "biological indicators"),
    ("longitudinal trajectory", "change over time"),
    ("disease progression", "how the condition changes over time"),
]


def _apply_tone(text, tone="accessible"):  # DEPRECATED
    """Simplify clinical jargon for an accessible tone.

    Only tone="accessible" is supported; any other value returns text unchanged.
    """
    if tone != "accessible":
        return text
    result = text
    for clinical, plain in _ACCESSIBLE_SUBSTITUTIONS:
        result = re.sub(re.escape(clinical), plain, result, flags=re.IGNORECASE)
    return result



_CACHE = {}

def load_gnn_outputs(results_dir):
    
    key = str(results_dir)
    if key in _CACHE:
        return _CACHE[key]

    rdir = Path(results_dir)
    data = {}

    with open(rdir / "metrics.json") as f:
        data["metrics"] = json.load(f)

    lm_path = rdir / "label_mapping.json"
    if lm_path.exists():
        with open(lm_path) as f:
            data["label_mapping"] = json.load(f)

    data["feature_importance"] = pd.read_csv(rdir / "feature_importance.csv")
    data["uncertainty"] = pd.read_csv(rdir / "uncertainty_estimates.csv")
    data["neighbors"] = pd.read_csv(rdir / "neighbor_influence.csv")
    data["counterfactuals"] = pd.read_csv(rdir / "counterfactual_explanations.csv")

    # class columns get compared as strings downstream, so normalize dtype here
    for col in ("pred_class", "true_class", "cf_class"):
        if col in data["uncertainty"].columns:
            data["uncertainty"][col] = data["uncertainty"][col].astype(str).str.strip()
        if col in data["counterfactuals"].columns:
            data["counterfactuals"][col] = data["counterfactuals"][col].astype(str).str.strip()
    if "class" in data["feature_importance"].columns:
        data["feature_importance"]["class"] = (
            data["feature_importance"]["class"].astype(str).str.strip()
        )

    _CACHE[key] = data
    return data


def load_patient_data(data_csv):
    
    return pd.read_csv(data_csv)


# Feature importance helpers

def get_top_features(fi_df, pred_class, top_k=5):
    """Return top-K features for a given predicted class from GNNExplainer."""
    cls_df = fi_df[fi_df["class"] == str(pred_class)].sort_values("rank")
    return cls_df.head(top_k)[["feature", "importance"]].to_dict("records")


# Longitudinal deltas

def compute_deltas(patient_df):
    """Compute first-to-last visit deltas for all numeric columns.

    Expects patient_df to contain 'clinical_session_id'. Returns dict with
    keys: deltas, n_visits, time_span_months.
    """
    if len(patient_df) < 2:
        return {"deltas": {}, "n_visits": len(patient_df), "time_span_months": 0}

    df = patient_df.copy()
    df["_visit_month"] = df["clinical_session_id"].apply(session_to_month)
    df = df.sort_values("_visit_month")
    first = df.iloc[0]
    last = df.iloc[-1]
    time_span = int(last["_visit_month"] - first["_visit_month"])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    skip = {"_visit_month", "clinical_GENDER", "clinical_APOE4_count", "clinical_EDUCAT"}
    deltas = {}
    for col in numeric_cols:
        if col in skip:
            continue
        v_first, v_last = first[col], last[col]
        if pd.notna(v_first) and pd.notna(v_last):
            deltas[col] = float(v_last - v_first)

    return {"deltas": deltas, "n_visits": len(df), "time_span_months": time_span}


# Assemble patient context

def get_patient_context(subject_id, node_idx, gnn, patient_df, session_id=None):
    """Build the full template context dict for one patient.

    If session_id is given, biomarkers are pulled from that visit instead
    of the latest one, falling back to the latest visit if no match is found.
    """
    # --- Uncertainty row ---
    unc_row = gnn["uncertainty"][gnn["uncertainty"]["node_idx"] == node_idx].iloc[0]
    pred_class = str(unc_row["pred_class"]).strip()
    if pred_class not in STAGE_DISPLAY:
        raise ValueError(
            f"Unexpected class code {pred_class!r}; expected one of "
            f"{list(STAGE_DISPLAY)}. Check label_mapping.json."
        )
    true_class = str(unc_row["true_class"]).strip()

    # --- Visit-specific (or latest) raw biomarkers ---
    pdf = patient_df.copy()
    pdf["_vm"] = pdf["clinical_session_id"].apply(session_to_month)
    if session_id is not None:
        match = pdf[pdf["clinical_session_id"].str.strip() == str(session_id).strip()]
        latest = match.iloc[0] if not match.empty else pdf.sort_values("_vm").iloc[-1]
    else:
        latest = pdf.sort_values("_vm").iloc[-1]

    # --- Top features ---
    top_feats = get_top_features(gnn["feature_importance"], pred_class, top_k=5)
    feat_list = []
    for row in top_feats:
        fname = row["feature"]
        raw_val = latest.get(fname, np.nan)
        feat_list.append({
            "feature": fname,
            "display_name": display_name(fname),
            "value": safe_fmt(raw_val),
            "importance": safe_fmt(row["importance"], 4),
        })

    # --- Neighbor influence (rank-1 row for this node) ---
    nb_df = gnn["neighbors"][gnn["neighbors"]["node_idx"] == node_idx]
    if not nb_df.empty:
        nb_top = nb_df.sort_values("rank").iloc[0]
        graph_agreement_pct = safe_fmt(nb_top.get("graph_agreement_pct", np.nan), 1)
        # Class distribution in neighborhood
        cls_pcts = {}
        classes = gnn.get("label_mapping", {}).get("classes", ["0", "0.5", "1+"])
        for c in classes:
            col = f"class_influence_pct_{c}"
            if col in nb_top.index:
                cls_pcts[c] = safe_fmt(nb_top[col], 1)
        if not cls_pcts:
            expected = [f"class_influence_pct_{c}" for c in classes]
            raise KeyError(
                f"None of {expected} found in neighbor_influence.csv columns: "
                f"{list(nb_top.index)}"
            )
        neighbor_dist_str = ", ".join(
            f"{STAGE_DISPLAY.get(c, c)}: {v}%" for c, v in cls_pcts.items()
        )
        # Dominant class
        def _pct(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0
        dominant_class = max(cls_pcts, key=lambda c: _pct(cls_pcts[c]))
        dominant_neighbor_class = STAGE_DISPLAY.get(dominant_class, dominant_class)
        top_neighbor_idx = int(nb_top["neighbor_idx"])
        top_neighbor_pct = safe_fmt(nb_top["influence_pct"], 1)
        agreement_val = float(nb_top.get("graph_agreement_pct", 0))
        agreement_label = "strong" if agreement_val >= 60 else ("moderate" if agreement_val >= 40 else "weak")
    else:
        graph_agreement_pct = "N/A"
        neighbor_dist_str = "N/A"
        dominant_neighbor_class = "N/A"
        top_neighbor_idx = "N/A"
        top_neighbor_pct = "N/A"
        agreement_label = "N/A"

    # --- Counterfactual ---
    cf_df = gnn["counterfactuals"][gnn["counterfactuals"]["node_idx"] == node_idx]
    has_cf = not cf_df.empty
    cf_changes = []
    cf_class_display = ""
    if has_cf:
        cf_row = cf_df.iloc[0]
        cf_class = str(cf_row.get("cf_class", "")).strip()
        cf_class_display = STAGE_DISPLAY.get(cf_class, cf_class)
        for i in (1, 2):
            feat_col = f"feat_{i}"
            dir_col = f"dir_{i}"
            pct_col = f"pct_{i}"
            feat_val = cf_row.get(feat_col)
            if pd.notna(feat_val) and str(feat_val).strip():
                cf_changes.append({
                    "feature": str(feat_val),
                    "feature_display": display_name(str(feat_val)),
                    "direction": "increase" if str(cf_row.get(dir_col, "+")).strip() == "+" else "decrease",
                    "magnitude": safe_fmt(cf_row.get(pct_col), 1),
                    "clinical_magnitude": clinical_magnitude(cf_row.get(pct_col)),
                    "is_fixed": str(feat_val) in FIXED_FEATURES,
                })

    # --- Longitudinal ---
    delta_info = compute_deltas(patient_df)

    # --- Assemble context ---
    ctx = {
        # Header
        "patient_id": subject_id,
        "pred_class": pred_class,
        "true_class": true_class,
        "clinical_status": STAGE_DISPLAY.get(pred_class, pred_class),
        "age": safe_fmt(latest.get("clinical_entry_age"), 1),
        "sex": sex_label(latest.get("clinical_GENDER")),
        "apoe4_status": apoe4_label(latest.get("clinical_APOE4_count")),
        "education_years": safe_fmt(latest.get("clinical_EDUCAT"), 0),

        # Prediction
        "pred_class_display": STAGE_DISPLAY.get(pred_class, pred_class),
        "confidence_pct": safe_fmt(float(unc_row["max_prob"]) * 100, 1),
        "certainty_label": certainty_label(unc_row["is_high_uncertainty"]),
        "is_high_uncertainty": bool(
            str(unc_row["is_high_uncertainty"]).strip().lower() == "true"
        ),

        # Top features
        "top_features": feat_list,

        # Neighborhood
        "neighbor_dist_str": neighbor_dist_str,
        "dominant_neighbor_class": dominant_neighbor_class,
        "graph_agreement_pct": graph_agreement_pct,
        "agreement_label": agreement_label,
        "top_neighbor_idx": top_neighbor_idx,
        "top_neighbor_pct": top_neighbor_pct,

        # Counterfactual
        "has_counterfactual": has_cf,
        "cf_class_display": cf_class_display,
        "pred_class_short": STAGE_SHORT.get(STAGE_DISPLAY.get(pred_class, pred_class), pred_class),
        "cf_class_short": STAGE_SHORT.get(cf_class_display, cf_class_display),
        "cf_changes": cf_changes,

        # Longitudinal
        "n_visits": delta_info["n_visits"],
        "time_span_months": delta_info["time_span_months"],
        "deltas": delta_info["deltas"],

        # Uncertainty
        "predictive_entropy": safe_fmt(unc_row["predictive_entropy"], 4),
        "epistemic_uncertainty": safe_fmt(unc_row["epistemic_uncertainty"], 6),
        "aleatoric_uncertainty": safe_fmt(unc_row["aleatoric_uncertainty"], 4),
        "pred_std": safe_fmt(unc_row["pred_std"], 6),

        # Raw biomarker values (for interpretation sections)
        "mmse": safe_fmt(latest.get("clinical_MMSCORE"), 0),
        "faq": safe_fmt(latest.get("clinical_FAQTOTAL"), 0),
        "ldel": safe_fmt(latest.get("clinical_LDELTOTAL"), 0),
        "trails_b": safe_fmt(latest.get("clinical_TRABSCOR"), 0),
        "hippo_vol": safe_fmt(latest.get("mri_hippocampus_vol_mean"), 2),
        "entorhinal_vol": safe_fmt(latest.get("mri_entorhinal_vol_mean"), 2),
        "amygdala_vol": safe_fmt(latest.get("mri_amygdala_vol_mean"), 2),
        "ventricle_vol": safe_fmt(latest.get("mri_lateral_ventricle_vol_mean"), 2),
        "abeta42": safe_fmt(latest.get("clinical_ABETA42"), 1),
        "ptau": safe_fmt(latest.get("pet_PTAU"), 1),
        "tau": safe_fmt(latest.get("pet_TAU"), 1),
    }
    # Derived qualitative fields (computed after dict is built)
    ctx["confidence_qualifier"] = confidence_qualifier(ctx)
    return ctx


# Stage-specific narrative builders

def build_stage_interpretation(ctx):
    """Return stage-conditional narrative text using hedged clinical language."""
    pc = ctx["pred_class"]
    lines = []

    if pc == "0":
        lines.append(
            "Cognition: Scores within expected range for healthy aging "
            "— cognition and function preserved."
        )
        if ctx["hippo_vol"] != "N/A":
            lines.append(
                "Structure: MRI of hippocampus and entorhinal cortex shows "
                "no significant atrophy beyond age norms."
            )
        else:
            lines.append("Structure: MRI data not available for this patient.")
        if ctx["ptau"] != "N/A":
            lines.append(
                "CSF: No significant Alzheimer's-type pathology indicated."
            )
        else:
            lines.append("CSF: Biomarker data not available for this patient.")

    elif pc == "0.5":
        lines.append(
            "Cognition: Mild deficits relative to age norms "
            "— consistent with mild cognitive impairment."
        )
        if ctx["hippo_vol"] != "N/A":
            lines.append(
                "Structure: MRI may show early medial temporal changes "
                "— warrants monitoring."
            )
        else:
            lines.append("Structure: MRI data not available for this patient.")
        if ctx["ptau"] != "N/A":
            lines.append(
                "CSF: Evaluate in context of possible early "
                "Alzheimer's-type changes."
            )
        else:
            lines.append("CSF: Biomarker data not available for this patient.")

    else:  # AD ("1+")
        lines.append(
            "Cognition: Impairment across global cognition, function, "
            "and memory — consistent with dementia-stage decline."
        )
        if ctx["hippo_vol"] != "N/A":
            lines.append(
                "Structure: MRI shows medial temporal changes "
                "consistent with neurodegeneration."
            )
        else:
            lines.append("Structure: MRI data not available for this patient.")
        if ctx["ptau"] != "N/A":
            lines.append(
                "CSF: Biomarkers support Alzheimer's-type pathology."
            )
        else:
            lines.append("CSF: Biomarker data not available for this patient.")

    return "\n".join(lines)


def build_longitudinal_text(ctx):
    """Return longitudinal context narrative with interpreted changes."""
    if ctx["n_visits"] < 2:
        return ("Only a single session is available; longitudinal change "
                "cannot be evaluated.")

    lines = [
        f"Data available from {ctx['n_visits']} visits spanning "
        f"{ctx['time_span_months']} months."
    ]
    deltas = ctx["deltas"]
    if not deltas:
        lines.append("No numeric changes could be computed across visits.")
        return "\n".join(lines)

    # Interpret key cognitive deltas qualitatively
    mmse_d = deltas.get("clinical_MMSCORE")
    if mmse_d is not None:
        try:
            v = float(mmse_d)
            if v <= -4:
                lines.append("Global cognition (MMSE): notable decline observed.")
            elif v <= -2:
                lines.append("Global cognition (MMSE): mild decline observed.")
            elif v >= 2:
                lines.append("Global cognition (MMSE): improvement noted.")
            else:
                lines.append("Global cognition (MMSE): stable across visits.")
        except (ValueError, TypeError):
            pass

    faq_d = deltas.get("clinical_FAQTOTAL")
    if faq_d is not None:
        try:
            v = float(faq_d)
            if v > 3:
                lines.append("Functional status (FAQ): measurable worsening observed.")
            elif v > 0:
                lines.append("Functional status (FAQ): mild worsening noted.")
            else:
                lines.append("Functional status (FAQ): stable or improved.")
        except (ValueError, TypeError):
            pass

    hippo_d = deltas.get("mri_hippocampus_vol_mean")
    if hippo_d is not None:
        try:
            v = float(hippo_d)
            if v < -200:
                lines.append("Hippocampal volume: reduction observed over time.")
            elif v < 0:
                lines.append("Hippocampal volume: mild change, compatible with aging.")
            else:
                lines.append("Hippocampal volume: stable.")
        except (ValueError, TypeError):
            pass

    ent_d = deltas.get("mri_entorhinal_vol_mean")
    if ent_d is not None:
        try:
            v = float(ent_d)
            if v < -0.2:
                lines.append("Entorhinal cortex: thinning observed.")
            else:
                lines.append("Entorhinal cortex: stable.")
        except (ValueError, TypeError):
            pass

    # Overall trajectory summary
    cog_decline = False
    try:
        if mmse_d is not None and float(mmse_d) <= -2:
            cog_decline = True
    except (ValueError, TypeError):
        pass
    struct_decline = False
    try:
        if hippo_d is not None and float(hippo_d) < -200:
            struct_decline = True
    except (ValueError, TypeError):
        pass

    if cog_decline and struct_decline:
        lines.append("Overall trajectory is consistent with progressive impairment.")
    elif cog_decline:
        lines.append("Overall trajectory suggests gradual cognitive decline.")
    elif struct_decline:
        lines.append("Structural changes are noted, though cognitive measures remain relatively stable.")
    else:
        lines.append("Overall trajectory is consistent with stable cognition.")

    return "\n".join(lines)


def build_risk_context(ctx):
    """Return risk context text covering genetic and demographic factors."""
    lines = []
    apoe = ctx["apoe4_status"]
    if "Non-carrier" in apoe:
        lines.append(f"Genetic: {apoe} — not an elevated genetic risk for late-onset AD.")
    elif "Heterozygous" in apoe:
        lines.append(
            f"Genetic: {apoe} — associated with moderately increased risk for "
            f"late-onset AD."
        )
    elif "Homozygous" in apoe:
        lines.append(
            f"Genetic: {apoe} — associated with substantially increased risk for "
            f"late-onset AD."
        )
    else:
        lines.append(f"Genetic: APOE4 status — {apoe}.")

    lines.append(f"Age: {ctx['age']} years at entry.")

    if ctx["education_years"] != "N/A":
        lines.append(f"Education: {ctx['education_years']} years (cognitive reserve factor).")

    return "\n".join(lines)


def build_progression_narrative(subj_ctx):
    """Generate the disease-progression narrative across visits.

Used for subject-level interpretations with longitudinal context.
    """
    stab = subj_ctx["stability"]
    n = subj_ctx["n_nodes"]
    span = subj_ctx.get("time_span_months", 0)
    if n > 1:
        first_m = subj_ctx["node_contexts"][0].get("visit_month", 0)
        last_m = subj_ctx["node_contexts"][-1].get("visit_month", 0)
        span = last_m - first_m

    label = stab["stability_label"]
    first_disp = STAGE_DISPLAY.get(stab["first_class"], stab["first_class"])
    last_disp = STAGE_DISPLAY.get(stab["last_class"], stab["last_class"])

    if label == "Stable":
        text = (
            f"Across {n} visits spanning {span} months,\n"
            f"the model consistently predicted {first_disp}."
        )
    elif label == "Progressed":
        lines = [f"Over {n} visits spanning {span} months, the predicted classification progressed:"]
        for fc, tc, fm, tm in stab["transitions"]:
            lines.append(
                f"  - {STAGE_DISPLAY.get(fc, fc)} to "
                f"{STAGE_DISPLAY.get(tc, tc)} (month {fm} to month {tm})"
            )
        text = "\n".join(lines)
    elif label == "Improved":
        lines = [f"Over {n} visits spanning {span} months, the predicted classification improved:"]
        for fc, tc, fm, tm in stab["transitions"]:
            lines.append(
                f"  - {STAGE_DISPLAY.get(fc, fc)} to "
                f"{STAGE_DISPLAY.get(tc, tc)} (month {fm} to month {tm})"
            )
        text = "\n".join(lines)
    else:  # Fluctuating
        text = (
            f"The predicted classification fluctuated across {n} visits "
            f"spanning {span} months:\n"
            f"  {stab['stability_detail']}\n"
            f"This inconsistency suggests the patient may be near a "
            f"classification boundary."
        )

    return text


def build_summary(ctx):
    """Return stage-appropriate synthesis paragraph."""
    pc = ctx["pred_class"]
    lines = []
    if pc == "0":
        lines.append("Profile: Consistent with healthy cognitive aging.")
        lines.append("No significant neurodegenerative change indicated.")
    elif pc == "0.5":
        lines.append("Profile: Consistent with mild cognitive impairment.")
        lines.append("Early changes detected — continued monitoring recommended.")
    else:
        lines.append("Profile: Consistent with dementia-stage neurodegenerative change.")
        lines.append("Supporting evidence: Cognitive, imaging, and biomarker findings.")

    if ctx["is_high_uncertainty"]:
        lines.append("Elevated uncertainty — interpret with caution.")

    lines.append("This is an automated research summary, not a clinical diagnosis.")
    return "\n".join(lines)


# Interpretation builders (rule-based, 1-2 sentences per section)

def build_neighborhood_interpretation(ctx):
   
    label = ctx.get("agreement_label", "N/A")
    if label == "N/A":
        return ""

    pred = ctx["pred_class_display"]
    dominant = ctx["dominant_neighbor_class"]

    if label == "strong":
        return (
            f"The local patient neighborhood is dominated by {pred}, "
            f"with strong agreement, indicating a consistent clinical "
            f"pattern among similar patients."
        )
    elif label == "moderate":
        return (
            f"The local patient neighborhood is dominated by {dominant}, "
            f"with moderate agreement, reflecting heterogeneity in "
            f"neighboring clinical stages."
        )
    else:
        return (
            f"The local patient neighborhood is dominated by {dominant}, "
            f"with weak agreement — most similar patients are classified "
            f"differently, suggesting this case may be atypical."
        )


def build_biomarker_interpretation(ctx):
    
    top_features = ctx.get("top_features", [])
    if not top_features:
        return ""

    cat_counts = defaultdict(int)
    for feat in top_features:
        cat = _categorize_feature(feat.get("feature", ""))
        cat_counts[cat] += 1

    ranked = sorted(cat_counts, key=cat_counts.get, reverse=True)
    dominant = ranked[0]
    dominant_count = cat_counts[dominant]
    total = len(top_features)

    if dominant_count >= 3:
        phrase = _DRIVER_PHRASES.get(dominant, dominant)
        return (f"The most influential features for this prediction are "
                f"{_CAT_LABELS.get(dominant, dominant)}, suggesting {phrase} "
                f"is the primary driver of this classification.")
    elif len(cat_counts) >= 3:
        labels = [_CAT_LABELS.get(c, c) for c in ranked[:3]]
        return (f"The prediction draws on a mix of {labels[0]}, {labels[1]}, "
                f"and {labels[2]}, indicating a multi-modal basis for "
                f"classification.")
    else:
        labels = [_CAT_LABELS.get(c, c) for c in ranked[:2]]
        return (f"The prediction is primarily driven by {labels[0]} and "
                f"{labels[1]}.")


def build_counterfactual_interpretation(ctx):
   
    if not ctx.get("has_counterfactual") or not ctx.get("cf_changes"):
        return ""

    magnitudes = []
    for ch in ctx["cf_changes"]:
        try:
            magnitudes.append(float(ch["magnitude"]))
        except (ValueError, TypeError):
            pass
    if not magnitudes:
        return ""

    avg_mag = sum(magnitudes) / len(magnitudes)
    pred_short = ctx.get("pred_class_short", ctx["pred_class_display"])
    cf_short = ctx.get("cf_class_short", ctx["cf_class_display"])
    n_fixed = sum(1 for ch in ctx["cf_changes"] if ch.get("is_fixed"))

    if avg_mag < 0.5:
        s = (f"Near the {pred_short}/{cf_short} boundary — "
             f"sensitive to small clinical changes.")
    elif avg_mag < 1.5:
        s = "Indicates moderate class separation in the model."
    else:
        s = "Indicates strong class separation in the model."

    if n_fixed > 0 and n_fixed == len(ctx["cf_changes"]):
        s += " All suggested changes involve non-modifiable features."

    return s


def build_uncertainty_interpretation(ctx):
   
    is_high = ctx.get("is_high_uncertainty", False)
    pred_class = ctx.get("pred_class", "")

    try:
        entropy = float(ctx["predictive_entropy"])
    except (ValueError, TypeError):
        entropy = None

    # Sentence 1: confidence level
    if is_high:
        s1 = ("Overall model confidence is low — ensemble seeds disagree "
              "substantially on this classification.")
    elif entropy is not None and entropy < 0.3:
        s1 = ("The model shows high confidence in this prediction, with "
              "strong agreement across ensemble seeds.")
    elif entropy is not None and entropy < 0.7:
        s1 = "The model shows moderate confidence in this prediction."
    else:
        s1 = ("Model confidence metrics are within an acceptable range for "
              "this prediction.")

    # Sentence 2: MCI borderline check
    s2 = ""
    if pred_class == "0.5":
        if is_high:
            s2 = ("MCI predictions are inherently more uncertain as this "
                  "category sits between normal cognition and dementia; "
                  "additional clinical follow-up is recommended.")
        elif entropy is not None and entropy > 0.5:
            s2 = ("As an MCI classification, this case may be near a "
                  "diagnostic boundary — periodic reassessment is advisable.")
    elif is_high:
        s2 = ("Consider additional clinical evaluation before acting on "
              "this classification.")

    return (s1 + " " + s2).strip() if s2 else s1


# Key Biomarker Values block

def render_key_biomarker_values(ctx):
    """Render a bulleted KEY BIOMARKER VALUES block 

    Only includes fields that have actual values (not N/A).
    Groups: cognitive → structural MRI → CSF biomarkers.
    """
    lines = []

    # --- Cognitive ---
    if ctx.get("mmse") != "N/A":
        lines.append(f"  \u2022 Global cognition (MMSE): {ctx['mmse']} / 30")
    if ctx.get("faq") != "N/A":
        lines.append(f"  \u2022 Functional assessment (FAQ): {ctx['faq']}")
    if ctx.get("ldel") != "N/A":
        lines.append(f"  \u2022 Logical memory (delayed recall): {ctx['ldel']}")
    if ctx.get("trails_b") != "N/A":
        lines.append(f"  \u2022 Executive function (Trail Making B): {ctx['trails_b']} s")

    # --- Structural MRI ---
    if ctx.get("hippo_vol") != "N/A":
        lines.append(f"  \u2022 Hippocampal volume (mean): {ctx['hippo_vol']} mm\u00b3")
    if ctx.get("entorhinal_vol") != "N/A":
        lines.append(f"  \u2022 Entorhinal cortex volume: {ctx['entorhinal_vol']} mm\u00b3")
    if ctx.get("amygdala_vol") != "N/A":
        lines.append(f"  \u2022 Amygdala volume: {ctx['amygdala_vol']} mm\u00b3")
    if ctx.get("ventricle_vol") != "N/A":
        lines.append(f"  \u2022 Lateral ventricle volume: {ctx['ventricle_vol']} mm\u00b3")

    # --- CSF biomarkers ---
    if ctx.get("abeta42") != "N/A":
        lines.append(f"  \u2022 CSF Abeta42: {ctx['abeta42']} pg/mL")
    if ctx.get("ptau") != "N/A":
        lines.append(f"  \u2022 CSF Phosphorylated-Tau: {ctx['ptau']} pg/mL")
    if ctx.get("tau") != "N/A":
        lines.append(f"  \u2022 CSF Total-Tau: {ctx['tau']} pg/mL")

    # Only return block if we have at least one value
    if not lines:
        return ""

    lines.append("")
    return "\n".join(lines)


# Jinja2 template

LEAFLET_TEMPLATE = Template("""\
BRAIN HEALTH BIO-LEAFLET
Patient ID: {{ patient_id }}
Clinical Status: {{ clinical_status }} (Dx = {{ pred_class }})
Age: {{ age }} years, Sex: {{ sex }}
APOE genotype (genetic susceptibility marker): {{ apoe4_status }}
Education: {{ education_years }} years

{% if key_biomarker_values %}\
KEY BIOMARKER VALUES :
{{ key_biomarker_values }}
{% endif %}\
PREDICTION :
  Predicted stage: {{ pred_class_display }}
  Model confidence: {{ confidence_pct }}% ({{ confidence_qualifier }} reliability)
  Prediction certainty: {{ certainty_label }}
{% if is_high_uncertainty %}\
  Note: Elevated uncertainty — additional clinical evaluation is recommended.
{% endif %}

{% if progression_timeline %}\
DISEASE PROGRESSION :
{{ progression_timeline }}
{{ progression_narrative }}

{% endif %}\
INTERPRETATION :
{{ stage_interpretation }}

GRAPH CONTEXT :
  Graph agreement: {{ graph_agreement_pct }}%  (neighbours sharing the predicted class)
  Neighbourhood composition: {{ neighbor_dist_str }}
{{ neighborhood_interpretation }}

{% if has_counterfactual %}\
WHAT-IF ANALYSIS :
To shift prediction from {{ pred_class_short }} to {{ cf_class_short }}:
{% for ch in cf_changes %}\
  \u2022 {{ ch.feature_display }}: {{ ch.clinical_magnitude }} {{ ch.direction }}{% if ch.is_fixed %} (non-modifiable){% endif %}
{% endfor %}\
{% if counterfactual_interpretation %}
{{ counterfactual_interpretation }}
{% endif %}

{% endif %}\
LONGITUDINAL CONTEXT :
{{ longitudinal_text }}

UNCERTAINTY ANALYSIS :
  Predictive entropy: {{ predictive_entropy }}
  Epistemic uncertainty: {{ epistemic_uncertainty }}  (model disagreement)
  Aleatoric uncertainty: {{ aleatoric_uncertainty }}  (data noise)
{% if uncertainty_interpretation %}\
  {{ uncertainty_interpretation }}
{% endif %}

RISK CONTEXT :
{{ risk_context_text }}

SUMMARY :
{{ summary_text }}

Generated by Bio-Leaflet System | ADNI Spectral GCN Pipeline
This automated research summary is non-diagnostic and should not replace clinical evaluation.
""")


# Leaflet rendering

def render_leaflet(ctx):
    """Render the Jinja2 template with a fully assembled context dict."""
    # Key biomarker values (bulleted block)
    ctx.setdefault("key_biomarker_values", render_key_biomarker_values(ctx))
    # Build narrative sections
    ctx["stage_interpretation"] = build_stage_interpretation(ctx)
    ctx["longitudinal_text"] = build_longitudinal_text(ctx)
    ctx["risk_context_text"] = build_risk_context(ctx)
    ctx["summary_text"] = build_summary(ctx)
    # Interpretation sentences for GNN explanation sections
    ctx["neighborhood_interpretation"] = build_neighborhood_interpretation(ctx)
    ctx["counterfactual_interpretation"] = build_counterfactual_interpretation(ctx)
    ctx["uncertainty_interpretation"] = build_uncertainty_interpretation(ctx)
    # Disease progression (subject-level only — populated by build_subject_context)
    if "progression_timeline" in ctx:
        ctx["progression_narrative"] = build_progression_narrative(ctx)
    else:
        ctx.setdefault("progression_timeline", "")
        ctx.setdefault("progression_narrative", "")
    return LEAFLET_TEMPLATE.render(**ctx)


def render_leaflet_with_t5(ctx, t5_stage_interpretation=None, t5_summary=None):
    """Render the full Jinja2 template, using T5 text for INTERPRETATION and
    SUMMARY when provided (falls back to the deterministic builders
    otherwise). All other sections are always rendered deterministically.
    """
    # Key biomarker values (bulleted block)
    ctx.setdefault("key_biomarker_values", render_key_biomarker_values(ctx))
    # All deterministic narrative sections
    ctx["longitudinal_text"] = build_longitudinal_text(ctx)
    ctx["risk_context_text"] = build_risk_context(ctx)
    ctx["neighborhood_interpretation"] = build_neighborhood_interpretation(ctx)
    ctx["counterfactual_interpretation"] = build_counterfactual_interpretation(ctx)
    ctx["uncertainty_interpretation"] = build_uncertainty_interpretation(ctx)
    # Disease progression (subject-level only)
    if "progression_timeline" in ctx:
        ctx["progression_narrative"] = build_progression_narrative(ctx)
    else:
        ctx.setdefault("progression_timeline", "")
        ctx.setdefault("progression_narrative", "")
    # T5 override: INTERPRETATION section
    if t5_stage_interpretation:
        ctx["stage_interpretation"] = t5_stage_interpretation
    else:
        ctx["stage_interpretation"] = build_stage_interpretation(ctx)
    # T5 override: SUMMARY section
    if t5_summary:
        ctx["summary_text"] = t5_summary
    else:
        ctx["summary_text"] = build_summary(ctx)
    return LEAFLET_TEMPLATE.render(**ctx)


# Factual verification

def extract_leaflet_facts(text):
    """Regex-extract factual claims from a rendered leaflet."""
    facts = {}

    m = re.search(r"Patient ID:\s*(\S+)", text)
    if m:
        facts["patient_id"] = m.group(1)

    m = re.search(r"Age:\s*([\d.]+)", text)
    if m:
        facts["age"] = float(m.group(1))

    m = re.search(r"Sex:\s*(\w+)", text)
    if m:
        facts["sex"] = m.group(1)

    m = re.search(r"APOE4 Status:\s*(.+?)(?:\n|$)", text)
    if m:
        facts["apoe4"] = m.group(1).strip()

    m = re.search(r"(?:Model|Prediction) confidence[^:]*:\s*([\d.]+)%", text)
    if m:
        facts["confidence_pct"] = float(m.group(1))

    m = re.search(r"MMSE.*?:\s*(\d+)\s*/\s*30", text)
    if m:
        facts["mmse"] = int(m.group(1))

    m = re.search(r"Predictive entropy:\s*([\d.]+)", text)
    if m:
        facts["predictive_entropy"] = float(m.group(1))

    m = re.search(r"Graph agreement:\s*([\d.]+)%", text)
    if m:
        facts["graph_agreement_pct"] = float(m.group(1))

    return facts


def extract_source_facts(ctx):
    """Extract ground-truth facts from the assembled patient context."""
    facts = {}
    facts["patient_id"] = ctx["patient_id"]
    try:
        facts["age"] = float(ctx["age"])
    except (ValueError, TypeError):
        pass
    facts["sex"] = ctx["sex"]
    facts["apoe4"] = ctx["apoe4_status"]
    try:
        facts["confidence_pct"] = float(ctx["confidence_pct"])
    except (ValueError, TypeError):
        pass
    try:
        facts["mmse"] = int(float(ctx["mmse"]))
    except (ValueError, TypeError):
        pass
    try:
        facts["predictive_entropy"] = float(ctx["predictive_entropy"])
    except (ValueError, TypeError):
        pass
    try:
        facts["graph_agreement_pct"] = float(ctx["graph_agreement_pct"])
    except (ValueError, TypeError):
        pass
    return facts


def compute_fact_metrics(ground_truth, generated):
    """Compare extracted facts: return (n_checked, n_correct, issues)."""
    issues = []
    n_checked = 0
    n_correct = 0

    # Tolerances for numeric fields
    tolerances = {
        "age": 1.0,
        "confidence_pct": 0.5,
        "predictive_entropy": 0.01,
        "graph_agreement_pct": 0.5,
    }

    for key in ground_truth:
        if key not in generated:
            continue
        n_checked += 1
        gt_val = ground_truth[key]
        gen_val = generated[key]

        if isinstance(gt_val, (int, float)) and isinstance(gen_val, (int, float)):
            tol = tolerances.get(key, 0)
            if abs(gt_val - gen_val) <= tol:
                n_correct += 1
            else:
                issues.append(
                    f"{key}: expected {gt_val}, got {gen_val} "
                    f"(tolerance ±{tol})"
                )
        else:
            if str(gt_val).strip().lower() == str(gen_val).strip().lower():
                n_correct += 1
            else:
                issues.append(f"{key}: expected '{gt_val}', got '{gen_val}'")

    return n_checked, n_correct, issues


_SECTION_HEADER_RE = re.compile(r"^([A-Z][A-Z &/-]*)\s*:\s*$")


def extract_narrative_sections(leaflet_text):
    """Return only the INTERPRETATION and SUMMARY section bodies from a
    rendered leaflet, excluding deterministic template sections (e.g.
    GRAPH CONTEXT) that legitimately name other classes."""
    lines = leaflet_text.splitlines()
    headers = []
    for i, line in enumerate(lines):
        m = _SECTION_HEADER_RE.match(line)
        if m:
            headers.append((i, m.group(1).strip()))

    def section_text(name):
        for idx, (line_idx, hname) in enumerate(headers):
            if hname == name:
                start = line_idx + 1
                end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
                return "\n".join(lines[start:end]).strip()
        return None

    parts = [t for t in (section_text("INTERPRETATION"), section_text("SUMMARY")) if t is not None]
    return "\n".join(parts)


def verify_leaflet(leaflet_text, ctx):
    """Run factual verification on a rendered leaflet.

    """
    gen_facts = extract_leaflet_facts(leaflet_text)
    src_facts = extract_source_facts(ctx)
    n_checked, n_correct, issues = compute_fact_metrics(src_facts, gen_facts)

    # Hallucination check — scan rendered text for forbidden claims
    text_lower = leaflet_text.lower()
    forbidden = [
        "diagnosed with alzheimer", "confirmed dementia",
        "definitive diagnosis", "alzheimer's confirmed",
        "clinical diagnosis of", "we diagnose",
        "patient has alzheimer",
    ]
    for term in forbidden:
        if term in text_lower:
            issues.append(f"HALLUCINATION: contains '{term}'")

    # Stage consistency check
    pred = ctx.get("pred_class", "")
    narrative = extract_narrative_sections(leaflet_text)
    if not narrative:
        narrative = leaflet_text
        issues.append("NOTE: narrative sections not found; stage check ran on full text")
    narrative_lower = narrative.lower()
    if pred == "0" and ("alzheimer's disease" in narrative_lower or "ad-level" in narrative_lower):
        ctx_start = max(0, narrative_lower.find("alzheimer") - 20)
        ctx_end = narrative_lower.find("alzheimer")
        if "not" not in narrative_lower[ctx_start:ctx_end]:
            issues.append("STAGE_MISMATCH: CN patient described with AD language")
    if pred == "1+" and "normal aging" in narrative_lower:
        issues.append("STAGE_MISMATCH: AD patient described as normal aging")

    precision = n_correct / n_checked if n_checked > 0 else 0.0
    scores = {
        "fields_checked": n_checked,
        "fields_correct": n_correct,
        "precision": round(precision, 4),
    }
    has_hallucination = any("HALLUCINATION" in i for i in issues)
    status = "PASS" if (precision >= 0.95 and not has_hallucination) else "FAIL"
    return status, issues, scores


# Main generation pipeline

def generate_leaflet(subject_id, node_idx, gnn, data_df, verbose=False):
    """Full pipeline: assemble context → render → verify → return text."""
    # Get all visits for this subject
    patient_df = data_df[data_df["subject_id"] == subject_id]
    if patient_df.empty:
        warnings.warn(f"Subject {subject_id} not found in data CSV; skipping.")
        return None

    ctx = get_patient_context(subject_id, node_idx, gnn, patient_df)
    leaflet = render_leaflet(ctx)

    status, issues, scores = verify_leaflet(leaflet, ctx)
    if verbose:
        print(f"  [{subject_id}] verification: {status} "
              f"({scores['fields_correct']}/{scores['fields_checked']} correct)")
        if issues:
            for iss in issues:
                print(f"    ! {iss}")

    if status == "FAIL":
        leaflet += f"\n\n[VERIFICATION WARNING: {len(issues)} inconsistencies detected]\n"
        for iss in issues:
            leaflet += f"  - {iss}\n"

    return leaflet, status, scores


def generate_all_leaflets(results_dir, data_csv, out_dir, verbose=False):
    """Batch-generate leaflets for all test nodes."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gnn = load_gnn_outputs(results_dir)
    data_df = load_patient_data(data_csv)

    unc_df = gnn["uncertainty"]
    n_total = len(unc_df)
    n_generated = 0
    n_pass = 0
    n_high_unc = 0
    n_no_cf = 0
    verification_rows = []

    print(f"Generating bio-leaflets for {n_total} test nodes...")
    print(f"  Results dir:  {results_dir}")
    print(f"  Data CSV:     {data_csv}")
    print(f"  Output dir:   {out_dir}")
    print()

    for _, row in unc_df.iterrows():
        node_idx = int(row["node_idx"])
        subject_id = str(row["subject_id"]).strip()

        if verbose:
            print(f"Processing node {node_idx} ({subject_id})...")

        result = generate_leaflet(subject_id, node_idx, gnn, data_df, verbose)
        if result is None:
            continue

        leaflet, status, scores = result
        n_generated += 1
        if status == "PASS":
            n_pass += 1

        is_high = str(row["is_high_uncertainty"]).strip().lower() == "true"
        if is_high:
            n_high_unc += 1

        cf_exists = not gnn["counterfactuals"][
            gnn["counterfactuals"]["node_idx"] == node_idx
        ].empty
        if not cf_exists:
            n_no_cf += 1

        safe_id = re.sub(r"[^\w\-.]", "_", subject_id)
        fname = f"{safe_id}_node{node_idx}_leaflet.txt"
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
            "is_high_uncertainty": is_high,
        })

    ver_df = pd.DataFrame(verification_rows)
    ver_df.to_csv(out_path / "verification_summary.csv", index=False)

    print()
    print("=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Total test nodes:        {n_total}")
    print(f"  Leaflets generated:      {n_generated}")
    print(f"  Verification PASS:       {n_pass}")
    print(f"  Verification FAIL:       {n_generated - n_pass}")
    print(f"  High-uncertainty flagged: {n_high_unc}")
    print(f"  Missing counterfactual:  {n_no_cf}")
    print(f"  Output directory:        {out_dir}")
    print(f"  Verification CSV:        {out_path / 'verification_summary.csv'}")
    print("=" * 60)


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Generate patient bio-leaflets from ADNI Spectral GCN outputs."
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory containing paper8_1.py output files "
             "(metrics.json, feature_importance.csv, etc.)"
    )
    parser.add_argument(
        "--data_csv", type=str, required=True,
        help="Path to the source patient data CSV (e.g., ADNI_version1.csv)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results",
        help="Output directory for generated leaflets (default: ./results)"
    )
    parser.add_argument(
        "--patient", type=str, default=None,
        help="Generate leaflet for a single patient (subject_id). "
             "If omitted, generates for all test nodes."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-patient verification details."
    )
    args = parser.parse_args()

    if args.patient:
        gnn = load_gnn_outputs(args.results_dir)
        data_df = load_patient_data(args.data_csv)
        # Find node_idx for this subject
        unc_df = gnn["uncertainty"]
        matches = unc_df[unc_df["subject_id"].str.strip() == args.patient]
        if matches.empty:
            print(f"ERROR: Subject '{args.patient}' not found in uncertainty_estimates.csv")
            return
        for _, row in matches.iterrows():
            node_idx = int(row["node_idx"])
            result = generate_leaflet(args.patient, node_idx, gnn, data_df, verbose=True)
            if result:
                leaflet, status, scores = result
                print(leaflet)
    else:
        generate_all_leaflets(
            args.results_dir, args.data_csv, args.out_dir, verbose=args.verbose
        )


if __name__ == "__main__":
    main()
