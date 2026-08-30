#!/usr/bin/env python3
"""Generate and evaluate grounded ADNI GNN bio-leaflets.

This is a self-contained reconstruction of the thesis/paper evaluation protocol:
ROUGE for narrative sections plus rule-based factual and explanation grounding.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "adni_diagnosis_dxsum_no_cdr.csv"
DEFAULT_RUN = ROOT / "results_target_comparison" / "diagnosis_all_explainer"
DEFAULT_CDR = ROOT / "results_audit" / "diagnosis_vs_cdr_agreement.csv"
DEFAULT_OUT = ROOT / "results_bioleaflets" / "diagnosis_all_explainer"

LABELS = {1: "NC", 2: "MCI", 3: "AD"}
MODEL_LABELS = {
    "0": "NC",
    "0.0": "NC",
    "NC": "NC",
    "0.5": "MCI",
    "MCI": "MCI",
    "1+": "AD",
    "AD": "AD",
}
CLASS_LONG = {
    "NC": "cognitively normal",
    "MCI": "mild cognitive impairment",
    "AD": "Alzheimer's disease / dementia",
}
FEATURE_NAMES = {
    "clinical_entry_age": "entry age",
    "visit_age": "visit age",
    "visit_month": "visit month",
    "clinical_GENDER": "sex",
    "clinical_EDUCAT": "education",
    "clinical_APOE4_count": "APOE4 allele count",
    "clinical_MMSCORE": "MMSE",
    "clinical_FAQTOTAL": "FAQ total",
    "clinical_LDELTOTAL": "Logical Memory delayed recall",
    "clinical_TRABSCOR": "Trail Making Test B",
    "mri_hippocampus_vol_mean": "hippocampal volume",
    "mri_entorhinal_vol_mean": "entorhinal volume",
    "mri_amygdala_vol_mean": "amygdala volume",
    "mri_inferior_temporal_vol_mean": "inferior temporal volume",
    "mri_middle_temporal_vol_mean": "middle temporal volume",
    "mri_lateral_ventricle_vol_mean": "lateral ventricle volume",
    "mri_inf_lat_vent_vol_mean": "inferior lateral ventricle volume",
    "clinical_ABETA42": "CSF A-beta42",
    "pet_PTAU": "p-tau",
    "pet_TAU": "t-tau",
}
IMMUTABLE_FEATURES = {"clinical_APOE4_count", "clinical_entry_age", "clinical_GENDER", "clinical_EDUCAT"}
RESTRICTED_PATTERNS = [
    r"\bdiagnosed with\b",
    r"\bdefinitively\b",
    r"\btreatment should\b",
    r"\bmedication should\b",
    r"\bwill develop\b",
    r"\bguarantee[sd]?\b",
    r"\btrue[- ]class neighbourhood\b",
]


def parse_visit_month(value: object) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"bl", "sc", "screen", "screening"}:
        return 0.0
    if s.startswith("m") and s[1:].isdigit():
        return float(int(s[1:]))
    return None


def fmt_num(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "not available"
    try:
        value_f = float(value)
    except Exception:
        return str(value)
    if abs(value_f - round(value_f)) < 1e-9:
        return str(int(round(value_f)))
    return f"{value_f:.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value) * 100:.{digits}f}%"


def sex_label(value: object) -> str:
    if pd.isna(value):
        return "not available"
    s = str(value).strip().lower()
    if s in {"1", "1.0", "m", "male"}:
        return "male"
    if s in {"2", "2.0", "f", "female"}:
        return "female"
    return str(value)


def feature_label(feature: object) -> str:
    if feature is None or pd.isna(feature) or str(feature).strip() == "":
        return ""
    return FEATURE_NAMES.get(str(feature), str(feature).replace("_", " "))


def model_label(value: object) -> object:
    """Normalize exported model classes without reinterpreting source labels."""
    if value is None or pd.isna(value):
        return pd.NA
    value_str = str(value).strip()
    return MODEL_LABELS.get(value_str, value_str)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def prf(overlap: int, pred_total: int, ref_total: int) -> dict[str, float]:
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(reference: str, generated: str, n: int) -> dict[str, float]:
    ref = ngrams(tokenize(reference), n)
    pred = ngrams(tokenize(generated), n)
    overlap = sum((ref & pred).values())
    return prf(overlap, sum(pred.values()), sum(ref.values()))


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, generated: str) -> dict[str, float]:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(generated)
    return prf(lcs_len(ref_tokens, pred_tokens), len(pred_tokens), len(ref_tokens))


def mean_dict(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ["precision", "recall", "f1"]:
        values = [row[metric] for row in rows if not math.isnan(row[metric])]
        out[f"{prefix}_{metric}"] = float(sum(values) / len(values)) if values else float("nan")
    return out


@dataclass
class Leaflet:
    node_idx: int
    subject_id: str
    clinical_session_id: str
    true_class: str
    pred_class: str
    text: str
    interpretation: str
    summary: str
    reference_interpretation: str
    reference_summary: str
    expected_fields: dict[str, object]


def load_context(source_path: Path, run_dir: Path, cdr_path: Path) -> pd.DataFrame:
    source = pd.read_csv(source_path, low_memory=False)
    source["clinical_session_id"] = source["clinical_session_id"].astype(str)
    source["visit_month"] = source["clinical_session_id"].map(parse_visit_month)
    source["visit_age"] = source["clinical_entry_age"] + source["visit_month"].fillna(0) / 12.0
    source["diagnosis_label"] = source["DIAGNOSIS"].map(LABELS)

    unc = pd.read_csv(run_dir / "uncertainty_estimates.csv")
    for col in ["true_class", "pred_class"]:
        if col in unc.columns:
            unc[col] = unc[col].map(model_label)
    evidence_path = run_dir / "patient_evidence_all_test_nodes.csv"
    if not evidence_path.exists():
        evidence_path = run_dir / "patient_evidence.csv"
    evidence = pd.read_csv(evidence_path)
    for col in ["true_class", "pred_class"]:
        if col in evidence.columns:
            evidence[col] = evidence[col].map(model_label)
    feature_importance = pd.read_csv(run_dir / "feature_importance.csv")
    feature_importance["class"] = feature_importance["class"].map(model_label)
    cf = pd.read_csv(run_dir / "counterfactual_explanations.csv")
    for col in ["true_class", "pred_class", "cf_class"]:
        if col in cf.columns:
            cf[col] = cf[col].map(model_label)
    nb = pd.read_csv(run_dir / "neighbor_influence.csv")
    for col in [
        "node_predicted_class",
        "node_true_class",
        "neighbor_predicted_class",
        "neighbor_true_class",
    ]:
        if col in nb.columns:
            nb[col] = nb[col].map(model_label)
    nb = nb.rename(
        columns={
            "class_influence_pct_0": "class_influence_pct_NC",
            "class_influence_pct_0.5": "class_influence_pct_MCI",
            "class_influence_pct_1+": "class_influence_pct_AD",
        }
    )

    top_rows = []
    for node_idx, group in evidence.sort_values(["node_idx", "rank"]).groupby("node_idx"):
        top_rows.append(
            {
                "node_idx": node_idx,
                "top_features": [
                    {
                        "feature": str(row.feature),
                        "label": feature_label(row.feature),
                        "importance": float(row.importance),
                        "raw_value": None if pd.isna(row.raw_value) else row.raw_value,
                    }
                    for row in group.head(3).itertuples()
                ],
                "top_feature_scope": "patient-specific GNNExplainer attribution signals",
                "top_feature_source": evidence_path.name,
            }
        )
    top_features = pd.DataFrame(top_rows)

    class_top_features: dict[str, list[dict[str, object]]] = {}
    for cls, group in feature_importance.sort_values(["class", "rank"]).groupby("class"):
        class_top_features[str(cls)] = [
            {
                "feature": str(row.feature),
                "label": feature_label(row.feature),
                "importance": float(row.importance),
                "raw_value": None,
            }
            for row in group.head(3).itertuples()
        ]

    nb_first = nb.sort_values(["node_idx", "rank"]).groupby("node_idx", as_index=False).first()
    nb_first = nb_first[["node_idx", "graph_agreement_pct", "class_influence_pct_AD", "class_influence_pct_MCI", "class_influence_pct_NC"]]

    merged = unc.merge(source, on=["subject_id", "clinical_session_id"], how="left", validate="m:1")
    merged = merged.merge(top_features, on="node_idx", how="left")
    merged = merged.merge(cf, on=["node_idx", "subject_id"], how="left", suffixes=("", "_cf"))
    merged = merged.merge(nb_first, on="node_idx", how="left")

    def fill_feature_scope(row: pd.Series) -> pd.Series:
        patient_features = row.get("top_features")
        if isinstance(patient_features, list) and patient_features:
            return pd.Series(
                {
                    "top_features": patient_features,
                    "top_feature_scope": row.get("top_feature_scope"),
                    "top_feature_source": row.get("top_feature_source"),
                }
            )
        return pd.Series(
            {
                "top_features": class_top_features.get(str(row.get("pred_class")), []),
                "top_feature_scope": "predicted-class global GNN importance signals",
                "top_feature_source": "feature_importance.csv",
            }
        )

    feature_scope = merged.apply(fill_feature_scope, axis=1)
    merged["top_features"] = feature_scope["top_features"]
    merged["top_feature_scope"] = feature_scope["top_feature_scope"]
    merged["top_feature_source"] = feature_scope["top_feature_source"]

    if cdr_path.exists():
        cdr = pd.read_csv(cdr_path)
        cdr["diagnosis_cdr_agree"] = cdr["DIAGNOSIS"] == cdr["DIAGNOSIS_CDR_STAGE"]
        cdr["cdr_stage_label"] = cdr["DIAGNOSIS_CDR_STAGE"].map(LABELS)
        merged = merged.merge(
            cdr[
                [
                    "subject_id",
                    "clinical_session_id",
                    "clinical_CDGLOBAL",
                    "DIAGNOSIS_CDR_STAGE",
                    "cdr_stage_label",
                    "diagnosis_cdr_agree",
                ]
            ],
            on=["subject_id", "clinical_session_id"],
            how="left",
        )
    else:
        merged["clinical_CDGLOBAL"] = pd.NA
        merged["DIAGNOSIS_CDR_STAGE"] = pd.NA
        merged["cdr_stage_label"] = pd.NA
        merged["diagnosis_cdr_agree"] = pd.NA

    return merged


def top_feature_text(top_features: object) -> str:
    if not isinstance(top_features, list) or not top_features:
        return "no feature attribution values were available"
    parts = []
    for feat in top_features:
        val = feat.get("raw_value")
        label = feat.get("label")
        if val is None or pd.isna(val):
            parts.append(str(label))
        else:
            parts.append(f"{label} ({fmt_num(val, 2)})")
    return ", ".join(parts)


def expected_top_feature_names(top_features: object) -> list[str]:
    if not isinstance(top_features, list):
        return []
    return [feat["label"] for feat in top_features if feat.get("label")]


def cdr_status_text(row: pd.Series) -> str:
    if pd.isna(row.get("clinical_CDGLOBAL")):
        return "CDR comparison was not available for this visit."
    cdr = fmt_num(row.get("clinical_CDGLOBAL"), 1)
    cdr_stage = row.get("cdr_stage_label")
    if bool(row.get("diagnosis_cdr_agree")):
        return f"The DXSUM diagnosis agrees with CDR global {cdr} ({cdr_stage})."
    return f"The DXSUM diagnosis differs from CDR global {cdr} ({cdr_stage}), marking a label-severity discordance."


def counterfactual_text(row: pd.Series) -> tuple[str, list[str]]:
    cf_class = row.get("cf_class")
    if pd.isna(cf_class):
        return "No one- or two-feature counterfactual was found within the tested perturbation range.", []
    features: list[str] = []
    chunks: list[str] = []
    for idx in [1, 2]:
        feat = row.get(f"feat_{idx}")
        direction = row.get(f"dir_{idx}")
        step = row.get(f"pct_{idx}")
        if pd.isna(feat):
            continue
        label = feature_label(feat)
        features.append(label)
        direction_word = "increase" if str(direction) == "+" else "decrease"
        chunks.append(f"{direction_word} {label} by {fmt_num(step, 1)} standard deviations")
    if not chunks:
        return f"A counterfactual target class {cf_class} was identified, but no feature details were available.", []
    return f"Changing the model prediction toward {cf_class} would require: " + "; ".join(chunks) + ".", features


def certainty_label(row: pd.Series) -> str:
    if bool(row.get("is_high_uncertainty")):
        return "high uncertainty"
    if float(row.get("max_prob", 0)) >= 0.75:
        return "higher confidence"
    return "moderate confidence"


def build_leaflet(row: pd.Series) -> Leaflet:
    top_text = top_feature_text(row.get("top_features"))
    top_scope = str(row.get("top_feature_scope") or "GNN explanation signals")
    cf_text, cf_features = counterfactual_text(row)
    graph_agree = row.get("graph_agreement_pct")
    same_class_col = f"class_influence_pct_{row.get('pred_class')}"
    same_class = row.get(same_class_col) if same_class_col in row else graph_agree
    confidence = fmt_pct(row.get("max_prob"))
    entropy = fmt_num(row.get("predictive_entropy"), 3)
    certainty = certainty_label(row)
    true_class = str(row.get("true_class"))
    pred_class = str(row.get("pred_class"))
    correct = true_class == pred_class
    cdr_text = cdr_status_text(row)

    interpretation = clean_text(
        f"The model classified this visit as {pred_class} with {confidence} confidence, "
        f"which indicates {certainty}. The leading {top_scope} were {top_text}. "
        f"Graph context showed {fmt_num(graph_agree, 1)}% agreement with neighbours sharing the predicted class. "
        f"{cdr_text} {cf_text} Predictive entropy was {entropy}."
    )
    if correct:
        result_phrase = f"The predicted class matched the DXSUM diagnostic label ({true_class})."
    else:
        result_phrase = f"The prediction differed from the DXSUM diagnostic label ({true_class}), so this visit should be interpreted as model-discordant."
    summary = clean_text(
        f"{result_phrase} This automated research bio-leaflet is non-diagnostic and summarizes model-derived evidence for review."
    )

    reference_interpretation = clean_text(
        f"Prediction {pred_class}; confidence {confidence}; uncertainty status {certainty}; "
        f"top features {top_text}; graph agreement {fmt_num(graph_agree, 1)}%; "
        f"CDR status: {cdr_text}; counterfactual: {cf_text}; entropy {entropy}."
    )
    reference_summary = clean_text(
        f"True label {true_class}; predicted label {pred_class}; correct prediction {correct}; non-diagnostic research summary."
    )

    header = [
        "BRAIN HEALTH BIO-LEAFLET",
        f"Patient/session: {row.get('subject_id')} / {row.get('clinical_session_id')}",
        f"Age: {fmt_num(row.get('visit_age'), 1)} years; Sex: {sex_label(row.get('clinical_GENDER'))}; Education: {fmt_num(row.get('clinical_EDUCAT'), 0)} years; APOE4: {fmt_num(row.get('clinical_APOE4_count'), 0)}",
    ]
    biomarkers = [
        "KEY BIOMARKER VALUES:",
        f"- MMSE: {fmt_num(row.get('clinical_MMSCORE'), 0)} / 30",
        f"- FAQ total: {fmt_num(row.get('clinical_FAQTOTAL'), 1)}",
        f"- Logical Memory delayed recall: {fmt_num(row.get('clinical_LDELTOTAL'), 1)}",
        f"- Trail Making Test B: {fmt_num(row.get('clinical_TRABSCOR'), 1)} seconds",
        f"- Hippocampal volume: {fmt_num(row.get('mri_hippocampus_vol_mean'), 1)}",
        f"- Entorhinal volume: {fmt_num(row.get('mri_entorhinal_vol_mean'), 1)}",
        f"- CSF A-beta42: {fmt_num(row.get('clinical_ABETA42'), 1)}; p-tau: {fmt_num(row.get('pet_PTAU'), 2)}; t-tau: {fmt_num(row.get('pet_TAU'), 1)}",
    ]
    prediction = [
        "PREDICTION:",
        f"- DXSUM diagnostic label: {true_class}",
        f"- Model prediction: {pred_class} ({confidence}; {certainty})",
        f"- Predictive entropy: {entropy}; aleatoric: {fmt_num(row.get('aleatoric_uncertainty'), 3)}; epistemic: {fmt_num(row.get('epistemic_uncertainty'), 3)}",
    ]
    explanation = [
        "INTERPRETATION:",
        interpretation,
        "GRAPH CONTEXT:",
        f"Neighbourhood agreement was {fmt_num(graph_agree, 1)}%, with same-class influence {fmt_num(same_class, 1)}%.",
        "WHAT-IF ANALYSIS:",
        cf_text,
        "SUMMARY:",
        summary,
    ]
    text = "\n".join(header + [""] + biomarkers + [""] + prediction + [""] + explanation)

    expected_fields = {
        "subject_id": str(row.get("subject_id")),
        "clinical_session_id": str(row.get("clinical_session_id")),
        "true_class": true_class,
        "pred_class": pred_class,
        "confidence_pct": confidence,
        "certainty": certainty,
        "entropy": entropy,
        "top_feature_labels": expected_top_feature_names(row.get("top_features")),
        "top_feature_scope": top_scope,
        "graph_agreement_basis": "predicted class",
        "counterfactual_features": cf_features,
        "cf_class": None if pd.isna(row.get("cf_class")) else str(row.get("cf_class")),
        "cdr_status_available": not pd.isna(row.get("clinical_CDGLOBAL")),
        "diagnosis_cdr_agree": None if pd.isna(row.get("diagnosis_cdr_agree")) else bool(row.get("diagnosis_cdr_agree")),
        "non_diagnostic_disclaimer": "non-diagnostic",
    }
    return Leaflet(
        node_idx=int(row["node_idx"]),
        subject_id=str(row["subject_id"]),
        clinical_session_id=str(row["clinical_session_id"]),
        true_class=true_class,
        pred_class=pred_class,
        text=text,
        interpretation=interpretation,
        summary=summary,
        reference_interpretation=reference_interpretation,
        reference_summary=reference_summary,
        expected_fields=expected_fields,
    )


def contains_all(text: str, values: Iterable[str]) -> bool:
    lowered = text.lower()
    return all(str(value).lower() in lowered for value in values if value)


def verify_leaflet(leaflet: Leaflet) -> list[dict[str, object]]:
    text = leaflet.text
    fields = leaflet.expected_fields
    checks: list[tuple[str, bool, str]] = []
    checks.append(("subject_id", str(fields["subject_id"]) in text, str(fields["subject_id"])))
    checks.append(("clinical_session_id", str(fields["clinical_session_id"]) in text, str(fields["clinical_session_id"])))
    checks.append(("true_class", str(fields["true_class"]) in text, str(fields["true_class"])))
    checks.append(("pred_class", str(fields["pred_class"]) in text, str(fields["pred_class"])))
    checks.append(("confidence_pct", str(fields["confidence_pct"]) in text, str(fields["confidence_pct"])))
    checks.append(("certainty", str(fields["certainty"]).lower() in text.lower(), str(fields["certainty"])))
    checks.append(("entropy", str(fields["entropy"]) in text, str(fields["entropy"])))
    checks.append(("top_feature_scope", str(fields["top_feature_scope"]).lower() in text.lower(), str(fields["top_feature_scope"])))
    checks.append(("top_features", contains_all(text, fields["top_feature_labels"]), "; ".join(fields["top_feature_labels"])))
    checks.append(("graph_agreement_basis", "predicted class" in text.lower(), "predicted class"))
    if fields["cf_class"]:
        checks.append(("counterfactual_class", str(fields["cf_class"]) in text, str(fields["cf_class"])))
        checks.append(("counterfactual_features", contains_all(text, fields["counterfactual_features"]), "; ".join(fields["counterfactual_features"])))
    else:
        checks.append(("counterfactual_absent_statement", "No one- or two-feature counterfactual" in text, "no counterfactual"))
    if fields["cdr_status_available"]:
        expected = "agrees" if fields["diagnosis_cdr_agree"] else "differs"
        checks.append(("cdr_agreement_status", expected in text, expected))
    checks.append(("non_diagnostic_disclaimer", "non-diagnostic" in text.lower(), "non-diagnostic"))
    restricted = [pattern for pattern in RESTRICTED_PATTERNS if re.search(pattern, text, flags=re.IGNORECASE)]
    checks.append(("restricted_language_absent", not restricted, ", ".join(restricted)))
    return [
        {
            "node_idx": leaflet.node_idx,
            "subject_id": leaflet.subject_id,
            "clinical_session_id": leaflet.clinical_session_id,
            "field": field,
            "correct": bool(ok),
            "expected": expected,
        }
        for field, ok, expected in checks
    ]


def rouge_rows(leaflets: list[Leaflet]) -> pd.DataFrame:
    rows = []
    for leaflet in leaflets:
        sections = [
            ("interpretation", leaflet.reference_interpretation, leaflet.interpretation),
            ("summary", leaflet.reference_summary, leaflet.summary),
            (
                "interpretation_summary",
                leaflet.reference_interpretation + " " + leaflet.reference_summary,
                leaflet.interpretation + " " + leaflet.summary,
            ),
        ]
        for section, reference, generated in sections:
            r1 = rouge_n(reference, generated, 1)
            r2 = rouge_n(reference, generated, 2)
            rl = rouge_l(reference, generated)
            rows.append(
                {
                    "node_idx": leaflet.node_idx,
                    "section": section,
                    "rouge1_f1": r1["f1"],
                    "rouge2_f1": r2["f1"],
                    "rougeL_f1": rl["f1"],
                    "rougeLsum_f1": rl["f1"],
                    "rouge1_recall": r1["recall"],
                    "rouge2_recall": r2["recall"],
                    "rougeL_recall": rl["recall"],
                }
            )
    return pd.DataFrame(rows)


def fact_prf(checks: pd.DataFrame) -> dict[str, float]:
    tp = int(checks["correct"].sum())
    fp = int((~checks["correct"]).sum())
    fn = 0
    scores = prf(tp, tp + fp, tp + fn)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, **scores}


def write_outputs(leaflets: list[Leaflet], checks: pd.DataFrame, rouge: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "texts").mkdir(exist_ok=True)
    rows = []
    for leaflet in leaflets:
        safe_name = f"node_{leaflet.node_idx}_{leaflet.subject_id}_{leaflet.clinical_session_id}.txt".replace("/", "_")
        (out_dir / "texts" / safe_name).write_text(leaflet.text, encoding="utf-8")
        rows.append(
            {
                "node_idx": leaflet.node_idx,
                "subject_id": leaflet.subject_id,
                "clinical_session_id": leaflet.clinical_session_id,
                "true_class": leaflet.true_class,
                "pred_class": leaflet.pred_class,
                "top_feature_scope": leaflet.expected_fields.get("top_feature_scope"),
                "interpretation": leaflet.interpretation,
                "summary": leaflet.summary,
                "reference_interpretation": leaflet.reference_interpretation,
                "reference_summary": leaflet.reference_summary,
                "text_path": str((out_dir / "texts" / safe_name).relative_to(ROOT)),
                "bioleaflet_text": leaflet.text,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "bioleaflets.csv", index=False)
    checks.to_csv(out_dir / "factual_checks.csv", index=False)
    rouge.to_csv(out_dir / "rouge_scores.csv", index=False)

    field_summary = (
        checks.groupby("field", as_index=False)
        .agg(n=("correct", "size"), correct=("correct", "sum"))
        .assign(accuracy=lambda d: d["correct"] / d["n"])
        .sort_values(["accuracy", "n"], ascending=[True, False])
    )
    field_summary.to_csv(out_dir / "field_accuracy_summary.csv", index=False)

    rouge_cols = [
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "rougeLsum_f1",
        "rouge1_recall",
        "rouge2_recall",
        "rougeL_recall",
    ]
    rouge_summary = rouge.groupby("section", as_index=False)[rouge_cols].mean()
    rouge_summary.to_csv(out_dir / "rouge_summary.csv", index=False)

    per_leaflet = checks.groupby("node_idx")["correct"].all().rename("pass").reset_index()
    leaflets_df = pd.DataFrame(rows)
    per_leaflet = per_leaflet.merge(
        leaflets_df[["node_idx", "true_class", "pred_class"]],
        on="node_idx",
        how="left",
    )
    per_leaflet["correct_prediction"] = per_leaflet["true_class"] == per_leaflet["pred_class"]
    node_meta = pd.DataFrame(
        [
            {
                "node_idx": leaflet.node_idx,
                "true_class": leaflet.true_class,
                "pred_class": leaflet.pred_class,
                "diagnosis_cdr_agree": leaflet.expected_fields.get("diagnosis_cdr_agree"),
            }
            for leaflet in leaflets
        ]
    )
    per_leaflet = per_leaflet.merge(node_meta[["node_idx", "diagnosis_cdr_agree"]], on="node_idx", how="left")
    strata_rows = []
    for name, group_cols in {
        "true_class": ["true_class"],
        "pred_class": ["pred_class"],
        "correct_prediction": ["correct_prediction"],
        "diagnosis_cdr_agree": ["diagnosis_cdr_agree"],
    }.items():
        for key, group in per_leaflet.groupby(group_cols, dropna=False):
            if isinstance(key, tuple):
                label = " / ".join(str(x) for x in key)
            else:
                label = str(key)
            strata_rows.append(
                {
                    "stratum": name,
                    "group": label,
                    "n": int(len(group)),
                    "pass_rate": float(group["pass"].mean()),
                }
            )
    stratified = pd.DataFrame(strata_rows)
    stratified.to_csv(out_dir / "stratified_pass_rates.csv", index=False)

    overall = {
        "evaluation_layer": "corrected deterministic visit-level BioLeaflet template",
        "rouge_scope": (
            "template-to-reference overlap from this deterministic audit; "
            "not the held-out FLAN-T5 narrative evaluation reported separately"
        ),
        "n_leaflets": len(leaflets),
        "n_checked_fields": int(len(checks)),
        "field_accuracy": float(checks["correct"].mean()),
        "pass_rate": float(per_leaflet["pass"].mean()),
        "fact_prf": fact_prf(checks),
        "restricted_language_violations": int((checks["field"].eq("restricted_language_absent") & ~checks["correct"]).sum()),
        "feature_scope_counts": pd.Series(
            [leaflet.expected_fields.get("top_feature_scope") for leaflet in leaflets]
        ).value_counts(dropna=False).to_dict(),
        "rouge": {
            row["section"]: {
                "rouge1_f1": float(row["rouge1_f1"]),
                "rouge2_f1": float(row["rouge2_f1"]),
                "rougeL_f1": float(row["rougeL_f1"]),
                "rougeLsum_f1": float(row["rougeLsum_f1"]),
            }
            for row in rouge_summary.to_dict("records")
        },
    }
    (out_dir / "bioleaflet_evaluation_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    lines = ["# Bio-Leaflet Evaluation Summary", ""]
    lines.append(f"- Leaflets generated: {overall['n_leaflets']}.")
    lines.append(f"- Checked fields: {overall['n_checked_fields']}.")
    lines.append(f"- Field-level factual accuracy: {overall['field_accuracy']:.3f}.")
    lines.append(f"- Zero-error pass rate: {overall['pass_rate']:.3f}.")
    prf_scores = overall["fact_prf"]
    lines.append(
        f"- Fact precision/recall/F1: {prf_scores['precision']:.3f} / {prf_scores['recall']:.3f} / {prf_scores['f1']:.3f}."
    )
    lines.append(f"- Restricted-language violations: {overall['restricted_language_violations']}.")
    if overall["feature_scope_counts"]:
        lines.append("- Feature evidence scope:")
        for scope, n in overall["feature_scope_counts"].items():
            lines.append(f"  - {scope}: {n}.")
    lines.append("")
    lines.append("## ROUGE F1")
    lines.append("")
    lines.append("| Section | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |")
    lines.append("|---|---:|---:|---:|---:|")
    for section, scores in overall["rouge"].items():
        lines.append(
            f"| {section} | {scores['rouge1_f1']:.3f} | {scores['rouge2_f1']:.3f} | "
            f"{scores['rougeL_f1']:.3f} | {scores['rougeLsum_f1']:.3f} |"
        )
    lines.append("")
    lines.append("## Field Accuracy")
    lines.append("")
    lines.append("| Field | N | Accuracy |")
    lines.append("|---|---:|---:|")
    for row in field_summary.to_dict("records"):
        lines.append(f"| {row['field']} | {int(row['n'])} | {float(row['accuracy']):.3f} |")
    lines.append("")
    lines.append("## Stratified Pass Rates")
    lines.append("")
    lines.append("| Stratum | Group | N | Pass rate |")
    lines.append("|---|---|---:|---:|")
    for row in stratified.to_dict("records"):
        lines.append(f"| {row['stratum']} | {row['group']} | {int(row['n'])} | {float(row['pass_rate']):.3f} |")
    lines.append("")
    lines.append("## Note")
    lines.append(
        "This first implementation evaluates template-generated bio-leaflets. It recreates the thesis/paper protocol "
        "and can be reused on FLAN-T5 outputs when those generated summaries or the original code become available."
    )
    (out_dir / "bioleaflet_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate grounded ADNI GNN bio-leaflets.")
    parser.add_argument("--source_csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cdr_csv", type=Path, default=DEFAULT_CDR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_context(args.source_csv, args.run_dir, args.cdr_csv)
    if args.limit is not None:
        context = context.head(args.limit)
    leaflets = [build_leaflet(row) for _, row in context.iterrows()]
    check_rows = [item for leaflet in leaflets for item in verify_leaflet(leaflet)]
    checks = pd.DataFrame(check_rows)
    rouge = rouge_rows(leaflets)
    write_outputs(leaflets, checks, rouge, args.out_dir)
    print(f"Wrote {args.out_dir / 'bioleaflet_evaluation_summary.md'}")
    print(f"Wrote {args.out_dir / 'bioleaflets.csv'}")
    print(f"Wrote {args.out_dir / 'factual_checks.csv'}")
    print(f"Wrote {args.out_dir / 'rouge_scores.csv'}")


if __name__ == "__main__":
    main()
