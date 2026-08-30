#!/usr/bin/env python3
"""Challenge-set evaluation for the BioLeaflet and RAG verifiers.

Clean generated outputs serve as negative controls. Deterministic mutations
introduce one known error at a time. Only aggregate detection counts are
written; patient identifiers and generated text remain in memory.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import bio_leaflet
import run_bioleaflet_rag_prototype as rag


def replace_numeric(text, pattern, delta):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    original = float(match.group(1))
    replacement = f"{original + delta:.1f}"
    start, end = match.span(1)
    return text[:start] + replacement + text[end:]


def mutate_leaflet(text, ctx):
    mutations = {
        "wrong_patient_id": re.sub(
            r"Patient ID:\s*\S+", "Patient ID: CORRUPTED", text, count=1
        ),
        "forbidden_diagnostic_claim": (
            text + "\n\nINTERPRETATION :\nWe diagnose this patient with Alzheimer's disease."
        ),
        "contradictory_age": text + "\nAge: 99.9 years at entry.",
        "contradictory_confidence": text + "\nModel confidence: 99.9%.",
        "contradictory_graph_agreement": text + "\nGraph agreement: 1.1%.",
        "missing_age": re.sub(
            r"Age:\s*[\d.]+\s*years(?:\s+at entry)?(?:,\s*)?",
            "",
            text,
            flags=re.IGNORECASE,
        ),
    }
    numeric_specs = {
        "wrong_age": (r"Age:\s*([\d.]+)", 10.0),
        "wrong_confidence": (r"Model confidence:\s*([\d.]+)%", 15.0),
        "wrong_mmse": (r"MMSE\):\s*(\d+)\s*/\s*30", 5.0),
        "wrong_entropy": (r"Predictive entropy:\s*([\d.]+)", 0.5),
        "wrong_graph_agreement": (r"Graph agreement:\s*([\d.]+)%", 20.0),
    }
    for name, (pattern, delta) in numeric_specs.items():
        mutated = replace_numeric(text, pattern, delta)
        if mutated is not None:
            mutations[name] = mutated

    sex = str(ctx.get("sex", "")).lower()
    if sex in {"male", "female"}:
        replacement = "Female" if sex == "male" else "Male"
        mutations["wrong_sex"] = re.sub(
            r"Sex:\s*\w+", f"Sex: {replacement}", text, count=1, flags=re.IGNORECASE
        )

    pred = str(ctx.get("pred_class", ""))
    if pred == "0":
        mutations["stage_mismatch"] = text.replace(
            "INTERPRETATION :",
            "INTERPRETATION :\nThis profile reflects Alzheimer's disease.",
            1,
        )
    elif pred == "1+":
        mutations["stage_mismatch"] = text.replace(
            "INTERPRETATION :",
            "INTERPRETATION :\nThis profile reflects normal aging.",
            1,
        )
    return mutations


def run_leaflet_challenges(results_dir, data_csv):
    gnn = bio_leaflet.load_gnn_outputs(results_dir)
    data = bio_leaflet.load_patient_data(data_csv)
    clean_total = 0
    clean_pass = 0
    challenge_counts = defaultdict(lambda: {"n": 0, "detected": 0})

    for _, row in gnn["uncertainty"].iterrows():
        node_idx = int(row["node_idx"])
        subject_id = str(row["subject_id"]).strip()
        patient_rows = data[data["subject_id"].astype(str).eq(subject_id)]
        ctx = bio_leaflet.get_patient_context(subject_id, node_idx, gnn, patient_rows)
        clean_text = bio_leaflet.render_leaflet(ctx)
        status, _, _ = bio_leaflet.verify_leaflet(clean_text, ctx)
        clean_total += 1
        clean_pass += int(status == "PASS")

        for challenge_type, mutated in mutate_leaflet(clean_text, ctx).items():
            mutated_status, _, _ = bio_leaflet.verify_leaflet(mutated, ctx)
            challenge_counts[challenge_type]["n"] += 1
            challenge_counts[challenge_type]["detected"] += int(mutated_status == "FAIL")

    corrupt_total = sum(item["n"] for item in challenge_counts.values())
    detected_total = sum(item["detected"] for item in challenge_counts.values())
    return {
        "clean_outputs": clean_total,
        "clean_passed": clean_pass,
        "specificity": clean_pass / clean_total if clean_total else 0.0,
        "corrupted_outputs": corrupt_total,
        "corruptions_detected": detected_total,
        "sensitivity": detected_total / corrupt_total if corrupt_total else 0.0,
        "by_challenge": {
            name: {
                **counts,
                "detection_rate": counts["detected"] / counts["n"] if counts["n"] else 0.0,
            }
            for name, counts in sorted(challenge_counts.items())
        },
    }


def parse_rag_facts(row):
    answer = str(row["answer"])
    confidence = rag.first_match(r"confidence\s+([0-9]+(?:\.[0-9]+)?)%", answer)
    entropy = rag.first_match(r"entropy\s+([0-9]+(?:\.[0-9]+)?)", answer)
    graph = rag.first_match(r"neighbourhood agreement of\s+([0-9]+(?:\.[0-9]+)?)%", answer)
    return {
        "subject_id": str(row["subject_id"]),
        "clinical_session_id": str(row["clinical_session_id"]),
        "true_class": str(row["true_class"]),
        "pred_class": str(row["pred_class"]),
        "confidence_pct": confidence,
        "entropy": entropy,
        "graph_agreement_pct": graph,
        "top_features": "",
        "counterfactual": "",
    }


def mutate_rag_answer(answer, row, facts):
    mutations = {
        "wrong_patient_id": answer.replace(facts["subject_id"], "CORRUPTED", 1),
        "wrong_true_label": re.sub(
            r"DXSUM label\s+\S+", "DXSUM label CORRUPTED", answer, count=1
        ),
        "wrong_prediction": re.sub(
            r"model prediction\s+\S+", "model prediction CORRUPTED", answer, count=1
        ),
        "invalid_citation": re.sub(
            r"\[[A-Z]+-\d{3}\]", "[BAD-999]", answer, count=1
        ),
        "missing_citations": re.sub(r"\s*\[[A-Z]+-\d{3}\]", "", answer),
        "missing_non_diagnostic_label": answer.replace("non-diagnostic", "diagnostic", 1),
        "restricted_clinical_language": answer + " You have Alzheimer's disease.",
        "unsupported_uncited_sentence": answer + " Biomarker certainty is absolute.",
        "contradictory_true_label": (
            answer + " The source BioLeaflet reports DXSUM label CORRUPTED."
        ),
        "contradictory_prediction": (
            answer + " The source BioLeaflet reports model prediction CORRUPTED."
        ),
        "contradictory_confidence": (
            answer + " The source BioLeaflet reports confidence 99.9%."
        ),
    }
    if str(row["question_type"]) == "graph_context" and facts["graph_agreement_pct"]:
        wrong_value = f"{float(facts['graph_agreement_pct']) + 10.0:.1f}"
        mutations["wrong_graph_agreement"] = answer.replace(
            f"{facts['graph_agreement_pct']}%", f"{wrong_value}%", 1
        )
        mutations["contradictory_graph_agreement"] = (
            answer
            + " The source BioLeaflet reports neighbourhood agreement of 1.1%."
        )
    return mutations


def run_rag_challenges(answers_csv, knowledge_base_json):
    answers = pd.read_csv(answers_csv, dtype=str).fillna("")
    with open(knowledge_base_json, encoding="utf-8") as handle:
        knowledge_base = json.load(handle)
    by_id = {entry["id"]: entry for entry in knowledge_base}
    kb_ids = set(by_id)
    clean_total = 0
    clean_pass = 0
    challenge_counts = defaultdict(lambda: {"n": 0, "detected": 0})

    for _, row in answers.iterrows():
        facts = parse_rag_facts(row)
        question = {
            "answer_id": str(row["answer_id"]),
            "question_type": str(row["question_type"]),
            "question": str(row["question"]),
        }
        evidence_ids = [item for item in str(row["evidence_ids"]).split(";") if item]
        evidence = [(by_id[item], 1.0) for item in evidence_ids if item in by_id]
        answer = str(row["answer"])
        clean = rag.verify_answer(answer, facts, question, evidence, kb_ids)
        clean_total += 1
        clean_pass += int(clean["overall_pass"] == 1)

        for challenge_type, mutated in mutate_rag_answer(answer, row, facts).items():
            result = rag.verify_answer(mutated, facts, question, evidence, kb_ids)
            challenge_counts[challenge_type]["n"] += 1
            challenge_counts[challenge_type]["detected"] += int(result["overall_pass"] == 0)

    corrupt_total = sum(item["n"] for item in challenge_counts.values())
    detected_total = sum(item["detected"] for item in challenge_counts.values())
    return {
        "clean_outputs": clean_total,
        "clean_passed": clean_pass,
        "specificity": clean_pass / clean_total if clean_total else 0.0,
        "corrupted_outputs": corrupt_total,
        "corruptions_detected": detected_total,
        "sensitivity": detected_total / corrupt_total if corrupt_total else 0.0,
        "by_challenge": {
            name: {
                **counts,
                "detection_rate": counts["detected"] / counts["n"] if counts["n"] else 0.0,
            }
            for name, counts in sorted(challenge_counts.items())
        },
    }


def summary_rows(component, result):
    rows = [
        {
            "component": component,
            "challenge": "clean_negative_control",
            "n": result["clean_outputs"],
            "detected_or_passed": result["clean_passed"],
            "rate": result["specificity"],
        }
    ]
    for name, counts in result["by_challenge"].items():
        rows.append(
            {
                "component": component,
                "challenge": name,
                "n": counts["n"],
                "detected_or_passed": counts["detected"],
                "rate": counts["detection_rate"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--rag_answers", required=True)
    parser.add_argument("--rag_knowledge_base", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaflet_result = run_leaflet_challenges(args.results_dir, args.data_csv)
    rag_result = run_rag_challenges(args.rag_answers, args.rag_knowledge_base)
    aggregate = {
        "schema_version": 1,
        "design": "clean negative controls plus deterministic single-error mutations",
        "bioleaflet_verifier": leaflet_result,
        "rag_verifier": rag_result,
    }
    with open(output_dir / "verifier_stress_test.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    rows = summary_rows("BioLeaflet", leaflet_result) + summary_rows("RAG", rag_result)
    pd.DataFrame(rows).to_csv(output_dir / "verifier_stress_test.csv", index=False)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
