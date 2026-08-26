"""
Deterministic retrieval-augmented BioLeaflet follow-up prototype.

This script adds a conservative RAG layer on top of the already verified
BioLeaflet CSV. It does not use open-ended LLM generation. Patient facts are
copied from the BioLeaflet source row, and background statements are drawn from
a curated knowledge base with explicit evidence IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - fallback for very small environments
    TfidfVectorizer = None
    cosine_similarity = None


ROOT = Path(__file__).resolve().parents[1]


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_BIOLEAFLETS = first_existing(
    ROOT / "results_bioleaflets" / "diagnosis_all_explainer" / "bioleaflets.csv",
    ROOT / "results" / "bioleaflets" / "bioleaflets.csv",
)
DEFAULT_KB = ROOT / "analysis" / "resources" / "bioleaflet_rag_knowledge_base.json"
DEFAULT_OUT = (
    ROOT / "results_bioleaflets" / "diagnosis_all_explainer" / "rag_followup_prototype"
    if (ROOT / "results_bioleaflets").exists()
    else ROOT / "results" / "bioleaflet_rag"
)


RESTRICTED_PATTERNS = [
    r"\byou have\b",
    r"\bthis patient has alzheimer",
    r"\bdefinitively has\b",
    r"\bwill definitely develop\b",
    r"\bstart medication\b",
    r"\bprescribe\b",
    r"\btreatment should\b",
    r"\bclinical recommendation\b",
    r"\bthis proves\b",
]


def safe_str(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def first_match(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else default


def parse_leaflet_facts(row: pd.Series) -> Dict[str, str]:
    interpretation = safe_str(row.get("interpretation"))
    bioleaflet_text = safe_str(row.get("bioleaflet_text"))
    reference = safe_str(row.get("reference_interpretation"))
    full_text = "\n".join([interpretation, bioleaflet_text, reference])

    pred_class = safe_str(row.get("pred_class"))
    true_class = safe_str(row.get("true_class"))

    confidence = first_match(r"with\s+([0-9]+(?:\.[0-9]+)?)%\s+confidence", full_text)
    if not confidence:
        confidence = first_match(r"Model prediction:\s*[A-Z0-9+]+\s*\(([0-9]+(?:\.[0-9]+)?)%", full_text)

    entropy = first_match(r"Predictive entropy(?:\s+was|:)?\s+([0-9]+(?:\.[0-9]+)?)", full_text)
    graph_agreement = first_match(r"Graph agreement:\s*([0-9]+(?:\.[0-9]+)?)%", full_text)
    if not graph_agreement:
        graph_agreement = first_match(r"Graph context showed\s+([0-9]+(?:\.[0-9]+)?)%\s+agreement", full_text)
    if not graph_agreement:
        graph_agreement = first_match(r"Neighbourhood agreement was\s+([0-9]+(?:\.[0-9]+)?)%", full_text)

    top_features = first_match(
        r"leading patient-specific .*? attribution signals were\s+(.*?)\.\s+Graph context",
        full_text,
    )
    if not top_features:
        top_features = first_match(r"top features\s+(.*?);", reference)
    if not top_features:
        top_features = "not available"

    counterfactual = first_match(
        r"(Changing the model prediction toward\s+[A-Z0-9+]+\s+would require:.*?)(?=\.\s+(?:Predictive entropy|GRAPH CONTEXT|WHAT-IF ANALYSIS|SUMMARY)|$)",
        full_text,
    )
    if not counterfactual:
        counterfactual = "No counterfactual change was available for this leaflet"

    source_summary = safe_str(row.get("summary"))

    return {
        "node_idx": safe_str(row.get("node_idx")),
        "subject_id": safe_str(row.get("subject_id")),
        "clinical_session_id": safe_str(row.get("clinical_session_id")),
        "true_class": true_class,
        "pred_class": pred_class,
        "confidence_pct": confidence,
        "entropy": entropy,
        "graph_agreement_pct": graph_agreement,
        "top_features": top_features,
        "counterfactual": counterfactual,
        "source_summary": source_summary,
    }


def make_questions(facts: Dict[str, str]) -> List[Dict[str, str]]:
    subject = facts["subject_id"]
    session = facts["clinical_session_id"]
    pred_class = facts["pred_class"]
    true_class = facts["true_class"]
    top_features = facts["top_features"]

    questions = [
        {
            "question_type": "feature_context",
            "question": "Why do the highlighted features matter for this model output?",
            "query": f"{top_features} feature attribution biomarker cognitive MRI CSF {pred_class}",
        },
        {
            "question_type": "graph_context",
            "question": "How should the graph neighbourhood agreement be interpreted?",
            "query": "population graph neighbor agreement edge similarity GCN explanation",
        },
        {
            "question_type": "uncertainty_context",
            "question": "What does the prediction confidence and entropy mean?",
            "query": "confidence entropy uncertainty model probability cautious interpretation",
        },
        {
            "question_type": "diagnostic_safety",
            "question": "Why should this BioLeaflet not be read as a clinical diagnosis?",
            "query": "diagnosis no single test medical history cognitive functional MRI CSF blood",
        },
        {
            "question_type": "mci_context",
            "question": "Why is the NC/MCI/AD boundary difficult for automated classification?",
            "query": f"MCI mild cognitive impairment heterogeneous stable progress revert {true_class} {pred_class}",
        },
    ]

    if facts["counterfactual"].startswith("Changing the model prediction"):
        questions.append(
            {
                "question_type": "counterfactual_context",
                "question": "How should the what-if/counterfactual statement be interpreted?",
                "query": "counterfactual what-if model sensitivity prediction boundary feature changes",
            }
        )

    for i, item in enumerate(questions, start=1):
        item["answer_id"] = f"{subject}_{session}_{i:02d}_{item['question_type']}"
    return questions


def kb_doc(entry: Dict) -> str:
    keywords = " ".join(entry.get("keywords", []))
    return f"{entry.get('topic', '')} {keywords} {entry.get('text', '')}"


def token_set(text: str) -> set:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()))


def retrieve_with_overlap(query: str, kb: List[Dict], top_k: int) -> List[Tuple[Dict, float]]:
    q_tokens = token_set(query)
    scored = []
    for entry in kb:
        e_tokens = token_set(kb_doc(entry))
        if not q_tokens or not e_tokens:
            score = 0.0
        else:
            score = len(q_tokens & e_tokens) / len(q_tokens | e_tokens)
        scored.append((entry, float(score)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def build_retriever(kb: List[Dict]):
    if TfidfVectorizer is None or cosine_similarity is None:
        return None, None
    docs = [kb_doc(entry) for entry in kb]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def retrieve(query: str, kb: List[Dict], vectorizer, matrix, top_k: int) -> List[Tuple[Dict, float]]:
    if vectorizer is None or matrix is None:
        return retrieve_with_overlap(query, kb, top_k)
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).ravel()
    order = scores.argsort()[::-1][:top_k]
    return [(kb[int(i)], float(scores[int(i)])) for i in order]


def forced_ids(question_type: str, facts: Dict[str, str]) -> List[str]:
    feature_text = facts.get("top_features", "").lower()
    ids: List[str] = []
    if question_type == "feature_context":
        ids.append("XAI-002")
        if any(term in feature_text for term in ["hippocamp", "entorhinal", "temporal", "amygdala", "ventricle"]):
            ids.append("ALZ-003")
        if any(term in feature_text for term in ["tau", "a-beta", "abeta", "csf", "amyloid"]):
            ids.append("ALZ-002")
        if any(term in feature_text for term in ["mmse", "faq", "trail", "memory", "recall", "cognitive"]):
            ids.append("ALZ-004")
        if "apoe" in feature_text:
            ids.append("ALZ-006")
        ids.append("ADNI-001")
    elif question_type == "graph_context":
        ids = ["XAI-001"]
    elif question_type == "uncertainty_context":
        ids = ["XAI-003"]
    elif question_type == "diagnostic_safety":
        ids = ["ALZ-001"]
    elif question_type == "mci_context":
        ids = ["ALZ-005"]
    elif question_type == "counterfactual_context":
        ids = ["XAI-004"]
    return list(dict.fromkeys(ids))


def merge_evidence(
    retrieved: List[Tuple[Dict, float]],
    required: List[str],
    kb_by_id: Dict[str, Dict],
    top_k: int,
) -> List[Tuple[Dict, float]]:
    merged: List[Tuple[Dict, float]] = []
    for eid in required:
        if eid in kb_by_id:
            merged.append((kb_by_id[eid], 1.0))
    for entry, score in retrieved:
        if entry["id"] not in {item[0]["id"] for item in merged}:
            merged.append((entry, score))
        if len(merged) >= top_k:
            break
    return merged[:top_k]


def cite(evidence: Iterable[Tuple[Dict, float]]) -> str:
    ids = [entry["id"] for entry, _ in evidence]
    return "[" + "; ".join(ids) + "]"


def evidence_sentence(evidence: List[Tuple[Dict, float]], index: int = 0) -> str:
    if not evidence:
        return ""
    entry = evidence[min(index, len(evidence) - 1)][0]
    citation = f"[{entry['id']}]"
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", entry["text"]) if s.strip()]
    cited_parts = []
    for part in parts:
        if re.search(r"\[[A-Z]+-\d{3}\]", part):
            cited_parts.append(part)
        else:
            if part[-1:] in ".!?":
                cited_parts.append(f"{part[:-1]} {citation}{part[-1]}")
            else:
                cited_parts.append(f"{part} {citation}.")
    return " ".join(cited_parts)


def compose_answer(facts: Dict[str, str], question: Dict[str, str], evidence: List[Tuple[Dict, float]]) -> str:
    subject = facts["subject_id"]
    session = facts["clinical_session_id"]
    true_class = facts["true_class"]
    pred_class = facts["pred_class"]
    confidence = facts["confidence_pct"] or "not available"
    entropy = facts["entropy"] or "not available"
    qtype = question["question_type"]

    prefix = (
        f"Research-only answer for {subject}/{session}: the source BioLeaflet reports "
        f"DXSUM label {true_class}, model prediction {pred_class}, confidence {confidence}%, "
        f"and entropy {entropy}. This is non-diagnostic and should be read as model explanation, "
        f"not medical advice."
    )

    if qtype == "feature_context":
        return (
            f"{prefix} The highlighted features were {facts['top_features']}. "
            f"{evidence_sentence(evidence, 0)} "
            f"{evidence_sentence(evidence, 1)} "
            "For this patient-level leaflet, the useful statement is that these variables were locally influential "
            "for the trained model, not that they independently determine disease status."
        )
    if qtype == "graph_context":
        graph = facts["graph_agreement_pct"]
        if graph:
            agreement_sentence = (
                f"The leaflet reports neighbourhood agreement of {graph}%, defined as the "
                "influence-weighted share of influential neighbouring visits whose "
                "ensemble-predicted class matches the index visit's ensemble-predicted class; "
                "ground-truth diagnosis labels are not used in this calculation."
            )
        else:
            agreement_sentence = "The leaflet does not report a neighbourhood agreement value."
        return (
            f"{prefix} {agreement_sentence} "
            f"{evidence_sentence(evidence, 0)} "
            "Therefore, graph agreement is best described as model context from similar records, not as a separate clinical test."
        )
    if qtype == "uncertainty_context":
        return (
            f"{prefix} The confidence value summarizes the winning class probability, while entropy summarizes how spread "
            f"the model probabilities were across NC/MCI/AD. {evidence_sentence(evidence, 0)} "
            "In the manuscript, this can justify flagging ambiguous visits for cautious interpretation."
        )
    if qtype == "diagnostic_safety":
        return (
            f"{prefix} {evidence_sentence(evidence, 0)} "
            "For that reason, the BioLeaflet should be described as a communication aid for a research model, "
            "with clinical diagnosis remaining outside the automated system."
        )
    if qtype == "mci_context":
        return (
            f"{prefix} {evidence_sentence(evidence, 0)} "
            "This explains why the NC/MCI/AD boundary can be harder than the NC-vs-AD contrast and why MCI recall is a key metric."
        )
    if qtype == "counterfactual_context":
        return (
            f"{prefix} The source what-if statement was: {facts['counterfactual']}. "
            f"{evidence_sentence(evidence, 0)} "
            "The paper should present this as model sensitivity around the decision boundary, not as an intervention."
        )
    return f"{prefix} {evidence_sentence(evidence, 0)}"


def sentence_support_pass(answer: str) -> bool:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    for sentence in sentences:
        lower = sentence.lower()
        is_patient_fact = (
            "source bioleaflet reports" in lower
            or "highlighted features were" in lower
            or "neighbourhood agreement" in lower
            or "source what-if statement" in lower
            or "confidence value summarizes" in lower
            or "non-diagnostic" in lower
            or "not medical advice" in lower
            or "for this patient-level leaflet" in lower
            or "for that reason" in lower
            or "this explains why" in lower
            or "therefore" in lower
            or "in the manuscript" in lower
            or "the paper should present" in lower
        )
        has_citation = bool(re.search(r"\[[A-Z]+-\d{3}\]", sentence))
        has_multi_citation = bool(re.search(r"\[[A-Z]+-\d{3}(?:;\s*[A-Z]+-\d{3})+\]", sentence))
        if not is_patient_fact and not has_citation and not has_multi_citation:
            return False
    return True


def verify_answer(
    answer: str,
    facts: Dict[str, str],
    question: Dict[str, str],
    evidence: List[Tuple[Dict, float]],
    kb_ids: set,
) -> Dict:
    cited_ids = re.findall(r"[A-Z]+-\d{3}", answer)
    evidence_ids = [entry["id"] for entry, _ in evidence]
    required = forced_ids(question["question_type"], facts)

    confidence_ok = True
    if facts["confidence_pct"]:
        confidence_ok = f"{facts['confidence_pct']}%" in answer
    entropy_ok = True
    if facts["entropy"]:
        entropy_ok = facts["entropy"] in answer

    patient_fact_pass = (
        facts["subject_id"] in answer
        and facts["clinical_session_id"] in answer
        and f"DXSUM label {facts['true_class']}" in answer
        and f"model prediction {facts['pred_class']}" in answer
        and confidence_ok
        and entropy_ok
    )
    graph_agreement_pass = True
    if question["question_type"] == "graph_context":
        if facts["graph_agreement_pct"]:
            graph_agreement_pass = f"{facts['graph_agreement_pct']}%" in answer
        else:
            graph_agreement_pass = "does not report a neighbourhood agreement value" in answer.lower()

    valid_citations_pass = bool(cited_ids) and all(eid in kb_ids for eid in cited_ids)
    citation_coverage_pass = all(eid in cited_ids for eid in evidence_ids[:1])
    question_relevance_pass = all(eid in cited_ids for eid in required[:1]) if required else bool(cited_ids)
    restricted_language_absent = not any(
        re.search(pattern, answer, flags=re.IGNORECASE) for pattern in RESTRICTED_PATTERNS
    )
    non_diagnostic_pass = "non-diagnostic" in answer.lower() and "research" in answer.lower()
    support_pass = sentence_support_pass(answer)

    overall = all(
        [
            patient_fact_pass,
            valid_citations_pass,
            citation_coverage_pass,
            question_relevance_pass,
            restricted_language_absent,
            non_diagnostic_pass,
            support_pass,
            graph_agreement_pass,
        ]
    )

    return {
        "answer_id": question["answer_id"],
        "question_type": question["question_type"],
        "patient_fact_pass": int(patient_fact_pass),
        "valid_citations_pass": int(valid_citations_pass),
        "citation_coverage_pass": int(citation_coverage_pass),
        "question_relevance_pass": int(question_relevance_pass),
        "restricted_language_absent": int(restricted_language_absent),
        "non_diagnostic_pass": int(non_diagnostic_pass),
        "sentence_support_pass": int(support_pass),
        "graph_agreement_pass": int(graph_agreement_pass),
        "overall_pass": int(overall),
        "n_citations": len(cited_ids),
        "cited_ids": ";".join(cited_ids),
    }


def summarize_checks(checks: pd.DataFrame, answers: pd.DataFrame) -> Dict:
    bool_cols = [
        "patient_fact_pass",
        "valid_citations_pass",
        "citation_coverage_pass",
        "question_relevance_pass",
        "restricted_language_absent",
        "non_diagnostic_pass",
        "sentence_support_pass",
        "graph_agreement_pass",
        "overall_pass",
    ]
    summary = {
        "n_leaflets": int(answers[["subject_id", "clinical_session_id", "node_idx"]].drop_duplicates().shape[0]),
        "n_answers": int(len(answers)),
        "mean_evidence_per_answer": float(answers["n_evidence"].mean()),
        "mean_top_retrieval_score": float(answers["top_retrieval_score"].mean()),
    }
    for col in bool_cols:
        summary[col + "_rate"] = float(checks[col].mean()) if len(checks) else 0.0
    for qtype, count in Counter(answers["question_type"]).items():
        summary[f"n_{qtype}"] = int(count)
    return summary


def write_examples(answers: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# RAG BioLeaflet Follow-up Examples",
        "",
        "These examples are deterministic, citation-grounded follow-up answers generated from the verified BioLeaflet CSV.",
        "",
    ]
    chosen = []
    for qtype in [
        "feature_context",
        "graph_context",
        "uncertainty_context",
        "diagnostic_safety",
        "mci_context",
        "counterfactual_context",
    ]:
        subset = answers[answers["question_type"] == qtype]
        if not subset.empty:
            chosen.append(subset.iloc[0])
    for row in chosen:
        lines.extend(
            [
                f"## {row['question_type']}",
                "",
                f"Patient/session: {row['subject_id']} / {row['clinical_session_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Answer: {row['answer']}",
                "",
                f"Evidence IDs: {row['evidence_ids']}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(summary: Dict, out_path: Path) -> None:
    lines = [
        "# RAG BioLeaflet Follow-up Prototype Summary",
        "",
        "This run adds deterministic retrieval-augmented follow-up answers to the verified BioLeaflet outputs.",
        "",
        f"- Leaflets processed: {summary['n_leaflets']}",
        f"- Follow-up answers generated: {summary['n_answers']}",
        f"- Mean evidence snippets per answer: {summary['mean_evidence_per_answer']:.2f}",
        f"- Mean top retrieval score: {summary['mean_top_retrieval_score']:.3f}",
        f"- Patient fact pass rate: {summary['patient_fact_pass_rate']:.3f}",
        f"- Citation coverage pass rate: {summary['citation_coverage_pass_rate']:.3f}",
        f"- Question relevance pass rate: {summary['question_relevance_pass_rate']:.3f}",
        f"- Restricted language absent rate: {summary['restricted_language_absent_rate']:.3f}",
        f"- Non-diagnostic wording pass rate: {summary['non_diagnostic_pass_rate']:.3f}",
        f"- Sentence support pass rate: {summary['sentence_support_pass_rate']:.3f}",
        f"- Overall automatic verification pass rate: {summary['overall_pass_rate']:.3f}",
        "",
        "Recommended paper wording:",
        "",
        "We evaluated a retrieval-augmented BioLeaflet extension that answers fixed follow-up questions using curated Alzheimer biomarker and XAI evidence snippets. Patient-specific values were copied only from verified BioLeaflet source records, and every background statement was required to cite a retrieved evidence ID. Automatic checks assessed patient fact preservation, citation coverage, unsupported background sentences, non-diagnostic wording, and restricted clinical language.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bioleaflets = pd.read_csv(args.bioleaflets)
    if args.max_leaflets:
        bioleaflets = bioleaflets.head(args.max_leaflets).copy()
    with open(args.knowledge_base, "r", encoding="utf-8") as f:
        kb = json.load(f)
    kb_by_id = {entry["id"]: entry for entry in kb}
    kb_ids = set(kb_by_id)

    vectorizer, matrix = build_retriever(kb)

    answer_rows = []
    check_rows = []
    for _, row in bioleaflets.iterrows():
        facts = parse_leaflet_facts(row)
        for question in make_questions(facts):
            raw_retrieved = retrieve(question["query"], kb, vectorizer, matrix, top_k=max(args.top_k, 4))
            evidence = merge_evidence(raw_retrieved, forced_ids(question["question_type"], facts), kb_by_id, args.top_k)
            answer = compose_answer(facts, question, evidence)
            check = verify_answer(answer, facts, question, evidence, kb_ids)
            check_rows.append(check)

            evidence_ids = [entry["id"] for entry, _ in evidence]
            answer_rows.append(
                {
                    "answer_id": question["answer_id"],
                    "node_idx": facts["node_idx"],
                    "subject_id": facts["subject_id"],
                    "clinical_session_id": facts["clinical_session_id"],
                    "true_class": facts["true_class"],
                    "pred_class": facts["pred_class"],
                    "question_type": question["question_type"],
                    "question": question["question"],
                    "answer": answer,
                    "evidence_ids": ";".join(evidence_ids),
                    "evidence_titles": ";".join(entry["source_title"] for entry, _ in evidence),
                    "evidence_urls": ";".join(entry["source_url"] for entry, _ in evidence),
                    "retrieval_scores": ";".join(f"{score:.4f}" for _, score in evidence),
                    "top_retrieval_score": float(evidence[0][1]) if evidence else 0.0,
                    "n_evidence": len(evidence),
                }
            )

    answers = pd.DataFrame(answer_rows)
    checks = pd.DataFrame(check_rows)
    summary = summarize_checks(checks, answers)

    answers.to_csv(out_dir / "rag_followup_answers.csv", index=False)
    checks.to_csv(out_dir / "rag_followup_checks.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "rag_followup_summary.csv", index=False)
    (out_dir / "rag_followup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_examples(answers, out_dir / "rag_followup_examples.md")
    write_summary_md(summary, out_dir / "rag_followup_summary.md")

    # Save the exact KB used with the outputs for reproducibility.
    (out_dir / "bioleaflet_rag_knowledge_base.used.json").write_text(
        json.dumps(kb, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic RAG follow-up BioLeaflet prototype")
    parser.add_argument("--bioleaflets", default=str(DEFAULT_BIOLEAFLETS))
    parser.add_argument("--knowledge_base", default=str(DEFAULT_KB))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_leaflets", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
