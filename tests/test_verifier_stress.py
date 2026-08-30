import unittest

import bio_leaflet
import run_bioleaflet_rag_prototype as rag
import verifier_stress_test as stress


class VerifierStressTests(unittest.TestCase):
    def test_missing_source_fact_is_an_error(self):
        checked, correct, issues = bio_leaflet.compute_fact_metrics(
            {"patient_id": "A", "age": 70.0}, {"patient_id": "A"}
        )
        self.assertEqual(checked, 2)
        self.assertEqual(correct, 1)
        self.assertTrue(any("required fact is missing" in issue for issue in issues))

    def test_numeric_mutation_changes_value(self):
        original = "Age: 70.0 years"
        mutated = stress.replace_numeric(original, r"Age:\s*([\d.]+)", 10.0)
        self.assertEqual(mutated, "Age: 80.0 years")

    def test_stage_mismatch_changes_verification_status(self):
        context = {
            "patient_id": "A",
            "age": "70.0",
            "sex": "Male",
            "apoe4_status": "Non-carrier (0 copies)",
            "confidence_pct": "60.0",
            "mmse": "28",
            "predictive_entropy": "0.5000",
            "graph_agreement_pct": "60.0",
            "pred_class": "0",
        }
        text = """Patient ID: A
Age: 70.0 years, Sex: Male
APOE genotype (genetic susceptibility marker): Non-carrier (0 copies)
Global cognition (MMSE): 28 / 30
Model confidence: 60.0%
INTERPRETATION :
This profile reflects Alzheimer's disease.
GRAPH CONTEXT :
Graph agreement: 60.0%
UNCERTAINTY ANALYSIS :
Predictive entropy: 0.5000
SUMMARY :
Automated research summary.
"""
        status, issues, _ = bio_leaflet.verify_leaflet(text, context)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("STAGE_MISMATCH" in issue for issue in issues))

    def test_appended_contradictory_age_is_rejected(self):
        context = {
            "patient_id": "A",
            "age": "70.0",
            "sex": "Male",
            "apoe4_status": "Non-carrier (0 copies)",
            "confidence_pct": "60.0",
            "mmse": "28",
            "predictive_entropy": "0.5000",
            "graph_agreement_pct": "60.0",
            "pred_class": "0.5",
        }
        text = """Patient ID: A
Age: 70.0 years, Sex: Male
APOE genotype: Non-carrier (0 copies)
MMSE: 28 / 30
Model confidence: 60.0%
Predictive entropy: 0.5000
Graph agreement: 60.0%
INTERPRETATION :
Mild cognitive impairment.
SUMMARY :
Research summary.
Age: 99.9 years at entry.
"""
        status, issues, _ = bio_leaflet.verify_leaflet(text, context)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("CONTRADICTORY_FACT" in issue for issue in issues))

    def test_counterfactual_target_is_not_read_as_current_prediction(self):
        facts = {
            "subject_id": "A",
            "clinical_session_id": "bl",
            "true_class": "AD",
            "pred_class": "MCI",
            "confidence_pct": "44.5",
            "entropy": "1.028",
            "graph_agreement_pct": "100.0",
        }
        question = {
            "answer_id": "A_bl_counterfactual",
            "question_type": "counterfactual_context",
        }
        answer = (
            "Research-only answer for A/bl: the source BioLeaflet reports "
            "DXSUM label AD, model prediction MCI, confidence 44.5%, and "
            "entropy 1.028. This is non-diagnostic. Changing the model "
            "prediction toward AD is a sensitivity statement [XAI-004]."
        )
        result = rag.verify_answer(
            answer,
            facts,
            question,
            [({"id": "XAI-004"}, 1.0)],
            {"XAI-004"},
        )
        self.assertEqual(result["patient_fact_consistency_pass"], 1)

    def test_unsupported_pathology_statement_is_rejected(self):
        context = {
            "patient_id": "A",
            "age": "70.0",
            "sex": "Male",
            "apoe4_status": "Non-carrier (0 copies)",
            "confidence_pct": "60.0",
            "mmse": "28",
            "predictive_entropy": "0.5000",
            "graph_agreement_pct": "60.0",
            "pred_class": "1+",
        }
        text = """Patient ID: A
Age: 70.0 years, Sex: Male
APOE genotype: Non-carrier (0 copies)
MMSE: 28 / 30
Model confidence: 60.0%
Predictive entropy: 0.5000
Graph agreement: 60.0%
INTERPRETATION :
Biomarkers support Alzheimer's-type pathology.
SUMMARY :
Research summary.
"""
        status, issues, _ = bio_leaflet.verify_leaflet(text, context)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("HALLUCINATION" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
