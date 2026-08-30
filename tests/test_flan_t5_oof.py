import unittest

import pandas as pd

from flan_t5_oof import (
    UNCERTAINTY_WARNING,
    _repair_uncertainty_language,
    make_outer_splits,
    summarize_oof,
)


class FlanT5OutOfFoldTests(unittest.TestCase):
    def _corpus(self):
        rows = []
        classes = ["0", "0.5", "1+"]
        for subject_index in range(15):
            subject = f"S{subject_index:02d}"
            pred_class = classes[subject_index % len(classes)]
            for visit in range(2):
                rows.append(
                    {
                        "input": f"input {subject} {visit}",
                        "target": "CURRENT STATUS: grounded text SUMMARY: safe summary text",
                        "pred_class": pred_class,
                        "subject_id": subject,
                        "node_idx": subject_index * 2 + visit,
                        "mode": "node",
                    }
                )
            rows.append(
                {
                    "input": f"variant {subject}",
                    "target": "CURRENT STATUS: grounded text SUMMARY: safe summary text",
                    "pred_class": pred_class,
                    "subject_id": subject,
                    "node_idx": subject_index * 2,
                    "mode": "node_v1",
                }
            )
        return pd.DataFrame(rows)

    def test_outer_splits_are_subject_disjoint_and_exhaustive(self):
        corpus = self._corpus()
        splits = make_outer_splits(corpus, n_splits=5, seed=7)
        seen_subjects = set()
        seen_nodes = set()
        for split in splits:
            train_subjects = set(split["train_subjects"])
            test_subjects = set(split["test_subjects"])
            self.assertFalse(train_subjects & test_subjects)
            self.assertFalse(seen_subjects & test_subjects)
            seen_subjects.update(test_subjects)
            node_rows = corpus.loc[
                corpus["mode"].eq("node")
                & corpus["subject_id"].isin(test_subjects)
            ]
            self.assertFalse(seen_nodes & set(node_rows["node_idx"]))
            seen_nodes.update(node_rows["node_idx"])

        original = corpus.loc[corpus["mode"].eq("node")]
        self.assertEqual(seen_subjects, set(original["subject_id"]))
        self.assertEqual(seen_nodes, set(original["node_idx"]))

    def test_summary_reports_guardrails_without_identifiers(self):
        predictions = pd.DataFrame(
            {
                "subject_id": ["S1", "S1", "S2"],
                "fold": [1, 1, 2],
                "pred_class": ["0", "0", "0.5"],
                "raw_prediction": ["alpha beta", "gamma delta", "epsilon zeta"],
                "prediction": ["alpha beta", "gamma delta", "epsilon zeta"],
                "target": ["alpha beta", "gamma theta", "epsilon zeta"],
                "candidate_accepted": [True, False, True],
                "targeted_uncertainty_repair": [False, True, False],
                "targeted_uncertainty_repair_success": [False, True, False],
                "post_repair_candidate_pass": [True, True, True],
                "fell_back_to_template": [False, False, False],
                "final_report_pass": [True, True, True],
                "headers_present": [True, False, True],
                "target_guardrail_pass": [True, False, True],
                "semantic_guardrail_pass": [True, False, True],
                "hybrid_report_pass": [True, True, True],
            }
        )
        summary = summarize_oof(predictions, bootstrap_replicates=100, seed=3)
        self.assertEqual(summary["n_participants"], 2)
        self.assertEqual(summary["n_visit_reports"], 3)
        self.assertEqual(summary["guardrails"]["candidate_accepted"], 2)
        self.assertEqual(summary["guardrails"]["targeted_uncertainty_repairs"], 1)
        self.assertEqual(
            summary["guardrails"]["candidates_passing_after_targeted_repair"], 3
        )
        self.assertEqual(summary["guardrails"]["template_fallbacks"], 0)
        self.assertEqual(summary["guardrails"]["final_reports_passing_verification"], 3)
        self.assertNotIn("subject_id", summary)

    def test_uncertainty_repair_removes_only_unsupported_warning(self):
        text = (
            "CURRENT STATUS: Grounded statement. "
            "SUMMARY: This is an automated research summary, not a clinical diagnosis. "
            "Elevated uncertainty. The accompanying sections summarize uncertainty."
        )
        repaired, changed = _repair_uncertainty_language(text, False)
        self.assertTrue(changed)
        self.assertNotIn("Elevated uncertainty", repaired)
        self.assertIn("sections summarize uncertainty", repaired)

    def test_uncertainty_repair_adds_required_warning_once(self):
        text = (
            "CURRENT STATUS: Grounded statement. "
            "SUMMARY: This is an automated research summary, not a clinical diagnosis. "
            "The accompanying sections summarize uncertainty."
        )
        repaired, changed = _repair_uncertainty_language(text, True)
        self.assertTrue(changed)
        self.assertEqual(repaired.count(UNCERTAINTY_WARNING), 1)

    def test_uncertainty_repair_leaves_unrecognized_prose_for_fallback(self):
        text = (
            "CURRENT STATUS: Grounded statement. "
            "SUMMARY: Elevated uncertainty was inferred from unrelated prose."
        )
        repaired, changed = _repair_uncertainty_language(text, False)
        self.assertFalse(changed)
        self.assertEqual(repaired, text)


if __name__ == "__main__":
    unittest.main()
