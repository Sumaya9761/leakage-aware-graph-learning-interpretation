import unittest

import numpy as np
import pandas as pd

import generate_and_evaluate_bioleaflets as evaluation


class BioLeafletEvaluationTests(unittest.TestCase):
    def test_exported_model_labels_are_normalized(self):
        self.assertEqual(evaluation.model_label("0"), "NC")
        self.assertEqual(evaluation.model_label(0.5), "MCI")
        self.assertEqual(evaluation.model_label("1+"), "AD")

    def test_graph_agreement_is_described_against_predicted_class(self):
        row = pd.Series(
            {
                "node_idx": 1,
                "subject_id": "A",
                "clinical_session_id": "bl",
                "true_class": "AD",
                "pred_class": "MCI",
                "top_features": [
                    {
                        "feature": "clinical_MMSCORE",
                        "label": "MMSE",
                        "importance": 0.9,
                        "raw_value": 25,
                    }
                ],
                "top_feature_scope": "patient-specific attribution",
                "graph_agreement_pct": 75.0,
                "class_influence_pct_MCI": 75.0,
                "max_prob": 0.6,
                "predictive_entropy": 0.8,
                "aleatoric_uncertainty": 0.7,
                "epistemic_uncertainty": 0.1,
                "is_high_uncertainty": False,
                "cf_class": np.nan,
                "clinical_CDGLOBAL": np.nan,
            }
        )
        leaflet = evaluation.build_leaflet(row)
        self.assertIn("neighbours sharing the predicted class", leaflet.text)
        self.assertNotIn("true class neighbourhood", leaflet.text.lower())
        checks = evaluation.verify_leaflet(leaflet)
        graph_check = next(
            item for item in checks if item["field"] == "graph_agreement_basis"
        )
        self.assertTrue(graph_check["correct"])


if __name__ == "__main__":
    unittest.main()
