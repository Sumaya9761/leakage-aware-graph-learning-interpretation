import tempfile
import unittest
from pathlib import Path

import pandas as pd

from paired_ablation_validation import (
    align_candidate,
    load_probabilities,
    paired_participant_bootstrap,
)


def probability_rows(predictions: list[str]) -> list[dict[str, float]]:
    probabilities = {
        "0": {"prob_NC": 0.8, "prob_MCI": 0.1, "prob_AD": 0.1},
        "0.5": {"prob_NC": 0.1, "prob_MCI": 0.8, "prob_AD": 0.1},
        "1+": {"prob_NC": 0.1, "prob_MCI": 0.1, "prob_AD": 0.8},
    }
    return [probabilities[prediction] for prediction in predictions]


class PairedAblationValidationTests(unittest.TestCase):
    def make_frame(self, predictions: list[str]) -> pd.DataFrame:
        labels = ["0"] * 4 + ["0.5"] * 4 + ["1+"] * 4
        frame = pd.DataFrame(
            {
                "subject_id": [f"S{index:02d}" for index in range(12)],
                "clinical_session_id": ["bl"] * 12,
                "true_class": labels,
            }
        )
        return pd.concat([frame, pd.DataFrame(probability_rows(predictions))], axis=1)

    def write_and_load(self, frame: pd.DataFrame, directory: Path, name: str) -> pd.DataFrame:
        path = directory / name
        frame.to_csv(path, index=False)
        return load_probabilities(path)

    def test_paired_bootstrap_detects_reference_advantage(self):
        labels = ["0"] * 4 + ["0.5"] * 4 + ["1+"] * 4
        candidate_predictions = labels.copy()
        candidate_predictions[0] = "0.5"
        candidate_predictions[4] = "1+"
        candidate_predictions[8] = "0"
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            reference = self.write_and_load(self.make_frame(labels), directory, "reference.csv")
            candidate = self.write_and_load(
                self.make_frame(candidate_predictions), directory, "candidate.csv"
            )
            result = paired_participant_bootstrap(reference, candidate, replicates=500, seed=7)

        difference = result["differences"]["balanced_accuracy"]
        self.assertGreater(result["accepted_replicates"], 0)
        self.assertGreater(difference["mean_difference"], 0)
        self.assertGreater(difference["probability_reference_gt_candidate"], 0.5)

    def test_alignment_rejects_changed_visit_keys(self):
        labels = ["0"] * 4 + ["0.5"] * 4 + ["1+"] * 4
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            reference = self.write_and_load(self.make_frame(labels), directory, "reference.csv")
            changed = self.make_frame(labels)
            changed.loc[0, "clinical_session_id"] = "m06"
            candidate = self.write_and_load(changed, directory, "candidate.csv")
            with self.assertRaisesRegex(ValueError, "same ordered participant-session rows"):
                align_candidate(reference, candidate, "changed")


if __name__ == "__main__":
    unittest.main()
