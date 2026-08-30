import tempfile
import unittest
from pathlib import Path

import pandas as pd

import hybrid_gnn


class FixedSubjectSplitTests(unittest.TestCase):
    def test_reuses_assignments_and_ignores_extra_subjects(self):
        cohort = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "C", "D"],
                "DIAGNOSIS": ["0", "0", "0.5", "1+", "0.5"],
            }
        )
        assignments = pd.DataFrame(
            {
                "subject_id": ["A", "B", "C", "D", "EXTRA"],
                "split": ["train", "val", "test", "train", "test"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            split_file = Path(tmp) / "split.csv"
            assignments.to_csv(split_file, index=False)
            train, val, test = hybrid_gnn.subject_split_from_file(
                cohort, "subject_id", split_file
            )

        self.assertEqual(train.tolist(), [True, True, False, False, True])
        self.assertEqual(val.tolist(), [False, False, True, False, False])
        self.assertEqual(test.tolist(), [False, False, False, True, False])

    def test_missing_subject_assignment_is_rejected(self):
        cohort = pd.DataFrame({"subject_id": ["A", "B", "C"]})
        assignments = pd.DataFrame(
            {"subject_id": ["A", "B"], "split": ["train", "val"]}
        )

        with tempfile.TemporaryDirectory() as tmp:
            split_file = Path(tmp) / "split.csv"
            assignments.to_csv(split_file, index=False)
            with self.assertRaisesRegex(ValueError, "does not assign"):
                hybrid_gnn.subject_split_from_file(cohort, "subject_id", split_file)


class TargetLabelTests(unittest.TestCase):
    def tearDown(self):
        hybrid_gnn.configure_class_labels("diagnosis")

    def test_cdr_exports_do_not_use_diagnosis_names(self):
        hybrid_gnn.configure_class_labels("cdr")
        self.assertEqual(hybrid_gnn.CLASS_DISPLAY["0.5"], "CDR 0.5")
        self.assertEqual(hybrid_gnn.CLASS_EXPORT["0.5"], "CDR0_5")


if __name__ == "__main__":
    unittest.main()
