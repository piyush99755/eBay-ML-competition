"""
Unit tests for Phase 29B: 3-Fold Stratified & Grouped Partitioning.
"""

import unittest
from src.data.build_examples import load_training_data, build_examples
from src.data.make_splits import split_examples
from src.data.make_cv_folds import get_cv_folds, verify_fold_integrity


class TestCVFolds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = load_training_data()
        cls.examples = build_examples(df)
        cls.train_examples, cls.val_examples = split_examples(cls.examples, random_state=42)

    def test_fold_integrity_and_disjointness(self):
        folds = get_cv_folds(self.train_examples, n_splits=3, random_state=42)
        self.assertEqual(len(folds), 3)

        report = verify_fold_integrity(folds, self.val_examples)
        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(report["total_val_titles"], 4000)

        # Check fold sizes
        summaries = report["fold_summaries"]
        # Fold 0 has 1333, Fold 1 has 1334, Fold 2 has 1333
        self.assertEqual(summaries[0]["val_size"], 1333)
        self.assertEqual(summaries[1]["val_size"], 1334)
        self.assertEqual(summaries[2]["val_size"], 1333)

        # Check category stratification
        for s in summaries:
            # Each fold has 666-667 of Cat 1 and 666-667 of Cat 2
            self.assertIn(s["cat_counts"][1], (666, 667))
            self.assertIn(s["cat_counts"][2], (666, 667))

        # Check near duplicate group (1779 and 4204)
        # They should be together in fold 0
        self.assertTrue(summaries[0]["contains_grouped_pair"])
        self.assertFalse(summaries[1]["contains_grouped_pair"])
        self.assertFalse(summaries[2]["contains_grouped_pair"])


if __name__ == "__main__":
    unittest.main()
