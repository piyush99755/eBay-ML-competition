"""
Unit tests for Phase 39A: Official Competition Metric Evaluation Utility.
Tests all requirements specified in Phase 39A Task 2:
1. Argument ordering and non-commutativity.
2. Exact entity and aspect matching.
3. Aspect-specific precision, recall, and F0.2.
4. Ground-truth frequency weighting per category.
5. Category macro-averaging (50/50).
6. Zero-prediction aspects.
7. False-positive-only aspects.
8. Multiset duplicate handling.
9. Distinguishes Global Micro F0.2 vs Official-Style Hierarchical F0.2.
"""

import unittest
from src.evaluation.phase_39a_official_metric import (
    BETA,
    compute_fbeta,
    count_entity_matches,
    calculate_official_competition_metrics,
    calculate_global_micro_metrics,
)


class TestPhase39AOfficialMetric(unittest.TestCase):

    def test_compute_fbeta_mathematical_precision(self):
        """Test F0.2 calculation with known precision/recall."""
        # When P=1.0, R=1.0 -> F0.2=1.0
        self.assertAlmostEqual(compute_fbeta(1.0, 1.0, beta=0.2), 1.0, places=6)
        # When P=0.0 or R=0.0 -> F0.2=0.0
        self.assertAlmostEqual(compute_fbeta(0.0, 1.0, beta=0.2), 0.0, places=6)
        self.assertAlmostEqual(compute_fbeta(1.0, 0.0, beta=0.2), 0.0, places=6)

        # High precision, low recall: P=0.90, R=0.50
        # F0.2 = 1.04 * (0.90 * 0.50) / (0.04 * 0.90 + 0.50) = 1.04 * 0.45 / (0.036 + 0.50) = 0.468 / 0.536 = 0.873134
        expected = 1.04 * (0.90 * 0.50) / (0.04 * 0.90 + 0.50)
        self.assertAlmostEqual(compute_fbeta(0.90, 0.50, beta=0.2), expected, places=5)

        # Low precision, high recall: P=0.50, R=0.90
        # F0.2 = 1.04 * (0.50 * 0.90) / (0.04 * 0.50 + 0.90) = 1.04 * 0.45 / (0.020 + 0.90) = 0.468 / 0.920 = 0.508696
        expected_low_p = 1.04 * (0.50 * 0.90) / (0.04 * 0.50 + 0.90)
        self.assertAlmostEqual(compute_fbeta(0.50, 0.90, beta=0.2), expected_low_p, places=5)
        # Verify 25x precision preference: F0.2 with high precision is much higher
        self.assertGreater(expected, expected_low_p + 0.35)

    def test_argument_ordering_asymmetry(self):
        """Verify that count_entity_matches is NOT symmetric: swapping (pred, gold) swaps FP and FN."""
        pred = [("Hersteller", "BOSCH"), ("Hersteller", "ATE")]
        gold = [("Hersteller", "BOSCH")]

        tp, fp, fn = count_entity_matches(pred, gold)
        self.assertEqual((tp, fp, fn), (1, 1, 0))

        # Inverted call
        tp_inv, fp_inv, fn_inv = count_entity_matches(gold, pred)
        self.assertEqual((tp_inv, fp_inv, fn_inv), (1, 0, 1))

    def test_multiset_duplicate_handling(self):
        """Test repeated identical entities per Annexure.pdf Notes 6 & 7."""
        # Gold has 2 "Golf" models; model predicts 3 "Golf" models
        pred = [
            ("Kompatibles_Fahrzeug_Modell", "Golf"),
            ("Kompatibles_Fahrzeug_Modell", "Golf"),
            ("Kompatibles_Fahrzeug_Modell", "Golf"),
        ]
        gold = [
            ("Kompatibles_Fahrzeug_Modell", "Golf"),
            ("Kompatibles_Fahrzeug_Modell", "Golf"),
        ]
        tp, fp, fn = count_entity_matches(pred, gold)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

    def test_synthetic_hierarchical_scoring(self):
        """Verify complete category weighting and macro-average against manually calculated truth."""
        records = [
            {
                "category": 1,
                "gold_entities": [
                    ("Hersteller", "BOSCH"),
                    ("Produktart", "Bremsscheibe"),
                    ("Einbauposition", "vorne"),
                ],
                "predicted_entities": [
                    ("Hersteller", "BOSCH"),        # TP
                    ("Produktart", "Bremsscheibe"),  # TP
                    ("Einbauposition", "hinten"),   # FP (and FN for 'vorne')
                ],
            },
            {
                "category": 2,
                "gold_entities": [
                    ("Hersteller", "Conti"),         # TP
                    ("Produktart", "Zahnriemen"),    # FN
                ],
                "predicted_entities": [
                    ("Hersteller", "Conti"),         # TP
                    ("Produktart", "Keilriemen"),    # FP
                ],
            },
        ]

        res = calculate_official_competition_metrics(records)

        # Category 1 breakdown:
        # Total gold in Cat 1 = 3 entities (each aspect has 1 gold entity -> weight = 1/3)
        # Hersteller: TP=1, FP=0, FN=0 -> P=1, R=1, F0.2=1.0 -> weighted = 1/3 * 1.0 = 0.333333
        # Produktart: TP=1, FP=0, FN=0 -> P=1, R=1, F0.2=1.0 -> weighted = 1/3 * 1.0 = 0.333333
        # Einbauposition: TP=0, FP=1, FN=1 -> P=0, R=0, F0.2=0.0 -> weighted = 0.0
        # Cat 1 Score = 1/3 + 1/3 + 0 = 2/3 = 0.666667
        cat1 = res["categories"][1]
        self.assertAlmostEqual(cat1["score"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(cat1["aspects"]["Hersteller"]["weight"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(cat1["aspects"]["Hersteller"]["f_beta"], 1.0, places=5)

        # Category 2 breakdown:
        # Total gold in Cat 2 = 2 entities (Hersteller: 1, Produktart: 1 -> weight = 1/2 each)
        # Hersteller: TP=1, FP=0, FN=0 -> F0.2=1.0 -> weighted = 0.5
        # Produktart: TP=0, FP=1, FN=1 -> F0.2=0.0 -> weighted = 0.0
        # Cat 2 Score = 0.5 + 0.0 = 0.50
        cat2 = res["categories"][2]
        self.assertAlmostEqual(cat2["score"], 0.50, places=5)

        # Official score = (Cat 1 Score + Cat 2 Score) / 2 = (2/3 + 1/2) / 2 = (7/6) / 2 = 7/12 = 0.583333
        expected_official = (2.0 / 3.0 + 0.50) / 2.0
        self.assertAlmostEqual(res["official_score"], expected_official, places=5)

    def test_difference_between_micro_and_official(self):
        """
        Verify that Global Micro F0.2 differs from Official Hierarchical F0.2
        when categories have unequal sizes or aspect distributions.
        """
        records = [
            # Category 1 has 10 identical perfect items
            {
                "category": 1,
                "gold_entities": [("Hersteller", "BOSCH")] * 10,
                "predicted_entities": [("Hersteller", "BOSCH")] * 10,
            },
            # Category 2 has 1 item with 0 true positives
            {
                "category": 2,
                "gold_entities": [("Hersteller", "Conti")],
                "predicted_entities": [("Hersteller", "Gates")],
            },
        ]

        official = calculate_official_competition_metrics(records)
        micro = calculate_global_micro_metrics(records)

        # Official: Cat 1 score = 1.0, Cat 2 score = 0.0 -> Official = (1.0 + 0.0) / 2 = 0.50
        self.assertAlmostEqual(official["official_score"], 0.50, places=5)

        # Global Micro: Total TP = 10, Total FP = 1, Total FN = 1
        # P = 10/11 = 0.909091, R = 10/11 = 0.909091 -> F0.2 = 0.909091
        # Notice that global micro is heavily dominated by Cat 1 (0.909 vs 0.500)!
        self.assertAlmostEqual(micro["global_f_beta"], 10.0 / 11.0, places=5)
        self.assertNotAlmostEqual(official["official_score"], micro["global_f_beta"], places=2)

    def test_zero_support_and_false_positive_only_aspects(self):
        """Test aspects with zero gold entities but some spurious predictions."""
        records = [
            {
                "category": 1,
                "gold_entities": [("Hersteller", "BOSCH")],
                "predicted_entities": [
                    ("Hersteller", "BOSCH"),
                    ("Farbe", "rot"),  # Gold has 0 'Farbe' entities
                ],
            }
        ]
        res = calculate_official_competition_metrics(records)
        cat1 = res["categories"][1]
        farbe = cat1["aspects"]["Farbe"]
        self.assertEqual(farbe["gold_count"], 0)
        self.assertEqual(farbe["pred_count"], 1)
        self.assertEqual(farbe["fp"], 1)
        self.assertEqual(farbe["tp"], 0)
        self.assertEqual(farbe["weight"], 0.0)
        self.assertEqual(farbe["weighted_f_beta"], 0.0)


if __name__ == "__main__":
    unittest.main()
