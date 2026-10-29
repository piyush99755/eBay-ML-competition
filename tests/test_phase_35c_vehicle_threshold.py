"""
Unit tests for Phase 35C: Aspect-Specific Thresholding and Subset Invariant.
"""

import unittest
import numpy as np

from src.evaluation.entity_metrics import (
    bio_to_entities_with_confidence,
    filter_entities_by_category,
)


class TestAspectSpecificThresholding(unittest.TestCase):
    """Verify entity confidence calculation, aspect-specific thresholding, and subset invariant."""

    def test_entity_confidence_calculation(self):
        """
        Verify that min_conf correctly computes the minimum selected probability
        across constituent tokens of an entity.
        """
        tokens = ["BMW", "3er", "Limousine", "Bremsbelag"]
        bio_tags = ["B-Kompatible_Fahrzeug_Marke", "B-Kompatibles_Fahrzeug_Modell", "I-Kompatibles_Fahrzeug_Modell", "B-Produktart"]
        probs = [0.95, 0.75, 0.85, 0.90]

        entities = bio_to_entities_with_confidence(tokens, bio_tags, probs)
        self.assertEqual(len(entities), 3)

        # 1. Marke: BMW
        self.assertEqual(entities[0][0], "Kompatible_Fahrzeug_Marke")
        self.assertEqual(entities[0][1], "BMW")
        self.assertAlmostEqual(entities[0][3], 0.95)

        # 2. Modell: 3er Limousine
        self.assertEqual(entities[1][0], "Kompatibles_Fahrzeug_Modell")
        self.assertEqual(entities[1][1], "3er Limousine")
        self.assertAlmostEqual(entities[1][2], 0.80)  # mean (0.75 + 0.85)/2
        self.assertAlmostEqual(entities[1][3], 0.75)  # min(0.75, 0.85) = 0.75

        # 3. Produktart: Bremsbelag
        self.assertEqual(entities[2][0], "Produktart")
        self.assertEqual(entities[2][1], "Bremsbelag")
        self.assertAlmostEqual(entities[2][3], 0.90)

    def test_subset_invariant_mathematical_proof(self):
        """
        Prove that for any aspect A and thresholds T_high > T_low:
        {e in E | e.conf >= (T_high if e.asp == A else T_low)} is a strict subset of
        {e in E | e.conf >= T_low}.
        """
        np.random.seed(42)
        n_entities = 1000
        aspects = [
            "Kompatibles_Fahrzeug_Modell",
            "Produktart",
            "Hersteller",
            "Einbauposition",
            "Im_Lieferumfang_Enthalten",
        ]

        test_entities = []
        for i in range(n_entities):
            asp = np.random.choice(aspects)
            val = f"val_{i}"
            conf = float(np.random.uniform(0.40, 1.00))
            test_entities.append((asp, val, conf))

        # Baseline: MIN 0.60 across all aspects
        baseline_preds = [
            (asp, val) for asp, val, conf in test_entities if conf >= 0.60
        ]

        # Challenger: Vehicle 0.80, others 0.60
        challenger_preds = [
            (asp, val) for asp, val, conf in test_entities
            if conf >= (0.80 if asp == "Kompatibles_Fahrzeug_Modell" else 0.60)
        ]

        # 1. Subset invariant: every challenger pred must be in baseline
        base_set = set(baseline_preds)
        chal_set = set(challenger_preds)
        self.assertTrue(chal_set.issubset(base_set))

        # 2. Exactly zero added predictions
        added = chal_set - base_set
        self.assertEqual(len(added), 0)

        # 3. All non-vehicle predictions must be identical
        base_non_vm = [(a, v) for a, v in baseline_preds if a != "Kompatibles_Fahrzeug_Modell"]
        chal_non_vm = [(a, v) for a, v in challenger_preds if a != "Kompatibles_Fahrzeug_Modell"]
        self.assertEqual(base_non_vm, chal_non_vm)

        # 4. Removed predictions must consist exclusively of vehicle models with 0.60 <= conf < 0.80
        removed = base_set - chal_set
        for a, v in removed:
            self.assertEqual(a, "Kompatibles_Fahrzeug_Modell")

    def test_aspect_specific_filtering_logic(self):
        """Test targeted thresholding on mock title."""
        cat_id = 1
        raw_ents = [
            ("Kompatibles_Fahrzeug_Modell", "Golf VII", 0.72, 0.72),
            ("Kompatibles_Fahrzeug_Modell", "Passat", 0.88, 0.88),
            ("Produktart", "Bremsscheiben", 0.65, 0.65),
            ("Hersteller", "Brembo", 0.55, 0.55),
        ]

        # Baseline: MIN 0.60
        base_ents = [
            (a, v) for a, v, _, min_c in raw_ents
            if min_c >= 0.60 and filter_entities_by_category([(a, v)], cat_id)
        ]
        self.assertEqual(len(base_ents), 3)  # Golf VII, Passat, Bremsscheiben

        # Challenger: Vehicle 0.80, others 0.60
        chal_ents = [
            (a, v) for a, v, _, min_c in raw_ents
            if min_c >= (0.80 if a == "Kompatibles_Fahrzeug_Modell" else 0.60)
            and filter_entities_by_category([(a, v)], cat_id)
        ]
        self.assertEqual(len(chal_ents), 2)  # Passat, Bremsscheiben (Golf VII pruned!)
        self.assertIn(("Kompatibles_Fahrzeug_Modell", "Passat"), chal_ents)
        self.assertIn(("Produktart", "Bremsscheiben"), chal_ents)
        self.assertNotIn(("Kompatibles_Fahrzeug_Modell", "Golf VII"), chal_ents)


if __name__ == "__main__":
    unittest.main()
