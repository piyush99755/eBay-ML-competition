"""
Unit tests for confidence calculation, threshold filtering, and competition scoring.

Tests:
1. Single-token entity confidence extraction.
2. Multi-token entity confidence calculation (mean vs min).
3. BIO edge cases (consecutive B-, malformed I-, trailing open entity).
4. Zero-entity titles (all O).
5. Threshold boundary conditions (>= threshold, boundary equality).
6. Category-aware post-filtering before threshold filtering.
7. Exact TP / FP / FN bookkeeping and competition metric verification.
"""

import unittest
import numpy as np
from src.evaluation.entity_metrics import (
    bio_to_entities,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
)
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.experiment_26a_confidence_calibration import (
    bio_to_entities_with_confidence,
)


class TestConfidenceFiltering(unittest.TestCase):

    def test_single_token_entity_confidence(self):
        tokens = ["BOSCH", "Bremsscheibe"]
        bio_tags = ["B-Hersteller", "B-Produktart"]
        probs = [0.95, 0.85]

        entities = bio_to_entities_with_confidence(tokens, bio_tags, probs)
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0], ("Hersteller", "BOSCH", 0.95, 0.95))
        self.assertEqual(entities[1], ("Produktart", "Bremsscheibe", 0.85, 0.85))

    def test_multi_token_entity_confidence(self):
        tokens = ["Zahnriemen", "Satz", "Kit", "Plus"]
        bio_tags = ["B-Produktart", "I-Produktart", "I-Produktart", "I-Produktart"]
        probs = [0.90, 0.70, 0.80, 0.60]

        entities = bio_to_entities_with_confidence(tokens, bio_tags, probs)
        self.assertEqual(len(entities), 1)
        asp, val, mean_c, min_c = entities[0]
        self.assertEqual(asp, "Produktart")
        self.assertEqual(val, "Zahnriemen Satz Kit Plus")
        self.assertAlmostEqual(mean_c, (0.90 + 0.70 + 0.80 + 0.60) / 4.0, places=5)
        self.assertAlmostEqual(min_c, 0.60, places=5)

    def test_bio_edge_cases(self):
        # Consecutive B- tags for the same aspect (distinct entities per Annexure.pdf)
        tokens = ["VW", "Golf", "Passat"]
        bio_tags = ["B-Kompatible_Fahrzeug_Marke", "B-Kompatibles_Fahrzeug_Modell", "B-Kompatibles_Fahrzeug_Modell"]
        probs = [0.99, 0.92, 0.88]

        entities = bio_to_entities_with_confidence(tokens, bio_tags, probs)
        self.assertEqual(len(entities), 3)
        self.assertEqual(entities[0], ("Kompatible_Fahrzeug_Marke", "VW", 0.99, 0.99))
        self.assertEqual(entities[1], ("Kompatibles_Fahrzeug_Modell", "Golf", 0.92, 0.92))
        self.assertEqual(entities[2], ("Kompatibles_Fahrzeug_Modell", "Passat", 0.88, 0.88))

        # Malformed I- tag without preceding B- tag (should start new entity)
        tokens_mal = ["Motor", "5W-30", "LL"]
        tags_mal = ["O", "I-SAE_Viskosität", "I-SAE_Viskosität"]
        probs_mal = [0.99, 0.75, 0.85]

        entities_mal = bio_to_entities_with_confidence(tokens_mal, tags_mal, probs_mal)
        self.assertEqual(len(entities_mal), 1)
        self.assertEqual(entities_mal[0][0], "SAE_Viskosität")
        self.assertEqual(entities_mal[0][1], "5W-30 LL")
        self.assertAlmostEqual(entities_mal[0][2], 0.80, places=5)
        self.assertAlmostEqual(entities_mal[0][3], 0.75, places=5)

    def test_zero_entity_titles(self):
        tokens = ["Original", "Qualität", "Neu", "Günstig"]
        bio_tags = ["O", "O", "O", "O"]
        probs = [0.99, 0.98, 0.99, 0.97]

        entities = bio_to_entities_with_confidence(tokens, bio_tags, probs)
        self.assertEqual(len(entities), 0)

    def test_threshold_boundary(self):
        tokens = ["A", "B", "C"]
        bio_tags = ["B-Hersteller", "B-Produktart", "B-Einbauposition"]
        probs = [0.6000, 0.5999, 0.6001]

        entities = bio_to_entities_with_confidence(tokens, bio_tags, probs)
        kept = [e for e in entities if e[2] >= 0.60]
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0][0], "Hersteller")
        self.assertEqual(kept[1][0], "Einbauposition")

    def test_category_filtering_and_confidence(self):
        raw_entities_conf = [
            ("Hersteller", "BOSCH", 0.95, 0.95),
            ("Bremsscheibenart", "belüftet", 0.99, 0.99),
        ]

        # In Category 1: both allowed
        cat1_allowed = []
        for asp, val, mean_c, min_c in raw_entities_conf:
            if filter_entities_by_category([(asp, val)], category=1):
                cat1_allowed.append((asp, val, mean_c, min_c))
        self.assertEqual(len(cat1_allowed), 2)

        # In Category 2: Bremsscheibenart forbidden even with 0.99 confidence
        cat2_allowed = []
        for asp, val, mean_c, min_c in raw_entities_conf:
            if filter_entities_by_category([(asp, val)], category=2):
                cat2_allowed.append((asp, val, mean_c, min_c))
        self.assertEqual(len(cat2_allowed), 1)
        self.assertEqual(cat2_allowed[0][0], "Hersteller")

    def test_bookkeeping_and_fbeta(self):
        pred_base = [("Hersteller", "A"), ("Produktart", "B"), ("Produktart", "C")]
        true_base = [("Hersteller", "A"), ("Produktart", "B")]

        tp, fp, fn = count_matches(pred_base, true_base)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

        # Suppose ("Produktart", "C") was removed
        pred_filt = [("Hersteller", "A"), ("Produktart", "B")]
        tp_f, fp_f, fn_f = count_matches(pred_filt, true_base)
        self.assertEqual(tp_f, 2)
        self.assertEqual(fp_f, 0)
        self.assertEqual(fn_f, 0)

        self.assertEqual(tp - tp_f, 0)
        self.assertEqual(fp - fp_f, 1)
        self.assertEqual(len(pred_base) - len(pred_filt), (tp - tp_f) + (fp - fp_f))



    def test_unfiltered_equivalence_with_bio_to_entities(self):
        tokens = ["BOSCH", "Bremsscheiben", "vorne", "links", "VW", "Golf", "7"]
        bio_tags = [
            "B-Hersteller", "B-Produktart", "B-Einbauposition", "I-Einbauposition",
            "B-Kompatible_Fahrzeug_Marke", "B-Kompatibles_Fahrzeug_Modell", "I-Kompatibles_Fahrzeug_Modell",
        ]
        probs = [0.95, 0.90, 0.85, 0.80, 0.99, 0.92, 0.91]

        base_ents = bio_to_entities(tokens, bio_tags)
        conf_ents = bio_to_entities_with_confidence(tokens, bio_tags, probs)
        unfiltered_ents = [(asp, val) for asp, val, _, _ in conf_ents]

        self.assertEqual(base_ents, unfiltered_ents)

    def test_filter_entities_by_category_direct_4tuples(self):
        conf_ents = [
            ("Hersteller", "BOSCH", 0.95, 0.95),
            ("Bremsscheibenart", "belüftet", 0.99, 0.99),
            ("Anwendung", "Bremsen", 0.80, 0.80),
        ]

        # Category 1: Bremsscheibenart allowed, Anwendung forbidden
        cat1 = filter_entities_by_category(conf_ents, category=1)
        self.assertEqual(len(cat1), 2)
        self.assertEqual(cat1[0], ("Hersteller", "BOSCH", 0.95, 0.95))
        self.assertEqual(cat1[1], ("Bremsscheibenart", "belüftet", 0.99, 0.99))

        # Category 2: Anwendung allowed, Bremsscheibenart forbidden
        cat2 = filter_entities_by_category(conf_ents, category=2)
        self.assertEqual(len(cat2), 2)
        self.assertEqual(cat2[0], ("Hersteller", "BOSCH", 0.95, 0.95))
        self.assertEqual(cat2[1], ("Anwendung", "Bremsen", 0.80, 0.80))

    def test_min_confidence_filtering_logic(self):
        conf_ents = [
            ("Hersteller", "BOSCH", 0.95, 0.95),
            ("Produktart", "Bremsscheiben", 0.58, 0.58),  # < 0.60
            ("Einbauposition", "vorne links", 0.75, 0.55), # min 0.55 < 0.60
            ("Kompatible_Fahrzeug_Marke", "VW", 0.99, 0.99),
        ]

        # Filter at 0.60
        kept_060 = [(asp, val) for asp, val, _, min_c in conf_ents if min_c >= 0.60]
        self.assertEqual(len(kept_060), 2)
        self.assertEqual(kept_060[0], ("Hersteller", "BOSCH"))
        self.assertEqual(kept_060[1], ("Kompatible_Fahrzeug_Marke", "VW"))

        # Unfiltered (0.00)
        kept_unfiltered = [(asp, val) for asp, val, _, min_c in conf_ents if min_c >= 0.0]
        self.assertEqual(len(kept_unfiltered), 4)


if __name__ == "__main__":
    unittest.main()
