"""
Unit tests for Phase 31A Boundary-Aware Decoding:
- Transition mask legality
- Viterbi dynamic programming constraint enforcement
- Candidate span generation & collision avoidance
- Joint likelihood mathematical invariance
"""

import unittest
import numpy as np

from src.evaluation.experiment_31a_boundary_decoding import (
    build_bio_transition_mask,
    viterbi_decode_sequence,
    refine_vehicle_model_boundaries,
)


class TestBoundaryDecoding(unittest.TestCase):

    def setUp(self):
        # 5 labels: 0: B-Modell, 1: I-Modell, 2: B-Marke, 3: I-Marke, 4: O
        self.id2label = {
            0: "B-Kompatibles_Fahrzeug_Modell",
            1: "I-Kompatibles_Fahrzeug_Modell",
            2: "B-Kompatible_Fahrzeug_Marke",
            3: "I-Kompatible_Fahrzeug_Marke",
            4: "O",
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.legal_trans = build_bio_transition_mask(self.id2label)

    def test_transition_mask_legality(self):
        # O (4) -> I-Modell (1) must be False
        self.assertFalse(self.legal_trans[4, 1])
        # O (4) -> B-Modell (0) must be True
        self.assertTrue(self.legal_trans[4, 0])
        # B-Modell (0) -> I-Modell (1) must be True
        self.assertTrue(self.legal_trans[0, 1])
        # B-Marke (2) -> I-Modell (1) must be False
        self.assertFalse(self.legal_trans[2, 1])
        # I-Modell (1) -> I-Marke (3) must be False
        self.assertFalse(self.legal_trans[1, 3])
        # I-Modell (1) -> O (4) must be True
        self.assertTrue(self.legal_trans[1, 4])

    def test_viterbi_prevents_illegal_start_with_i_tag(self):
        # 1-token sequence where I-Modell has highest probability
        # Token 0: probs = [0.1, 0.7, 0.05, 0.05, 0.1]
        probs = np.array([[0.1, 0.7, 0.05, 0.05, 0.1]], dtype=np.float32)
        best_tags, selected_probs = viterbi_decode_sequence(probs, self.id2label, self.legal_trans)
        # Cannot start with I-Modell (1)! Must pick second highest legal tag (B-Modell or O)
        self.assertNotEqual(best_tags[0], "I-Kompatibles_Fahrzeug_Modell")
        self.assertIn(best_tags[0], ["B-Kompatibles_Fahrzeug_Modell", "O"])

    def test_viterbi_enforces_continuation_consistency(self):
        # 2 tokens:
        # Token 0: argmax is O (4)
        # Token 1: argmax is I-Modell (1)
        # Raw argmax would produce [O, I-Modell], which is an illegal transition!
        probs = np.array([
            [0.1, 0.0, 0.0, 0.0, 0.9],   # O is 0.9
            [0.35, 0.40, 0.0, 0.0, 0.25] # I-Modell is 0.40, B-Modell is 0.35, O is 0.25
        ], dtype=np.float32)

        best_tags, selected_probs = viterbi_decode_sequence(probs, self.id2label, self.legal_trans)
        # Viterbi must either pick [B-Modell, I-Modell] or [O, B-Modell] or [O, O], but never [O, I-Modell]
        self.assertFalse(best_tags[0] == "O" and best_tags[1] == "I-Kompatibles_Fahrzeug_Modell")

    def test_candidate_generation_does_not_cross_neighbors(self):
        tokens = ["Bremsscheibe", "VW", "Golf", "7", "Vorne"]
        # VW is Marke at [1, 1], Golf is Modell at [2, 2], Vorne is Einbauposition at [4, 4]
        pred_entities = [
            ("Kompatible_Fahrzeug_Marke", "VW", 1, 1, 0.95, 0.95),
            ("Kompatibles_Fahrzeug_Modell", "Golf", 2, 2, 0.90, 0.90),
            ("Einbauposition", "Vorne", 4, 4, 0.85, 0.85),
        ]
        # Fake uniform probabilities
        probs = np.full((len(tokens), len(self.id2label)), 0.2, dtype=np.float32)

        refined, num_changed = refine_vehicle_model_boundaries(
            tokens=tokens,
            pred_entities=pred_entities,
            probs=probs,
            label2id=self.label2id,
            scoring_mode="window_joint",
            margin=0.5,
        )
        # Golf at [2, 2] cannot extend left to [1, 2] because token 1 is occupied by VW!
        for ent in refined:
            if ent[0] == "Kompatibles_Fahrzeug_Modell":
                self.assertGreaterEqual(ent[2], 2)  # s must be >= 2

    def test_joint_window_likelihood_argmax_invariance(self):
        # If token probabilities strictly peak at the original span,
        # joint window likelihood difference delta <= 0, resulting in 0 changes with margin >= 0.
        tokens = ["Golf", "Plus"]
        probs = np.array([
            [0.90, 0.01, 0.01, 0.01, 0.07], # B-Modell is peak
            [0.01, 0.85, 0.01, 0.01, 0.12], # I-Modell is peak
        ], dtype=np.float32)

        pred_entities = [("Kompatibles_Fahrzeug_Modell", "Golf Plus", 0, 1, 0.875, 0.85)]
        refined, num_changed = refine_vehicle_model_boundaries(
            tokens=tokens,
            pred_entities=pred_entities,
            probs=probs,
            label2id=self.label2id,
            scoring_mode="window_joint",
            margin=0.0,
        )
        self.assertEqual(num_changed, 0)
        self.assertEqual(refined[0][1], "Golf Plus")


if __name__ == "__main__":
    unittest.main()
