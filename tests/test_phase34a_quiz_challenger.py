"""
Unit Tests for Phase 34A Official EvalAI Quiz Challenger Pipeline.
"""

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.submission.predict_quiz_phase34a_dapt import (
    CHECKPOINTS,
    WEIGHTS,
    CONF_THRESHOLD,
    QUIZ_START_RECORD,
    QUIZ_END_RECORD,
    EXPECTED_CHAMPION_SHA256,
    CHAMPION_QUIZ_PATH,
    compute_sha256,
)


class TestPhase34AQuizChallenger(unittest.TestCase):

    def test_ensemble_weights_sum_to_one(self):
        total_weight = sum(WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        self.assertEqual(WEIGHTS["dapt_gbert"], 0.40)
        self.assertEqual(WEIGHTS["xlmr_seed42"], 0.20)
        self.assertEqual(WEIGHTS["xlmr_seed7"], 0.20)
        self.assertEqual(WEIGHTS["xlmr_seed21"], 0.20)

    def test_confidence_threshold_is_min_060(self):
        self.assertEqual(CONF_THRESHOLD, 0.60)

    def test_quiz_record_range(self):
        self.assertEqual(QUIZ_START_RECORD, 5001)
        self.assertEqual(QUIZ_END_RECORD, 30000)
        self.assertEqual(QUIZ_END_RECORD - QUIZ_START_RECORD + 1, 25000)

    def test_all_checkpoints_exist(self):
        for name, ckpt_path in CHECKPOINTS.items():
            self.assertTrue(ckpt_path.exists(), f"Checkpoint {name} not found at {ckpt_path}")
            st_path = ckpt_path / "model.safetensors"
            self.assertTrue(st_path.exists(), f"Safetensors file missing in {ckpt_path}")

    def test_champion_submission_integrity(self):
        self.assertTrue(CHAMPION_QUIZ_PATH.exists(), f"Champion Quiz file not found: {CHAMPION_QUIZ_PATH}")
        champ_hash = compute_sha256(CHAMPION_QUIZ_PATH)
        self.assertEqual(champ_hash, EXPECTED_CHAMPION_SHA256, "Champion Quiz file hash has been modified!")

    def test_production_pipeline_unmodified(self):
        prod_script = PROJECT_ROOT / "src" / "submission" / "predict_test_champion.py"
        self.assertTrue(prod_script.exists())


if __name__ == "__main__":
    unittest.main()
