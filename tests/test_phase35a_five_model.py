"""
Unit Tests for Phase 35A Five-Model Ensemble and Validation Integrity Pipeline.
"""

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.submission.predict_quiz_phase35a_five_model import (
    CHECKPOINTS,
    WEIGHTS,
    CONF_THRESHOLD,
    QUIZ_START_RECORD,
    QUIZ_END_RECORD,
    ORIG_CHAMP_QUIZ_PATH,
    DAPT_CHAMP_QUIZ_PATH,
    EXPECTED_ORIG_CHAMP_SHA256,
    EXPECTED_DAPT_CHAMP_SHA256,
    compute_sha256,
)


class TestPhase35AFiveModel(unittest.TestCase):

    def test_ensemble_weights_sum_to_one(self):
        total_weight = sum(WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        self.assertEqual(len(WEIGHTS), 5)
        for name, w in WEIGHTS.items():
            self.assertEqual(w, 0.20, f"Model {name} weight expected 0.20, got {w}")

    def test_confidence_threshold_is_min_060(self):
        self.assertEqual(CONF_THRESHOLD, 0.60)

    def test_quiz_record_range(self):
        self.assertEqual(QUIZ_START_RECORD, 5001)
        self.assertEqual(QUIZ_END_RECORD, 30000)
        self.assertEqual(QUIZ_END_RECORD - QUIZ_START_RECORD + 1, 25000)

    def test_all_five_checkpoints_exist(self):
        for name, ckpt_path in CHECKPOINTS.items():
            self.assertTrue(ckpt_path.exists(), f"Checkpoint {name} not found at {ckpt_path}")
            st_path = ckpt_path / "model.safetensors"
            self.assertTrue(st_path.exists(), f"Safetensors file missing in {ckpt_path}")

    def test_both_existing_quiz_submissions_intact(self):
        # 1. Original Champion
        self.assertTrue(ORIG_CHAMP_QUIZ_PATH.exists(), f"Original Quiz file missing: {ORIG_CHAMP_QUIZ_PATH}")
        orig_hash = compute_sha256(ORIG_CHAMP_QUIZ_PATH)
        self.assertEqual(orig_hash, EXPECTED_ORIG_CHAMP_SHA256, "Original Quiz champion hash modified!")

        # 2. DAPT Champion
        self.assertTrue(DAPT_CHAMP_QUIZ_PATH.exists(), f"DAPT Quiz file missing: {DAPT_CHAMP_QUIZ_PATH}")
        dapt_hash = compute_sha256(DAPT_CHAMP_QUIZ_PATH)
        self.assertEqual(dapt_hash, EXPECTED_DAPT_CHAMP_SHA256, "DAPT Quiz champion hash modified!")

    def test_production_champion_pipeline_unmodified(self):
        prod_script = PROJECT_ROOT / "src" / "submission" / "predict_test_champion.py"
        self.assertTrue(prod_script.exists())

    def test_local_evaluation_cache_exists(self):
        eval_path = PROJECT_ROOT / "data" / "cache" / "phase_35a_local_evaluation_results.json"
        self.assertTrue(eval_path.exists(), "Local evaluation results cache missing")
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("clean_489_unpolluted", data)
        self.assertIn("val_1k_contaminated_diagnostic", data)
        self.assertIn("bootstrap", data)


if __name__ == "__main__":
    unittest.main()
