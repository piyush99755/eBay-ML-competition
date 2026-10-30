"""
Unit tests for Phase 48 Hard Example Mining and Precision-Aware Fine-Tuning.

Verifies:
  1. Phase 40 official champion file integrity and SHA-256 preservation.
  2. Training-only hard example mining manifest properties and zero-leakage disjointness.
  3. Trained challenger checkpoint existence and parameter footprint.
  4. 3-epoch training history consistency and best epoch tracking.
  5. Scientific evaluation reproduction, statistical bootstrap properties, and verdict.
  6. Disk space safety floor compliance (>= 5.0 GB).
"""

import json
from pathlib import Path
import shutil
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = REPO_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"

HARD_MANIFEST_PATH = REPO_ROOT / "data" / "cache" / "phase_48_hard_examples_manifest.json"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "gbert_phase48_hard_mined_best"
TRAINING_METRICS_PATH = REPO_ROOT / "data" / "cache" / "phase48_training_history.json"
EVALUATION_RESULTS_PATH = REPO_ROOT / "data" / "cache" / "phase_48_evaluation_results.json"
CLEAN_HOLDOUT_MANIFEST_PATH = REPO_ROOT / "data" / "cache" / "phase_35b_clean_holdout_manifest.json"

from src.evaluation.phase_43_part_number_calibration import compute_sha256


class TestPhase48HardExampleMining(unittest.TestCase):
    def test_01_champion_baseline_integrity(self):
        """Phase 40 champion must exist and retain its exact verified SHA-256."""
        self.assertTrue(CHAMPION_PATH.exists(), f"Champion missing at {CHAMPION_PATH}")
        h = compute_sha256(CHAMPION_PATH)
        self.assertEqual(
            h,
            EXPECTED_CHAMPION_SHA256,
            f"Phase 40 Champion hash altered! Expected {EXPECTED_CHAMPION_SHA256}, got {h}",
        )

    def test_02_hard_examples_manifest_provenance(self):
        """Hard examples manifest must exist, contain 1,858 titles, and be disjoint from holdouts."""
        self.assertTrue(HARD_MANIFEST_PATH.exists(), f"Manifest missing at {HARD_MANIFEST_PATH}")
        with open(HARD_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["phase"], "48")
        self.assertEqual(manifest["total_train_titles"], 4000)
        self.assertEqual(manifest["hard_titles_count"], 1858)
        self.assertEqual(len(manifest["hard_record_ids"]), 1858)
        self.assertEqual(manifest["category_distribution"]["1"], 952)
        self.assertEqual(manifest["category_distribution"]["2"], 906)

        # Verify zero leakage into clean holdout
        with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            clean_holdout_ids = set(json.load(f)["record_ids"])

        hard_ids = set(manifest["hard_record_ids"])
        leakage = hard_ids.intersection(clean_holdout_ids)
        self.assertEqual(len(leakage), 0, f"Data leakage detected! {len(leakage)} IDs overlap with holdout.")

    def test_03_trained_checkpoint_structure(self):
        """Trained Phase 48 GBERT checkpoint must exist with valid weights."""
        self.assertTrue(CHECKPOINT_DIR.exists(), f"Checkpoint directory missing at {CHECKPOINT_DIR}")
        st_file = CHECKPOINT_DIR / "model.safetensors"
        self.assertTrue(st_file.exists(), f"Missing model.safetensors in {CHECKPOINT_DIR}")
        size_mb = st_file.stat().st_size / (1024 ** 2)
        self.assertGreater(size_mb, 400.0)
        self.assertLess(size_mb, 450.0)

        cfg_file = CHECKPOINT_DIR / "config.json"
        self.assertTrue(cfg_file.exists(), f"Missing config.json in {CHECKPOINT_DIR}")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(len(cfg["id2label"]), 48)

    def test_04_training_metrics_consistency(self):
        """Training metrics must reflect 3 completed epochs and best epoch 2."""
        self.assertTrue(TRAINING_METRICS_PATH.exists(), f"Training metrics missing at {TRAINING_METRICS_PATH}")
        with open(TRAINING_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertEqual(metrics["phase"], "48")
        self.assertEqual(len(metrics["training_history"]), 3)
        self.assertEqual(metrics["best_epoch"], 2)
        self.assertGreater(metrics["best_val_f02"], 0.90)
        self.assertLess(metrics["total_training_time_minutes"], 240.0)  # Under 4 hours budget

    def test_05_evaluation_results_and_verdict(self):
        """Evaluation results must reproduce baseline, demonstrate non-significance, and conclude RETAIN PHASE 40."""
        self.assertTrue(EVALUATION_RESULTS_PATH.exists(), f"Evaluation results missing at {EVALUATION_RESULTS_PATH}")
        with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        # Baseline reproduction
        base = eval_data["baseline_clean_holdout"]
        self.assertAlmostEqual(base["official_score"], 0.927771, places=5)
        self.assertEqual(base["tp"], 3021)
        self.assertEqual(base["fp"], 165)
        self.assertEqual(base["fn"], 752)

        # Ensemble results
        ens = eval_data["ensemble_clean_holdout"]
        self.assertAlmostEqual(ens["delta_overall"], 0.000015, places=5)
        self.assertLess(ens["delta_cat1"], 0.0)  # Active Category 1 regression

        # Bootstrap non-significance
        boot = eval_data["bootstrap"]
        self.assertFalse(boot["is_significant"])
        self.assertLess(boot["ci_lower"], 0.0)
        self.assertGreater(boot["ci_upper"], 0.0)
        self.assertGreater(boot["p_value"], 0.50)

        # Verdict
        self.assertEqual(eval_data["verdict"], "RETAIN PHASE 40 — NO IMPROVEMENT")

    def test_06_disk_space_floor(self):
        """Workspace must maintain at least 5.0 GB free disk space."""
        _, _, free_bytes = shutil.disk_usage(REPO_ROOT)
        free_gb = free_bytes / (1024 ** 3)
        self.assertGreaterEqual(
            free_gb,
            5.0,
            f"Free disk space {free_gb:.2f} GB is below the 5.0 GB safety threshold!",
        )


if __name__ == "__main__":
    unittest.main()
