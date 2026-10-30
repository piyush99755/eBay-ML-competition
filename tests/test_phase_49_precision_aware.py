"""
Unit tests for Phase 49: Precision-Aware Fine-Tuning & Methodology Audit.

Validates:
1. Phase 40 official champion file integrity and SHA-256.
2. Phase 49 audit results (Part A sampling flaw metrics and Part B FP taxonomy).
3. Leak-free internal train / dev partition isolation from validation & clean holdouts.
4. Trained checkpoint integrity and file completeness.
5. Generalization, complementarity, and bootstrap evaluation persistence.
6. Official decision verdict consistency.
7. Workspace free disk space limit (>= 5.0 GB).
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.data.make_internal_split import get_internal_split, audit_internal_split

CHAMPION_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"

AUDIT_RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_49_audit_results.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_49_evaluation_results.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "gbert_phase49_precision_aware_best"

INTERNAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "cache" / "internal_split_3200_800.json"
CLEAN_HOLDOUT_MANIFEST_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35b_clean_holdout_manifest.json"


class TestPhase49PrecisionAware(unittest.TestCase):
    """Test suite verifying all Phase 49 components and safety invariants."""

    def test_01_champion_integrity(self):
        """Verify Phase 40 champion file exists and matches exact SHA-256."""
        self.assertTrue(CHAMPION_PATH.exists(), f"Champion missing: {CHAMPION_PATH}")
        content = CHAMPION_PATH.read_bytes()
        actual_sha256 = hashlib.sha256(content).hexdigest().upper()
        self.assertEqual(
            actual_sha256,
            EXPECTED_CHAMPION_SHA256,
            f"Champion SHA-256 mismatch! Expected {EXPECTED_CHAMPION_SHA256}, got {actual_sha256}",
        )

    def test_02_audit_results_integrity(self):
        """Verify Phase 48 audit findings and Part B FP root cause analysis."""
        self.assertTrue(AUDIT_RESULTS_PATH.exists(), f"Audit results missing: {AUDIT_RESULTS_PATH}")
        with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
            audit = json.load(f)

        self.assertEqual(audit["phase"], "49")

        # Part A sampling flaw assertions
        part_a = audit["part_a_methodology_audit"]
        sampling = part_a["sampling_audit"]
        self.assertAlmostEqual(sampling["mean_exposures_per_standard_title"], 3.922, places=2)
        self.assertAlmostEqual(sampling["mean_exposures_per_hard_title"], 1.938, places=2)
        self.assertAlmostEqual(sampling["exposure_inversion_ratio"], 2.024, places=2)
        self.assertEqual(sampling["total_never_sampled_count"], 122)

        # Part B false-positive anatomy assertions
        part_b = audit["part_b_fp_root_cause_analysis"]
        counts = part_b["fp_class_counts"]
        total_fps = part_b["total_fps"]
        self.assertEqual(total_fps, 2134)
        self.assertEqual(counts["B_wrong_boundaries"], 1378)
        self.assertEqual(counts["C_wrong_aspect_label"], 444)
        self.assertEqual(counts["A_spurious_prediction"], 291)
        self.assertEqual(counts["D_omitted_valid_entity"], 21)

        # Wrong boundaries account for > 60%
        b_pct = (counts["B_wrong_boundaries"] / total_fps) * 100
        self.assertGreater(b_pct, 60.0)

        # Class B vehicle model dominance
        b_asps = part_b["fp_by_aspect"]["Kompatibles_Fahrzeug_Modell"]["B"]
        self.assertEqual(b_asps, 957)

    def test_03_leak_free_split_isolation(self):
        """Verify internal train / dev partitions are completely isolated from validation & clean holdouts."""
        self.assertTrue(CLEAN_HOLDOUT_MANIFEST_PATH.exists(), "Clean holdout manifest missing")
        with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            clean_recs = set(json.load(f)["record_ids"])

        df = load_training_data()
        examples = build_examples(df)
        train_examples, val_examples = split_examples(examples)

        self.assertEqual(len(train_examples), 4000)
        self.assertEqual(len(val_examples), 1000)

        # Deterministic 3200/800 split
        train_3200, dev_800, manifest = get_internal_split(train_examples, random_state=42)
        self.assertEqual(len(train_3200), 3200)
        self.assertEqual(len(dev_800), 800)

        # Verify audit pass without exception
        audit_res = audit_internal_split(train_3200, dev_800, val_examples)
        self.assertEqual(audit_res["internal_overlap"], 0)
        self.assertEqual(audit_res["holdout_overlap"], 0)
        self.assertTrue(audit_res["is_leakage_free"])

        # Strict isolation from 489 clean holdout
        train_rec_ids = {ex["record_number"] for ex in train_3200}
        dev_rec_ids = {ex["record_number"] for ex in dev_800}
        self.assertEqual(len(train_rec_ids & clean_recs), 0, "Train overlaps with clean holdout!")
        self.assertEqual(len(dev_rec_ids & clean_recs), 0, "Dev overlaps with clean holdout!")

    def test_04_checkpoint_artifacts(self):
        """Verify the trained Phase 49 model checkpoint exists and has all required files."""
        self.assertTrue(CHECKPOINT_PATH.exists(), f"Checkpoint folder missing: {CHECKPOINT_PATH}")
        config_path = CHECKPOINT_PATH / "config.json"
        model_path = CHECKPOINT_PATH / "model.safetensors"
        tok_path = CHECKPOINT_PATH / "tokenizer.json"

        self.assertTrue(config_path.exists(), "config.json missing")
        self.assertTrue(model_path.exists(), "model.safetensors missing")
        self.assertTrue(tok_path.exists(), "tokenizer.json missing")

        # Verify weights size is ~400MB
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        self.assertGreater(model_size_mb, 350.0)
        self.assertLess(model_size_mb, 500.0)

    def test_05_evaluation_metrics_and_verdict(self):
        """Verify primary dev evaluation, secondary clean diagnostic, bootstrap, and decision verdict."""
        self.assertTrue(EVAL_RESULTS_PATH.exists(), f"Evaluation results missing: {EVAL_RESULTS_PATH}")
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        # Internal dev evaluation
        dev_eval = res["internal_dev_evaluation"]
        matched_base = dev_eval["matched_baseline"]
        p49_dev = dev_eval["phase49_standalone"]
        self.assertEqual(dev_eval["title_count"], 800)
        self.assertGreater(matched_base["official_score"], 0.940)
        self.assertLess(p49_dev["delta_overall"], -0.030)  # Precision penalty degraded recall

        # Dev bootstrap
        dev_boot = dev_eval["bootstrap"]
        self.assertEqual(dev_boot["n_resamples"], 1000)
        self.assertLess(dev_boot["ci_upper"], 0.0)
        self.assertFalse(dev_boot["is_significant"])

        # Clean holdout secondary diagnostic
        clean_diag = res["clean_holdout_secondary_diagnostic"]
        self.assertEqual(clean_diag["title_count"], 489)
        p40_base = clean_diag["phase40_champion"]
        self.assertAlmostEqual(p40_base["official_score"], 0.927771, places=5)

        # Predeclared ensemble
        ens_clean = clean_diag["ensemble_90_10"]
        self.assertAlmostEqual(ens_clean["official_score"], 0.927707, places=5)
        clean_boot = clean_diag["bootstrap"]
        self.assertFalse(clean_boot["is_significant"])
        self.assertLess(clean_boot["ci_lower"], 0.0)
        self.assertGreater(clean_boot["ci_upper"], 0.0)

        # Official decision verdict
        self.assertEqual(res["verdict"], "RETAIN PHASE 40 — NO RELIABLE IMPROVEMENT")

    def test_06_disk_space_floor(self):
        """Verify disk space remains above 5.0 GB limit."""
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
        self.assertGreaterEqual(free_gb, 5.0, f"Free disk space fell below 5.0 GB: {free_gb:.2f} GB")


if __name__ == "__main__":
    unittest.main()
