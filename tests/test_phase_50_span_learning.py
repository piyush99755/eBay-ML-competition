"""
Unit tests for Phase 50: Boundary-Aware Span Learning.

Validates:
1. Phase 40 official champion file integrity and SHA-256.
2. Official metric algebraic equivalence and Phase 49 error taxonomy verification.
3. Span-width coverage (>= 99.5% at W_max=8) and 100% subword boundary alignment.
4. Strict leak-free partition isolation (3,200 train / 800 dev).
5. Trained SpanNERModel checkpoint structure and evaluation results persistence.
6. Workspace free disk space limit (>= 5.0 GB).
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

CHAMPION_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"

FEASIBILITY_RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_50_span_feasibility.json"
AUDIT_RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_49_audit_results.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_50_evaluation_results.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "gbert_phase50_span_best"
CLEAN_HOLDOUT_MANIFEST_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35b_clean_holdout_manifest.json"


class TestPhase50SpanLearning(unittest.TestCase):
    """Test suite verifying all Phase 50 span components and safety invariants."""

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

    def test_02_metric_equivalence_and_error_taxonomy(self):
        """Verify algebraic equivalence of F0.2 formula and verified error taxonomy."""
        self.assertTrue(FEASIBILITY_RESULTS_PATH.exists(), "Feasibility results missing")
        with open(FEASIBILITY_RESULTS_PATH, "r", encoding="utf-8") as f:
            feas = json.load(f)

        self.assertTrue(feas["metric_audit"]["is_equivalent"])
        for case in feas["metric_audit"]["test_cases"]:
            self.assertTrue(case["matches"], f"Mismatch for case: {case}")

        # Check Phase 49 error taxonomy numbers
        self.assertTrue(AUDIT_RESULTS_PATH.exists(), "Audit results missing")
        with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
            audit = json.load(f)
        part_b = audit["part_b_fp_root_cause_analysis"]
        self.assertEqual(part_b["total_fps"], 2134)
        self.assertEqual(part_b["fp_class_counts"]["B_wrong_boundaries"], 1378)
        self.assertEqual(part_b["fp_class_counts"]["C_wrong_aspect_label"], 444)
        self.assertEqual(part_b["fp_class_counts"]["A_spurious_prediction"], 291)
        self.assertEqual(part_b["fp_class_counts"]["D_omitted_valid_entity"], 21)
        self.assertEqual(part_b["fp_by_aspect"]["Kompatibles_Fahrzeug_Modell"]["B"], 957)

    def test_03_span_width_coverage_and_alignment(self):
        """Verify span width coverage >= 99.5% at W_max=8 and 100% subword boundary alignment."""
        with open(FEASIBILITY_RESULTS_PATH, "r", encoding="utf-8") as f:
            feas = json.load(f)

        span_dist = feas["span_distribution"]
        cov_8 = span_dist["width_coverage"]["8"]
        self.assertGreater(cov_8["coverage_pct"], 99.5)
        self.assertEqual(span_dist["overlapping_spans_count"], 0)
        self.assertEqual(span_dist["nested_spans_count"], 0)

        subword_align = feas["subword_alignment"]
        self.assertEqual(subword_align["unaligned_spans"], 0)
        self.assertEqual(subword_align["subword_mapping_success_rate"], 100.0)

    def test_04_split_isolation(self):
        """Verify internal train / dev partition strict isolation."""
        from src.data.build_examples import build_examples, load_training_data
        from src.data.make_splits import split_examples
        from src.data.make_internal_split import get_internal_split, audit_internal_split

        with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            clean_recs = set(json.load(f)["record_ids"])

        df = load_training_data()
        examples = build_examples(df)
        train_examples, val_examples = split_examples(examples, random_state=42)

        train_3200, dev_800, _ = get_internal_split(train_examples, random_state=42)
        self.assertEqual(len(train_3200), 3200)
        self.assertEqual(len(dev_800), 800)

        audit_res = audit_internal_split(train_3200, dev_800, val_examples)
        self.assertEqual(audit_res["internal_overlap"], 0)
        self.assertEqual(audit_res["holdout_overlap"], 0)
        self.assertTrue(audit_res["is_leakage_free"])

        train_ids = {ex["record_number"] for ex in train_3200}
        dev_ids = {ex["record_number"] for ex in dev_800}
        self.assertEqual(len(train_ids & clean_recs), 0)
        self.assertEqual(len(dev_ids & clean_recs), 0)

    def test_05_span_checkpoint_and_eval_persistence(self):
        """Verify SpanNERModel checkpoint structure and evaluation results."""
        self.assertTrue(CHECKPOINT_PATH.exists(), f"Span checkpoint folder missing: {CHECKPOINT_PATH}")
        self.assertTrue((CHECKPOINT_PATH / "span_model_checkpoint.pt").exists(), "span_model_checkpoint.pt missing")
        self.assertTrue((CHECKPOINT_PATH / "config.json").exists(), "config.json missing")
        self.assertTrue((CHECKPOINT_PATH / "tokenizer.json").exists(), "tokenizer.json missing")
        self.assertTrue((CHECKPOINT_PATH / "training_meta.json").exists(), "training_meta.json missing")

        # Verify evaluation results
        if EVAL_RESULTS_PATH.exists():
            with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
                res = json.load(f)
            self.assertEqual(res["phase"], "50")
            self.assertIn(
                res["verdict"],
                [
                    "PROMISING SPAN MODEL — FURTHER VALIDATION JUSTIFIED",
                    "RETAIN PHASE 40 — SPAN MODEL NOT COMPETITIVE",
                    "STOP — INSUFFICIENT DATA, VALIDATION, OR RESOURCES",
                ],
            )

    def test_06_disk_space_floor(self):
        """Verify disk space remains above 5.0 GB limit."""
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
        self.assertGreaterEqual(free_gb, 5.0, f"Free disk space fell below 5.0 GB: {free_gb:.2f} GB")


if __name__ == "__main__":
    unittest.main()
