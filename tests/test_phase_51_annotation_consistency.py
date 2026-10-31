"""
Unit tests for Phase 51: Annotation Consistency & Boundary Convention Discovery.
Verifies champion preservation, dataset partition integrity, knowledge base schema,
empirical inconsistency findings, and system invariants.
"""

import hashlib
import json
from pathlib import Path
import shutil
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "analysis" / "phase_51_boundary_patterns.json"


class TestPhase51AnnotationConsistency(unittest.TestCase):
    """Test suite for Phase 51 discovery deliverables and invariants."""

    def test_champion_integrity(self):
        """Phase 40 champion submission must be completely untouched with exact SHA-256."""
        self.assertTrue(CHAMPION_PATH.exists(), f"Champion submission not found at {CHAMPION_PATH}")
        sha256 = hashlib.sha256(CHAMPION_PATH.read_bytes()).hexdigest().upper()
        self.assertEqual(
            sha256,
            EXPECTED_CHAMPION_SHA256,
            f"Champion SHA-256 mismatch! Got {sha256}, expected {EXPECTED_CHAMPION_SHA256}",
        )

    def test_disk_space_invariant(self):
        """Free disk space must remain >= 5.0 GB."""
        _, _, free_bytes = shutil.disk_usage(PROJECT_ROOT)
        free_gb = free_bytes / (1024 ** 3)
        self.assertGreaterEqual(
            free_gb,
            5.0,
            f"Disk space invariant violated: only {free_gb:.2f} GB available (< 5.0 GB limit).",
        )

    def test_data_partition_integrity_and_isolation(self):
        """Primary discovery dataset must be exactly 4,000 titles and disjoint from 1,000 val."""
        from src.data.build_examples import build_examples, load_training_data
        from src.data.make_splits import split_examples

        df = load_training_data()
        examples = build_examples(df)
        train_4k, val_1k = split_examples(examples, random_state=42)

        self.assertEqual(len(train_4k), 4000, "Train partition must contain exactly 4,000 titles")
        self.assertEqual(len(val_1k), 1000, "Val partition must contain exactly 1,000 titles")

        train_recs = set(e["record_number"] for e in train_4k)
        val_recs = set(e["record_number"] for e in val_1k)
        self.assertEqual(len(train_recs.intersection(val_recs)), 0, "Train and Val must be completely disjoint")

    def test_knowledge_base_schema(self):
        """Knowledge base JSON must exist and satisfy all schema requirements."""
        self.assertTrue(KNOWLEDGE_BASE_PATH.exists(), f"Knowledge base missing at {KNOWLEDGE_BASE_PATH}")
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)

        required_keys = [
            "phase",
            "timestamp",
            "data_provenance",
            "gold_entity_inventory",
            "pattern_taxonomy",
            "pattern_statistics",
            "surface_model_statistics",
            "oof_error_reconciliation",
            "out_of_sample_verification",
            "decision_verdict",
        ]
        for key in required_keys:
            self.assertIn(key, kb, f"Missing key in knowledge base: {key}")

        self.assertEqual(kb["phase"], 51)
        self.assertIn("Kompatibles_Fahrzeug_Modell", kb["gold_entity_inventory"]["aspect_counts"])

    def test_decision_verdict_validity(self):
        """Decision verdict must be one of three allowed options and match empirical findings."""
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)

        allowed_verdicts = [
            "CONSISTENT PATTERNS FOUND — CONTROLLED LEARNING EXPERIMENT JUSTIFIED",
            "ANNOTATIONS TOO AMBIGUOUS — RETAIN PHASE 40",
            "INCONCLUSIVE — ADDITIONAL EVIDENCE REQUIRED",
        ]
        verdict = kb["decision_verdict"]["verdict"]
        self.assertIn(verdict, allowed_verdicts, f"Invalid verdict: {verdict}")
        self.assertEqual(verdict, "ANNOTATIONS TOO AMBIGUOUS — RETAIN PHASE 40")

        inconsistent_fraction = kb["decision_verdict"]["inconsistent_fraction"]
        self.assertGreater(inconsistent_fraction, 0.80, f"Expected >80% inconsistent errors, got {inconsistent_fraction}")

    def test_taxonomy_and_inconsistent_patterns(self):
        """Key automotive patterns must be identified as truly inconsistent."""
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)

        patterns = kb["pattern_statistics"]
        self.assertGreaterEqual(len(patterns), 15, "Taxonomy must define at least 15 patterns")

        inconsistent_pats = ["chassis_audi_vag", "chassis_bmw_e", "chassis_merc_w_s_c_x", "gen_arabic_num", "chassis_parens"]
        for p in inconsistent_pats:
            self.assertIn(p, patterns, f"Pattern {p} not found in statistics")
            self.assertEqual(
                patterns[p]["classification"],
                "TRULY_INCONSISTENT",
                f"Pattern {p} should be TRULY_INCONSISTENT, got {patterns[p]['classification']}",
            )

    def test_out_of_sample_correlation(self):
        """Out-of-sample verification must show strong correlation (r >= 0.90) between train and val."""
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)

        oos = kb["out_of_sample_verification"]
        self.assertGreaterEqual(
            oos["pearson_correlation"],
            0.90,
            f"Expected train/val correlation >= 0.90, got {oos['pearson_correlation']}",
        )
        self.assertLessEqual(
            oos["mean_absolute_difference"],
            0.10,
            f"Expected mean absolute difference <= 0.10, got {oos['mean_absolute_difference']}",
        )


if __name__ == "__main__":
    unittest.main()
