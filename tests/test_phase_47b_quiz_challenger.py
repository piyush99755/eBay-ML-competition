"""
Unit tests for Phase 47B Quiz Challenger Generation and Validation.

Verifies:
  1. Champion baseline integrity (Phase 40 SHA-256).
  2. Challenger file presence and cryptographic hash reproducibility.
  3. Submission format compliance using competition validator.
  4. 100% bit-for-bit invariance of Category 1 predictions against Phase 40 champion.
  5. Category 2 delta sanity, allowed aspects compliance, and change statistics.
  6. Audit cache persistence and consistency.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_submission import (
    OFFICIAL_ASPECTS_CAT1,
    OFFICIAL_ASPECTS_CAT2,
    validate_submission_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = REPO_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"

CHALLENGER_PATH = REPO_ROOT / "submissions" / "submission_quiz_phase47_category2_specialist010.tsv"
EXPECTED_CHALLENGER_SHA256 = "58471EAC63A31D542EA578BD89012E796116EF1AACE7A615313994ECECF96794"

STATS_CACHE_PATH = REPO_ROOT / "data" / "cache" / "phase_47b_quiz_challenger_stats.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def load_tsv_by_category(path: Path):
    cat1_rows = []
    cat2_rows = []
    cat1_by_rec = defaultdict(list)
    cat2_by_rec = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for parts in reader:
            if len(parts) == 4:
                rec_id = int(parts[0])
                cat_id = int(parts[1])
                asp = parts[2]
                val = parts[3]
                if cat_id == 1:
                    cat1_rows.append((rec_id, cat_id, asp, val))
                    cat1_by_rec[rec_id].append((asp, val))
                elif cat_id == 2:
                    cat2_rows.append((rec_id, cat_id, asp, val))
                    cat2_by_rec[rec_id].append((asp, val))

    return header, cat1_rows, cat2_rows, cat1_by_rec, cat2_by_rec


class TestPhase47BQuizChallenger(unittest.TestCase):
    def test_01_champion_baseline_integrity(self):
        """Phase 40 champion must exist and retain its exact verified SHA-256."""
        self.assertTrue(CHAMPION_PATH.exists(), f"Champion missing at {CHAMPION_PATH}")
        h = compute_sha256(CHAMPION_PATH)
        self.assertEqual(
            h,
            EXPECTED_CHAMPION_SHA256,
            f"Phase 40 Champion hash altered! Expected {EXPECTED_CHAMPION_SHA256}, got {h}",
        )

    def test_02_challenger_file_presence_and_hash(self):
        """Phase 47B challenger must exist and match its expected cryptographic hash."""
        self.assertTrue(CHALLENGER_PATH.exists(), f"Challenger missing at {CHALLENGER_PATH}")
        h = compute_sha256(CHALLENGER_PATH)
        self.assertEqual(
            h,
            EXPECTED_CHALLENGER_SHA256,
            f"Challenger hash mismatch! Expected {EXPECTED_CHALLENGER_SHA256}, got {h}",
        )

    def test_03_submission_format_compliance(self):
        """Challenger must pass all official competition schema and range checks."""
        is_valid, report = validate_submission_file(
            CHALLENGER_PATH,
            expected_min_record=5001,
            expected_max_record=30000,
        )
        self.assertTrue(is_valid, f"Validation failed with errors: {report.get('errors')}")
        self.assertEqual(report["total_data_rows"], 143828)
        self.assertEqual(report["unique_record_numbers"], 24897)
        self.assertEqual(report["min_record_number"], 5001)
        self.assertEqual(report["max_record_number"], 30000)
        self.assertEqual(report["category_distribution"][1], 76404)
        self.assertEqual(report["category_distribution"][2], 67424)

    def test_04_category1_bit_for_bit_invariance(self):
        """Category 1 rows in Challenger must be 100% bit-for-bit identical to Phase 40 champion."""
        _, p40_c1, _, p40_c1_by_rec, _ = load_tsv_by_category(CHAMPION_PATH)
        _, chal_c1, _, chal_c1_by_rec, _ = load_tsv_by_category(CHALLENGER_PATH)

        self.assertEqual(len(chal_c1), 76404)
        self.assertEqual(len(p40_c1), 76404)
        self.assertEqual(chal_c1, p40_c1, "Category 1 rows do not match Phase 40 bit-for-bit!")

        # Verify record by record
        self.assertEqual(set(chal_c1_by_rec.keys()), set(p40_c1_by_rec.keys()))
        for rec_id in p40_c1_by_rec:
            self.assertEqual(chal_c1_by_rec[rec_id], p40_c1_by_rec[rec_id])

    def test_05_category2_aspect_rules_and_deltas(self):
        """Category 2 rows must strictly adhere to Category 2 allowed aspects and exhibit expected delta."""
        _, _, p40_c2, _, p40_c2_by_rec = load_tsv_by_category(CHAMPION_PATH)
        _, _, chal_c2, _, chal_c2_by_rec = load_tsv_by_category(CHALLENGER_PATH)

        self.assertEqual(len(p40_c2), 67311)
        self.assertEqual(len(chal_c2), 67424)
        self.assertEqual(len(chal_c2) - len(p40_c2), +113)

        # Verify all aspects in Category 2 are valid Category 2 aspects
        for _, _, asp, val in chal_c2:
            self.assertIn(
                asp,
                OFFICIAL_ASPECTS_CAT2,
                f"Invalid Category 2 aspect emitted: {asp}",
            )
            self.assertTrue(len(val.strip()) > 0, "Empty aspect value detected!")

        # Verify changed records count
        all_c2_recs = set(p40_c2_by_rec.keys()) | set(chal_c2_by_rec.keys())
        changed_recs = [r for r in all_c2_recs if p40_c2_by_rec.get(r, []) != chal_c2_by_rec.get(r, [])]
        self.assertEqual(len(changed_recs), 1303)

    def test_06_audit_stats_cache_consistency(self):
        """Stats JSON cache must exist and match verified properties."""
        self.assertTrue(STATS_CACHE_PATH.exists(), f"Stats cache missing at {STATS_CACHE_PATH}")
        with open(STATS_CACHE_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.assertEqual(stats["phase"], "47B")
        self.assertEqual(stats["champion_sha256"], EXPECTED_CHAMPION_SHA256)
        self.assertEqual(stats["challenger_sha256"], EXPECTED_CHALLENGER_SHA256)
        self.assertEqual(stats["total_rows"], 143828)
        self.assertEqual(stats["cat1_rows"], 76404)
        self.assertEqual(stats["cat2_rows"], 67424)
        self.assertTrue(stats["cat1_invariance"])
        self.assertEqual(stats["cat2_changed_records"], 1303)
        self.assertEqual(stats["cat2_added_entities"], 747)
        self.assertEqual(stats["cat2_removed_entities"], 634)


if __name__ == "__main__":
    unittest.main()
