"""
Unit and regression tests for Phase 36C: Produktart 0.80 Quiz Challenger Verification.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase35c_vehicle080.tsv"
CHALLENGER_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase36c_produktart080.tsv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35c_quiz_dapt_decoded_entities.json"

EXPECTED_BASELINE_SHA256 = "4B591EF6A9072C9319FF1C0134DFC9736E5A436431ED512FE47B980ECCC01382"
EXPECTED_CHALLENGER_SHA256 = "9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87"
EXPECTED_BASELINE_ROWS = 147794
EXPECTED_CHALLENGER_ROWS = 145396
EXPECTED_REMOVED_ROWS = 2398


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def load_tsv_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip("\r\n").split("\t")
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) == 4:
                rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return header, rows


class TestPhase36CProduktartChallenger(unittest.TestCase):
    """Verify integrity, subset invariants, and aspect consistency of Phase 36C submission."""

    def test_file_existence_and_hashes(self):
        """Verify that baseline and challenger exist with exact expected SHA-256 digests."""
        self.assertTrue(BASELINE_TSV.exists(), "Phase 35C baseline TSV missing!")
        self.assertTrue(CHALLENGER_TSV.exists(), "Phase 36C challenger TSV missing!")

        base_hash = compute_sha256(BASELINE_TSV)
        chal_hash = compute_sha256(CHALLENGER_TSV)

        self.assertEqual(base_hash, EXPECTED_BASELINE_SHA256, "Baseline SHA-256 altered!")
        self.assertEqual(chal_hash, EXPECTED_CHALLENGER_SHA256, "Challenger SHA-256 mismatch!")

    def test_row_counts_and_headers(self):
        """Verify headers, row counts, and exact row reduction count."""
        h_base, r_base = load_tsv_rows(BASELINE_TSV)
        h_chal, r_chal = load_tsv_rows(CHALLENGER_TSV)

        expected_header = ["Record Number", "Category", "Aspect Name", "Aspect Value"]
        self.assertEqual(h_base, expected_header)
        self.assertEqual(h_chal, expected_header)

        self.assertEqual(len(r_base), EXPECTED_BASELINE_ROWS)
        self.assertEqual(len(r_chal), EXPECTED_CHALLENGER_ROWS)
        self.assertEqual(len(r_base) - len(r_chal), EXPECTED_REMOVED_ROWS)

    def test_non_produktart_aspects_identical(self):
        """Verify that all non-Produktart predictions are 100% identical row-for-row."""
        _, r_base = load_tsv_rows(BASELINE_TSV)
        _, r_chal = load_tsv_rows(CHALLENGER_TSV)

        non_pa_base = [r for r in r_base if r[2] != "Produktart"]
        non_pa_chal = [r for r in r_chal if r[2] != "Produktart"]

        self.assertEqual(non_pa_base, non_pa_chal, "Non-Produktart predictions modified!")

    def test_multiset_subset_invariant(self):
        """Verify that Phase 36C is a strict multi-set subset of Phase 35C (zero new rows)."""
        _, r_base = load_tsv_rows(BASELINE_TSV)
        _, r_chal = load_tsv_rows(CHALLENGER_TSV)

        c_base = Counter(r_base)
        c_chal = Counter(r_chal)

        for item, chal_count in c_chal.items():
            base_count = c_base[item]
            self.assertGreaterEqual(base_count, chal_count, f"Prediction {item} has higher count in challenger!")

    def test_confidence_of_removed_predictions(self):
        """Verify every removed prediction corresponds to Produktart with confidence in [0.60, 0.80)."""
        self.assertTrue(CACHE_PATH.exists(), "Quiz decoded cache missing!")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)

        _, r_base = load_tsv_rows(BASELINE_TSV)
        _, r_chal = load_tsv_rows(CHALLENGER_TSV)

        c_base = Counter(r_base)
        c_chal = Counter(r_chal)
        removed_counter = c_base - c_chal

        # Verify all removed items are Produktart
        for item, count in removed_counter.items():
            rec_id, cat_id, asp, val = item
            self.assertEqual(asp, "Produktart", f"Non-Produktart prediction removed: {item}")

        # Check against cache
        cache_by_id = {r["record_id"]: r["entities"] for r in records}
        for (rec_id, cat_id, asp, val), count in removed_counter.items():
            ents = cache_by_id.get(rec_id, [])
            matching = [
                e for e in ents
                if e["aspect"] == asp and e["value"] == val and 0.60 <= e["min_conf"] < 0.80
            ]
            self.assertTrue(len(matching) >= count, f"Filtered entity not found in [0.60, 0.80) for record {rec_id}")


if __name__ == "__main__":
    unittest.main()
