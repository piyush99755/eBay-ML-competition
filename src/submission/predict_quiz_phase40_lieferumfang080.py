"""
Phase 40: Official Quiz Challenger Pipeline (Im_Lieferumfang_Enthalten 0.80).

Generates:
  submissions/submission_quiz_phase40_lieferumfang080.tsv

Ensemble Architecture (Frozen):
  - DAPT GBERT seed 42 (checkpoints/phase33b_dapt_gbert_best):   Weight = 0.40
  - XLM-R seed 42 (checkpoints/xlmr_baseline_best):              Weight = 0.20
  - XLM-R seed 7 (checkpoints/xlmr_seed_7_best):                 Weight = 0.20
  - XLM-R seed 21 (checkpoints/xlmr_seed_21_best):               Weight = 0.20

Threshold Configuration:
  - Kompatibles_Fahrzeug_Modell: MIN Token Confidence >= 0.80
  - Produktart:                  MIN Token Confidence >= 0.80
  - Im_Lieferumfang_Enthalten:   MIN Token Confidence >= 0.80
  - All remaining 26 aspects:    MIN Token Confidence >= 0.60
  - Category-Aware Filtering:    Official Annexure category rules

Reuses verified decoded Quiz entities from data/cache/phase_35c_quiz_dapt_decoded_entities.json
with zero GPU compute and 100% mathematical determinism.
"""

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_submission import (
    OFFICIAL_HEADER,
    print_validation_report,
    validate_submission_file,
)

# Constants & Paths
QUIZ_START_RECORD = 5001
QUIZ_END_RECORD = 30000
EXPECTED_RECORD_COUNT = 25000

CHAMPION_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase36c_produktart080.tsv"
EXPECTED_CHAMPION_SHA256 = "9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87"

CHALLENGER_OUT_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
DECODED_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35c_quiz_dapt_decoded_entities.json"

CHECKPOINT_PATHS = [
    PROJECT_ROOT / "checkpoints" / "phase33b_dapt_gbert_best",
    PROJECT_ROOT / "checkpoints" / "xlmr_baseline_best",
    PROJECT_ROOT / "checkpoints" / "xlmr_seed_7_best",
    PROJECT_ROOT / "checkpoints" / "xlmr_seed_21_best",
]

ASPECT_THRESHOLDS = {
    "Kompatibles_Fahrzeug_Modell": 0.80,
    "Produktart": 0.80,
    "Im_Lieferumfang_Enthalten": 0.80,
}
DEFAULT_THRESHOLD = 0.60


def compute_sha256(path: Path) -> str:
    """Compute uppercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def load_tsv_rows(path: Path) -> Tuple[List[str], List[Tuple[int, int, str, str]]]:
    """Load TSV header and prediction rows."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip("\r\n").split("\t")
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) == 4:
                rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return header, rows


def generate_phase_40_challenger() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 40: OFFICIAL QUIZ CHALLENGER GENERATION (LIEFERUMFANG 0.80)")
    print("=" * 80)

    # 1. Verify Champion Baseline Integrity
    print(f"\n[Step 1] Verifying Phase 36C champion baseline at {CHAMPION_PATH}...")
    assert CHAMPION_PATH.exists(), f"Champion file missing: {CHAMPION_PATH}"
    initial_champ_hash = compute_sha256(CHAMPION_PATH)
    print(f"Phase 36C TSV SHA-256: {initial_champ_hash}")
    assert initial_champ_hash == EXPECTED_CHAMPION_SHA256, (
        f"Champion SHA-256 mismatch! Expected {EXPECTED_CHAMPION_SHA256}, got {initial_champ_hash}"
    )
    print("MATCH CONFIRMED: Phase 36C champion TSV is verified and intact.")

    # Check checkpoints
    for cp in CHECKPOINT_PATHS:
        assert cp.exists(), f"Checkpoint missing: {cp}"
    print(f"VERIFIED: All {len(CHECKPOINT_PATHS)} checkpoints are intact.")

    # 2. Inspect and Load Decoded Cache
    print(f"\n[Step 2] Loading verified Quiz decoded entities from {DECODED_CACHE_PATH}...")
    assert DECODED_CACHE_PATH.exists(), f"Quiz decoded cache missing: {DECODED_CACHE_PATH}"

    with open(DECODED_CACHE_PATH, "r", encoding="utf-8") as f:
        cache_records = json.load(f)

    print(f"Loaded {len(cache_records):,} records from cache.")
    assert len(cache_records) == EXPECTED_RECORD_COUNT, (
        f"Expected {EXPECTED_RECORD_COUNT:,} records, got {len(cache_records):,}"
    )

    # Check ID universe
    record_ids = [r["record_id"] for r in cache_records]
    assert min(record_ids) == QUIZ_START_RECORD and max(record_ids) == QUIZ_END_RECORD, (
        f"Record ID range error: {min(record_ids)} to {max(record_ids)}"
    )
    assert len(set(record_ids)) == EXPECTED_RECORD_COUNT, "Duplicate record IDs detected in cache!"
    print(f"Record universe confirmed: exactly {EXPECTED_RECORD_COUNT:,} unique records ({QUIZ_START_RECORD} to {QUIZ_END_RECORD}).")

    # 3. Reconstruct Baseline and Filter Challenger
    print("\n[Step 3] Reconstructing baseline and filtering Phase 40 challenger predictions...")
    reconstructed_p36c_rows: List[Tuple[int, int, str, str]] = []
    challenger_rows: List[Tuple[int, int, str, str]] = []
    removed_lieferumfang: List[Tuple[int, int, str, str, float]] = []
    affected_titles: Set[int] = set()

    for r in cache_records:
        rec_id = int(r["record_id"])
        cat_id = int(r["category"])

        for e in r["entities"]:
            asp = e["aspect"]
            val = e["value"]
            min_c = float(e["min_conf"])

            # Phase 36C condition: Vehicle 0.80, Produktart 0.80, all others 0.60
            thresh_36c = 0.80 if asp in ("Kompatibles_Fahrzeug_Modell", "Produktart") else DEFAULT_THRESHOLD
            in_36c = (min_c >= thresh_36c)

            if in_36c:
                reconstructed_p36c_rows.append((rec_id, cat_id, asp, val))

            # Phase 40 condition: Vehicle 0.80, Produktart 0.80, Lieferumfang 0.80, all others 0.60
            thresh_40 = ASPECT_THRESHOLDS.get(asp, DEFAULT_THRESHOLD)
            in_40 = (min_c >= thresh_40)

            if in_40:
                challenger_rows.append((rec_id, cat_id, asp, val))
            elif in_36c:
                assert asp == "Im_Lieferumfang_Enthalten", f"Unexpected aspect filtered: {asp}"
                assert DEFAULT_THRESHOLD <= min_c < 0.80, f"Filtered confidence out of range: {min_c}"
                removed_lieferumfang.append((rec_id, cat_id, asp, val, min_c))
                affected_titles.add(rec_id)

    # 4. Strict Validation against Baseline
    print("\n[Step 4] Running rigorous comparative validation against Phase 36C champion...")
    actual_header, actual_champ_rows = load_tsv_rows(CHAMPION_PATH)

    assert len(reconstructed_p36c_rows) == len(actual_champ_rows), (
        f"Reconstructed baseline count ({len(reconstructed_p36c_rows):,}) != actual ({len(actual_champ_rows):,})"
    )
    assert reconstructed_p36c_rows == actual_champ_rows, (
        "Reconstructed baseline does not match Phase 36C TSV row-for-row!"
    )
    print("VERIFIED: Cached entities reproduce Phase 36C champion TSV row-for-row with 100% bit-for-bit identity.")

    # Check non-Lieferumfang identity
    non_lie_36c = [r for r in actual_champ_rows if r[2] != "Im_Lieferumfang_Enthalten"]
    non_lie_40 = [r for r in challenger_rows if r[2] != "Im_Lieferumfang_Enthalten"]
    assert non_lie_36c == non_lie_40, "Non-Lieferumfang predictions differ between Phase 36C and Phase 40!"
    print("VERIFIED: All predictions for aspects other than Im_Lieferumfang_Enthalten are 100% identical.")

    # Check subset invariant
    base_counter = Counter(actual_champ_rows)
    chal_counter = Counter(challenger_rows)
    for item, c_count in chal_counter.items():
        assert c_count <= base_counter[item], f"New prediction introduced: {item}"
    print("VERIFIED: Phase 40 is a strict multi-set subset of Phase 36C (zero new predictions introduced).")

    # Check removed count
    expected_diff = len(actual_champ_rows) - len(challenger_rows)
    assert expected_diff == len(removed_lieferumfang), (
        f"Row difference ({expected_diff}) != removed Lieferumfang count ({len(removed_lieferumfang)})"
    )
    print(f"VERIFIED: Exactly {len(removed_lieferumfang):,} Im_Lieferumfang_Enthalten predictions removed (confidence in [0.60, 0.80)).")

    cat1_removed = sum(1 for r in removed_lieferumfang if r[1] == 1)
    cat2_removed = sum(1 for r in removed_lieferumfang if r[1] == 2)
    print(f"  Category 1 removed: {cat1_removed:,}")
    print(f"  Category 2 removed: {cat2_removed:,}")
    print(f"  Affected Quiz titles: {len(affected_titles):,} ({len(affected_titles)/EXPECTED_RECORD_COUNT*100:.2f}%)")

    # 5. Write Challenger TSV
    print(f"\n[Step 5] Writing Phase 40 challenger to {CHALLENGER_OUT_PATH}...")
    CHALLENGER_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHALLENGER_OUT_PATH, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\t".join(OFFICIAL_HEADER) + "\n")
        for rec_id, cat_id, asp, val in challenger_rows:
            f_out.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")

    chal_size = CHALLENGER_OUT_PATH.stat().st_size
    chal_sha256 = compute_sha256(CHALLENGER_OUT_PATH)

    print(f"Challenger file: {CHALLENGER_OUT_PATH}")
    print(f"File size:       {chal_size:,} bytes")
    print(f"SHA-256:         {chal_sha256}")
    print(f"Total rows:      {len(challenger_rows):,}")

    # 6. Official Submission Validator
    print("\n[Step 6] Running official submission validator on Phase 40 TSV...")
    is_valid, val_report = validate_submission_file(
        CHALLENGER_OUT_PATH,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(val_report)
    assert is_valid, "Submission validator reported validation failure!"
    print("VERIFIED: Phase 40 submission file passed all official validation checks 100%.")

    # 7. Re-verify Champion Hash After Generation
    post_champ_hash = compute_sha256(CHAMPION_PATH)
    assert post_champ_hash == EXPECTED_CHAMPION_SHA256, "Champion hash modified during run!"
    print("VERIFIED: Phase 36C champion TSV SHA-256 remains intact and untouched.")

    return {
        "status": "READY",
        "champion_sha256_before": initial_champ_hash,
        "champion_sha256_after": post_champ_hash,
        "challenger_path": str(CHALLENGER_OUT_PATH),
        "challenger_sha256": chal_sha256,
        "challenger_size_bytes": chal_size,
        "original_total_predictions": len(actual_champ_rows),
        "challenger_total_predictions": len(challenger_rows),
        "removed_predictions_count": len(removed_lieferumfang),
        "removed_category_1": cat1_removed,
        "removed_category_2": cat2_removed,
        "affected_titles_count": len(affected_titles),
        "pct_predictions_removed": (len(removed_lieferumfang) / len(actual_champ_rows)) * 100.0,
        "added_predictions_count": 0,
        "changed_non_target_aspects_count": 0,
        "validator_passed": is_valid,
    }


if __name__ == "__main__":
    res = generate_phase_40_challenger()
    print("\nPhase 40 Challenger Generation Complete.")
    print(f"Challenger TSV: {res['challenger_path']}")
    print(f"Challenger SHA-256: {res['challenger_sha256']}")
    print(f"Total Predictions: {res['challenger_total_predictions']:,}")
    print(f"Removed Predictions: {res['removed_predictions_count']:,} ({res['pct_predictions_removed']:.3f}%)")
