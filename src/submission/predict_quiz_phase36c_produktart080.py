"""
Phase 36C: Produktart 0.80 Quiz Challenger Pipeline.

Generates:
submissions/submission_quiz_phase36c_produktart080.tsv

Ensemble Architecture:
- DAPT GBERT seed 42 (checkpoints/phase33b_dapt_gbert_best):   Weight = 0.40
- XLM-R seed 42 (checkpoints/xlmr_baseline_best):              Weight = 0.20
- XLM-R seed 7 (checkpoints/xlmr_seed_7_best):                 Weight = 0.20
- XLM-R seed 21 (checkpoints/xlmr_seed_21_best):               Weight = 0.20

Threshold Configuration:
- Kompatibles_Fahrzeug_Modell: MIN Token Confidence >= 0.80
- Produktart:                  MIN Token Confidence >= 0.80
- All other 27 aspects:        MIN Token Confidence >= 0.60
- Category-Aware Filtering:    Method A (Official Annexure category rules)

Reuses verified decoded Quiz entities from data/cache/phase_35c_quiz_dapt_decoded_entities.json
with zero GPU compute and 100% mathematical determinism.
"""

import csv
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_submission import (
    OFFICIAL_HEADER,
    print_validation_report,
    validate_submission_file,
)

# Input and Output Paths
QUIZ_START_RECORD = 5001
QUIZ_END_RECORD = 30000
EXPECTED_RECORD_COUNT = 25000

CHALLENGER_OUT_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase36c_produktart080.tsv"
BASELINE_CHAMP_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase35c_vehicle080.tsv"
EXPECTED_BASELINE_SHA256 = "4B591EF6A9072C9319FF1C0134DFC9736E5A436431ED512FE47B980ECCC01382"

DECODED_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35c_quiz_dapt_decoded_entities.json"

ASPECT_THRESHOLDS = {
    "Kompatibles_Fahrzeug_Modell": 0.80,
    "Produktart": 0.80,
}
DEFAULT_THRESHOLD = 0.60


def compute_sha256(path: Path) -> str:
    """Compute uppercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def generate_phase_36c_challenger() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 36C: PRODUKTART 0.80 QUIZ CHALLENGER GENERATION")
    print("=" * 80)

    # 1. Verify Baseline TSV Integrity
    print(f"\n[Step 1] Verifying Phase 35C champion baseline at {BASELINE_CHAMP_PATH}...")
    assert BASELINE_CHAMP_PATH.exists(), f"Baseline file missing: {BASELINE_CHAMP_PATH}"
    actual_base_hash = compute_sha256(BASELINE_CHAMP_PATH)
    print(f"Phase 35C TSV SHA-256: {actual_base_hash}")
    assert actual_base_hash == EXPECTED_BASELINE_SHA256, (
        f"Baseline SHA-256 mismatch! Expected {EXPECTED_BASELINE_SHA256}, got {actual_base_hash}"
    )
    print("MATCH CONFIRMED: Phase 35C champion TSV is verified and intact.")

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
    print("\n[Step 3] Reconstructing baseline and filtering Phase 36C challenger predictions...")
    baseline_rows = []
    challenger_rows = []
    removed_produktart = []

    cat_counts_35c = Counter()
    cat_counts_36c = Counter()
    aspect_counts_35c = Counter()
    aspect_counts_36c = Counter()

    for r in cache_records:
        rec_id = int(r["record_id"])
        cat_id = int(r["category"])

        for e in r["entities"]:
            asp = e["aspect"]
            val = e["value"]
            min_c = float(e["min_conf"])

            # Baseline condition: Vehicle 0.80, all others 0.60
            thresh_35c = 0.80 if asp == "Kompatibles_Fahrzeug_Modell" else DEFAULT_THRESHOLD
            in_35c = (min_c >= thresh_35c)

            if in_35c:
                baseline_rows.append((rec_id, cat_id, asp, val))
                cat_counts_35c[cat_id] += 1
                aspect_counts_35c[asp] += 1

            # Challenger condition: Vehicle 0.80, Produktart 0.80, all others 0.60
            thresh_36c = ASPECT_THRESHOLDS.get(asp, DEFAULT_THRESHOLD)
            in_36c = (min_c >= thresh_36c)

            if in_36c:
                challenger_rows.append((rec_id, cat_id, asp, val))
                cat_counts_36c[cat_id] += 1
                aspect_counts_36c[asp] += 1
            elif in_35c:
                assert asp == "Produktart", f"Unexpected aspect filtered: {asp}"
                assert DEFAULT_THRESHOLD <= min_c < 0.80, f"Filtered confidence out of range: {min_c}"
                removed_produktart.append((rec_id, cat_id, asp, val, min_c))

    # 4. Strict Validation against Baseline
    print("\n[Step 4] Running rigorous comparative validation against Phase 35C champion...")
    # Load actual baseline TSV rows
    actual_base_rows = []
    with open(BASELINE_CHAMP_PATH, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) == 4:
                actual_base_rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))

    assert len(baseline_rows) == len(actual_base_rows), (
        f"Reconstructed baseline count ({len(baseline_rows):,}) != actual ({len(actual_base_rows):,})"
    )
    assert baseline_rows == actual_base_rows, "Reconstructed baseline does not match Phase 35C TSV row-for-row!"
    print("VERIFIED: Cached entities reproduce Phase 35C champion TSV row-for-row with 100% identity.")

    # Check non-Produktart identity
    non_pa_35c = [r for r in baseline_rows if r[2] != "Produktart"]
    non_pa_36c = [r for r in challenger_rows if r[2] != "Produktart"]
    assert non_pa_35c == non_pa_36c, "Non-Produktart predictions differ between Phase 35C and Phase 36C!"
    print("VERIFIED: All predictions for aspects other than Produktart are 100% identical.")

    # Check subset invariant
    base_counter = Counter(baseline_rows)
    chal_counter = Counter(challenger_rows)
    for item, c_count in chal_counter.items():
        assert c_count <= base_counter[item], f"New prediction introduced: {item}"
    print("VERIFIED: Phase 36C is a strict multi-set subset of Phase 35C (zero new predictions introduced).")

    # Check removed count
    expected_diff = len(baseline_rows) - len(challenger_rows)
    assert expected_diff == len(removed_produktart), (
        f"Row difference ({expected_diff}) != removed Produktart count ({len(removed_produktart)})"
    )
    print(f"VERIFIED: Exactly {len(removed_produktart):,} Produktart predictions removed (confidence in [0.60, 0.80)).")

    # 5. Write Challenger TSV
    print(f"\n[Step 5] Writing Phase 36C challenger to {CHALLENGER_OUT_PATH}...")
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
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 80)
    valid, report = validate_submission_file(
        CHALLENGER_OUT_PATH,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(report)
    assert valid, "Challenger submission validation failed!"

    # 7. Re-verify Baseline SHA-256 Intact
    final_base_hash = compute_sha256(BASELINE_CHAMP_PATH)
    assert final_base_hash == EXPECTED_BASELINE_SHA256, "Baseline TSV was altered during execution!"
    print("FINAL INTEGRITY CHECK: Phase 35C champion TSV SHA-256 is 100% unchanged.")

    return {
        "status": "COMPLETE",
        "challenger_tsv": str(CHALLENGER_OUT_PATH),
        "sha256": chal_sha256,
        "file_size": chal_size,
        "total_rows": len(challenger_rows),
        "baseline_rows": len(baseline_rows),
        "removed_produktart": len(removed_produktart),
        "cat1_rows": cat_counts_36c[1],
        "cat2_rows": cat_counts_36c[2],
        "baseline_sha256_verified": True,
    }


if __name__ == "__main__":
    generate_phase_36c_challenger()
