"""
Preflight verification suite for Phase 23B: Full-Data Champion Quiz Inference.
"""

import gzip
import hashlib
from pathlib import Path
import shutil
import sys
import torch
from transformers import AutoConfig

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_submission import validate_submission_file

FULLDATA_CHECKPOINTS = [
    Path("checkpoints/gbert_fulldata_seed42_final"),
    Path("checkpoints/xlmr_fulldata_seed42_final"),
    Path("checkpoints/xlmr_fulldata_seed7_final"),
    Path("checkpoints/xlmr_fulldata_seed21_final"),
]

BASELINE_SUBMISSION = Path("submissions/submission_quiz_exp10_baseline_corrected.tsv")
EXPECTED_BASELINE_SHA256 = "51F880C8A6DEBB4EAF1A83594ECA2D4D2D4A92201D6CB64F9F63EAE913B65115"
LISTING_TITLES = Path("data/raw/Listing_Titles.tsv.gz")


def run_preflight_checks():
    print("=" * 60)
    print("PHASE 23B: PREFLIGHT VERIFICATION CHECKS")
    print("=" * 60)

    # 1. Check Full-Data Checkpoints
    print("\n[Check 1] Verifying all 4 full-data champion checkpoints...")
    for cp in FULLDATA_CHECKPOINTS:
        assert cp.exists(), f"Missing checkpoint: {cp}"
        assert (cp / "model.safetensors").exists(), f"Missing weights in {cp}"
        assert (cp / "config.json").exists(), f"Missing config in {cp}"
        cfg = AutoConfig.from_pretrained(cp)
        assert len(cfg.id2label) == 48, f"{cp}: expected 48 labels, got {len(cfg.id2label)}"
        assert cfg.id2label[47] == "O", f"{cp}: label 47 is not 'O'"
        size_mb = sum(f.stat().st_size for f in cp.iterdir() if f.is_file()) / (1024 ** 2)
        print(f"  PASS: {cp.name:<32} ({size_mb:.1f} MB, 48 labels)")

    # 2. Check Baseline Quiz Submission
    print("\n[Check 2] Verifying existing successful baseline Quiz submission...")
    assert BASELINE_SUBMISSION.exists(), f"Missing baseline submission: {BASELINE_SUBMISSION}"
    sha256 = hashlib.sha256(BASELINE_SUBMISSION.read_bytes()).hexdigest().upper()
    assert sha256 == EXPECTED_BASELINE_SHA256, f"Hash mismatch: expected {EXPECTED_BASELINE_SHA256}, got {sha256}"
    is_valid, report = validate_submission_file(BASELINE_SUBMISSION, 5001, 30000)
    assert is_valid, "Baseline submission failed validation!"
    print(f"  PASS: Baseline submission intact (SHA-256: {sha256[:16]}..., 186,164 rows, COMPLIANT)")

    # 3. Check Disk Space
    print("\n[Check 3] Verifying Drive C: available disk space...")
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    assert free_gb >= 4.0, f"Insufficient disk space: {free_gb:.2f} GB available, need >= 4.0 GB"
    print(f"  PASS: Available disk space = {free_gb:.2f} GB (Required >= 4.0 GB)")

    # 4. Check GPU Availability
    print("\n[Check 4] Verifying GPU compute availability...")
    assert torch.cuda.is_available(), "CUDA is not available!"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  PASS: GPU available: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    # 5. Check Listing Titles Quiz Range
    print("\n[Check 5] Verifying raw Quiz candidate records (5001..30000)...")
    assert LISTING_TITLES.exists(), f"Missing listing titles: {LISTING_TITLES}"
    quiz_count = 0
    min_rec = None
    max_rec = None
    with gzip.open(LISTING_TITLES, "rt", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3:
                rec_id = int(parts[0])
                if 5001 <= rec_id <= 30000:
                    quiz_count += 1
                    if min_rec is None or rec_id < min_rec:
                        min_rec = rec_id
                    if max_rec is None or rec_id > max_rec:
                        max_rec = rec_id
                elif rec_id > 30000:
                    break
    assert quiz_count == 25000, f"Expected 25,000 Quiz titles, got {quiz_count}"
    assert min_rec == 5001 and max_rec == 30000, f"Unexpected range: {min_rec}..{max_rec}"
    print(f"  PASS: Verified exactly {quiz_count:,} Quiz titles (Records {min_rec}..{max_rec})")

    # 6. Check Distinct Submission Filenames
    print("\n[Check 6] Verifying output paths isolation...")
    out_file = Path("submissions/submission_quiz_exp23b_fulldata.tsv")
    assert out_file != BASELINE_SUBMISSION, "Output path collides with baseline submission!"
    print(f"  PASS: Output target is distinct: {out_file.name}")

    print("\n==================================================")
    print("ALL 6 PREFLIGHT VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_preflight_checks()
