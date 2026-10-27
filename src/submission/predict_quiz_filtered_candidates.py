"""
Phase 26B: Generate Filtered Champion Quiz Candidate Submissions.

Generates two candidate files for official Quiz evaluation:
1. submissions/submission_quiz_exp26b_min060.tsv (MIN confidence >= 0.60)
2. submissions/submission_quiz_exp26b_min080.tsv (MIN confidence >= 0.80)

Uses the exact frozen Exp 10 ensemble (GBERT 0.40 + 3-seed XLM-R 0.60)
on Quiz records 5,001..30,000 from data/raw/Listing_Titles.tsv.gz.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import time
import sys
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.submission.predict_test_champion import (
    CHAMPION_CHECKPOINTS,
    CandidateTitleDataset,
    collect_model_probabilities,
    compute_sha256,
    load_candidate_records,
    BATCH_SIZE,
    O_LABEL_NAME,
)
from src.evaluation.entity_metrics import filter_entities_by_category
from src.evaluation.experiment_26a_confidence_calibration import (
    bio_to_entities_with_confidence,
)
from scripts.validate_submission import (
    OFFICIAL_HEADER,
    print_validation_report,
    validate_submission_file,
)

CANDIDATE_OUT_MIN060 = Path("submissions/submission_quiz_exp26b_min060.tsv")
CANDIDATE_OUT_MIN080 = Path("submissions/submission_quiz_exp26b_min080.tsv")
REFERENCE_QUIZ_BASELINE = Path("submissions/submission_quiz_exp10_baseline_corrected.tsv")
EXPECTED_BASELINE_SHA256 = "51F880C8A6DEBB4EAF1A83594ECA2D4D2D4A92201D6CB64F9F63EAE913B65115"

QUIZ_START_RECORD = 5001
QUIZ_END_RECORD = 30000


def main():
    print("=" * 75)
    print("PHASE 26B: GENERATING FILTERED QUIZ CANDIDATE SUBMISSIONS")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")

    # 1. Load 25,000 Quiz Records
    input_file = Path("data/raw/Listing_Titles.tsv.gz")
    print(f"\nLoading Quiz candidate records from {input_file} (records {QUIZ_START_RECORD}..{QUIZ_END_RECORD})...")
    candidate_records = load_candidate_records(
        file_path=input_file,
        start_record=QUIZ_START_RECORD,
        end_record=QUIZ_END_RECORD,
    )
    assert len(candidate_records) == 25000, f"Expected 25,000 Quiz records, got {len(candidate_records)}"
    print(f"Loaded {len(candidate_records):,} Quiz records. ID Range: [{candidate_records[0][0]}, {candidate_records[-1][0]}].")

    # 2. BIO Label Mapping from Reference Champion
    ref_cfg = AutoConfig.from_pretrained(CHAMPION_CHECKPOINTS["gbert_seed42"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    o_label_id = label2id[O_LABEL_NAME]
    print(f"BIO label space: {len(id2label)} classes ('{O_LABEL_NAME}' ID = {o_label_id})")

    # 3. Model Inference Sequentially
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Model 1: GBERT (0.40)
    print("\n--- [1/4] GBERT Inference (Weight = 0.40) ---")
    gbert_cache = cache_dir / "quiz_probs_gbert_exp26b.npy"
    gbert_tok = AutoTokenizer.from_pretrained(CHAMPION_CHECKPOINTS["gbert_seed42"])
    gbert_ds = CandidateTitleDataset(candidate_records, gbert_tok)
    t0_gb = time.time()
    gbert_probs = collect_model_probabilities(
        checkpoint_path=CHAMPION_CHECKPOINTS["gbert_seed42"],
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=gbert_cache,
    )
    print(f"GBERT completed in {time.time() - t0_gb:.1f}s.")
    del gbert_ds, gbert_tok

    # Shared XLM-R Dataset
    print("\nPreparing shared XLM-R dataset for all 3 seeds...")
    xlmr_tok = AutoTokenizer.from_pretrained(CHAMPION_CHECKPOINTS["xlmr_seed42"])
    xlmr_ds = CandidateTitleDataset(candidate_records, xlmr_tok)

    # Model 2: XLM-R Seed 42 (0.20)
    print("\n--- [2/4] XLM-R Seed 42 Inference (Weight = 0.20) ---")
    s42_cache = cache_dir / "quiz_probs_xlmr_s42_exp26b.npy"
    t0_42 = time.time()
    probs_42 = collect_model_probabilities(
        checkpoint_path=CHAMPION_CHECKPOINTS["xlmr_seed42"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=s42_cache,
    )
    print(f"XLM-R seed 42 completed in {time.time() - t0_42:.1f}s.")
    xlmr_accum = [p * (1.0 / 3.0) for p in probs_42]
    del probs_42

    # Model 3: XLM-R Seed 7 (0.20)
    print("\n--- [3/4] XLM-R Seed 7 Inference (Weight = 0.20) ---")
    s7_cache = cache_dir / "quiz_probs_xlmr_s7_exp26b.npy"
    t0_7 = time.time()
    probs_7 = collect_model_probabilities(
        checkpoint_path=CHAMPION_CHECKPOINTS["xlmr_seed7"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=s7_cache,
    )
    print(f"XLM-R seed 7 completed in {time.time() - t0_7:.1f}s.")
    for i in range(len(xlmr_accum)):
        xlmr_accum[i] += probs_7[i] * (1.0 / 3.0)
    del probs_7

    # Model 4: XLM-R Seed 21 (0.20)
    print("\n--- [4/4] XLM-R Seed 21 Inference (Weight = 0.20) ---")
    s21_cache = cache_dir / "quiz_probs_xlmr_s21_exp26b.npy"
    t0_21 = time.time()
    probs_21 = collect_model_probabilities(
        checkpoint_path=CHAMPION_CHECKPOINTS["xlmr_seed21"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=s21_cache,
    )
    print(f"XLM-R seed 21 completed in {time.time() - t0_21:.1f}s.")
    for i in range(len(xlmr_accum)):
        xlmr_accum[i] += probs_21[i] * (1.0 / 3.0)
    del probs_21
    del xlmr_ds, xlmr_tok

    # 4. Generate Candidate Files
    print("\n" + "=" * 75)
    print("GENERATING CANDIDATE SUBMISSION TSV FILES")
    print("=" * 75)

    CANDIDATE_OUT_MIN060.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_OUT_MIN080.parent.mkdir(parents=True, exist_ok=True)

    f_060 = open(CANDIDATE_OUT_MIN060, "w", encoding="utf-8", newline="\n")
    f_080 = open(CANDIDATE_OUT_MIN080, "w", encoding="utf-8", newline="\n")

    f_060.write("\t".join(OFFICIAL_HEADER) + "\n")
    f_080.write("\t".join(OFFICIAL_HEADER) + "\n")

    rows_060 = 0
    rows_080 = 0
    titles_with_ents_060 = 0
    titles_with_ents_080 = 0

    aspect_counts_060: Dict[str, int] = {}
    aspect_counts_080: Dict[str, int] = {}
    cat_counts_060 = {1: 0, 2: 0}
    cat_counts_080 = {1: 0, 2: 0}

    for idx, (rec_id, cat_id, tokens) in enumerate(candidate_records):
        p_gb = gbert_probs[idx]
        p_xl = xlmr_accum[idx]

        assert len(tokens) == len(p_gb) == len(p_xl)

        p_fused = 0.40 * p_gb + 0.60 * p_xl
        pred_ids = np.argmax(p_fused, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]

        # Extract confidence assigned to each token's selected label
        selected_probs = [float(p_fused[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        # Reconstruct entities with mean & min confidence
        raw_conf_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

        # Apply Category Post-Filtering
        cat_valid_ents = []
        for asp, val, mean_c, min_c in raw_conf_ents:
            if filter_entities_by_category([(asp, val)], cat_id):
                cat_valid_ents.append((asp, val, mean_c, min_c))

        # Filter for Candidate 1: MIN >= 0.60
        ents_060 = [(asp, val) for asp, val, _, min_c in cat_valid_ents if min_c >= 0.60]
        if ents_060:
            titles_with_ents_060 += 1
            for asp, val in ents_060:
                f_060.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")
                rows_060 += 1
                aspect_counts_060[asp] = aspect_counts_060.get(asp, 0) + 1
                cat_counts_060[cat_id] += 1

        # Filter for Candidate 2: MIN >= 0.80
        ents_080 = [(asp, val) for asp, val, _, min_c in cat_valid_ents if min_c >= 0.80]
        if ents_080:
            titles_with_ents_080 += 1
            for asp, val in ents_080:
                f_080.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")
                rows_080 += 1
                aspect_counts_080[asp] = aspect_counts_080.get(asp, 0) + 1
                cat_counts_080[cat_id] += 1

    f_060.close()
    f_080.close()

    # 5. Clean up temporary cache files
    print("\nCleaning temporary probability caches...")
    for c_path in [gbert_cache, s42_cache, s7_cache, s21_cache]:
        if c_path.exists():
            c_path.unlink()
    print("Probability caches deleted. Disk space preserved.")

    # 6. Compute Hashes and Stats
    sha256_060 = compute_sha256(CANDIDATE_OUT_MIN060)
    sha256_080 = compute_sha256(CANDIDATE_OUT_MIN080)
    sha256_base = compute_sha256(REFERENCE_QUIZ_BASELINE) if REFERENCE_QUIZ_BASELINE.exists() else "N/A"

    BASE_QUIZ_ROWS = 186164
    removed_060 = BASE_QUIZ_ROWS - rows_060
    pct_rem_060 = (removed_060 / BASE_QUIZ_ROWS) * 100.0
    removed_080 = BASE_QUIZ_ROWS - rows_080
    pct_rem_080 = (removed_080 / BASE_QUIZ_ROWS) * 100.0

    print("\n" + "=" * 75)
    print("CANDIDATE FILES SUMMARY")
    print("=" * 75)
    print(f"Original Baseline (Unfiltered Exp 10):")
    print(f"  Path:            {REFERENCE_QUIZ_BASELINE}")
    print(f"  SHA-256:         {sha256_base}")
    print(f"  Rows:            {BASE_QUIZ_ROWS:,}")
    print(f"  Titles Covered:  25,000 (100.00%)")

    print(f"\nCandidate 1 (MIN >= 0.60):")
    print(f"  Path:            {CANDIDATE_OUT_MIN060}")
    print(f"  SHA-256:         {sha256_060}")
    print(f"  Total Rows:      {rows_060:,}")
    print(f"  Entities Pruned: {removed_060:,} ({pct_rem_060:.2f}%)")
    print(f"  Titles Covered:  {titles_with_ents_060:,} ({(titles_with_ents_060/25000)*100:.2f}%)")
    print(f"  Zero-Entity:     {25000 - titles_with_ents_060:,} ({((25000 - titles_with_ents_060)/25000)*100:.2f}%)")
    print(f"  Category Dist:   {cat_counts_060}")

    print(f"\nCandidate 2 (MIN >= 0.80):")
    print(f"  Path:            {CANDIDATE_OUT_MIN080}")
    print(f"  SHA-256:         {sha256_080}")
    print(f"  Total Rows:      {rows_080:,}")
    print(f"  Entities Pruned: {removed_080:,} ({pct_rem_080:.2f}%)")
    print(f"  Titles Covered:  {titles_with_ents_080:,} ({(titles_with_ents_080/25000)*100:.2f}%)")
    print(f"  Zero-Entity:     {25000 - titles_with_ents_080:,} ({((25000 - titles_with_ents_080)/25000)*100:.2f}%)")
    print(f"  Category Dist:   {cat_counts_080}")

    # 7. Strict Standalone Validation
    print("\n" + "=" * 75)
    print("RUNNING STRICT SUBMISSION VALIDATOR ON CANDIDATE 1 (MIN 0.60)")
    print("=" * 75)
    valid_060, report_060 = validate_submission_file(
        CANDIDATE_OUT_MIN060,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(report_060)
    assert valid_060, "Candidate 1 validation failed!"

    print("\n" + "=" * 75)
    print("RUNNING STRICT SUBMISSION VALIDATOR ON CANDIDATE 2 (MIN 0.80)")
    print("=" * 75)
    valid_080, report_080 = validate_submission_file(
        CANDIDATE_OUT_MIN080,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(report_080)
    assert valid_080, "Candidate 2 validation failed!"

    print("\n>>> BOTH CANDIDATE FILES PASSED 100% OF VALIDATION CHECKS <<<")


if __name__ == "__main__":
    main()
