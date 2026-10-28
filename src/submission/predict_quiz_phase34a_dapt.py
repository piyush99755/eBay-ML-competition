"""
Phase 34A: Official EvalAI Quiz Challenger Submission Generation.

Generates:
submissions/submission_quiz_phase34a_dapt_min060.tsv

Ensemble Architecture:
- DAPT GBERT (checkpoints/phase33b_dapt_gbert_best):   Weight = 0.40
- XLM-R Seed 42 (checkpoints/xlmr_baseline_best):       Weight = 0.20
- XLM-R Seed 7 (checkpoints/xlmr_seed_7_best):          Weight = 0.20
- XLM-R Seed 21 (checkpoints/xlmr_seed_21_best):        Weight = 0.20

Threshold:
- MIN Token Confidence >= 0.60
- Category-Aware Filtering: Method A

Evaluates Quiz records 5,001..30,000 from data/raw/Listing_Titles.tsv.gz.
Preserves champion checkpoints and scripts with 100% isolation.
"""

import csv
from collections import Counter, defaultdict
import gzip
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.submission.predict_test_champion import (
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

# Input and Output Paths
QUIZ_INPUT_FILE = Path("data/raw/Listing_Titles.tsv.gz")
QUIZ_START_RECORD = 5001
QUIZ_END_RECORD = 30000

CHALLENGER_OUT_PATH = Path("submissions/submission_quiz_phase34a_dapt_min060.tsv")
CHAMPION_QUIZ_PATH = Path("submissions/submission_quiz_exp26b_min060.tsv")
EXPECTED_CHAMPION_SHA256 = "049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA"

# Checkpoints
CHECKPOINTS = {
    "dapt_gbert": Path("checkpoints/phase33b_dapt_gbert_best"),
    "xlmr_seed42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr_seed7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr_seed21": Path("checkpoints/xlmr_seed_21_best"),
}

WEIGHTS = {
    "dapt_gbert": 0.40,
    "xlmr_seed42": 0.20,
    "xlmr_seed7": 0.20,
    "xlmr_seed21": 0.20,
}

CONF_THRESHOLD = 0.60


def verify_champion_integrity():
    """Verify that champion checkpoints and submission remain completely intact."""
    print("Verifying champion submission and checkpoint integrity...")
    assert CHAMPION_QUIZ_PATH.exists(), f"Champion Quiz file missing: {CHAMPION_QUIZ_PATH}"
    champ_hash = compute_sha256(CHAMPION_QUIZ_PATH)
    assert champ_hash == EXPECTED_CHAMPION_SHA256, (
        f"Champion hash mismatch! Expected {EXPECTED_CHAMPION_SHA256}, got {champ_hash}"
    )
    print(f"Champion file verified intact: {CHAMPION_QUIZ_PATH} (SHA-256: {champ_hash[:16]}...)")

    for name, ckpt_path in CHECKPOINTS.items():
        assert ckpt_path.exists(), f"Checkpoint path does not exist: {ckpt_path}"
        st_file = ckpt_path / "model.safetensors"
        assert st_file.exists(), f"Missing model.safetensors in {ckpt_path}"
    print("All four model checkpoints verified present and valid.")


def run_quiz_inference():
    print("=" * 80)
    print("PHASE 34A: OFFICIAL EVALAI QUIZ CHALLENGER GENERATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB total VRAM)")

    # 1. Preflight Safety Check
    verify_champion_integrity()

    # 2. Load 25,000 Quiz Records
    print(f"\nLoading Quiz candidate records from {QUIZ_INPUT_FILE} ({QUIZ_START_RECORD}..{QUIZ_END_RECORD})...")
    candidate_records = load_candidate_records(
        file_path=QUIZ_INPUT_FILE,
        start_record=QUIZ_START_RECORD,
        end_record=QUIZ_END_RECORD,
    )
    assert len(candidate_records) == 25000, f"Expected 25,000 Quiz records, got {len(candidate_records)}"
    print(f"Loaded {len(candidate_records):,} Quiz records. ID Range: [{candidate_records[0][0]}, {candidate_records[-1][0]}].")

    # 3. Label Mapping from Reference Configuration
    ref_cfg = AutoConfig.from_pretrained(CHECKPOINTS["dapt_gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    o_label_id = label2id[O_LABEL_NAME]
    print(f"BIO Label Space: {len(id2label)} classes ('{O_LABEL_NAME}' ID = {o_label_id})")

    # 4. Sequentially Collect Probabilities
    t_start = time.time()

    # Model 1: DAPT GBERT (0.40)
    print("\n--- [1/4] DAPT GBERT Inference (Weight = 0.40) ---")
    gbert_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["dapt_gbert"])
    gbert_ds = CandidateTitleDataset(candidate_records, gbert_tok)
    t0 = time.time()
    dapt_probs = collect_model_probabilities(
        checkpoint_path=CHECKPOINTS["dapt_gbert"],
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"DAPT GBERT completed in {time.time() - t0:.1f}s.")
    del gbert_ds, gbert_tok

    # Shared XLM-R Pre-tokenized Dataset
    print("\nPreparing shared XLM-R dataset for all 3 seeds...")
    xlmr_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["xlmr_seed42"])
    xlmr_ds = CandidateTitleDataset(candidate_records, xlmr_tok)

    # Model 2: XLM-R Seed 42 (0.20)
    print("\n--- [2/4] XLM-R Seed 42 Inference (Weight = 0.20) ---")
    t0 = time.time()
    probs_42 = collect_model_probabilities(
        checkpoint_path=CHECKPOINTS["xlmr_seed42"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 42 completed in {time.time() - t0:.1f}s.")
    xlmr_accum = [p * (1.0 / 3.0) for p in probs_42]
    del probs_42

    # Model 3: XLM-R Seed 7 (0.20)
    print("\n--- [3/4] XLM-R Seed 7 Inference (Weight = 0.20) ---")
    t0 = time.time()
    probs_7 = collect_model_probabilities(
        checkpoint_path=CHECKPOINTS["xlmr_seed7"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 7 completed in {time.time() - t0:.1f}s.")
    for i in range(len(xlmr_accum)):
        xlmr_accum[i] += probs_7[i] * (1.0 / 3.0)
    del probs_7

    # Model 4: XLM-R Seed 21 (0.20)
    print("\n--- [4/4] XLM-R Seed 21 Inference (Weight = 0.20) ---")
    t0 = time.time()
    probs_21 = collect_model_probabilities(
        checkpoint_path=CHECKPOINTS["xlmr_seed21"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 21 completed in {time.time() - t0:.1f}s.")
    for i in range(len(xlmr_accum)):
        xlmr_accum[i] += probs_21[i] * (1.0 / 3.0)
    del probs_21
    del xlmr_ds, xlmr_tok

    total_inference_time = time.time() - t_start
    print(f"\nAll 4 models completed inference in {total_inference_time:.1f}s ({total_inference_time/60:.2f}m).")

    # 5. Probability Fusion & Challenger TSV Generation
    print("\n" + "=" * 80)
    print("FUSING PROBABILITIES & GENERATING CHALLENGER SUBMISSION TSV")
    print("=" * 80)

    CHALLENGER_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHALLENGER_OUT_PATH, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\t".join(OFFICIAL_HEADER) + "\n")

        total_rows = 0
        titles_with_ents = 0
        aspect_counts: Dict[str, int] = {}
        cat_counts = {1: 0, 2: 0}
        unique_records: Set[int] = set()

        for idx, (rec_id, cat_id, tokens) in enumerate(candidate_records):
            p_gb = dapt_probs[idx]
            p_xl = xlmr_accum[idx]

            assert len(tokens) == len(p_gb) == len(p_xl)

            p_fused = 0.40 * p_gb + 0.60 * p_xl
            pred_ids = np.argmax(p_fused, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]

            selected_probs = [float(p_fused[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]
            raw_conf_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

            # Category-aware post-filtering
            cat_valid_ents = []
            for asp, val, mean_c, min_c in raw_conf_ents:
                if filter_entities_by_category([(asp, val)], cat_id):
                    cat_valid_ents.append((asp, val, mean_c, min_c))

            # MIN confidence filtering
            ents_filtered = [(asp, val) for asp, val, _, min_c in cat_valid_ents if min_c >= CONF_THRESHOLD]

            if ents_filtered:
                titles_with_ents += 1
                for asp, val in ents_filtered:
                    f_out.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")
                    total_rows += 1
                    aspect_counts[asp] = aspect_counts.get(asp, 0) + 1
                    cat_counts[cat_id] += 1
                    unique_records.add(rec_id)

    # 6. Verification and Validation
    challenger_sha256 = compute_sha256(CHALLENGER_OUT_PATH)
    challenger_size = CHALLENGER_OUT_PATH.stat().st_size

    print("\n" + "=" * 80)
    print("RUNNING STRICT SUBMISSION VALIDATOR")
    print("=" * 80)
    valid, report = validate_submission_file(
        CHALLENGER_OUT_PATH,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(report)
    assert valid, "Challenger submission validation failed!"

    # 7. Detailed Comparison against Accepted Champion
    stats = compute_and_save_comparison_stats(
        champ_path=CHAMPION_QUIZ_PATH,
        chal_path=CHALLENGER_OUT_PATH,
        expected_champ_hash=EXPECTED_CHAMPION_SHA256,
        total_inference_time=total_inference_time,
        validation_passed=valid,
    )
    return stats


def compute_and_save_comparison_stats(
    champ_path: Path = CHAMPION_QUIZ_PATH,
    chal_path: Path = CHALLENGER_OUT_PATH,
    expected_champ_hash: str = EXPECTED_CHAMPION_SHA256,
    total_inference_time: float = 2363.8,
    validation_passed: bool = True,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: CHALLENGER VS ACCEPTED CHAMPION")
    print("=" * 80)

    # Load champion predictions (robust TSV parsing without quote escaping)
    champ_rows: List[Tuple[int, int, str, str]] = []
    with open(champ_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            r = line.strip("\r\n").split("\t")
            if len(r) == 4:
                champ_rows.append((int(r[0]), int(r[1]), r[2], r[3]))

    chal_rows: List[Tuple[int, int, str, str]] = []
    with open(chal_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            r = line.strip("\r\n").split("\t")
            if len(r) == 4:
                chal_rows.append((int(r[0]), int(r[1]), r[2], r[3]))

    challenger_size = chal_path.stat().st_size
    challenger_sha256 = compute_sha256(chal_path)

    champ_set = set(champ_rows)
    chal_set = set(chal_rows)

    common_preds = champ_set.intersection(chal_set)
    added_preds = chal_set - champ_set
    removed_preds = champ_set - chal_set

    # Group by record for changed predictions
    champ_by_rec = defaultdict(set)
    for r in champ_rows:
        champ_by_rec[r[0]].add((r[2], r[3]))
    chal_by_rec = defaultdict(set)
    for r in chal_rows:
        chal_by_rec[r[0]].add((r[2], r[3]))

    records_with_diff = 0
    for rec_id in range(QUIZ_START_RECORD, QUIZ_END_RECORD + 1):
        if champ_by_rec[rec_id] != chal_by_rec[rec_id]:
            records_with_diff += 1

    champ_unique_recs = len(champ_by_rec)
    chal_unique_recs = len(chal_by_rec)

    champ_cat_counts = Counter([r[1] for r in champ_rows])
    chal_cat_counts = Counter([r[1] for r in chal_rows])

    champ_asp_counts = Counter([r[2] for r in champ_rows])
    chal_asp_counts = Counter([r[2] for r in chal_rows])

    print(f"Champion Path:              {champ_path}")
    print(f"Champion Data Rows:         {len(champ_rows):,}")
    print(f"Champion Unique Records:    {champ_unique_recs:,} / 25,000 ({(champ_unique_recs/25000)*100:.2f}%)")
    print(f"Champion SHA-256:           {expected_champ_hash}")

    print(f"\nChallenger Path:            {chal_path}")
    print(f"Challenger Data Rows:       {len(chal_rows):,}")
    print(f"Challenger Unique Records:  {chal_unique_recs:,} / 25,000 ({(chal_unique_recs/25000)*100:.2f}%)")
    print(f"Challenger File Size:       {challenger_size:,} bytes")
    print(f"Challenger SHA-256:         {challenger_sha256}")

    print(f"\nRow Delta (Chal - Champ):   {len(chal_rows) - len(champ_rows):+,} rows")
    print(f"Common Predictions:         {len(common_preds):,} ({(len(common_preds)/len(champ_rows))*100:.2f}% of champion)")
    print(f"Added Predictions:          {len(added_preds):,}")
    print(f"Removed Predictions:        {len(removed_preds):,}")
    print(f"Records with Differences:   {records_with_diff:,} / 25,000 ({(records_with_diff/25000)*100:.2f}%)")

    print(f"\nCategory Breakdown:")
    print(f"  Category 1: Champion = {champ_cat_counts[1]:,}, Challenger = {chal_cat_counts[1]:,} (Delta: {chal_cat_counts[1] - champ_cat_counts[1]:+,})")
    print(f"  Category 2: Champion = {champ_cat_counts[2]:,}, Challenger = {chal_cat_counts[2]:,} (Delta: {chal_cat_counts[2] - champ_cat_counts[2]:+,})")

    # Top aspects with differences
    print(f"\nTop Aspects with Differences:")
    all_aspects = sorted(set(list(champ_asp_counts.keys()) + list(chal_asp_counts.keys())))
    diff_table = []
    for asp in all_aspects:
        c_count = champ_asp_counts.get(asp, 0)
        ch_count = chal_asp_counts.get(asp, 0)
        delta = ch_count - c_count
        diff_table.append((asp, c_count, ch_count, delta))
    diff_table.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"  {'Aspect Name':<35} | {'Champion':<10} | {'Challenger':<10} | {'Delta':<8}")
    print("  " + "-" * 70)
    for asp, c_count, ch_count, delta in diff_table[:15]:
        print(f"  {asp:<35} | {c_count:<10,} | {ch_count:<10,} | {delta:+8,}")

    stats = {
        "status": "SUCCESS",
        "challenger_path": str(chal_path),
        "challenger_sha256": challenger_sha256,
        "challenger_size_bytes": challenger_size,
        "challenger_rows": len(chal_rows),
        "champion_rows": len(champ_rows),
        "row_delta": len(chal_rows) - len(champ_rows),
        "common_predictions": len(common_preds),
        "added_predictions": len(added_preds),
        "removed_predictions": len(removed_preds),
        "records_with_differences": records_with_diff,
        "challenger_unique_records": chal_unique_recs,
        "total_inference_time_seconds": round(total_inference_time, 1),
        "category_counts": {1: chal_cat_counts[1], 2: chal_cat_counts[2]},
        "validation_passed": validation_passed,
        "aspect_differences": [
            {"aspect": item[0], "champion": item[1], "challenger": item[2], "delta": item[3]}
            for item in diff_table
        ],
    }

    # Save stats summary for reporting
    stats_path = Path("data/cache/phase34a_quiz_challenger_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, indent=2)
    print(f"\nChallenger stats saved to {stats_path}")

    return stats


if __name__ == "__main__":
    run_quiz_inference()
