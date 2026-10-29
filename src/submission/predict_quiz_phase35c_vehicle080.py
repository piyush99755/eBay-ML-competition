"""
Phase 35C: Precision-Optimized Vehicle Model Thresholding Quiz Challenger Pipeline.

Generates:
submissions/submission_quiz_phase35c_vehicle080.tsv

Ensemble Architecture:
- DAPT GBERT seed 42 (checkpoints/phase33b_dapt_gbert_best):   Weight = 0.40
- XLM-R seed 42 (checkpoints/xlmr_baseline_best):              Weight = 0.20
- XLM-R seed 7 (checkpoints/xlmr_seed_7_best):                 Weight = 0.20
- XLM-R seed 21 (checkpoints/xlmr_seed_21_best):               Weight = 0.20

Threshold Configuration:
- Kompatibles_Fahrzeug_Modell: MIN Token Confidence >= 0.80
- All other aspects:           MIN Token Confidence >= 0.60
- Category-Aware Filtering: Method A

Evaluates Quiz records 5,001..30,000 from data/raw/Listing_Titles.tsv.gz.
Preserves champion checkpoints, scripts, and previous Quiz submissions with 100% isolation.
"""

import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.submission.predict_test_champion import (
    CandidateTitleDataset,
    BATCH_SIZE,
    O_LABEL_NAME,
    load_candidate_records,
)
from src.evaluation.entity_metrics import (
    bio_to_entities_with_confidence,
    filter_entities_by_category,
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

CHALLENGER_OUT_PATH = Path("submissions/submission_quiz_phase35c_vehicle080.tsv")
ORIG_CHAMP_QUIZ_PATH = Path("submissions/submission_quiz_exp26b_min060.tsv")
DAPT_CHAMP_QUIZ_PATH = Path("submissions/submission_quiz_phase34a_dapt_min060.tsv")

EXPECTED_ORIG_CHAMP_SHA256 = "049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA"
EXPECTED_DAPT_CHAMP_SHA256 = "B40B03D5772EC89ED957993610B31B59F06F7C6D4FB84EB44C61C0CBF9D57258"

# Checkpoints
CHECKPOINTS = {
    "dapt_gbert": Path("checkpoints/phase33b_dapt_gbert_best"),
    "xlmr_seed42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr_seed7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr_seed21": Path("checkpoints/xlmr_seed_21_best"),
}

EXPECTED_CHECKPOINT_HASHES = {
    "dapt_gbert": "446637E3BB3B1CA09FC998290C5576619A0E68E26F1503BD9F8ADC20277DF0E0",
    "xlmr_seed42": "E1933989C5625E78D63D27CFEE8781D311DBF01D9D24E122A97ADA97EEC67EB3",
    "xlmr_seed7": "49437DFF72FA00DAB4E5C3E045AD2C21D76DB4F4207E0293E1898AAE35998170",
    "xlmr_seed21": "638BAA567ECE3AE9B363FE0869454F300FE2FF1CB20E0B57A8B1D9C838BC5338",
}

WEIGHTS = {
    "dapt_gbert": 0.40,
    "xlmr_seed42": 0.20,
    "xlmr_seed7": 0.20,
    "xlmr_seed21": 0.20,
}

ASPECT_THRESHOLDS = {
    "Kompatibles_Fahrzeug_Modell": 0.80,
}
DEFAULT_THRESHOLD = 0.60

CACHE_DIR = Path("data/cache")
DECODED_CACHE_PATH = CACHE_DIR / "phase_35c_quiz_dapt_decoded_entities.json"
STATS_OUT_PATH = CACHE_DIR / "phase_35c_quiz_challenger_stats.json"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_system_integrity():
    """Verify that previous Quiz submissions and checkpoints are strictly unaltered."""
    print("Verifying existing Quiz submissions and checkpoint integrity...")
    assert ORIG_CHAMP_QUIZ_PATH.exists(), f"Original Champion file missing: {ORIG_CHAMP_QUIZ_PATH}"
    orig_hash = compute_sha256(ORIG_CHAMP_QUIZ_PATH)
    assert orig_hash == EXPECTED_ORIG_CHAMP_SHA256, (
        f"Original Champion hash mismatch! Expected {EXPECTED_ORIG_CHAMP_SHA256}, got {orig_hash}"
    )
    print(f"Original Champion verified intact: {ORIG_CHAMP_QUIZ_PATH} (SHA-256: {orig_hash[:16]}...)")

    assert DAPT_CHAMP_QUIZ_PATH.exists(), f"DAPT Champion file missing: {DAPT_CHAMP_QUIZ_PATH}"
    dapt_hash = compute_sha256(DAPT_CHAMP_QUIZ_PATH)
    assert dapt_hash == EXPECTED_DAPT_CHAMP_SHA256, (
        f"DAPT Champion hash mismatch! Expected {EXPECTED_DAPT_CHAMP_SHA256}, got {dapt_hash}"
    )
    print(f"DAPT Champion verified intact: {DAPT_CHAMP_QUIZ_PATH} (SHA-256: {dapt_hash[:16]}...)")

    for name, ckpt_path in CHECKPOINTS.items():
        assert ckpt_path.exists(), f"Checkpoint path does not exist: {ckpt_path}"
        st_file = ckpt_path / "model.safetensors"
        assert st_file.exists(), f"Missing model.safetensors in {ckpt_path}"
        st_hash = compute_sha256(st_file)
        expected_h = EXPECTED_CHECKPOINT_HASHES[name]
        assert st_hash == expected_h, (
            f"Checkpoint hash mismatch for {name}! Expected {expected_h}, got {st_hash}"
        )
    print("All four model checkpoints verified present and byte-exact.")


def collect_model_probabilities_fast(
    checkpoint_path: Path,
    dataset: CandidateTitleDataset,
    device: torch.device,
    o_label_id: int,
    num_classes: int = 48,
    batch_size: int = BATCH_SIZE,
) -> List[np.ndarray]:
    """Optimized batch-level softmax probability extraction without per-row CUDA synchronizations."""
    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    data_collator = DataCollatorForTokenClassification(tokenizer=dataset.tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    all_probabilities: List[np.ndarray] = [None] * len(dataset)
    total_batches = len(dataloader)
    start_time = time.time()

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader, start=1):
            batch_dev = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if k != "example_idx"
            }
            outputs = model(**batch_dev)
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
            example_indices = batch["example_idx"].tolist()
            batch_size_actual = logits.shape[0]

            for row_idx in range(batch_size_actual):
                ex_idx = example_indices[row_idx]
                num_tokens = dataset.token_lengths[ex_idx]
                subword_map = dataset.token_to_subword_maps[ex_idx]

                row_probs = np.zeros((num_tokens, num_classes), dtype=np.float32)
                row_probs[:, o_label_id] = 1.0

                if subword_map:
                    w_indices = list(subword_map.keys())
                    s_indices = list(subword_map.values())
                    row_probs[w_indices] = batch_probs[row_idx, s_indices]

                all_probabilities[ex_idx] = row_probs

            if b_idx % 200 == 0 or b_idx == total_batches:
                elapsed = time.time() - start_time
                speed = (b_idx * batch_size) / max(1.0, elapsed)
                print(f"  Processed {b_idx}/{total_batches} batches ({speed:.1f} titles/sec)...")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_probabilities


def run_quiz_inference():
    print("=" * 80)
    print("PHASE 35C: PRECISION-OPTIMIZED VEHICLE MODEL THRESHOLDING (MIN 0.80)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB total VRAM)")

    # 1. Preflight Safety Check
    verify_system_integrity()

    # 2. Load 25,000 Quiz Records
    print(f"\nLoading Quiz candidate records from {QUIZ_INPUT_FILE} ({QUIZ_START_RECORD}..{QUIZ_END_RECORD})...")
    candidate_records = load_candidate_records(
        QUIZ_INPUT_FILE,
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

    # 4. Initialize In-Memory Accumulator for Fused Probabilities
    # fused_probs[idx] stores: 0.40 * dapt_gbert + 0.20 * xlmr42 + 0.20 * xlmr7 + 0.20 * xlmr21
    fused_probs = [
        np.zeros((len(candidate_records[i][2]), len(id2label)), dtype=np.float32)
        for i in range(len(candidate_records))
    ]

    t_start = time.time()

    # --- Model 1: DAPT GBERT (Weight = 0.40) ---
    print("\n--- [1/4] DAPT GBERT Seed 42 Inference (Weight = 0.40) ---")
    gbert_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["dapt_gbert"])
    gbert_ds = CandidateTitleDataset(candidate_records, gbert_tok)
    t0 = time.time()
    p1 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["dapt_gbert"],
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"DAPT GBERT completed in {time.time() - t0:.1f}s. Accumulating with weight 0.40...")
    for i in range(len(candidate_records)):
        fused_probs[i] += 0.40 * p1[i]
    del p1, gbert_ds, gbert_tok

    # --- XLM-R Group: Shared Pre-tokenized Dataset for all 3 Seeds ---
    print("\nPreparing shared XLM-R dataset for all 3 XLM-R seeds...")
    xlmr_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["xlmr_seed42"])
    xlmr_ds = CandidateTitleDataset(candidate_records, xlmr_tok)

    # Model 2: XLM-R Seed 42 (Weight = 0.20)
    print("\n--- [2/4] XLM-R Seed 42 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p2 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed42"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 42 completed in {time.time() - t0:.1f}s. Accumulating with weight 0.20...")
    for i in range(len(candidate_records)):
        fused_probs[i] += 0.20 * p2[i]
    del p2

    # Model 3: XLM-R Seed 7 (Weight = 0.20)
    print("\n--- [3/4] XLM-R Seed 7 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p3 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed7"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 7 completed in {time.time() - t0:.1f}s. Accumulating with weight 0.20...")
    for i in range(len(candidate_records)):
        fused_probs[i] += 0.20 * p3[i]
    del p3

    # Model 4: XLM-R Seed 21 (Weight = 0.20)
    print("\n--- [4/4] XLM-R Seed 21 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p4 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed21"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 21 completed in {time.time() - t0:.1f}s. Accumulating with weight 0.20...")
    for i in range(len(candidate_records)):
        fused_probs[i] += 0.20 * p4[i]
    del p4, xlmr_ds, xlmr_tok

    total_inference_time = time.time() - t_start
    print(f"\nAll 4 models successfully evaluated in {total_inference_time:.1f}s ({total_inference_time/60:.2f}m).")

    # 5. Decode Predictions & Apply Aspect-Specific Thresholding
    print("\n" + "=" * 80)
    print("DECODING ENTITIES & APPLYING ASPECT-SPECIFIC THRESHOLDING")
    print("=" * 80)

    CHALLENGER_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    decoded_cache_records = []
    baseline_check_rows = []
    challenger_rows = []

    titles_with_ents = 0
    cat_counts = {1: 0, 2: 0}
    aspect_counts: Dict[str, int] = defaultdict(int)
    removed_by_category = {1: 0, 2: 0}
    removed_vehicle_models = 0
    records_affected = set()

    for idx, (rec_id, cat_id, tokens) in enumerate(candidate_records):
        p_fused = fused_probs[idx]
        assert len(tokens) == len(p_fused)

        pred_ids = np.argmax(p_fused, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]

        selected_probs = [float(p_fused[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]
        raw_conf_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

        # Category-aware post-filtering
        cat_valid_ents = []
        for asp, val, mean_c, min_c in raw_conf_ents:
            if filter_entities_by_category([(asp, val)], cat_id):
                cat_valid_ents.append((asp, val, mean_c, min_c))

        has_chal_ent = False
        title_cached_ents = []

        for asp, val, mean_c, min_c in cat_valid_ents:
            title_cached_ents.append({
                "aspect": asp,
                "value": val,
                "min_conf": min_c,
                "mean_conf": mean_c,
            })

            # Check baseline threshold (MIN 0.60)
            in_baseline = (min_c >= DEFAULT_THRESHOLD)
            if in_baseline:
                baseline_check_rows.append((rec_id, cat_id, asp, val))

            # Check challenger threshold
            thresh = ASPECT_THRESHOLDS.get(asp, DEFAULT_THRESHOLD)
            in_challenger = (min_c >= thresh)

            if in_challenger:
                challenger_rows.append((rec_id, cat_id, asp, val))
                has_chal_ent = True
                cat_counts[cat_id] += 1
                aspect_counts[asp] += 1

            # Track removed predictions
            if in_baseline and not in_challenger:
                removed_by_category[cat_id] += 1
                records_affected.add(rec_id)
                if asp == "Kompatibles_Fahrzeug_Modell":
                    removed_vehicle_models += 1

        if has_chal_ent:
            titles_with_ents += 1

        decoded_cache_records.append({
            "record_id": rec_id,
            "category": cat_id,
            "entities": title_cached_ents,
        })

    # Save decoded cache
    print(f"\nSaving decoded candidate entities with confidence scores to {DECODED_CACHE_PATH}...")
    with open(DECODED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(decoded_cache_records, f)

    # 6. Verify Baseline Identity against Phase 34A Leader
    print("\n--- Verifying Exact Identity of Baseline against Phase 34A Leader ---")
    print(f"Decoded baseline rows at MIN 0.60: {len(baseline_check_rows):,}")
    phase34a_rows = []
    with open(DAPT_CHAMP_QUIZ_PATH, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) == 4:
                phase34a_rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    print(f"Phase 34A DAPT leader rows:       {len(phase34a_rows):,}")

    assert len(baseline_check_rows) == len(phase34a_rows), (
        f"Row count mismatch! Baseline has {len(baseline_check_rows)}, Phase 34A has {len(phase34a_rows)}"
    )
    assert baseline_check_rows == phase34a_rows, (
        "Baseline predictions do not match Phase 34A leader row-for-row!"
    )
    print("MATCH CONFIRMED: Baseline MIN 0.60 is 100% identical to Phase 34A leader!")

    # 7. Write Challenger TSV
    print(f"\nWriting Phase 35C challenger to {CHALLENGER_OUT_PATH}...")
    with open(CHALLENGER_OUT_PATH, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\t".join(OFFICIAL_HEADER) + "\n")
        for rec_id, cat_id, asp, val in challenger_rows:
            f_out.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")

    chal_size = CHALLENGER_OUT_PATH.stat().st_size
    chal_sha256 = compute_sha256(CHALLENGER_OUT_PATH)

    print(f"Generated challenger TSV: {CHALLENGER_OUT_PATH}")
    print(f"File size: {chal_size:,} bytes")
    print(f"SHA-256:   {chal_sha256}")

    # 8. Strict Submission Validator
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

    # 9. Verify Mandatory Subset Invariant
    print("\n" + "=" * 80)
    print("MANDATORY SUBSET INVARIANT VERIFICATION")
    print("=" * 80)

    chal_counter = Counter(challenger_rows)
    p34a_counter = Counter(phase34a_rows)

    # Invariant 1: Multi-set subset condition
    # For every item in challenger, count in challenger <= count in Phase 34A
    is_subset = True
    added_count = 0
    for item, c_count in chal_counter.items():
        base_count = p34a_counter.get(item, 0)
        if c_count > base_count:
            is_subset = False
            added_count += (c_count - base_count)

    print(f"Phase 34A total predictions: {len(phase34a_rows):,}")
    print(f"Phase 35C total predictions: {len(challenger_rows):,}")
    total_removed = len(phase34a_rows) - len(challenger_rows)
    print(f"Total predictions removed:   {total_removed:,}")
    print(f"Total predictions added:     {added_count}")
    print(f"Strict subset invariant:     {'PASS' if is_subset else 'FAIL'}")
    assert is_subset, f"SUBSET INVARIANT FAILED! {added_count} new predictions detected!"

    # Invariant 2: Exactly zero non-vehicle predictions removed
    # All removed items must have aspect == "Kompatibles_Fahrzeug_Modell"
    removed_aspect_counts = defaultdict(int)
    for item, base_count in p34a_counter.items():
        c_count = chal_counter.get(item, 0)
        if base_count > c_count:
            removed_aspect_counts[item[2]] += (base_count - c_count)

    print(f"\nBreakdown of removed predictions by aspect:")
    for asp, cnt in sorted(removed_aspect_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asp:<35}: {cnt:,} removed")

    assert len(removed_aspect_counts) == 1 and "Kompatibles_Fahrzeug_Modell" in removed_aspect_counts, (
        f"VIOLATION: Removed aspects other than Kompatibles_Fahrzeug_Modell! {dict(removed_aspect_counts)}"
    )
    print("INVARIANT CONFIRMED: 100% of removed predictions are 'Kompatibles_Fahrzeug_Modell'!")

    # 10. Summary Statistics
    stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_path": str(CHALLENGER_OUT_PATH),
        "challenger_sha256": chal_sha256,
        "challenger_size_bytes": chal_size,
        "total_phase34a_predictions": len(phase34a_rows),
        "total_phase35c_predictions": len(challenger_rows),
        "total_removed": total_removed,
        "total_added": added_count,
        "removed_vehicle_model_predictions": removed_vehicle_models,
        "removed_by_category": {str(k): v for k, v in removed_by_category.items()},
        "records_affected": len(records_affected),
        "titles_with_predictions": titles_with_ents,
        "total_inference_time_seconds": total_inference_time,
        "aspect_counts": dict(aspect_counts),
        "validation_passed": valid,
        "subset_invariant_passed": is_subset,
    }

    with open(STATS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved challenger stats to {STATS_OUT_PATH}")

    return stats


if __name__ == "__main__":
    run_quiz_inference()
