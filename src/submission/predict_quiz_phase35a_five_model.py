"""
Phase 35A: Five-Model Ensemble Official EvalAI Quiz Challenger Pipeline.

Generates:
submissions/submission_quiz_phase35a_five_model_min060.tsv

Ensemble Architecture:
- Model 1: Original GBERT seed 42 (checkpoints/gbert_baseline_best):       Weight = 0.20
- Model 2: DAPT GBERT seed 42 (checkpoints/phase33b_dapt_gbert_best):     Weight = 0.20
- Model 3: XLM-R seed 42 (checkpoints/xlmr_baseline_best):                 Weight = 0.20
- Model 4: XLM-R seed 7 (checkpoints/xlmr_seed_7_best):                    Weight = 0.20
- Model 5: XLM-R seed 21 (checkpoints/xlmr_seed_21_best):                  Weight = 0.20

Confidence Metric:
- MIN Token Confidence >= 0.60
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

CHALLENGER_OUT_PATH = Path("submissions/submission_quiz_phase35a_five_model_min060.tsv")
ORIG_CHAMP_QUIZ_PATH = Path("submissions/submission_quiz_exp26b_min060.tsv")
DAPT_CHAMP_QUIZ_PATH = Path("submissions/submission_quiz_phase34a_dapt_min060.tsv")

EXPECTED_ORIG_CHAMP_SHA256 = "049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA"
EXPECTED_DAPT_CHAMP_SHA256 = "B40B03D5772EC89ED957993610B31B59F06F7C6D4FB84EB44C61C0CBF9D57258"

# Checkpoints
CHECKPOINTS = {
    "orig_gbert": Path("checkpoints/gbert_baseline_best"),
    "dapt_gbert": Path("checkpoints/phase33b_dapt_gbert_best"),
    "xlmr_seed42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr_seed7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr_seed21": Path("checkpoints/xlmr_seed_21_best"),
}

WEIGHTS = {
    "orig_gbert": 0.20,
    "dapt_gbert": 0.20,
    "xlmr_seed42": 0.20,
    "xlmr_seed7": 0.20,
    "xlmr_seed21": 0.20,
}

CONF_THRESHOLD = 0.60


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_system_integrity():
    """Verify that both existing Quiz submissions and all 5 checkpoints remain completely intact."""
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
    print("All five model checkpoints verified present and valid.")


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
            # Batched softmax on GPU and single transfer to host NumPy per batch
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

    # Free GPU memory immediately
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_probabilities


def run_quiz_inference():
    print("=" * 80)
    print("PHASE 35A: FIVE-MODEL OFFICIAL EVALAI QUIZ CHALLENGER GENERATION")
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
    ref_cfg = AutoConfig.from_pretrained(CHECKPOINTS["orig_gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    o_label_id = label2id[O_LABEL_NAME]
    print(f"BIO Label Space: {len(id2label)} classes ('{O_LABEL_NAME}' ID = {o_label_id})")

    # 4. Initialize In-Memory Accumulator for 5-Model Probabilities
    # fused_probs[idx] will store: sum(0.20 * p_m)
    fused_probs = [
        np.zeros((len(candidate_records[i][2]), len(id2label)), dtype=np.float32)
        for i in range(len(candidate_records))
    ]

    t_start = time.time()

    # --- Group A: BERT-family Models (Shared Pre-tokenized Dataset) ---
    print("\nPreparing shared GBERT dataset for Original GBERT and DAPT GBERT...")
    gbert_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["orig_gbert"])
    gbert_ds = CandidateTitleDataset(candidate_records, gbert_tok)

    # Model 1: Original GBERT (Weight = 0.20)
    print("\n--- [1/5] Original GBERT Seed 42 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p1 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["orig_gbert"],
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"Original GBERT completed in {time.time() - t0:.1f}s.")
    for i in range(len(fused_probs)):
        fused_probs[i] += p1[i] * 0.20
    del p1

    # Model 2: DAPT GBERT (Weight = 0.20)
    print("\n--- [2/5] DAPT GBERT Seed 42 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p2 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["dapt_gbert"],
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"DAPT GBERT completed in {time.time() - t0:.1f}s.")
    for i in range(len(fused_probs)):
        fused_probs[i] += p2[i] * 0.20
    del p2
    del gbert_ds, gbert_tok

    # --- Group B: XLM-R Models (Shared Pre-tokenized Dataset) ---
    print("\nPreparing shared XLM-R dataset for all 3 seeds...")
    xlmr_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["xlmr_seed42"])
    xlmr_ds = CandidateTitleDataset(candidate_records, xlmr_tok)

    # Model 3: XLM-R Seed 42 (Weight = 0.20)
    print("\n--- [3/5] XLM-R Seed 42 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p3 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed42"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 42 completed in {time.time() - t0:.1f}s.")
    for i in range(len(fused_probs)):
        fused_probs[i] += p3[i] * 0.20
    del p3

    # Model 4: XLM-R Seed 7 (Weight = 0.20)
    print("\n--- [4/5] XLM-R Seed 7 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p4 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed7"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 7 completed in {time.time() - t0:.1f}s.")
    for i in range(len(fused_probs)):
        fused_probs[i] += p4[i] * 0.20
    del p4

    # Model 5: XLM-R Seed 21 (Weight = 0.20)
    print("\n--- [5/5] XLM-R Seed 21 Inference (Weight = 0.20) ---")
    t0 = time.time()
    p5 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed21"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 21 completed in {time.time() - t0:.1f}s.")
    for i in range(len(fused_probs)):
        fused_probs[i] += p5[i] * 0.20
    del p5
    del xlmr_ds, xlmr_tok

    total_inference_time = time.time() - t_start
    print(f"\nAll 5 models completed sequential inference in {total_inference_time:.1f}s ({total_inference_time/60:.2f}m).")

    # 5. Decode & Write 5-Model Submission File
    print("\n" + "=" * 80)
    print("GENERATING FIVE-MODEL SUBMISSION TSV")
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
            p_fused = fused_probs[idx]
            pred_ids = np.argmax(p_fused, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]

            selected_probs = [float(p_fused[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]
            raw_conf_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

            # Category-aware post-filtering & MIN 0.60 threshold
            valid_ents = []
            for asp, val, mean_c, min_c in raw_conf_ents:
                if min_c >= CONF_THRESHOLD:
                    if filter_entities_by_category([(asp, val)], cat_id):
                        valid_ents.append((asp, val))

            if valid_ents:
                titles_with_ents += 1
                unique_records.add(rec_id)

            for asp, val in valid_ents:
                f_out.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")
                total_rows += 1
                aspect_counts[asp] = aspect_counts.get(asp, 0) + 1
                cat_counts[cat_id] += 1

    del fused_probs

    challenger_size = CHALLENGER_OUT_PATH.stat().st_size
    challenger_sha256 = compute_sha256(CHALLENGER_OUT_PATH)

    print(f"\nFive-Model Challenger TSV generated:")
    print(f"  Path:       {CHALLENGER_OUT_PATH}")
    print(f"  Rows:       {total_rows:,}")
    print(f"  Size:       {challenger_size:,} bytes")
    print(f"  SHA-256:    {challenger_sha256}")
    print(f"  Titles:     {titles_with_ents:,} / 25,000 produce entities ({titles_with_ents/250:.2f}%)")
    print(f"  Category 1: {cat_counts[1]:,} rows")
    print(f"  Category 2: {cat_counts[2]:,} rows")

    # 6. Run Strict Independent Submission Validator
    print("\n" + "=" * 80)
    print("RUNNING STRICT SUBMISSION VALIDATOR")
    print("=" * 80)

    valid, report = validate_submission_file(
        CHALLENGER_OUT_PATH,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(report)
    assert valid, "Five-Model Challenger submission validation failed!"

    # 7. Detailed Comparison against Both Previous Quiz Submissions
    stats = compute_and_save_three_way_comparison(
        orig_path=ORIG_CHAMP_QUIZ_PATH,
        dapt_path=DAPT_CHAMP_QUIZ_PATH,
        five_path=CHALLENGER_OUT_PATH,
        total_inference_time=total_inference_time,
        validation_passed=valid,
    )
    return stats


def compute_and_save_three_way_comparison(
    orig_path: Path = ORIG_CHAMP_QUIZ_PATH,
    dapt_path: Path = DAPT_CHAMP_QUIZ_PATH,
    five_path: Path = CHALLENGER_OUT_PATH,
    total_inference_time: float = 0.0,
    validation_passed: bool = True,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("DETAILED THREE-WAY COMPARISON: FIVE-MODEL VS DAPT VS ORIGINAL CHAMPION")
    print("=" * 80)

    def load_tsv_rows(path: Path) -> List[Tuple[int, int, str, str]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            _ = f.readline()
            for line in f:
                p = line.strip("\r\n").split("\t")
                if len(p) == 4:
                    rows.append((int(p[0]), int(p[1]), p[2], p[3]))
        return rows

    orig_rows = load_tsv_rows(orig_path)
    dapt_rows = load_tsv_rows(dapt_path)
    five_rows = load_tsv_rows(five_path)

    set_orig = set(orig_rows)
    set_dapt = set(dapt_rows)
    set_five = set(five_rows)

    five_size = five_path.stat().st_size
    five_sha256 = compute_sha256(five_path)

    common_all = set_orig & set_dapt & set_five
    orig_dapt_common = set_orig & set_dapt
    orig_five_common = set_orig & set_five
    dapt_five_common = set_dapt & set_five

    unique_to_orig = set_orig - set_dapt - set_five
    unique_to_dapt = set_dapt - set_orig - set_five
    unique_to_five = set_five - set_orig - set_dapt

    print(f"Total Prediction Rows:")
    print(f"  Original Champion (Exp 26B):  {len(orig_rows):,} rows (SHA-256: {EXPECTED_ORIG_CHAMP_SHA256[:16]}...)")
    print(f"  DAPT Champion (Phase 34A):    {len(dapt_rows):,} rows (SHA-256: {EXPECTED_DAPT_CHAMP_SHA256[:16]}...)")
    print(f"  Five-Model Ensemble (Ph 35A): {len(five_rows):,} rows (SHA-256: {five_sha256})")

    print(f"\nVenn Overlap Distribution:")
    print(f"  Shared by ALL THREE systems:  {len(common_all):,} ({len(common_all)/len(five_rows)*100:.2f}% of 5-model)")
    print(f"  Five-Model & Orig Agreement:  {len(orig_five_common):,} ({len(orig_five_common)/len(orig_rows)*100:.2f}% of orig)")
    print(f"  Five-Model & DAPT Agreement:  {len(dapt_five_common):,} ({len(dapt_five_common)/len(dapt_rows)*100:.2f}% of DAPT)")
    print(f"  Unique to Orig Champion:      {len(unique_to_orig):,}")
    print(f"  Unique to DAPT Champion:      {len(unique_to_dapt):,}")
    print(f"  Unique to Five-Model:         {len(unique_to_five):,}")

    print(f"\nNet Deltas:")
    print(f"  Five-Model vs Orig Champion:  {len(five_rows) - len(orig_rows):+,} rows (Added: {len(set_five - set_orig):,}, Removed: {len(set_orig - set_five):,})")
    print(f"  Five-Model vs DAPT Champion:  {len(five_rows) - len(dapt_rows):+,} rows (Added: {len(set_five - set_dapt):,}, Removed: {len(set_dapt - set_five):,})")

    # Aspect comparison
    asp_orig = Counter([r[2] for r in orig_rows])
    asp_dapt = Counter([r[2] for r in dapt_rows])
    asp_five = Counter([r[2] for r in five_rows])

    all_asps = sorted(set(list(asp_orig.keys()) + list(asp_dapt.keys()) + list(asp_five.keys())))
    diff_table = []
    for asp in all_asps:
        co = asp_orig.get(asp, 0)
        cd = asp_dapt.get(asp, 0)
        cf = asp_five.get(asp, 0)
        diff_table.append((asp, co, cd, cf, cf - co, cf - cd))

    diff_table.sort(key=lambda x: abs(x[4]), reverse=True)

    print(f"\nTop Aspects Shift Table:")
    print(f"  {'Aspect Name':<32} | {'Orig':<8} | {'DAPT':<8} | {'5-Model':<8} | {'vs Orig':<9} | {'vs DAPT':<9}")
    print("  " + "-" * 85)
    for asp, co, cd, cf, d_orig, d_dapt in diff_table[:12]:
        print(f"  {asp:<32} | {co:<8,} | {cd:<8,} | {cf:<8,} | {d_orig:+9,} | {d_dapt:+9,}")

    stats = {
        "status": "SUCCESS",
        "challenger_path": str(five_path),
        "challenger_sha256": five_sha256,
        "challenger_size_bytes": five_size,
        "challenger_rows": len(five_rows),
        "orig_champion_rows": len(orig_rows),
        "dapt_champion_rows": len(dapt_rows),
        "shared_all_three": len(common_all),
        "unique_to_orig": len(unique_to_orig),
        "unique_to_dapt": len(unique_to_dapt),
        "unique_to_five": len(unique_to_five),
        "vs_orig": {
            "delta": len(five_rows) - len(orig_rows),
            "added": len(set_five - set_orig),
            "removed": len(set_orig - set_five),
            "common": len(orig_five_common),
        },
        "vs_dapt": {
            "delta": len(five_rows) - len(dapt_rows),
            "added": len(set_five - set_dapt),
            "removed": len(set_dapt - set_five),
            "common": len(dapt_five_common),
        },
        "total_inference_time_seconds": round(total_inference_time, 1),
        "validation_passed": validation_passed,
        "aspect_shifts": [
            {
                "aspect": item[0],
                "orig": item[1],
                "dapt": item[2],
                "five_model": item[3],
                "delta_vs_orig": item[4],
                "delta_vs_dapt": item[5],
            }
            for item in diff_table
        ],
    }

    stats_path = Path("data/cache/phase_35a_quiz_challenger_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nFive-Model Quiz stats saved to {stats_path}")

    return stats


if __name__ == "__main__":
    run_quiz_inference()
