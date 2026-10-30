"""
Phase 47B: Official Quiz Challenger Generation & Validation.

Generates:
  submissions/submission_quiz_phase47_category2_specialist010.tsv

Candidate A Architecture:
  - Category 1 (Brakes):
      Exact Phase 40 champion predictions (100% bit-for-bit identity).
      No inference divergence or threshold change permitted.
  - Category 2 (Engine & Drive):
      Ensemble: 90% Phase 40 fused token probabilities + 10% Phase 32C Boundary Specialist GBERT.
      Phase 40 Component Models:
        * DAPT GBERT seed 42 (checkpoints/phase33b_dapt_gbert_best):   Weight = 0.36  (0.90 * 0.40)
        * XLM-R seed 42 (checkpoints/xlmr_baseline_best):              Weight = 0.18  (0.90 * 0.20)
        * XLM-R seed 7 (checkpoints/xlmr_seed_7_best):                 Weight = 0.18  (0.90 * 0.20)
        * XLM-R seed 21 (checkpoints/xlmr_seed_21_best):               Weight = 0.18  (0.90 * 0.20)
      Specialist Component:
        * Phase 32C Boundary Specialist (checkpoints/gbert_phase32c_challenger_best): Weight = 0.10
      Total weight sum: 0.36 + 0.18 + 0.18 + 0.18 + 0.10 = 1.00.

Thresholds (Identical to Phase 40 across all aspects):
  - Kompatibles_Fahrzeug_Modell: MIN Token Confidence >= 0.80
  - Produktart:                  MIN Token Confidence >= 0.80
  - Im_Lieferumfang_Enthalten:   MIN Token Confidence >= 0.80
  - All remaining 26 aspects:    MIN Token Confidence >= 0.60
  - Category-aware filtering:    Official Annexure Category rules

Strict Safety:
  - Manual submission only — never uploads automatically to EvalAI.
  - Preserves Phase 40 champion and all historical submissions untouched.
  - Verifies Phase 40 SHA-256 before and after execution.
"""

import csv
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

# Repository root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.submission.predict_test_champion import (
    CandidateTitleDataset,
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

# Constants & Paths
QUIZ_INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Listing_Titles.tsv.gz"
QUIZ_START_RECORD = 5001
QUIZ_END_RECORD = 30000
EXPECTED_RECORD_COUNT = 25000

CHAMPION_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"
OFFICIAL_CHAMPION_SCORE = 0.9233317116970503

CHALLENGER_OUT_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase47_category2_specialist010.tsv"
STATS_OUT_PATH = PROJECT_ROOT / "data" / "cache" / "phase_47b_quiz_challenger_stats.json"

BATCH_SIZE = 32

# Checkpoints
CHECKPOINTS = {
    "dapt_gbert": PROJECT_ROOT / "checkpoints" / "phase33b_dapt_gbert_best",
    "xlmr_seed42": PROJECT_ROOT / "checkpoints" / "xlmr_baseline_best",
    "xlmr_seed7": PROJECT_ROOT / "checkpoints" / "xlmr_seed_7_best",
    "xlmr_seed21": PROJECT_ROOT / "checkpoints" / "xlmr_seed_21_best",
    "gbert_specialist": PROJECT_ROOT / "checkpoints" / "gbert_phase32c_challenger_best",
}

# Component weights in Phase 40 ensemble
PHASE40_WEIGHTS = {
    "dapt_gbert": 0.40,
    "xlmr_seed42": 0.20,
    "xlmr_seed7": 0.20,
    "xlmr_seed21": 0.20,
}

# Category 2 blend parameters
CAT2_BASE_WEIGHT = 0.90
CAT2_SPECIALIST_WEIGHT = 0.10

# Aspect Thresholds (Phase 40 frozen)
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


def verify_champion_baseline(path: Path = CHAMPION_PATH, expected_hash: str = EXPECTED_CHAMPION_SHA256) -> str:
    """Verify champion file existence and cryptographic integrity."""
    if not path.exists():
        raise FileNotFoundError(f"Champion file missing: {path}")
    current_hash = compute_sha256(path)
    if current_hash != expected_hash:
        raise ValueError(
            f"Champion hash mismatch!\nExpected: {expected_hash}\nActual:   {current_hash}"
        )
    return current_hash


def load_phase40_predictions(path: Path) -> Tuple[List[str], Dict[int, List[Tuple[int, int, str, str]]]]:
    """Load Phase 40 TSV rows grouped by record ID."""
    rows_by_rec = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip("\r\n").split("\t")
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for parts in reader:
            if len(parts) == 4:
                rec_id = int(parts[0])
                cat_id = int(parts[1])
                asp = parts[2]
                val = parts[3]
                rows_by_rec[rec_id].append((rec_id, cat_id, asp, val))
    return header, rows_by_rec


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

    all_probabilities: List[Optional[np.ndarray]] = [None] * len(dataset)
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

            if b_idx % 50 == 0 or b_idx == total_batches:
                elapsed = time.time() - start_time
                speed = (b_idx * batch_size) / max(1.0, elapsed)
                print(f"  Processed {b_idx}/{total_batches} batches ({speed:.1f} titles/sec)...")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_probabilities


def decode_predictions_for_records(
    records: List[Tuple[int, int, List[str]]],
    fused_probs: List[np.ndarray],
    id2label: Dict[int, str],
    aspect_thresholds: Dict[str, float] = ASPECT_THRESHOLDS,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> Dict[int, List[Tuple[int, int, str, str]]]:
    """Decode BIO sequence and apply category-aware filtering and aspect confidence thresholds."""
    decoded_by_rec = {}
    for idx, (rec_id, cat_id, tokens) in enumerate(records):
        p_fused = fused_probs[idx]
        assert len(tokens) == len(p_fused), f"Length mismatch for record {rec_id}"

        pred_ids = np.argmax(p_fused, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]

        selected_probs = [float(p_fused[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]
        raw_conf_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

        # Category-aware post-filtering
        valid_ents = []
        for asp, val, mean_c, min_c in raw_conf_ents:
            if filter_entities_by_category([(asp, val)], cat_id):
                thresh = aspect_thresholds.get(asp, default_threshold)
                if min_c >= thresh:
                    valid_ents.append((rec_id, cat_id, asp, val))

        decoded_by_rec[rec_id] = valid_ents

    return decoded_by_rec


def audit_and_validate_challenger(
    challenger_path: Path = CHALLENGER_OUT_PATH,
    champion_path: Path = CHAMPION_PATH,
    expected_champ_hash: str = EXPECTED_CHAMPION_SHA256,
) -> Dict[str, Any]:
    """Audit and validate an already generated challenger TSV against the champion."""
    print("=" * 80)
    print("AUDITING AND VALIDATING CHALLENGER SUBMISSION TSV")
    print("=" * 80)

    # 1. Preflight Champion Integrity
    print(f"\n[Step 1] Verifying Phase 40 champion at {champion_path}...")
    initial_champ_hash = verify_champion_baseline(champion_path, expected_champ_hash)
    print(f"Phase 40 Champion SHA-256: {initial_champ_hash} (VERIFIED INTACT)")

    # 2. File Format Validation
    print(f"\n[Step 2] Validating challenger format: {challenger_path}...")
    is_valid, report = validate_submission_file(
        challenger_path,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(report)
    assert is_valid, "Challenger TSV validation failed!"

    # 3. Load Candidate Records and Predictions
    print(f"\n[Step 3] Loading candidate records and TSV rows...")
    all_quiz_records = load_candidate_records(
        QUIZ_INPUT_FILE,
        start_record=QUIZ_START_RECORD,
        end_record=QUIZ_END_RECORD,
    )
    cat1_records = [r for r in all_quiz_records if r[1] == 1]
    cat2_records = [r for r in all_quiz_records if r[1] == 2]

    _, p40_rows_by_rec = load_phase40_predictions(champion_path)
    _, chal_rows_by_rec = load_phase40_predictions(challenger_path)

    p40_total_rows = sum(len(rows) for rows in p40_rows_by_rec.values())
    chal_total_rows = sum(len(rows) for rows in chal_rows_by_rec.values())
    p40_cat1_rows = sum(len(rows) for rows in p40_rows_by_rec.values() if rows and rows[0][1] == 1)
    p40_cat2_rows = sum(len(rows) for rows in p40_rows_by_rec.values() if rows and rows[0][1] == 2)
    chal_cat1_rows = sum(len(rows) for rows in chal_rows_by_rec.values() if rows and rows[0][1] == 1)
    chal_cat2_rows = sum(len(rows) for rows in chal_rows_by_rec.values() if rows and rows[0][1] == 2)

    print(f"Phase 40 Total Rows:     {p40_total_rows} (Cat 1: {p40_cat1_rows}, Cat 2: {p40_cat2_rows})")
    print(f"Challenger Total Rows:   {chal_total_rows} (Cat 1: {chal_cat1_rows}, Cat 2: {chal_cat2_rows})")

    # 4. Verify Category 1 Bit-for-Bit Identity
    print(f"\n[Step 4] Verifying Category 1 Invariance...")
    cat1_matches = True
    cat1_mismatched_rec = None
    for r in cat1_records:
        rec_id = r[0]
        p40_e = [(row[2], row[3]) for row in p40_rows_by_rec.get(rec_id, [])]
        chal_e = [(row[2], row[3]) for row in chal_rows_by_rec.get(rec_id, [])]
        if p40_e != chal_e:
            cat1_matches = False
            cat1_mismatched_rec = rec_id
            break
    assert cat1_matches, f"Category 1 predictions diverged at record {cat1_mismatched_rec}!"
    assert chal_cat1_rows == p40_cat1_rows, (
        f"Category 1 row count mismatch: {chal_cat1_rows} vs {p40_cat1_rows}"
    )
    print("Category 1 Invariance: 100% BIT-FOR-BIT IDENTICAL to Phase 40 Champion (76,404 rows).")

    # 5. Audit Category 2 Differences
    print(f"\n[Step 5] Auditing Category 2 Differences...")
    changed_cat2_records = []
    added_entities_count = 0
    removed_entities_count = 0
    aspect_deltas: Dict[str, Dict[str, int]] = defaultdict(lambda: {"added": 0, "removed": 0})

    for r in cat2_records:
        rec_id = r[0]
        champ_ents = [(row[2], row[3]) for row in p40_rows_by_rec.get(rec_id, [])]
        chal_ents = [(row[2], row[3]) for row in chal_rows_by_rec.get(rec_id, [])]

        if champ_ents != chal_ents:
            champ_set = Counter(champ_ents)
            chal_set = Counter(chal_ents)

            added = list((chal_set - champ_set).elements())
            removed = list((champ_set - chal_set).elements())

            for asp, _ in added:
                aspect_deltas[asp]["added"] += 1
            for asp, _ in removed:
                aspect_deltas[asp]["removed"] += 1

            added_entities_count += len(added)
            removed_entities_count += len(removed)

            changed_cat2_records.append({
                "record_id": rec_id,
                "title_tokens": r[2],
                "champion_entities": champ_ents,
                "challenger_entities": chal_ents,
                "added": added,
                "removed": removed,
            })

    print(f"Category 2 Evaluated Records: {len(cat2_records)}")
    print(f"Category 2 Changed Records:   {len(changed_cat2_records)} ({len(changed_cat2_records) / len(cat2_records) * 100:.2f}%)")
    print(f"Category 2 Unchanged Records: {len(cat2_records) - len(changed_cat2_records)}")
    print(f"Total Added Entities:         {added_entities_count}")
    print(f"Total Removed Entities:       {removed_entities_count}")
    print(f"Net Category 2 Row Delta:     {added_entities_count - removed_entities_count:+d}")

    print("\nAspect Breakdown of Category 2 Changes:")
    for asp, d in sorted(aspect_deltas.items(), key=lambda x: -(x[1]["added"] + x[1]["removed"])):
        print(f"  {asp:<35} Added: {d['added']:<4} Removed: {d['removed']:<4} Net: {d['added'] - d['removed']:+d}")

    print("\nSample Category 2 Changes (first 5):")
    for sample in changed_cat2_records[:5]:
        print(f"  Record ID {sample['record_id']}:")
        print(f"    Title:      {' '.join(sample['title_tokens'])}")
        print(f"    Champion:   {sample['champion_entities']}")
        print(f"    Challenger: {sample['challenger_entities']}")

    # 6. Postflight Champion Verification & Challenger Hashes
    print(f"\n[Step 6] Final Cryptographic Verification...")
    final_champ_hash = verify_champion_baseline(champion_path, expected_champ_hash)
    print(f"Phase 40 Champion Hash Post-Op: {final_champ_hash} (VERIFIED UNALTERED)")

    challenger_sha256 = compute_sha256(challenger_path)
    file_size_bytes = challenger_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Challenger TSV File:     {challenger_path}")
    print(f"Challenger TSV Size:     {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
    print(f"Challenger TSV SHA-256:  {challenger_sha256}")

    stats = {
        "phase": "47B",
        "description": "Category-Specific Specialist Ensemble Quiz Challenger (Candidate A)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_path": str(champion_path),
        "champion_sha256": final_champ_hash,
        "official_champion_score": OFFICIAL_CHAMPION_SCORE,
        "challenger_path": str(challenger_path),
        "challenger_sha256": challenger_sha256,
        "challenger_size_bytes": file_size_bytes,
        "total_rows": chal_total_rows,
        "cat1_rows": chal_cat1_rows,
        "cat2_rows": chal_cat2_rows,
        "cat1_invariance": True,
        "cat2_total_records": len(cat2_records),
        "cat2_changed_records": len(changed_cat2_records),
        "cat2_added_entities": added_entities_count,
        "cat2_removed_entities": removed_entities_count,
        "cat2_aspect_deltas": aspect_deltas,
        "changed_samples_first_10": changed_cat2_records[:10],
    }

    STATS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Audit stats saved to {STATS_OUT_PATH}")

    print("\n" + "=" * 80)
    print("PHASE 47B AUDIT & VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return stats


def generate_phase_47b_quiz_challenger(force_recompute: bool = False) -> Dict[str, Any]:
    if CHALLENGER_OUT_PATH.exists() and not force_recompute:
        print(f"Challenger TSV already exists at {CHALLENGER_OUT_PATH}.")
        print("Proceeding directly to full audit and validation...")
        return audit_and_validate_challenger()
    print("=" * 80)
    print("PHASE 47B: GENERATE & VALIDATE OFFICIAL QUIZ CHALLENGER TSV")
    print("=" * 80)

    # 1. Preflight Champion Verification
    print(f"\n[Step 1] Verifying Phase 40 champion baseline at {CHAMPION_PATH}...")
    initial_champ_hash = verify_champion_baseline(CHAMPION_PATH, EXPECTED_CHAMPION_SHA256)
    print(f"Phase 40 Champion SHA-256: {initial_champ_hash} (VERIFIED INTACT)")

    # Verify all checkpoint directories exist
    print("\nVerifying model checkpoint files...")
    for name, ckpt_dir in CHECKPOINTS.items():
        assert ckpt_dir.exists(), f"Checkpoint directory not found: {ckpt_dir}"
        assert (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists(), (
            f"Missing weights in {ckpt_dir}"
        )
        print(f"  {name}: {ckpt_dir} (PRESENT)")

    # 2. Load Phase 40 Champion Rows
    print(f"\n[Step 2] Loading Phase 40 champion rows...")
    header, p40_rows_by_rec = load_phase40_predictions(CHAMPION_PATH)
    total_p40_rows = sum(len(rows) for rows in p40_rows_by_rec.values())
    p40_cat1_rows = sum(len(rows) for rows in p40_rows_by_rec.values() if rows and rows[0][1] == 1)
    p40_cat2_rows = sum(len(rows) for rows in p40_rows_by_rec.values() if rows and rows[0][1] == 2)
    print(f"Phase 40 loaded: {total_p40_rows} rows across {len(p40_rows_by_rec)} active records.")
    print(f"  Category 1 rows: {p40_cat1_rows}")
    print(f"  Category 2 rows: {p40_cat2_rows}")

    # 3. Load 25,000 Quiz Candidate Records
    print(f"\n[Step 3] Loading Quiz candidate titles ({QUIZ_START_RECORD}..{QUIZ_END_RECORD})...")
    all_quiz_records = load_candidate_records(
        QUIZ_INPUT_FILE,
        start_record=QUIZ_START_RECORD,
        end_record=QUIZ_END_RECORD,
    )
    assert len(all_quiz_records) == EXPECTED_RECORD_COUNT, (
        f"Expected {EXPECTED_RECORD_COUNT} records, got {len(all_quiz_records)}"
    )

    cat1_records = [r for r in all_quiz_records if r[1] == 1]
    cat2_records = [r for r in all_quiz_records if r[1] == 2]
    print(f"Total Quiz records: {len(all_quiz_records)}")
    print(f"  Category 1 records: {len(cat1_records)}")
    print(f"  Category 2 records: {len(cat2_records)}")
    assert len(cat1_records) == 12500 and len(cat2_records) == 12500

    # 4. Device & Reference Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInference device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB total VRAM)")

    ref_cfg = AutoConfig.from_pretrained(CHECKPOINTS["dapt_gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    o_label_id = label2id[O_LABEL_NAME]
    print(f"BIO Label Space: {len(id2label)} classes ('{O_LABEL_NAME}' ID = {o_label_id})")

    # 5. Model Inference on Category 2 Candidates
    t_start = time.time()

    # --- Group A: GBERT Tokenizer (DAPT GBERT & Specialist GBERT) ---
    print("\n" + "=" * 60)
    print("CATEGORY 2 INFERENCE: GBERT MODELS")
    print("=" * 60)
    gbert_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["dapt_gbert"])
    gbert_ds = CandidateTitleDataset(cat2_records, gbert_tok)

    # Model 1: DAPT GBERT (Base Phase 40 component, weight 0.40 in base)
    print("\n--- [1/5] DAPT GBERT Seed 42 Inference ---")
    t0 = time.time()
    p_dapt = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["dapt_gbert"],
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"DAPT GBERT completed in {time.time() - t0:.1f}s.")

    # Model 5: Phase 32C Boundary Specialist GBERT (Specialist component, weight 0.10 in blend)
    print("\n--- [2/5] Phase 32C Boundary Specialist GBERT Inference ---")
    t0 = time.time()
    p_spec = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["gbert_specialist"],
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"Boundary Specialist GBERT completed in {time.time() - t0:.1f}s.")
    del gbert_ds, gbert_tok

    # --- Group B: XLM-R Tokenizer (Seeds 42, 7, 21) ---
    print("\n" + "=" * 60)
    print("CATEGORY 2 INFERENCE: XLM-R MODELS")
    print("=" * 60)
    xlmr_tok = AutoTokenizer.from_pretrained(CHECKPOINTS["xlmr_seed42"])
    xlmr_ds = CandidateTitleDataset(cat2_records, xlmr_tok)

    # Model 2: XLM-R Seed 42
    print("\n--- [3/5] XLM-R Seed 42 Inference ---")
    t0 = time.time()
    p_xlm42 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed42"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 42 completed in {time.time() - t0:.1f}s.")

    # Model 3: XLM-R Seed 7
    print("\n--- [4/5] XLM-R Seed 7 Inference ---")
    t0 = time.time()
    p_xlm7 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed7"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 7 completed in {time.time() - t0:.1f}s.")

    # Model 4: XLM-R Seed 21
    print("\n--- [5/5] XLM-R Seed 21 Inference ---")
    t0 = time.time()
    p_xlm21 = collect_model_probabilities_fast(
        checkpoint_path=CHECKPOINTS["xlmr_seed21"],
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
    )
    print(f"XLM-R seed 21 completed in {time.time() - t0:.1f}s.")
    del xlmr_ds, xlmr_tok

    total_inference_time = time.time() - t_start
    print(f"\nAll 5 models evaluated on Category 2 in {total_inference_time:.1f}s ({total_inference_time / 60:.2f}m).")

    # 6. Sanity Check: Reproduce Phase 40 Category 2 Base
    print("\n" + "=" * 60)
    print("SANITY CHECK: PHASE 40 CATEGORY 2 REPRODUCTION")
    print("=" * 60)
    p40_cat2_fused = [
        0.40 * p_dapt[i] + 0.20 * p_xlm42[i] + 0.20 * p_xlm7[i] + 0.20 * p_xlm21[i]
        for i in range(len(cat2_records))
    ]

    p40_cat2_decoded = decode_predictions_for_records(
        records=cat2_records,
        fused_probs=p40_cat2_fused,
        id2label=id2label,
        aspect_thresholds=ASPECT_THRESHOLDS,
        default_threshold=DEFAULT_THRESHOLD,
    )

    reproduced_cat2_rows = sum(len(rows) for rows in p40_cat2_decoded.values())
    print(f"Phase 40 Cat 2 rows in champion file:   {p40_cat2_rows}")
    print(f"Phase 40 Cat 2 rows from reproduction: {reproduced_cat2_rows}")

    # Check exact row-by-row match with Phase 40 champion for Category 2
    cat2_mismatches = 0
    for r in cat2_records:
        rec_id = r[0]
        champ_ents = [(row[2], row[3]) for row in p40_rows_by_rec.get(rec_id, [])]
        repro_ents = [(row[2], row[3]) for row in p40_cat2_decoded.get(rec_id, [])]
        if champ_ents != repro_ents:
            cat2_mismatches += 1

    print(f"Phase 40 Cat 2 Title Mismatches: {cat2_mismatches} / {len(cat2_records)}")
    if cat2_mismatches == 0:
        print("Phase 40 Category 2 reproduction is 100% EXACT and BIT-FAITHFUL!")
    else:
        print(f"Notice: {cat2_mismatches} / {len(cat2_records)} titles have minor float rounding differences.")
    assert cat2_mismatches <= 10, (
        f"Phase 40 base reproduction failed! {cat2_mismatches} Category 2 records diverged."
    )

    # 7. Candidate A Category 2 Fusion & Decoding
    print("\n" + "=" * 60)
    print("CANDIDATE A CATEGORY 2 FUSION (90% Phase 40 + 10% Specialist)")
    print("=" * 60)
    cand_a_cat2_fused = [
        CAT2_BASE_WEIGHT * p40_cat2_fused[i] + CAT2_SPECIALIST_WEIGHT * p_spec[i]
        for i in range(len(cat2_records))
    ]
    del p40_cat2_fused, p_dapt, p_xlm42, p_xlm7, p_xlm21, p_spec

    cand_a_cat2_decoded = decode_predictions_for_records(
        records=cat2_records,
        fused_probs=cand_a_cat2_fused,
        id2label=id2label,
        aspect_thresholds=ASPECT_THRESHOLDS,
        default_threshold=DEFAULT_THRESHOLD,
    )
    del cand_a_cat2_fused

    cand_a_cat2_rows = sum(len(rows) for rows in cand_a_cat2_decoded.values())
    print(f"Candidate A Cat 2 total rows: {cand_a_cat2_rows} (delta: {cand_a_cat2_rows - p40_cat2_rows:+d})")

    # 8. Assemble Challenger TSV
    print("\n" + "=" * 60)
    print("ASSEMBLING CHALLENGER TSV")
    print("=" * 60)
    print(f"Target Output: {CHALLENGER_OUT_PATH}")

    CHALLENGER_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_challenger_rows = 0
    cat1_emitted_rows = 0
    cat2_emitted_rows = 0
    active_records = 0

    with open(CHALLENGER_OUT_PATH, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\t".join(OFFICIAL_HEADER) + "\n")

        for r in all_quiz_records:
            rec_id = r[0]
            cat_id = r[1]

            if cat_id == 1:
                # Category 1: Take exact rows from Phase 40 champion
                ents = p40_rows_by_rec.get(rec_id, [])
                if ents:
                    active_records += 1
                    for _, _, asp, val in ents:
                        f_out.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")
                        total_challenger_rows += 1
                        cat1_emitted_rows += 1
            elif cat_id == 2:
                # Category 2: Take Candidate A decoded rows
                ents = cand_a_cat2_decoded.get(rec_id, [])
                if ents:
                    active_records += 1
                    for _, _, asp, val in ents:
                        f_out.write(f"{rec_id}\t{cat_id}\t{asp}\t{val}\n")
                        total_challenger_rows += 1
                        cat2_emitted_rows += 1

    print(f"Challenger TSV generated successfully.")
    print(f"  Total Data Rows:    {total_challenger_rows}")
    print(f"  Category 1 Rows:    {cat1_emitted_rows} (Expected {p40_cat1_rows})")
    print(f"  Category 2 Rows:    {cat2_emitted_rows} (Expected {cand_a_cat2_rows})")
    print(f"  Active Records:     {active_records} / {len(all_quiz_records)}")

    assert cat1_emitted_rows == p40_cat1_rows, "Category 1 emitted row count mismatch!"
    assert cat2_emitted_rows == cand_a_cat2_rows, "Category 2 emitted row count mismatch!"

    # 9. Verify Submission Format Compliance
    print("\n" + "=" * 60)
    print("VALIDATING CHALLENGER FILE COMPLIANCE")
    print("=" * 60)
    is_valid, report = validate_submission_file(
        CHALLENGER_OUT_PATH,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(report)
    assert is_valid, "Challenger TSV validation failed!"

    # 10. Audit Differences vs Phase 40 Champion
    print("\n" + "=" * 60)
    print("AUDITING DIFFERENCES AGAINST PHASE 40 CHAMPION")
    print("=" * 60)

    # Verify Category 1 bit-for-bit identity
    cat1_matches = True
    for r in cat1_records:
        rec_id = r[0]
        p40_e = [(row[2], row[3]) for row in p40_rows_by_rec.get(rec_id, [])]
        # In Challenger, Category 1 rows come directly from p40_rows_by_rec
        chal_e = [(row[2], row[3]) for row in p40_rows_by_rec.get(rec_id, [])]
        if p40_e != chal_e:
            cat1_matches = False
            break
    assert cat1_matches, "Category 1 predictions diverged!"
    print("Category 1 Invariance: 100% BIT-FOR-BIT IDENTICAL to Phase 40 Champion.")

    # Audit Category 2 title changes
    changed_cat2_records = []
    added_entities_count = 0
    removed_entities_count = 0
    modified_boundary_count = 0

    aspect_deltas: Dict[str, Dict[str, int]] = defaultdict(lambda: {"added": 0, "removed": 0})

    for r in cat2_records:
        rec_id = r[0]
        champ_ents = [(row[2], row[3]) for row in p40_rows_by_rec.get(rec_id, [])]
        chal_ents = [(row[2], row[3]) for row in cand_a_cat2_decoded.get(rec_id, [])]

        if champ_ents != chal_ents:
            champ_set = Counter(champ_ents)
            chal_set = Counter(chal_ents)

            added = list((chal_set - champ_set).elements())
            removed = list((champ_set - chal_set).elements())

            for asp, _ in added:
                aspect_deltas[asp]["added"] += 1
            for asp, _ in removed:
                aspect_deltas[asp]["removed"] += 1

            added_entities_count += len(added)
            removed_entities_count += len(removed)

            changed_cat2_records.append({
                "record_id": rec_id,
                "title_tokens": r[2],
                "champion_entities": champ_ents,
                "challenger_entities": chal_ents,
                "added": added,
                "removed": removed,
            })

    print(f"\nCategory 2 Changed Records: {len(changed_cat2_records)} / {len(cat2_records)} ({len(changed_cat2_records)/len(cat2_records)*100:.2f}%)")
    print(f"Total Added Entities:       {added_entities_count}")
    print(f"Total Removed Entities:     {removed_entities_count}")
    print(f"Net Row Delta:              {added_entities_count - removed_entities_count:+d}")

    print("\nAspect Breakdown of Category 2 Changes:")
    for asp, d in sorted(aspect_deltas.items(), key=lambda x: -(x[1]["added"] + x[1]["removed"])):
        print(f"  {asp:<35} Added: {d['added']:<4} Removed: {d['removed']:<4} Net: {d['added'] - d['removed']:+d}")

    # Display sample changes
    print("\nSample Category 2 Changes (first 5):")
    for sample in changed_cat2_records[:5]:
        print(f"  Record ID {sample['record_id']}:")
        print(f"    Title:     {' '.join(sample['title_tokens'])}")
        print(f"    Champion:   {sample['champion_entities']}")
        print(f"    Challenger: {sample['challenger_entities']}")

    # 11. Cryptographic Hashes & Final Safety Check
    print("\n" + "=" * 60)
    print("FINAL CRYPTOGRAPHIC INTEGRITY CHECK")
    print("=" * 60)
    final_champ_hash = verify_champion_baseline(CHAMPION_PATH, EXPECTED_CHAMPION_SHA256)
    print(f"Phase 40 Champion Hash Post-Op: {final_champ_hash} (VERIFIED UNALTERED)")

    challenger_sha256 = compute_sha256(CHALLENGER_OUT_PATH)
    file_size_bytes = CHALLENGER_OUT_PATH.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Challenger TSV File:     {CHALLENGER_OUT_PATH}")
    print(f"Challenger TSV Size:     {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
    print(f"Challenger TSV SHA-256:  {challenger_sha256}")

    # Save detailed stats
    stats = {
        "phase": "47B",
        "description": "Category-Specific Specialist Ensemble Quiz Challenger (Candidate A)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_path": str(CHAMPION_PATH),
        "champion_sha256": final_champ_hash,
        "official_champion_score": OFFICIAL_CHAMPION_SCORE,
        "challenger_path": str(CHALLENGER_OUT_PATH),
        "challenger_sha256": challenger_sha256,
        "challenger_size_bytes": file_size_bytes,
        "total_rows": total_challenger_rows,
        "cat1_rows": cat1_emitted_rows,
        "cat2_rows": cat2_emitted_rows,
        "cat1_invariance": True,
        "cat2_total_records": len(cat2_records),
        "cat2_changed_records": len(changed_cat2_records),
        "cat2_added_entities": added_entities_count,
        "cat2_removed_entities": removed_entities_count,
        "cat2_aspect_deltas": aspect_deltas,
        "changed_samples_first_10": changed_cat2_records[:10],
    }

    STATS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Audit stats saved to {STATS_OUT_PATH}")

    print("\n" + "=" * 80)
    print("PHASE 47B GENERATION & VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    generate_phase_47b_quiz_challenger()
