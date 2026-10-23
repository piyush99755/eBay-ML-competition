"""
Phase 21A: Baseline Quiz Inference Pipeline for eBay 2026 Machine Learning Competition.

Objective:
Execute unlabelled candidate inference on Quiz records (5,001 to 30,000) using the verified
Experiment 10 champion ensemble:
- GBERT (seed 42): 0.40
- XLM-R (seed 42): 0.20
- XLM-R (seed 7):  0.20
- XLM-R (seed 21): 0.20

Sequential GPU Execution:
Loads one model at a time onto GPU, collects original-token probability distributions,
caches probabilities to disk, releases GPU memory via torch.cuda.empty_cache(), and repeats.
Designed for 6 GB VRAM safety.

Submission Format:
Outputs official UTF-8 Tab-Separated Values (TSV) submission with header:
Record Number\tCategory\tAspect Name\tAspect Value
Strictly literal text, csv.QUOTE_NONE, no 'O' tags, multi-token entities joined by single space.

Privacy:
Outputs aggregate metrics only. Never prints raw titles, tokens, or private competition data.
"""

import gzip
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from src.evaluation.entity_metrics import bio_to_entities
from scripts.validate_submission import (
    OFFICIAL_HEADER,
    validate_submission_file,
    print_validation_report,
)

# Champion Checkpoints
GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
XLMR_SEED_42_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")
XLMR_SEED_7_CHECKPOINT = Path("checkpoints/xlmr_seed_7_best")
XLMR_SEED_21_CHECKPOINT = Path("checkpoints/xlmr_seed_21_best")

RAW_LISTING_TITLES = Path("data/raw/Listing_Titles.tsv.gz")
SUBMISSIONS_DIR = Path("submissions")
OUTPUT_SUBMISSION_TSV = SUBMISSIONS_DIR / "submission_quiz_exp10_baseline.tsv"

CACHE_DIR = Path("data/cache")
GBERT_CACHE_FILE = CACHE_DIR / "quiz_probs_gbert.npy"
XLMR_42_CACHE_FILE = CACHE_DIR / "quiz_probs_xlmr_seed42.npy"
XLMR_7_CACHE_FILE = CACHE_DIR / "quiz_probs_xlmr_seed7.npy"
XLMR_21_CACHE_FILE = CACHE_DIR / "quiz_probs_xlmr_seed21.npy"

QUIZ_START_RECORD = 5001
QUIZ_END_RECORD = 30000
EXPECTED_QUIZ_COUNT = 25000

BATCH_SIZE = 32
O_LABEL_NAME = "O"


class QuizCandidateDataset(Dataset):
    """
    Dataset for unlabelled candidate titles.
    Pre-tokenizes titles and builds a direct mapping from original token index to
    the first subword index for that token.
    For any original token that produces 0 subwords (e.g. lone unencodable character),
    the mapping records absence so it can be defaulted to 'O' probability.
    """

    def __init__(self, records: List[Tuple[int, int, List[str]]], tokenizer):
        self.records = records  # list of (record_id, category_id, tokens)
        self.tokenizer = tokenizer

        self.token_lengths: List[int] = []
        self.token_to_subword_maps: List[Dict[int, int]] = []
        self.items: List[dict] = []

        print(f"Pre-tokenizing {len(records)} Quiz titles with {tokenizer.__class__.__name__}...")
        t0 = time.time()
        dropped_token_count = 0

        for idx, (_, _, tokens) in enumerate(records):
            if not tokens:
                tokens = ["[UNK]"]
            num_tokens = len(tokens)
            self.token_lengths.append(num_tokens)

            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )

            word_ids = encoded.word_ids()
            subword_map: Dict[int, int] = {}
            for s_idx, w_idx in enumerate(word_ids):
                if w_idx is not None and w_idx not in subword_map:
                    subword_map[w_idx] = s_idx

            if len(subword_map) < num_tokens:
                dropped_token_count += (num_tokens - len(subword_map))

            self.token_to_subword_maps.append(subword_map)

            item = {
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "example_idx": torch.tensor(idx, dtype=torch.long),
            }
            if "token_type_ids" in encoded:
                item["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)

            self.items.append(item)

        t_elapsed = time.time() - t0
        print(
            f"Pre-tokenization complete in {t_elapsed:.1f}s. "
            f"(Detected {dropped_token_count} unencodable tokens defaulted to 'O')"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def load_quiz_records(file_path: Path) -> List[Tuple[int, int, List[str]]]:
    """
    Load Quiz records (5,001 to 30,000) from Listing_Titles.tsv.gz.
    Returns: list of (record_id, category_id, tokens).
    """
    records = []
    print(f"Loading Quiz candidate titles from {file_path}...")
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
        # Read header: Record Number\tCategory\tTitle
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3:
                rec_id = int(parts[0])
                if rec_id < QUIZ_START_RECORD:
                    continue
                if rec_id > QUIZ_END_RECORD:
                    break
                cat_id = int(parts[1])
                title = "\t".join(parts[2:])
                tokens = title.split()
                records.append((rec_id, cat_id, tokens))

    if len(records) != EXPECTED_QUIZ_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUIZ_COUNT} Quiz records, got {len(records)}!"
        )

    print(f"Successfully loaded {len(records)} Quiz titles (Range: {records[0][0]}..{records[-1][0]}).")
    return records


def collect_model_probabilities(
    checkpoint_path: Path,
    dataset: QuizCandidateDataset,
    device: torch.device,
    o_label_id: int,
    num_classes: int = 48,
    batch_size: int = BATCH_SIZE,
    cache_file: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    Collect original-token probability distributions for all Quiz examples.
    Checks cache first; if absent, runs sequential GPU inference and saves to cache.
    Returns list of 2D numpy arrays [num_tokens, num_classes] per example.
    """
    # 1. Check cache
    if cache_file is not None and cache_file.exists():
        print(f"Loading cached probabilities from {cache_file}...")
        flat_probs = np.load(cache_file)
        splits = np.cumsum(dataset.token_lengths)[:-1]
        all_probabilities = np.split(flat_probs, splits)
        if len(all_probabilities) != len(dataset):
            raise ValueError(
                f"Cache length mismatch: got {len(all_probabilities)}, expected {len(dataset)}"
            )
        print(f"Loaded {len(all_probabilities)} example probability arrays from cache.")
        return all_probabilities

    # 2. Run GPU Inference
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
        pin_memory=True,
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
            logits = outputs.logits  # [B, S, num_classes]
            example_indices = batch["example_idx"].tolist()

            batch_size_actual = logits.shape[0]

            for row_idx in range(batch_size_actual):
                ex_idx = example_indices[row_idx]
                num_tokens = dataset.token_lengths[ex_idx]
                subword_map = dataset.token_to_subword_maps[ex_idx]

                # Default probability distribution: 100% "O" (Annexure.pdf compliance for dropped/obscure tokens)
                row_probs = np.zeros((num_tokens, num_classes), dtype=np.float32)
                row_probs[:, o_label_id] = 1.0

                if subword_map:
                    w_indices = list(subword_map.keys())
                    s_indices = list(subword_map.values())
                    token_logits = logits[row_idx, s_indices]
                    token_probs = torch.softmax(token_logits, dim=-1).cpu().numpy().astype(np.float32)
                    row_probs[w_indices] = token_probs

                all_probabilities[ex_idx] = row_probs

            if b_idx % 200 == 0 or b_idx == total_batches:
                elapsed = time.time() - start_time
                speed = (b_idx * batch_size) / max(1.0, elapsed)
                print(f"  Processed {b_idx}/{total_batches} batches ({speed:.1f} titles/sec)...")

    # Clean up GPU memory immediately
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Verify all examples are populated
    for i, p in enumerate(all_probabilities):
        if p is None or len(p) != dataset.token_lengths[i]:
            raise ValueError(f"Incomplete probability collection at index {i}!")

    # Cache to disk
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        flat_probs = np.concatenate(all_probabilities, axis=0)
        np.save(cache_file, flat_probs)
        file_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"Cached {flat_probs.shape[0]} token probabilities to {cache_file} ({file_mb:.1f} MB).")

    return all_probabilities


def run_quiz_inference_pipeline() -> dict:
    overall_start_time = time.time()
    print("==================================================")
    print("PHASE 21A: BASELINE QUIZ INFERENCE PIPELINE")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")

    # 1. Load Quiz Records
    quiz_records = load_quiz_records(RAW_LISTING_TITLES)

    # Load label mapping from reference checkpoint
    ref_cfg = AutoConfig.from_pretrained(GBERT_CHECKPOINT)
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    o_label_id = label2id[O_LABEL_NAME]
    print(f"Loaded {len(id2label)} label mappings from reference config. ('{O_LABEL_NAME}' ID = {o_label_id})")

    # ----------------------------------------------------
    # Model 1: GBERT (Weight = 0.40)
    # ----------------------------------------------------
    print("\n--- [1/4] GBERT INFERENCE (WEIGHT = 0.40) ---")
    gbert_tok = AutoTokenizer.from_pretrained(GBERT_CHECKPOINT)
    gbert_ds = QuizCandidateDataset(quiz_records, gbert_tok)
    t0_gbert = time.time()
    gbert_probs = collect_model_probabilities(
        checkpoint_path=GBERT_CHECKPOINT,
        dataset=gbert_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=GBERT_CACHE_FILE,
    )
    t_gbert = time.time() - t0_gbert
    print(f"GBERT completed in {t_gbert:.1f}s.")
    del gbert_ds, gbert_tok

    # ----------------------------------------------------
    # Prepare Shared XLM-R Dataset
    # ----------------------------------------------------
    print("\nPreparing shared XLM-R dataset for all 3 seeds...")
    xlmr_tok = AutoTokenizer.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    xlmr_ds = QuizCandidateDataset(quiz_records, xlmr_tok)

    # ----------------------------------------------------
    # Model 2: XLM-R Seed 42 (Weight = 0.20)
    # ----------------------------------------------------
    print("\n--- [2/4] XLM-R SEED 42 INFERENCE (WEIGHT = 0.20) ---")
    t0_42 = time.time()
    probs_42 = collect_model_probabilities(
        checkpoint_path=XLMR_SEED_42_CHECKPOINT,
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=XLMR_42_CACHE_FILE,
    )
    t_42 = time.time() - t0_42
    print(f"XLM-R seed 42 completed in {t_42:.1f}s.")

    # Initialize running XLM-R multiseed probabilities: (p42 + p7 + p21) / 3.0
    print("Accumulating XLM-R seed 42 probabilities...")
    xlmr_accum_probs = [p * (1.0 / 3.0) for p in probs_42]
    del probs_42

    # ----------------------------------------------------
    # Model 3: XLM-R Seed 7 (Weight = 0.20)
    # ----------------------------------------------------
    print("\n--- [3/4] XLM-R SEED 7 INFERENCE (WEIGHT = 0.20) ---")
    t0_7 = time.time()
    probs_7 = collect_model_probabilities(
        checkpoint_path=XLMR_SEED_7_CHECKPOINT,
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=XLMR_7_CACHE_FILE,
    )
    t_7 = time.time() - t0_7
    print(f"XLM-R seed 7 completed in {t_7:.1f}s.")

    print("Accumulating XLM-R seed 7 probabilities...")
    for i in range(len(xlmr_accum_probs)):
        xlmr_accum_probs[i] += probs_7[i] * (1.0 / 3.0)
    del probs_7

    # ----------------------------------------------------
    # Model 4: XLM-R Seed 21 (Weight = 0.20)
    # ----------------------------------------------------
    print("\n--- [4/4] XLM-R SEED 21 INFERENCE (WEIGHT = 0.20) ---")
    t0_21 = time.time()
    probs_21 = collect_model_probabilities(
        checkpoint_path=XLMR_SEED_21_CHECKPOINT,
        dataset=xlmr_ds,
        device=device,
        o_label_id=o_label_id,
        num_classes=len(id2label),
        batch_size=BATCH_SIZE,
        cache_file=XLMR_21_CACHE_FILE,
    )
    t_21 = time.time() - t0_21
    print(f"XLM-R seed 21 completed in {t_21:.1f}s.")

    print("Accumulating XLM-R seed 21 probabilities...")
    for i in range(len(xlmr_accum_probs)):
        xlmr_accum_probs[i] += probs_21[i] * (1.0 / 3.0)
    del probs_21
    del xlmr_ds, xlmr_tok

    # ----------------------------------------------------
    # Ensemble Probability Fusion & Entity Extraction
    # ----------------------------------------------------
    print("\n==================================================")
    print("ENSEMBLE PROBABILITY FUSION & SUBMISSION GENERATION")
    print("==================================================")
    print("Applying weights: 0.40 GBERT + 0.60 3-seed XLM-R...")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    aspect_counts: Dict[str, int] = {}
    titles_with_entities = 0
    titles_without_entities = 0

    with open(OUTPUT_SUBMISSION_TSV, "w", encoding="utf-8", newline="\n") as f:
        # Write official header
        f.write("\t".join(OFFICIAL_HEADER) + "\n")

        for idx, (rec_id, cat_id, tokens) in enumerate(quiz_records):
            p_gbert = gbert_probs[idx]
            p_xlmr = xlmr_accum_probs[idx]

            # Invariant check: tokens count matches probability rows
            if len(tokens) != len(p_gbert) or len(tokens) != len(p_xlmr):
                raise ValueError(
                    f"Token length mismatch at record {rec_id}: tokens={len(tokens)}, "
                    f"gbert={len(p_gbert)}, xlmr={len(p_xlmr)}"
                )

            # Combined probability: 0.40 * GBERT + 0.60 * XLMR_multiseed
            p_fused = 0.40 * p_gbert + 0.60 * p_xlmr
            pred_ids = np.argmax(p_fused, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]

            entities = bio_to_entities(tokens, pred_bio)
            non_o_entities = [(asp, val) for asp, val in entities if asp != "O"]

            if non_o_entities:
                titles_with_entities += 1
                for aspect_name, aspect_val in non_o_entities:
                    f.write(f"{rec_id}\t{cat_id}\t{aspect_name}\t{aspect_val}\n")
                    rows_written += 1
                    aspect_counts[aspect_name] = aspect_counts.get(aspect_name, 0) + 1
            else:
                titles_without_entities += 1

    total_runtime = time.time() - overall_start_time
    file_size_mb = OUTPUT_SUBMISSION_TSV.stat().st_size / (1024 * 1024)

    print(f"\nSubmission File Written Successfully:")
    print(f"  Output Path:            {OUTPUT_SUBMISSION_TSV}")
    print(f"  File Size:              {file_size_mb:.2f} MB")
    print(f"  Total Data Rows:        {rows_written}")
    print(f"  Titles with Entities:   {titles_with_entities} ({(titles_with_entities / EXPECTED_QUIZ_COUNT) * 100:.2f}%)")
    print(f"  Zero-Entity Titles:     {titles_without_entities} ({(titles_without_entities / EXPECTED_QUIZ_COUNT) * 100:.2f}%)")
    print(f"  Total Pipeline Runtime: {total_runtime:.1f}s ({total_runtime / 60:.2f} min)")

    # ----------------------------------------------------
    # Run Standalone Validator
    # ----------------------------------------------------
    print("\n==================================================")
    print("RUNNING STANDALONE SUBMISSION VALIDATOR")
    print("==================================================")
    is_valid, val_report = validate_submission_file(
        OUTPUT_SUBMISSION_TSV,
        expected_min_record=QUIZ_START_RECORD,
        expected_max_record=QUIZ_END_RECORD,
    )
    print_validation_report(val_report)

    if not is_valid:
        raise RuntimeError("Submission validation failed! Please check errors above.")

    print("\n>>> SUBMISSION VALIDATION PASSED 100% <<<")

    return {
        "output_path": str(OUTPUT_SUBMISSION_TSV),
        "total_titles": len(quiz_records),
        "total_rows": rows_written,
        "titles_with_entities": titles_with_entities,
        "titles_without_entities": titles_without_entities,
        "aspect_distribution": aspect_counts,
        "total_runtime_seconds": round(total_runtime, 1),
        "total_runtime_minutes": round(total_runtime / 60.0, 2),
        "validation_passed": is_valid,
        "validation_report": val_report,
    }


def main() -> None:
    run_quiz_inference_pipeline()


if __name__ == "__main__":
    main()
