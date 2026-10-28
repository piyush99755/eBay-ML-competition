"""
Phase 29B: Sequential 3-Fold GBERT Training Pipeline.

Trains deepset/gbert-base across 3 deterministic, category-stratified folds
derived from the 4,000 Split-A training titles.
Preserves the 1,000 Split-A validation titles as an external holdout.

Training Configurations:
- Architecture: deepset/gbert-base
- Learning Rate: 2e-5 (AdamW)
- Epochs: 3 (fixed 3-epoch policy for pilot parity)
- Batch Size: 4 (train), 8 (val)
- Folds: 3 deterministic folds (near-duplicates grouped into Fold 0)

Checkpoints:
- checkpoints/gbert_cv3_fold0_best
- checkpoints/gbert_cv3_fold1_best
- checkpoints/gbert_cv3_fold2_best
"""

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.data.make_cv_folds import get_cv_folds, verify_fold_integrity
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.evaluate_model import evaluate_model_entities


def evaluate_loss(model, dataloader, device: torch.device) -> float:
    """Calculate average loss on validation dataloader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch_dev)
            total_loss += outputs.loss.item()
    return total_loss / max(1, len(dataloader))

MODEL_NAME = "deepset/gbert-base"
RANDOM_SEED = 42
LEARNING_RATE = 2e-5
EPOCHS = 3
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
MIN_FREE_DISK_GB = 4.0

FOLD_CHECKPOINT_DIRS = {
    0: Path("checkpoints/gbert_cv3_fold0_best"),
    1: Path("checkpoints/gbert_cv3_fold1_best"),
    2: Path("checkpoints/gbert_cv3_fold2_best"),
}


def get_free_disk_space_gb() -> float:
    """Return free disk space in GB on workspace drive."""
    return round(shutil.disk_usage(".").free / (1024 ** 3), 2)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_fold_active_probabilities(
    model,
    dataloader: DataLoader,
    device: torch.device,
    ignore_label_id: int = -100,
) -> List[np.ndarray]:
    """Collect softmax probability distributions for active tokens in validation examples."""
    model.eval()
    all_example_probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch_dev)
            logits = outputs.logits
            labels = batch_dev["labels"]

            for row_idx in range(labels.shape[0]):
                row_labels = labels[row_idx]
                row_logits = logits[row_idx]
                active_mask = row_labels != ignore_label_id
                active_logits = row_logits[active_mask]
                active_probs = torch.softmax(active_logits, dim=-1).cpu().numpy().astype(np.float32)
                all_example_probs.append(active_probs)

    return all_example_probs


def train_single_fold(
    fold_idx: int,
    train_fold_examples: List[dict],
    val_fold_examples: List[dict],
    tokenizer,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
    checkpoint_dir: Path,
    cache_dir: Path,
) -> dict:
    """Train deepset/gbert-base for 3 epochs on a single fold partition."""
    print("\n" + "=" * 70)
    print(f"TRAINING GBERT FOLD {fold_idx}/3 (Train: {len(train_fold_examples):,} | Val: {len(val_fold_examples):,})")
    print("=" * 70)

    # Check free disk space preflight
    free_gb = get_free_disk_space_gb()
    print(f"Drive C: Free Space: {free_gb:.2f} GB (Required threshold: >= {MIN_FREE_DISK_GB} GB)")
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(f"Halt triggered: Free disk space {free_gb:.2f} GB is below safety threshold {MIN_FREE_DISK_GB} GB!")

    set_random_seed(RANDOM_SEED)

    # Prepare datasets
    print("Preparing tokenized training and validation datasets...")
    train_dataset, _ = prepare_split(train_fold_examples, tokenizer, label2id)
    val_dataset, _ = prepare_split(val_fold_examples, tokenizer, label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    # Build Model
    print(f"Loading {MODEL_NAME} for token classification ({len(label2id)} classes)...")
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        local_files_only=True,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    epoch_logs = []
    fold_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch_dev)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if step % 200 == 0 or step == len(train_loader):
                elapsed = time.time() - epoch_start_time
                print(f"  Epoch {epoch}/{EPOCHS} | Step {step}/{len(train_loader)} | Train Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")

        avg_train_loss = running_train_loss / len(train_loader)

        # Validation Loss
        val_loss = evaluate_loss(model, val_loader, device)

        # Validation Entity Evaluation
        val_eval = evaluate_model_entities(
            model=model,
            dataset=val_dataset,
            examples=val_fold_examples,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            batch_size=VAL_BATCH_SIZE,
        )

        metrics = val_eval["metrics"]
        f02_score = metrics["final_score"]
        cat1_score = metrics["categories"][1]["score"]
        cat2_score = metrics["categories"][2]["score"]

        # Calculate micro precision and recall
        tp_total = sum(asp["tp"] for cat_res in metrics["categories"].values() for asp in cat_res["aspects"].values())
        fp_total = sum(asp["fp"] for cat_res in metrics["categories"].values() for asp in cat_res["aspects"].values())
        fn_total = sum(asp["fn"] for cat_res in metrics["categories"].values() for asp in cat_res["aspects"].values())
        precision = tp_total / max(1, tp_total + fp_total)
        recall = tp_total / max(1, tp_total + fn_total)

        epoch_duration = time.time() - epoch_start_time

        print(f"\n>>> Epoch {epoch}/{EPOCHS} Summary (Fold {fold_idx}):")
        print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    Val F0.2:   {f02_score:.6f} (Cat 1: {cat1_score:.6f} | Cat 2: {cat2_score:.6f})")
        print(f"    Precision:  {precision:.4f} | Recall: {recall:.4f} (TP: {tp_total}, FP: {fp_total}, FN: {fn_total})")
        print(f"    Duration:   {epoch_duration:.1f}s")

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "f02": f02_score,
            "cat1_f02": cat1_score,
            "cat2_f02": cat2_score,
            "precision": precision,
            "recall": recall,
            "duration_seconds": epoch_duration,
        })

    # Save final Epoch 3 model per fixed 3-epoch pilot policy
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving Epoch 3 checkpoint to {checkpoint_dir}...")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Collect out-of-fold active token probabilities
    print("Collecting out-of-fold active token probabilities...")
    oof_probs = collect_fold_active_probabilities(model, val_loader, device)
    oof_cache_path = cache_dir / f"oof_probs_gbert_fold{fold_idx}.npy"
    flat_oof_probs = np.concatenate(oof_probs, axis=0)
    np.save(oof_cache_path, flat_oof_probs)
    print(f"Saved {len(oof_probs)} OOF arrays ({flat_oof_probs.shape[0]} tokens) to {oof_cache_path}")

    # Memory cleanup
    del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_total_time = time.time() - fold_start_time
    free_gb_after = get_free_disk_space_gb()
    print(f"Fold {fold_idx} complete in {fold_total_time:.1f}s ({fold_total_time/60:.2f} min). Free disk: {free_gb_after:.2f} GB.")

    return {
        "fold": fold_idx,
        "checkpoint_dir": str(checkpoint_dir),
        "total_duration_seconds": fold_total_time,
        "epochs": epoch_logs,
        "final_f02": epoch_logs[-1]["f02"],
        "free_disk_after_gb": free_gb_after,
    }


def main():
    print("=" * 75)
    print("PHASE 29B: THREE-FOLD GBERT SEQUENTIAL TRAINING PIPELINE")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")

    # 1. Load Data and Construct Folds
    print("\nLoading training data and creating Split A...")
    df = load_training_data()
    examples = build_examples(df)
    train_examples, val_examples = split_examples(examples, random_state=42)

    label2id, id2label = create_label_mappings(examples)
    print(f"Total labeled records: {len(examples)} (Train: {len(train_examples)}, Holdout: {len(val_examples)})")
    print(f"BIO Label Space: {len(label2id)} classes ('O' ID = {label2id['O']})")

    # Construct 3 Stratified & Grouped Folds
    print("\nPartitioning 4,000 training titles into 3 category-stratified folds (grouping near-duplicates)...")
    folds = get_cv_folds(train_examples, n_splits=3, random_state=42)

    # Verify Fold Integrity BEFORE Training
    print("Running mathematical partition integrity verification...")
    integrity_report = verify_fold_integrity(folds, val_examples)
    assert integrity_report["all_checks_passed"], "Partition integrity check failed!"
    print("Partition integrity 100% verified!")
    for s in integrity_report["fold_summaries"]:
        print(f"  Fold {s['fold']}: Train = {s['train_size']} | Val = {s['val_size']} | Cat Counts = {s['cat_counts']} | Grouped Pair = {s['contains_grouped_pair']}")

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    overall_start = time.time()
    training_history = []

    # 2. Sequential Fold Training
    for fold_idx in range(3):
        train_fold, val_fold = folds[fold_idx]
        ckpt_dir = FOLD_CHECKPOINT_DIRS[fold_idx]

        fold_res = train_single_fold(
            fold_idx=fold_idx,
            train_fold_examples=train_fold,
            val_fold_examples=val_fold,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            device=device,
            checkpoint_dir=ckpt_dir,
            cache_dir=cache_dir,
        )
        training_history.append(fold_res)

    total_training_time = time.time() - overall_start
    final_free_disk = get_free_disk_space_gb()

    # Save summary report
    history_file = cache_dir / "gbert_cv3_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_training_time_seconds": total_training_time,
            "final_free_disk_gb": final_free_disk,
            "folds": training_history,
        }, f, indent=2)

    print("\n" + "=" * 75)
    print("3-FOLD GBERT TRAINING COMPLETE")
    print("=" * 75)
    print(f"Total Sequential Training Time: {total_training_time:.1f}s ({total_training_time/60:.2f} min)")
    print(f"Final Drive C: Free Space:     {final_free_disk:.2f} GB")
    for res in training_history:
        print(f"  Fold {res['fold']}: Checkpoint = {res['checkpoint_dir']} | Epoch 3 Val F0.2 = {res['final_f02']:.6f}")


if __name__ == "__main__":
    main()
