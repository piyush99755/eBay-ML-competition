"""
Phase 25B: Training German ELECTRA Challenger (deepset/gelectra-base).

Trains ElectraForTokenClassification on Split A (4,000 train / 1,000 val)
using seed 42, 3 epochs, AdamW lr=2e-5, train batch size=4, val batch size=8.
Saves only the single best validation checkpoint to checkpoints/gelectra_baseline_best.
"""

from pathlib import Path
import random
import shutil
import time
import sys
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.evaluate_model import evaluate_model_entities
from src.evaluation.entity_metrics import count_matches, precision_recall_fbeta

GELECTRA_MODEL_NAME = "deepset/gelectra-base"
CHAMPION_GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
CHECKPOINT_DIR = Path("checkpoints/gelectra_baseline_best")

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_SEED = 42
MIN_DISK_HEADROOM_GB = 5.0


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_disk_space() -> float:
    total, used, free = shutil.disk_usage("C:")
    free_gb = free / (1024 ** 3)
    if free_gb < MIN_DISK_HEADROOM_GB:
        raise RuntimeError(
            f"Disk space critical: only {free_gb:.2f} GB free on C: (threshold: {MIN_DISK_HEADROOM_GB} GB). Aborting."
        )
    return free_gb


def evaluate_loss(model, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1
    return total_loss / max(1, total_batches)


def compute_overall_pr_fbeta(records: list[dict], beta: float = 0.2) -> tuple[float, float, float]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for r in records:
        pred = r["predicted_entities"]
        true = r["true_entities"]
        tp, fp, fn = count_matches(pred, true)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    metrics = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=beta)
    return metrics["precision"], metrics["recall"], metrics["f_beta"]


def main():
    print("=" * 60)
    print("PHASE 25B: TRAINING GERMAN ELECTRA CHALLENGER")
    print(f"Model: {GELECTRA_MODEL_NAME}")
    print(f"Target Checkpoint: {CHECKPOINT_DIR}")
    print("=" * 60)

    # 1. Disk & GPU Check
    free_gb = check_disk_space()
    print(f"Drive C: Free Space: {free_gb:.2f} GB (Safety Threshold: {MIN_DISK_HEADROOM_GB} GB)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats(0)

    # 2. Seed
    set_random_seed(RANDOM_SEED)

    # 3. Label Mapping from Reference Champion
    ref_cfg = AutoConfig.from_pretrained(CHAMPION_GBERT_CHECKPOINT)
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    print(f"Loaded {num_labels} BIO label mappings from reference champion.")

    # 4. Load Data & Create Split A
    print("\nLoading dataset and creating Split A (seed 42)...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_examples, val_examples = split_examples(all_examples, random_state=RANDOM_SEED)
    print(f"  Training titles:   {len(train_examples):,}")
    print(f"  Validation titles: {len(val_examples):,}")

    # 5. Tokenizer & Dataset Preparation
    print(f"\nInitializing tokenizer from {GELECTRA_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(GELECTRA_MODEL_NAME)

    print("Tokenizing and aligning training split...")
    train_dataset, train_stats = prepare_split(train_examples, tokenizer, label2id)
    print(f"  Train active labels: {train_stats['active_labels']:,}, max length: {train_stats['max_transformer_length']}")

    print("Tokenizing and aligning validation split...")
    val_dataset, val_stats = prepare_split(val_examples, tokenizer, label2id)
    print(f"  Val active labels: {val_stats['active_labels']:,}, max length: {val_stats['max_transformer_length']}")

    # 6. DataLoaders
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    # 7. Model Initialization
    print(f"\nBuilding ElectraForTokenClassification from {GELECTRA_MODEL_NAME}...")
    model = AutoModelForTokenClassification.from_pretrained(
        GELECTRA_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # 8. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    print(f"\nStarting training: {EPOCHS} epochs, {len(train_dataloader)} batches/epoch, lr={LEARNING_RATE}")
    best_f_beta = -1.0
    best_epoch = -1
    history = []

    overall_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*25} EPOCH {epoch}/{EPOCHS} {'='*25}")
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss_total = 0.0
        for step, batch in enumerate(train_dataloader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss_total += loss.item()

            if step % 250 == 0 or step == len(train_dataloader):
                elapsed = time.time() - epoch_start_time
                print(f"  Step {step}/{len(train_dataloader)} - Loss: {train_loss_total / step:.4f} ({elapsed:.1f}s)")

        train_loss_avg = train_loss_total / len(train_dataloader)
        training_time = time.time() - epoch_start_time

        # Validation Loss
        val_loss = evaluate_loss(model, val_dataloader, device)

        # Entity Metrics Evaluation
        eval_output = evaluate_model_entities(
            model=model,
            dataset=val_dataset,
            examples=val_examples,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            batch_size=VAL_BATCH_SIZE,
        )
        entity_results = eval_output["metrics"]
        val_f02 = entity_results["final_score"]
        overall_p, overall_r, overall_f02 = compute_overall_pr_fbeta(eval_output["records"], beta=0.2)

        peak_vram_gb = (
            torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        cat1_score = entity_results["categories"].get(1, {}).get("score", 0.0)
        cat2_score = entity_results["categories"].get(2, {}).get("score", 0.0)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "val_loss": val_loss,
            "f02": val_f02,
            "cat1_f02": cat1_score,
            "cat2_f02": cat2_score,
            "precision": overall_p,
            "recall": overall_r,
            "training_time": round(training_time, 1),
            "peak_vram_gb": round(peak_vram_gb, 2),
        }
        history.append(epoch_stats)

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"  Train Loss:       {train_loss_avg:.4f}")
        print(f"  Val Loss:         {val_loss:.4f}")
        print(f"  Official F0.2:    {val_f02:.6f} (Cat 1: {cat1_score:.6f}, Cat 2: {cat2_score:.6f})")
        print(f"  Micro Precision:  {overall_p:.6f}")
        print(f"  Micro Recall:     {overall_r:.6f}")
        print(f"  Training Time:    {training_time:.1f}s")
        print(f"  Peak GPU VRAM:    {peak_vram_gb:.2f} GB")

        # Save Best Checkpoint
        if val_f02 > best_f_beta:
            best_f_beta = val_f02
            best_epoch = epoch
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            print(f"  >>> SAVED NEW BEST CHECKPOINT (Epoch {best_epoch}, F0.2 = {best_f_beta:.6f}) to {CHECKPOINT_DIR} <<<")

        # Disk space monitoring
        post_epoch_free = check_disk_space()
        print(f"  Post-Epoch C: Free Space: {post_epoch_free:.2f} GB")

    total_time = time.time() - overall_start_time

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY: GERMAN ELECTRA CHALLENGER")
    print("=" * 60)
    print(f"Best Validation Epoch: {best_epoch}")
    print(f"Best Local F0.2:       {best_f_beta:.6f}")
    print(f"Total Time:            {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"Checkpoint Location:   {CHECKPOINT_DIR}")

    print("\nEpoch History Table:")
    print(f"{'Epoch':<6}{'TrainLoss':<11}{'ValLoss':<10}{'F0.2':<12}{'Cat1':<10}{'Cat2':<10}{'Prec':<10}{'Rec':<10}{'Time':<8}{'VRAM':<8}")
    for h in history:
        print(
            f"{h['epoch']:<6}{h['train_loss']:<11.4f}{h['val_loss']:<10.4f}{h['f02']:<12.6f}"
            f"{h['cat1_f02']:<10.6f}{h['cat2_f02']:<10.6f}{h['precision']:<10.4f}{h['recall']:<10.4f}"
            f"{h['training_time']:<8.1f}{h['peak_vram_gb']:<8.2f}"
        )

    return history


if __name__ == "__main__":
    main()
