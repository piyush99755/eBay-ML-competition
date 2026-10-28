"""
Phase 33B: Stage C — Matched Supervised Fine-Tuning of DAPT GBERT
Fine-tunes the domain-adapted encoder (checkpoints/phase33b_dapt_encoder/)
on the exact 3,200-title internal training partition under identical conditions
to the Phase 32C matched baseline.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_internal_split import get_internal_split
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.evaluate_model import evaluate_model_entities

DAPT_ENCODER_PATH = PROJECT_ROOT / "checkpoints" / "phase33b_dapt_encoder"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "phase33b_dapt_gbert_best"
METRICS_PATH = PROJECT_ROOT / "data" / "cache" / "phase33b_supervised_training_metrics.json"

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_SEED = 42
MIN_FREE_DISK_GB = 4.5


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_free_disk_space_gb() -> float:
    _, _, free_bytes = shutil.disk_usage(PROJECT_ROOT)
    return free_bytes / (1024 ** 3)


def evaluate_dev_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                token_type_ids=batch_dev.get("token_type_ids"),
            )
            logits = outputs.logits
            labels = batch_dev["labels"]
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def train_dapt_supervised(
    encoder_path: Path = DAPT_ENCODER_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    print("=" * 75)
    print("PHASE 33B: STAGE C — MATCHED SUPERVISED FINE-TUNING")
    print(f"Base Adapted Encoder: {encoder_path}")
    print(f"Output Best Checkpoint: {output_dir}")
    print(f"Training Budget: 3 Epochs | Batch Size: {TRAIN_BATCH_SIZE} | LR: {LEARNING_RATE}")
    print("=" * 75)

    # 1. Disk check
    free_disk = get_free_disk_space_gb()
    print(f"Drive C: Free Space: {free_disk:.2f} GB (Floor >= {MIN_FREE_DISK_GB} GB)")
    if free_disk < MIN_FREE_DISK_GB:
        raise RuntimeError(f"ABORT: Free disk {free_disk:.2f} GB below floor {MIN_FREE_DISK_GB} GB!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    set_random_seed(RANDOM_SEED)

    # 2. Load exact 3,200/800 internal partition
    print("\nLoading training data and internal 3,200/800 partition...")
    df = load_training_data()
    all_examples = build_examples(df)

    # Stratified 80/20 train/validation split (matching champion Split-A)
    train_4k, val_1k = split_examples(all_examples, random_state=RANDOM_SEED)
    train_examples, dev_examples, split_manifest = get_internal_split(
        train_4k, random_state=RANDOM_SEED
    )
    print(f"Internal Train Examples: {len(train_examples):,} titles")
    print(f"Internal Dev Examples:   {len(dev_examples):,} titles")

    # 3. Label Mapping from Champion Config
    ref_cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    label2id = {str(k): int(v) for k, v in ref_cfg.label2id.items()}
    id2label = {int(k): str(v) for k, v in ref_cfg.id2label.items()}
    print(f"Label Space: {len(label2id)} classes ('O' ID = {label2id['O']})")

    # 4. Tokenizer & Dataset Preparation
    print(f"\nLoading tokenizer from {encoder_path}...")
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)

    print("Tokenizing and aligning datasets...")
    train_dataset, _ = prepare_split(train_examples, tokenizer, label2id)
    dev_dataset, _ = prepare_split(dev_examples, tokenizer, label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    # 5. Instantiate Model from DAPT Encoder
    print(f"\nInstantiating AutoModelForTokenClassification from {encoder_path}...")
    config = AutoConfig.from_pretrained(
        encoder_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        encoder_path,
        config=config,
        local_files_only=True,
    )
    model.to(device)

    # 6. Loss Function: Uniform 1.00x weight across all classes (Matched Baseline)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 7. Training & Evaluation Loop
    best_dev_f02 = -1.0
    best_epoch = 0
    training_history = []
    total_start_time = time.time()
    peak_vram_mb = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0

        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        for step, batch in enumerate(train_loader, start=1):
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                token_type_ids=batch_dev.get("token_type_ids"),
            )
            logits = outputs.logits
            labels = batch_dev["labels"]
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if step % 200 == 0 or step == len(train_loader):
                if torch.cuda.is_available():
                    curr_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    peak_vram_mb = max(peak_vram_mb, curr_vram)
                print(f"  Step {step:4d}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)
        avg_dev_loss = evaluate_dev_loss(model, dev_loader, loss_fn, device)

        # Evaluate entity-level competition metrics on internal dev set
        dev_eval_output = evaluate_model_entities(
            model=model,
            dataset=dev_dataset,
            examples=dev_examples,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            batch_size=VAL_BATCH_SIZE,
        )

        dev_metrics = dev_eval_output["metrics"]
        dev_f02 = dev_metrics["final_score"]
        cat1_f02 = dev_metrics["categories"][1]["score"]
        cat2_f02 = dev_metrics["categories"][2]["score"]

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} Complete ({epoch_duration:.1f}s):")
        print(f"  Train Loss:     {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f}")
        print(f"  Dev Overall:    F0.2 = {dev_f02:.6f}")
        print(f"  Dev Category 1: F0.2 = {cat1_f02:.6f} | Category 2: F0.2 = {cat2_f02:.6f}")

        is_best = dev_f02 > best_dev_f02
        if is_best:
            best_dev_f02 = dev_f02
            best_epoch = epoch
            print(f"  >>> NEW BEST CHECKPOINT: F0.2 = {best_dev_f02:.6f} (Epoch {epoch}) <<<")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "dev_loss": round(avg_dev_loss, 4),
            "dev_f02": round(dev_f02, 6),
            "cat1_f02": round(cat1_f02, 6),
            "cat2_f02": round(cat2_f02, 6),
            "epoch_duration_seconds": round(epoch_duration, 1),
            "is_best": is_best,
        })

    total_training_time = time.time() - total_start_time
    print("\n" + "=" * 75)
    print("STAGE C TRAINING COMPLETE")
    print(f"Best Dev F0.2:      {best_dev_f02:.6f} (Epoch {best_epoch})")
    print(f"Reference Baseline: 0.866940 (Phase 32C)")
    print(f"Delta:              {best_dev_f02 - 0.866940:+.6f}")
    print(f"Total Runtime:      {total_training_time:.1f}s ({total_training_time/60:.2f}m)")
    print(f"Peak VRAM:          {peak_vram_mb:.1f} MB")
    print(f"Checkpoint Saved:   {output_dir}")
    print("=" * 75)

    summary = {
        "model_type": "dapt_gbert_challenger",
        "encoder_source": str(encoder_path),
        "best_epoch": best_epoch,
        "best_dev_f02": best_dev_f02,
        "reference_baseline_dev_f02": 0.866940,
        "delta_vs_baseline": round(best_dev_f02 - 0.866940, 6),
        "training_history": training_history,
        "total_runtime_seconds": round(total_training_time, 1),
        "total_runtime_minutes": round(total_training_time / 60, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "output_dir": str(output_dir),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", type=Path, default=DAPT_ENCODER_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    train_dapt_supervised(encoder_path=args.encoder_path, output_dir=args.output_dir)
