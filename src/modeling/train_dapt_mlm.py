"""
Phase 33B: Stage B — Controlled Domain-Adaptive Pretraining (DAPT)
Masked Language Modeling on 50,000 German automotive listing titles.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sample_dapt_corpus import sample_dapt_corpus

MODEL_NAME = "deepset/gbert-base"
SAMPLE_SIZE_PER_CATEGORY = 25000  # Total 50,000 titles
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 16
GRAD_ACCUM = 1  # Effective batch size = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
EPOCHS = 2
WARMUP_RATIO = 0.06
MLM_PROBABILITY = 0.15
SEED = 42
MIN_FREE_DISK_GB = 4.5

OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "phase33b_dapt_encoder"
METRICS_PATH = PROJECT_ROOT / "data" / "cache" / "dapt_training_metrics.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_free_disk_gb() -> float:
    _, _, free_bytes = shutil.disk_usage(PROJECT_ROOT)
    return free_bytes / (1024 ** 3)


class PretokenizedMLMDataset(Dataset):
    """Pre-tokenized dataset for high-throughput MLM training."""

    def __init__(self, examples: list, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        print(f"Pre-tokenizing {len(examples):,} listing titles for MLM...")
        t0 = time.time()
        self.items = []
        for ex in examples:
            enc = tokenizer(
                ex["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "special_tokens_mask": enc["special_tokens_mask"],
            })
        print(f"Pre-tokenization complete in {time.time() - t0:.2f}s ({len(self.items):,} examples).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_dapt():
    print("=" * 72)
    print("PHASE 33B: STAGE B — CONTROLLED DOMAIN-ADAPTIVE PRETRAINING (DAPT)")
    print("=" * 72)

    set_seed(SEED)

    # 1. Resource Pre-Check
    free_disk_start = get_free_disk_gb()
    print(f"Drive C: Free Space: {free_disk_start:.2f} GB (Strict Floor >= {MIN_FREE_DISK_GB} GB)")
    if free_disk_start < MIN_FREE_DISK_GB:
        raise RuntimeError(f"ABORT: Free disk space {free_disk_start:.2f} GB is below minimum safety floor {MIN_FREE_DISK_GB} GB!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats(0)

    # 2. Sample 50,000 Unlabeled Automotive Titles
    print("\n--- [Step 1/4] Sampling 50,000 Cleaned Automotive Titles ---")
    dapt_examples, sampling_stats = sample_dapt_corpus(
        sample_size_per_category=SAMPLE_SIZE_PER_CATEGORY,
        random_seed=SEED,
    )
    print(f"Sampled {len(dapt_examples):,} titles (Cat 1: {sampling_stats['sample_size_cat1']:,}, Cat 2: {sampling_stats['sample_size_cat2']:,})")
    print(f"Total lines scanned: {sampling_stats['total_lines_scanned']:,}")
    print(f"Skipped <= 30000:    {sampling_stats['skipped_due_to_record_id_le_30000']:,}")
    print(f"Skipped eval match:  {sampling_stats['skipped_due_to_normalized_eval_title_match']}")

    # 3. Tokenizer & Dataset
    print("\n--- [Step 2/4] Tokenization & DataLoader Setup ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    dataset = PretokenizedMLMDataset(dapt_examples, tokenizer, max_length=MAX_SEQ_LENGTH)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Model & Optimizer Setup
    print("\n--- [Step 3/4] Initializing Masked Language Model ---")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.to(device)
    model.train()

    total_steps = (len(dataloader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print(f"Model parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total steps:         {total_steps:,} (Warmup: {warmup_steps:,} steps)")
    print(f"Batch size:          {BATCH_SIZE} (GradAccum: {GRAD_ACCUM}, Effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"Learning rate:       {LEARNING_RATE} (Linear decay to 0)")
    print(f"Epochs:              {EPOCHS}")

    # 5. Training Loop
    print("\n--- [Step 4/4] Executing DAPT Training ---")
    overall_start_time = time.time()
    epoch_metrics = []
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        print(f"\n>>>>> Starting Epoch {epoch}/{EPOCHS} <<<<<")

        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / GRAD_ACCUM

            if not torch.isfinite(loss):
                raise RuntimeError(f"ABORT: Non-finite MLM loss detected at step {step}: {loss.item()}")

            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM
            step_count += 1

            if step % GRAD_ACCUM == 0 or step == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if step % 500 == 0 or step == len(dataloader):
                elapsed_epoch = time.time() - epoch_start_time
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = running_loss / step_count
                speed = (step * BATCH_SIZE) / max(1.0, elapsed_epoch)
                print(
                    f"  Epoch {epoch} | Step {step:5d}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Speed: {speed:.1f} ex/s | Elapsed: {elapsed_epoch/60:.1f}m"
                )

        epoch_duration = time.time() - epoch_start_time
        epoch_avg_loss = running_loss / max(1, step_count)
        epoch_perplexity = math.exp(min(20.0, epoch_avg_loss))

        # Check disk space mid-training
        free_disk_now = get_free_disk_gb()
        print(f"\n--- Epoch {epoch} Complete ---")
        print(f"  Average MLM Loss: {epoch_avg_loss:.4f}")
        print(f"  Perplexity:       {epoch_perplexity:.2f}")
        print(f"  Epoch Duration:   {epoch_duration:.1f}s ({epoch_duration/60:.2f}m)")
        print(f"  Free Disk Space:  {free_disk_now:.2f} GB")

        if free_disk_now < MIN_FREE_DISK_GB:
            raise RuntimeError(f"ABORT: Disk space dropped below safety threshold {MIN_FREE_DISK_GB} GB!")

        epoch_metrics.append({
            "epoch": epoch,
            "avg_mlm_loss": round(epoch_avg_loss, 4),
            "perplexity": round(epoch_perplexity, 2),
            "duration_seconds": round(epoch_duration, 1),
            "free_disk_gb": round(free_disk_now, 2),
        })

    total_training_time = time.time() - overall_start_time

    # 6. Save Adapted Encoder
    print("\n==================================================")
    print(f"SAVING ADAPTED ENCODER TO {OUTPUT_DIR}")
    print("==================================================")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    saved_size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*")) / (1024 ** 2)
    print(f"Adapted encoder successfully saved ({saved_size_mb:.2f} MB).")

    if torch.cuda.is_available():
        peak_vram_alloc = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        peak_vram_res = torch.cuda.max_memory_reserved(0) / (1024 ** 3)
    else:
        peak_vram_alloc, peak_vram_res = 0.0, 0.0

    print(f"Peak VRAM Allocated: {peak_vram_alloc:.2f} GB")
    print(f"Peak VRAM Reserved:  {peak_vram_res:.2f} GB")
    print(f"Total DAPT Runtime:  {total_training_time:.1f}s ({total_training_time/60:.2f}m)")

    free_disk_final = get_free_disk_gb()
    print(f"Final Free Disk:     {free_disk_final:.2f} GB")

    # 7. Write training metrics
    summary = {
        "base_model": MODEL_NAME,
        "sample_size_total": len(dapt_examples),
        "sampling_stats": sampling_stats,
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_seq_length": MAX_SEQ_LENGTH,
            "mlm_probability": MLM_PROBABILITY,
            "seed": SEED,
        },
        "epoch_metrics": epoch_metrics,
        "total_runtime_seconds": round(total_training_time, 1),
        "total_runtime_minutes": round(total_training_time / 60, 2),
        "peak_vram_allocated_gb": round(peak_vram_alloc, 2),
        "peak_vram_reserved_gb": round(peak_vram_res, 2),
        "saved_checkpoint_size_mb": round(saved_size_mb, 2),
        "saved_checkpoint_path": str(OUTPUT_DIR),
        "free_disk_final_gb": round(free_disk_final, 2),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training metrics written to: {METRICS_PATH}")
    print("\n>>> STAGE B DAPT TRAINING COMPLETE <<<")

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    train_dapt()
