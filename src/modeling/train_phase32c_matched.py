"""
Phase 32C: Controlled I-Modell Loss-Weight Experiment Training Script.

Trains two matched GBERT models on the 3,200-title internal partition:
1. Baseline:   I-Modell weight = 1.00 (uniform weights across all classes)
2. Challenger: I-Modell weight = 1.15 (all other classes at 1.00)

Configuration (Identical across both):
- Base model: deepset/gbert-base (local_files_only=True)
- Seed: 42
- Epochs: 3
- Batch size: 4 (train), 8 (val)
- Learning rate: 2e-5 (AdamW)
- Max sequence length: 512
- Evaluation & Checkpoint Selection: Best internal dev F0.2 (800 titles)
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
from src.data.make_splits import split_examples
from src.data.make_internal_split import get_internal_split
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.evaluate_model import evaluate_model_entities

MODEL_NAME = "deepset/gbert-base"
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_SEED = 42
MIN_FREE_DISK_GB = 4.0

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def set_random_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_free_disk_space_gb() -> float:
    """Return free disk space on Drive C: in GB."""
    _, _, free_bytes = shutil.disk_usage(PROJECT_ROOT)
    return free_bytes / (1024 ** 3)


def create_phase32c_loss_weights(
    label2id: Dict[str, int],
    i_modell_weight: float = 1.00,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create class weight vector for CrossEntropyLoss.
    Applies i_modell_weight strictly to I-Kompatibles_Fahrzeug_Modell.
    """
    num_labels = len(label2id)
    weights = torch.ones(num_labels, dtype=torch.float32, device=device)

    target_aspect_i = "I-Kompatibles_Fahrzeug_Modell"
    if target_aspect_i in label2id:
        i_id = label2id[target_aspect_i]
        weights[i_id] = float(i_modell_weight)

    return weights


def evaluate_dev_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Calculate average loss on the evaluation dataloader."""
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


def train_matched_model(
    model_type: str,
    i_modell_weight: float,
    train_examples: List[dict],
    dev_examples: List[dict],
    tokenizer,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Train a single matched model (baseline or challenger)."""
    print("\n" + "=" * 75)
    print(f"STARTING PHASE 32C TRAINING: {model_type.upper()}")
    print(f"I-Modell Class Weight: {i_modell_weight:.2f}x | All Other Classes: 1.00x")
    print(f"Internal Train Size:   {len(train_examples):,} titles")
    print(f"Internal Dev Size:     {len(dev_examples):,} titles")
    print(f"Output Checkpoint:     {output_dir}")
    print("=" * 75)

    # 1. Preflight Disk Check
    free_gb_start = get_free_disk_space_gb()
    print(f"Drive C: Free Space Pre-Training: {free_gb_start:.2f} GB (Threshold >= {MIN_FREE_DISK_GB} GB)")
    if free_gb_start < MIN_FREE_DISK_GB:
        raise RuntimeError(f"HALT: Free disk space {free_gb_start:.2f} GB is below safety threshold {MIN_FREE_DISK_GB} GB!")

    # 2. Reset Seeds
    set_random_seed(RANDOM_SEED)

    # 3. Prepare Datasets & DataLoaders
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

    # 4. Instantiate Pretrained GBERT
    print(f"Instantiating {MODEL_NAME} with {len(label2id)} classes...")
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

    # 5. Loss Function with Class Weights
    class_weights = create_phase32c_loss_weights(label2id, i_modell_weight=i_modell_weight, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    i_id = label2id.get("I-Kompatibles_Fahrzeug_Modell")
    print(f"Configured CrossEntropyLoss with ignore_index=-100.")
    print(f"Label ID for I-Modell: {i_id} (Assigned weight: {class_weights[i_id].item():.4f})")

    # 6. Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 7. Training Loop
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

        print(f"Epoch {epoch} Summary ({epoch_duration:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f}")
        print(f"  Dev F0.2:   {dev_f02:.6f} (Cat 1: {cat1_f02:.6f} | Cat 2: {cat2_f02:.6f})")

        is_new_best = dev_f02 > best_dev_f02
        if is_new_best:
            best_dev_f02 = dev_f02
            best_epoch = epoch
            print(f"  >>> NEW BEST MODEL! Saving checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": avg_dev_loss,
            "dev_f02": dev_f02,
            "cat1_f02": cat1_f02,
            "cat2_f02": cat2_f02,
            "duration_seconds": epoch_duration,
            "is_best": is_new_best,
        }
        training_history.append(epoch_log)

    total_duration = time.time() - total_start_time
    free_gb_end = get_free_disk_space_gb()

    summary = {
        "model_type": model_type,
        "i_modell_weight": i_modell_weight,
        "best_epoch": best_epoch,
        "best_dev_f02": best_dev_f02,
        "total_duration_seconds": total_duration,
        "peak_vram_mb": peak_vram_mb,
        "free_gb_start": free_gb_start,
        "free_gb_end": free_gb_end,
        "disk_consumed_mb": (free_gb_start - free_gb_end) * 1024,
        "checkpoint_path": str(output_dir),
        "epochs": training_history,
    }

    print("\n" + "=" * 75)
    print(f"TRAINING COMPLETE: {model_type.upper()}")
    print(f"Best Dev F0.2: {best_dev_f02:.6f} (at Epoch {best_epoch})")
    print(f"Total Time:    {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Peak VRAM:     {peak_vram_mb:.1f} MB")
    print(f"Drive C: Free: {free_gb_end:.2f} GB")
    print("=" * 75)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 32C Matched Training")
    parser.add_argument("--mode", choices=["all", "baseline", "challenger"], default="all",
                        help="Which model(s) to train: all, baseline, or challenger")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Data and Internal Split
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)

    train_3200, dev_800, manifest = get_internal_split(train_4k, random_state=42)
    print(f"Loaded 4,000 Split-A training titles.")
    print(f"Partitioned into {len(train_3200)} Internal Train and {len(dev_800)} Internal Dev titles.")

    # 2. Tokenizer & Labels
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    ref_cfg = AutoConfig.from_pretrained(Path("checkpoints/gbert_baseline_best"))
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    results = {}
    history_file = CACHE_DIR / "phase_32c_training_history.json"
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    # Train Baseline
    if args.mode in ["all", "baseline"]:
        base_dir = CHECKPOINTS_DIR / "gbert_phase32c_baseline_best"
        res_base = train_matched_model(
            model_type="baseline",
            i_modell_weight=1.00,
            train_examples=train_3200,
            dev_examples=dev_800,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            device=device,
            output_dir=base_dir,
        )
        results["baseline"] = res_base

    # Train Challenger
    if args.mode in ["all", "challenger"]:
        chal_dir = CHECKPOINTS_DIR / "gbert_phase32c_challenger_best"
        res_chal = train_matched_model(
            model_type="challenger",
            i_modell_weight=1.15,
            train_examples=train_3200,
            dev_examples=dev_800,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            device=device,
            output_dir=chal_dir,
        )
        results["challenger"] = res_chal

    # Save training history
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining history saved to {history_file}")


if __name__ == "__main__":
    main()
