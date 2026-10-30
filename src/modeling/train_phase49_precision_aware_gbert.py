"""
Phase 49: Precision-Aware Fine-Tuning of DAPT GBERT.

Objective:
  Train a precision-aware GBERT challenger using Cost-Sensitive Token Classification
  (Asymmetric O-class Precision Penalization: w_O = 1.5x) on the deterministic 3,200-title
  internal training partition, evaluated and selected strictly on the 800-title internal
  dev partition using the official hierarchical category-macro F0.2 metric.

Strict Validation & Resource Integrity:
  - No Quiz or Test labels.
  - The 1,000-title validation set (and 489 clean holdout) is STRICTLY EXCLUDED during training.
  - Checkpoint selection criterion: Official hierarchical category-macro F0.2 on 800 internal dev.
  - Architecture: Initialized from checkpoints/phase33b_dapt_encoder.
  - Hyperparameters: AdamW (lr=2e-5, weight_decay=0.01, linear warmup 10%), batch size 4, 3 epochs.
  - Safety: Disk floor >= 5.0 GB, max training time 2 hours.
  - Checkpoint output: checkpoints/gbert_phase49_precision_aware_best.
"""

import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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
    get_linear_schedule_with_warmup,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.data.make_internal_split import get_internal_split
from src.data.label_mapping import create_label_mappings
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.phase_35b_audit_and_error_analysis import extract_spanned_gold_entities
from src.evaluation.phase_39a_official_metric import (
    calculate_official_competition_metrics,
    calculate_global_micro_metrics,
)
from src.evaluation.phase_43_part_number_calibration import (
    decode_spanned_predictions_with_cat_thresholds,
    PHASE40_THRESHOLDS,
    DEFAULT_CONF_THRESH,
    verify_champion,
)

# Constants & Paths
DAPT_ENCODER_PATH = PROJECT_ROOT / "checkpoints" / "phase33b_dapt_encoder"
OUTPUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "gbert_phase49_precision_aware_best"
TRAINING_METRICS_PATH = PROJECT_ROOT / "data" / "cache" / "phase49_training_history.json"

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
WARMUP_RATIO = 0.10
RANDOM_SEED = 42
MIN_FREE_DISK_GB = 5.0

# Cost-Sensitive Precision Penalty: w_O = 1.50x
O_CLASS_WEIGHT = 1.50


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_resources() -> Tuple[float, torch.device]:
    _, _, free_bytes = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free_bytes / (1024 ** 3)
    print(f"Free Disk Space: {free_gb:.2f} GB (Requirement: >= {MIN_FREE_DISK_GB:.2f} GB)")
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(f"ABORT: Free disk {free_gb:.2f} GB below safety floor {MIN_FREE_DISK_GB} GB!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device:  {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Model:       {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")
    return free_gb, device


def evaluate_internal_dev(
    model: nn.Module,
    dev_dataset,
    dev_examples: List[dict],
    gold_tuples: List[List[Tuple[str, str]]],
    id2label: Dict[int, str],
    device: torch.device,
    loss_fn: nn.Module,
    batch_size: int = VAL_BATCH_SIZE,
) -> Dict[str, Any]:
    """Evaluate dev loss and official hierarchical category-macro F0.2 on the 800 internal dev set."""
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(DAPT_ENCODER_PATH, local_files_only=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    total_loss = 0.0
    total_batches = 0
    all_active_probs = []

    with torch.no_grad():
        ex_idx_global = 0
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

            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            for row_idx in range(labels.shape[0]):
                ex = dev_examples[ex_idx_global]
                ex_idx_global += 1

                row_labels = labels[row_idx].cpu().numpy()
                active_mask = (row_labels != -100)
                active_probs = batch_probs[row_idx, active_mask]

                assert len(active_probs) == len(ex["tokens"])
                all_active_probs.append(active_probs)

    avg_dev_loss = total_loss / max(1, total_batches)

    # Decode predictions under Phase 40 confidence thresholds
    pred_entities = decode_spanned_predictions_with_cat_thresholds(
        all_active_probs,
        examples=dev_examples,
        id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH,
        aspect_thresholds=PHASE40_THRESHOLDS,
    )
    pred_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in pred_entities]

    records = [
        {"category": dev_examples[i]["category"], "predicted_entities": pred_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(dev_examples))
    ]

    off_metrics = calculate_official_competition_metrics(records)
    mic_metrics = calculate_global_micro_metrics(records)

    return {
        "dev_loss": avg_dev_loss,
        "official_score": off_metrics["official_score"],
        "cat1_score": off_metrics["categories"][1]["score"],
        "cat2_score": off_metrics["categories"][2]["score"],
        "tp": mic_metrics["tp"],
        "fp": mic_metrics["fp"],
        "fn": mic_metrics["fn"],
        "predictions": pred_tuples,
        "active_probs": all_active_probs,
    }


def train_phase49_precision_aware_gbert() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 49: PRECISION-AWARE DAPT GBERT TRAINING")
    print("=" * 80)

    # 1. Preflight Safety & Champion Verification
    print("\n[Step 1] Auditing resources and champion baseline...")
    free_disk, device = check_resources()
    champ_info = verify_champion()
    print(f"Phase 40 Champion Hash: {champ_info['sha256']} (VERIFIED INTACT)")
    assert DAPT_ENCODER_PATH.exists(), f"Missing DAPT encoder: {DAPT_ENCODER_PATH}"

    set_seed(RANDOM_SEED)

    # 2. Load Data and Internal 3,200 / 800 Split
    print("\n[Step 2] Loading data and establishing internal 3,200 / 800 partition...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=RANDOM_SEED)
    train_3200, dev_800, manifest = get_internal_split(train_4k, random_state=RANDOM_SEED)

    print(f"Total Supervised Titles:     {len(all_examples)}")
    print(f"Internal Training Partition: {len(train_3200)} titles (1,600 Cat 1, 1,600 Cat 2)")
    print(f"Internal Dev Partition:      {len(dev_800)} titles (400 Cat 1, 400 Cat 2)")
    print(f"Validation Holdout Set:      {len(val_1k)} titles (100% ISOLATED)")

    # Gold entities for internal dev evaluation
    dev_gold_spanned = extract_spanned_gold_entities(dev_800)
    dev_gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in dev_gold_spanned]

    # 3. Label Space & Tokenizer
    print("\n[Step 3] Loading tokenizer and label mapping...")
    ref_cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    label2id = {str(k): int(v) for k, v in ref_cfg.label2id.items()}
    id2label = {int(k): str(v) for k, v in ref_cfg.id2label.items()}
    o_label_id = label2id["O"]
    print(f"Label Classes: {len(label2id)} classes ('O' ID = {o_label_id})")

    tokenizer = AutoTokenizer.from_pretrained(DAPT_ENCODER_PATH, local_files_only=True)

    print("Tokenizing and aligning internal train and dev datasets...")
    train_dataset, _ = prepare_split(train_3200, tokenizer, label2id)
    dev_dataset, _ = prepare_split(dev_800, tokenizer, label2id)

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Model Setup with Cost-Sensitive Token Loss
    print(f"\n[Step 4] Instantiating GBERT token classifier from {DAPT_ENCODER_PATH}...")
    config = AutoConfig.from_pretrained(
        DAPT_ENCODER_PATH,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        DAPT_ENCODER_PATH,
        config=config,
        local_files_only=True,
    )
    model.to(device)

    # Cost-Sensitive Loss: Assign w_O = 1.50x to penalize false-positive transitions from 'O'
    class_weights = torch.ones(len(label2id), dtype=torch.float32, device=device)
    class_weights[o_label_id] = O_CLASS_WEIGHT
    print(f"Loss Function: Cost-Sensitive CrossEntropyLoss with w_O = {O_CLASS_WEIGHT}x (all entity labels = 1.0x)")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Optimizer: AdamW | LR: {LEARNING_RATE} | Total Steps: {total_steps} | Warmup Steps: {warmup_steps}")

    # 5. Training Loop with Official Category-Macro F0.2 Checkpoint Selection
    print("\n" + "=" * 60)
    print("BEGINNING 3-EPOCH PRECISION-AWARE TRAINING")
    print("=" * 60)

    best_dev_f02 = -1.0
    best_epoch = 0
    training_history = []
    t_start_train = time.time()
    peak_vram_mb = 0.0

    OUTPUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t_epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        model.train()
        running_loss = 0.0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 200 == 0 or step == len(train_loader):
                if torch.cuda.is_available():
                    curr_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    peak_vram_mb = max(peak_vram_mb, curr_vram)
                lr_curr = scheduler.get_last_lr()[0]
                print(f"  Step {step:4d}/{len(train_loader)} | Batch Loss: {loss.item():.4f} | LR: {lr_curr:.2e}")

        avg_train_loss = running_loss / len(train_loader)
        epoch_train_time = time.time() - t_epoch_start

        # Internal Dev evaluation
        print(f"Evaluating official category-macro F0.2 on 800 internal dev titles...")
        dev_res = evaluate_internal_dev(
            model=model,
            dev_dataset=dev_dataset,
            dev_examples=dev_800,
            gold_tuples=dev_gold_tuples,
            id2label=id2label,
            device=device,
            loss_fn=loss_fn,
            batch_size=VAL_BATCH_SIZE,
        )

        dev_loss = dev_res["dev_loss"]
        dev_score = dev_res["official_score"]
        cat1_score = dev_res["cat1_score"]
        cat2_score = dev_res["cat2_score"]
        tp = dev_res["tp"]
        fp = dev_res["fp"]
        fn = dev_res["fn"]

        is_best = dev_score > best_dev_f02
        if is_best:
            best_dev_f02 = dev_score
            best_epoch = epoch
            print(f"--> New Best Epoch {epoch}! Internal Dev Official F0.2: {dev_score:.6f}. Saving checkpoint...")
            model.save_pretrained(OUTPUT_CHECKPOINT_DIR)
            tokenizer.save_pretrained(OUTPUT_CHECKPOINT_DIR)

        epoch_total_time = time.time() - t_epoch_start
        print(f"Epoch {epoch} Summary ({epoch_total_time:.1f}s):")
        print(f"  Train Loss:     {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
        print(f"  Official F0.2:  {dev_score:.6f} (Cat 1: {cat1_score:.6f}, Cat 2: {cat2_score:.6f})")
        print(f"  Micro Bookkeeping: TP={tp}, FP={fp}, FN={fn}")

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "official_score": dev_score,
            "cat1_score": cat1_score,
            "cat2_score": cat2_score,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "epoch_duration_seconds": epoch_total_time,
            "is_best": is_best,
        })

    total_training_time = time.time() - t_start_train
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s ({total_training_time / 60:.2f}m)")
    print(f"Best Epoch: {best_epoch} with Internal Dev Official F0.2 = {best_dev_f02:.6f}")
    print(f"Peak VRAM:  {peak_vram_mb:.1f} MB")
    print(f"Saved Checkpoint: {OUTPUT_CHECKPOINT_DIR}")
    print("=" * 60)

    # 6. Postflight Checkpoint Verification
    assert (OUTPUT_CHECKPOINT_DIR / "model.safetensors").exists(), "Missing model.safetensors in output dir!"
    ckpt_size_mb = (OUTPUT_CHECKPOINT_DIR / "model.safetensors").stat().st_size / (1024 ** 2)
    print(f"Saved model.safetensors size: {ckpt_size_mb:.1f} MB")

    final_free_disk, _ = check_resources()
    print(f"Final Free Disk Space: {final_free_disk:.2f} GB (Remains >= {MIN_FREE_DISK_GB} GB)")

    results = {
        "phase": "49",
        "model_name": "gbert_phase49_precision_aware_best",
        "technique": "Cost-Sensitive Token Classification (w_O = 1.50x)",
        "encoder_source": str(DAPT_ENCODER_PATH),
        "output_dir": str(OUTPUT_CHECKPOINT_DIR),
        "total_training_time_seconds": total_training_time,
        "total_training_time_minutes": total_training_time / 60,
        "peak_vram_mb": peak_vram_mb,
        "best_epoch": best_epoch,
        "best_dev_official_f02": best_dev_f02,
        "training_history": training_history,
        "hyperparameters": {
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "random_seed": RANDOM_SEED,
            "o_class_weight": O_CLASS_WEIGHT,
            "train_size": len(train_3200),
            "dev_size": len(dev_800),
        },
    }

    # Serialization helper
    def to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [to_serializable(i) for i in obj]
        return obj

    clean_results = to_serializable(results)
    TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"Training metrics saved to {TRAINING_METRICS_PATH}")

    return results


if __name__ == "__main__":
    train_phase49_precision_aware_gbert()
