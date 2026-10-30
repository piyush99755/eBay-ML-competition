"""
Phase 48: Hard Example Mined GBERT Token Classifier Training.

Objective:
  Train a genuinely complementary GBERT token classifier initialized from
  the domain-adapted encoder (checkpoints/phase33b_dapt_encoder) using a
  principled mixture of standard supervised examples (70%) and mined hard
  examples (30%) identified via 3-fold OOF cross-validation.

Design Specifications:
  - Base Backbone: checkpoints/phase33b_dapt_encoder (domain-adapted German BERT)
  - Training Data: 4,000 Split-A training titles
  - Hard Examples: 1,858 difficult titles mined in data/cache/phase_48_hard_examples_manifest.json
  - Sampling Scheme: 70% standard supervised examples (2,800) + 30% hard examples (1,200) per epoch
  - Epochs: 3
  - Batch Size: 4 (train), 8 (val)
  - Learning Rate: 2e-5 (AdamW, weight decay 0.01, linear warmup 10%)
  - Validation: 1,000 Split-A holdout titles
  - Safety: Disk floor >= 5.0 GB, max training budget 4 hours, only best checkpoint kept
  - Output Checkpoint: checkpoints/gbert_phase48_hard_mined_best
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

# Repository root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.data.label_mapping import create_label_mappings
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.entity_metrics import (
    bio_to_entities_with_confidence,
    filter_entities_by_category,
    count_matches,
)

# Paths & Constants
DAPT_ENCODER_PATH = PROJECT_ROOT / "checkpoints" / "phase33b_dapt_encoder"
HARD_MANIFEST_PATH = PROJECT_ROOT / "data" / "cache" / "phase_48_hard_examples_manifest.json"
CHAMPION_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"

OUTPUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "gbert_phase48_hard_mined_best"
TRAINING_METRICS_PATH = PROJECT_ROOT / "data" / "cache" / "phase48_training_history.json"

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
WARMUP_RATIO = 0.10
RANDOM_SEED = 42
MIN_FREE_DISK_GB = 5.0

# Sampling mixture: 70% standard, 30% hard
TOTAL_EPOCH_EXAMPLES = 4000
STANDARD_SAMPLE_COUNT = 2800
HARD_SAMPLE_COUNT = 1200

# Aspect Thresholds for Validation Monitoring (Phase 40 frozen)
ASPECT_THRESHOLDS = {
    "Kompatibles_Fahrzeug_Modell": 0.80,
    "Produktart": 0.80,
    "Im_Lieferumfang_Enthalten": 0.80,
}
DEFAULT_THRESHOLD = 0.60


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


def evaluate_validation(
    model: nn.Module,
    val_dataset,
    val_examples: List[dict],
    tokenizer,
    id2label: Dict[int, str],
    device: torch.device,
    loss_fn: nn.Module,
    batch_size: int = VAL_BATCH_SIZE,
) -> Dict[str, Any]:
    """Evaluate validation loss and exact entity metrics under Phase 40 confidence thresholds."""
    model.eval()
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    total_loss = 0.0
    total_batches = 0
    all_pred_entities = []

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
            batch_size_actual = labels.shape[0]

            for row_idx in range(batch_size_actual):
                ex = val_examples[ex_idx_global]
                ex_idx_global += 1

                row_labels = labels[row_idx].cpu().numpy()
                active_mask = (row_labels != -100)
                active_probs = batch_probs[row_idx, active_mask]

                assert len(active_probs) == len(ex["tokens"]), (
                    f"Token mismatch: {len(active_probs)} vs {len(ex['tokens'])}"
                )

                pred_ids = np.argmax(active_probs, axis=-1)
                pred_bio = [id2label[pid] for pid in pred_ids]
                selected_probs = [float(active_probs[t, pred_ids[t]]) for t in range(len(ex["tokens"]))]

                raw_conf_ents = bio_to_entities_with_confidence(ex["tokens"], pred_bio, selected_probs)

                # Filter by category and aspect thresholds
                filtered_ents = []
                for asp, val, mean_c, min_c in raw_conf_ents:
                    if filter_entities_by_category([(asp, val)], ex["category"]):
                        thresh = ASPECT_THRESHOLDS.get(asp, DEFAULT_THRESHOLD)
                        if min_c >= thresh:
                            filtered_ents.append((asp, val))

                all_pred_entities.append(filtered_ents)

    avg_val_loss = total_loss / max(1, total_batches)

    # Compute entity metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    cat_tp = {1: 0, 2: 0}
    cat_fp = {1: 0, 2: 0}
    cat_fn = {1: 0, 2: 0}

    for ex, preds in zip(val_examples, all_pred_entities):
        cat = ex["category"]
        gold_raw = bio_to_entities_with_confidence(ex["tokens"], ex["bio_tags"], [1.0] * len(ex["tokens"]))
        gold_tuples = filter_entities_by_category([(e[0], e[1]) for e in gold_raw], cat)

        tp, fp, fn = count_matches(preds, gold_tuples)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        cat_tp[cat] += tp
        cat_fp[cat] += fp
        cat_fn[cat] += fn

    def calc_f02(tp: int, fp: int, fn: int) -> float:
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        denom = 0.04 * p + r
        if denom == 0:
            return 0.0
        return (1.04 * p * r) / denom

    overall_f02 = calc_f02(total_tp, total_fp, total_fn)
    cat1_f02 = calc_f02(cat_tp[1], cat_fp[1], cat_fn[1])
    cat2_f02 = calc_f02(cat_tp[2], cat_fp[2], cat_fn[2])

    p_overall = total_tp / max(1, total_tp + total_fp)
    r_overall = total_tp / max(1, total_tp + total_fn)

    return {
        "val_loss": avg_val_loss,
        "f02": overall_f02,
        "cat1_f02": cat1_f02,
        "cat2_f02": cat2_f02,
        "precision": p_overall,
        "recall": r_overall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predictions": all_pred_entities,
    }


def train_phase48_hard_mined_gbert() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 48: HARD EXAMPLE MINED GBERT TRAINING")
    print("=" * 80)

    # 1. Preflight Safety & Resource Audit
    print("\n[Step 1] Auditing resources and checkpoint paths...")
    free_disk, device = check_resources()
    assert DAPT_ENCODER_PATH.exists(), f"Missing DAPT encoder: {DAPT_ENCODER_PATH}"
    assert HARD_MANIFEST_PATH.exists(), f"Missing hard examples manifest: {HARD_MANIFEST_PATH}"

    set_seed(RANDOM_SEED)

    # 2. Load Data and Manifest
    print("\n[Step 2] Loading supervised datasets and hard example manifest...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=RANDOM_SEED)

    with open(HARD_MANIFEST_PATH, "r", encoding="utf-8") as f:
        hard_manifest = json.load(f)

    hard_record_ids = set(hard_manifest["hard_record_ids"])
    hard_examples = [ex for ex in train_4k if ex["record_number"] in hard_record_ids]
    standard_examples = [ex for ex in train_4k if ex["record_number"] not in hard_record_ids]

    print(f"Total Split-A Training Set: {len(train_4k):,} titles")
    print(f"  Standard Examples:        {len(standard_examples):,} titles ({len(standard_examples)/len(train_4k)*100:.1f}%)")
    print(f"  Mined Hard Examples:      {len(hard_examples):,} titles ({len(hard_examples)/len(train_4k)*100:.1f}%)")
    print(f"Validation Holdout Set:     {len(val_1k):,} titles (STRICTLY ISOLATED)")

    # 3. Label Space & Tokenizer
    print("\n[Step 3] Loading tokenizer and label mapping...")
    ref_cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    label2id = {str(k): int(v) for k, v in ref_cfg.label2id.items()}
    id2label = {int(k): str(v) for k, v in ref_cfg.id2label.items()}
    print(f"Label Classes: {len(label2id)} classes ('O' ID = {label2id['O']})")

    tokenizer = AutoTokenizer.from_pretrained(DAPT_ENCODER_PATH, local_files_only=True)

    # Pre-tokenize validation set (static across epochs)
    print("Preparing validation dataset...")
    val_dataset, _ = prepare_split(val_1k, tokenizer, label2id)

    # 4. Model Setup
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

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * (TOTAL_EPOCH_EXAMPLES // TRAIN_BATCH_SIZE)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Optimizer: AdamW | LR: {LEARNING_RATE} | Total Steps: {total_steps} | Warmup Steps: {warmup_steps}")

    # 5. Training Loop
    print("\n" + "=" * 60)
    print("BEGINNING 3-EPOCH HARD-MINED TRAINING")
    print("=" * 60)

    best_val_f02 = -1.0
    best_epoch = 0
    training_history = []
    t_start_train = time.time()
    peak_vram_mb = 0.0

    OUTPUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    for epoch in range(1, EPOCHS + 1):
        t_epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        # Fixed mixture sampling: 70% standard, 30% hard
        rng = random.Random(RANDOM_SEED + epoch)
        epoch_standard = rng.choices(standard_examples, k=STANDARD_SAMPLE_COUNT)
        epoch_hard = rng.sample(hard_examples, k=HARD_SAMPLE_COUNT)
        epoch_examples = epoch_standard + epoch_hard
        rng.shuffle(epoch_examples)

        print(f"Constructed Epoch {epoch} training pool: {len(epoch_examples)} titles ({len(epoch_standard)} standard, {len(epoch_hard)} hard).")
        train_dataset, _ = prepare_split(epoch_examples, tokenizer, label2id)
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )

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

            if step % 250 == 0 or step == len(train_loader):
                if torch.cuda.is_available():
                    curr_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    peak_vram_mb = max(peak_vram_mb, curr_vram)
                lr_curr = scheduler.get_last_lr()[0]
                print(f"  Step {step:4d}/{len(train_loader)} | Batch Loss: {loss.item():.4f} | LR: {lr_curr:.2e}")

        avg_train_loss = running_loss / len(train_loader)
        epoch_train_time = time.time() - t_epoch_start

        # Validation evaluation
        print(f"Running validation on 1,000 holdout titles...")
        val_res = evaluate_validation(
            model=model,
            val_dataset=val_dataset,
            val_examples=val_1k,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            loss_fn=loss_fn,
            batch_size=VAL_BATCH_SIZE,
        )

        val_loss = val_res["val_loss"]
        val_f02 = val_res["f02"]
        cat1_f02 = val_res["cat1_f02"]
        cat2_f02 = val_res["cat2_f02"]
        p_val = val_res["precision"]
        r_val = val_res["recall"]

        is_best = val_f02 > best_val_f02
        if is_best:
            best_val_f02 = val_f02
            best_epoch = epoch
            print(f"--> New Best Epoch {epoch}! Validation F0.2: {val_f02:.6f}. Saving checkpoint...")
            model.save_pretrained(OUTPUT_CHECKPOINT_DIR)
            tokenizer.save_pretrained(OUTPUT_CHECKPOINT_DIR)

        epoch_total_time = time.time() - t_epoch_start
        print(f"Epoch {epoch} Summary ({epoch_total_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val F0.2:   {val_f02:.6f} (Cat 1: {cat1_f02:.6f}, Cat 2: {cat2_f02:.6f})")
        print(f"  Precision:  {p_val:.4f} | Recall: {r_val:.4f} | TP={val_res['tp']}, FP={val_res['fp']}, FN={val_res['fn']}")

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_f02": val_f02,
            "cat1_f02": cat1_f02,
            "cat2_f02": cat2_f02,
            "precision": p_val,
            "recall": r_val,
            "tp": val_res["tp"],
            "fp": val_res["fp"],
            "fn": val_res["fn"],
            "epoch_duration_seconds": epoch_total_time,
            "is_best": is_best,
        })

    total_training_time = time.time() - t_start_train
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s ({total_training_time / 60:.2f}m)")
    print(f"Best Epoch: {best_epoch} with Validation F0.2 = {best_val_f02:.6f}")
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
        "phase": "48",
        "model_name": "gbert_phase48_hard_mined_best",
        "encoder_source": str(DAPT_ENCODER_PATH),
        "output_dir": str(OUTPUT_CHECKPOINT_DIR),
        "total_training_time_seconds": total_training_time,
        "total_training_time_minutes": total_training_time / 60,
        "peak_vram_mb": peak_vram_mb,
        "best_epoch": best_epoch,
        "best_val_f02": best_val_f02,
        "training_history": training_history,
        "hyperparameters": {
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "random_seed": RANDOM_SEED,
            "total_epoch_examples": TOTAL_EPOCH_EXAMPLES,
            "standard_sample_count": STANDARD_SAMPLE_COUNT,
            "hard_sample_count": HARD_SAMPLE_COUNT,
        },
    }

    TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Training metrics saved to {TRAINING_METRICS_PATH}")

    return results


if __name__ == "__main__":
    train_phase48_hard_mined_gbert()
