"""
Phase 50: Train Lightweight Boundary-Aware Span Classifier on DAPT GBERT.

Constraints & Integrity Rules:
1. Initialized strictly from checkpoints/phase33b_dapt_encoder.
2. Trained strictly on the 3,200 internal train partition.
3. Evaluated and checkpoint selected strictly on the 800 internal dev partition.
4. Selection metric: Official competition hierarchical category-macro F0.2.
5. Mixed precision (fp16) and gradient clipping for fast, stable training.
6. Checkpoint saved to checkpoints/gbert_phase50_span_best.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.data.make_internal_split import get_internal_split
from src.data.mine_hard_examples import bio_to_spanned_entities
from src.evaluation.phase_39a_official_metric import (
    calculate_official_competition_metrics,
    calculate_global_micro_metrics,
)
from src.modeling.span_dataset import (
    SpanNERDataset,
    SpanCollateFn,
    NUM_CLASSES,
    ID2ASPECT,
    ASPECT2ID,
)
from src.modeling.span_model import SpanNERModel
from src.modeling.span_decoder import decode_spans_for_title

ENCODER_PATH = PROJECT_ROOT / "checkpoints" / "phase33b_dapt_encoder"
SAVE_DIR = PROJECT_ROOT / "checkpoints" / "gbert_phase50_span_best"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Hyperparameters
SEED = 42
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
MAX_WIDTH = 8
CONF_THRESH = 0.50


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_span_model_on_dev(
    model: SpanNERModel,
    dev_dataset: SpanNERDataset,
    collate_fn: SpanCollateFn,
    device: torch.device,
    conf_thresh: float = CONF_THRESH,
) -> Dict[str, Any]:
    """Evaluate SpanNERModel on dev set using official competition hierarchical F0.2."""
    model.eval()
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    all_decoded_records = []
    gold_records = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            span_batch_idx = batch["span_batch_idx"].to(device, non_blocking=True)
            span_sub_start = batch["span_sub_start"].to(device, non_blocking=True)
            span_sub_end = batch["span_sub_end"].to(device, non_blocking=True)
            span_width = batch["span_width"].to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    span_batch_idx=span_batch_idx,
                    span_sub_start=span_sub_start,
                    span_sub_end=span_sub_end,
                    span_width=span_width,
                )
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            # Group probabilities and metadata by batch index
            batch_spans_meta = defaultdict(list)
            batch_spans_probs = defaultdict(list)

            for idx, meta in enumerate(batch["span_meta"]):
                b_idx = meta["batch_idx"]
                batch_spans_meta[b_idx].append(meta)
                batch_spans_probs[b_idx].append(probs[idx])

            for b_idx, ex in enumerate(batch["batch_examples"]):
                cands = batch_spans_meta[b_idx]
                c_probs = np.array(batch_spans_probs[b_idx]) if cands else np.empty((0, NUM_CLASSES))
                decoded = decode_spans_for_title(
                    candidate_spans_meta=cands,
                    span_probs=c_probs,
                    tokens=ex["tokens"],
                    category=ex["category"],
                    conf_thresh=conf_thresh,
                )
                pred_tuples = [(e["aspect"], e["value"]) for e in decoded]

                gold_spanned = bio_to_spanned_entities(ex["tokens"], ex["bio_tags"]) if "bio_tags" in ex else []
                gold_tuples = [(e["aspect"], e["value"]) for e in gold_spanned]

                all_decoded_records.append({
                    "category": ex["category"],
                    "predicted_entities": pred_tuples,
                    "gold_entities": gold_tuples,
                })

    off_metrics = calculate_official_competition_metrics(all_decoded_records)
    mic_metrics = calculate_global_micro_metrics(all_decoded_records)

    return {
        "official_score": off_metrics["official_score"],
        "cat1_score": off_metrics["categories"][1]["score"],
        "cat2_score": off_metrics["categories"][2]["score"],
        "tp": mic_metrics["tp"],
        "fp": mic_metrics["fp"],
        "fn": mic_metrics["fn"],
        "precision": mic_metrics.get("global_precision", 0.0),
        "recall": mic_metrics.get("global_recall", 0.0),
        "global_micro_f02": mic_metrics.get("global_f_beta", 0.0),
    }


def train_phase50_span_model():
    set_seed(SEED)
    t0_start = time.time()

    print("=" * 80)
    print("PHASE 50: BOUNDARY-AWARE SPAN NER MODEL TRAINING")
    print("=" * 80)

    # 1. Resource Preflight
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
    print(f"Preflight Disk Free: {free_gb:.2f} GB (Requirement: >= 5.0 GB)")
    if free_gb < 5.0:
        raise RuntimeError(f"Disk space below 5.0 GB floor: {free_gb:.2f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Load and Partition Data
    print("\n[Step 1] Loading supervised data and building leak-free 3,200/800 partition...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=SEED)
    train_3200, dev_800, manifest = get_internal_split(train_4k, random_state=SEED)

    print(f"Internal Train Partition: {len(train_3200)} titles")
    print(f"Internal Dev Partition:   {len(dev_800)} titles")
    print(f"Historical 1K Val Holdout: {len(val_1k)} titles (Strictly Isolated)")

    # 3. Setup Datasets & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH, local_files_only=True)
    train_dataset = SpanNERDataset(
        examples=train_3200,
        tokenizer=tokenizer,
        max_width=MAX_WIDTH,
        is_training=True,
        seed=SEED,
    )
    dev_dataset = SpanNERDataset(
        examples=dev_800,
        tokenizer=tokenizer,
        max_width=MAX_WIDTH,
        is_training=False,  # Evaluates ALL candidate spans
        seed=SEED,
    )

    collate_fn = SpanCollateFn(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Model Initialization
    print(f"\n[Step 2] Initializing SpanNERModel from {ENCODER_PATH}...")
    model = SpanNERModel(
        encoder_path=str(ENCODER_PATH),
        num_classes=NUM_CLASSES,
        max_width=MAX_WIDTH,
        width_embed_dim=64,
        hidden_dim=256,
        dropout_prob=0.1,
    )
    model.to(device)

    # 5. Optimizer and Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    print(f"Total Training Steps: {total_steps} (Warmup Steps: {warmup_steps})")

    # 6. Training Loop
    best_dev_f02 = -1.0
    best_epoch = -1
    best_metrics = {}
    history = []

    print("\n[Step 3] Commencing 3-Epoch Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        t0_epoch = time.time()
        model.train()
        total_loss = 0.0
        step_count = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            span_batch_idx = batch["span_batch_idx"].to(device, non_blocking=True)
            span_sub_start = batch["span_sub_start"].to(device, non_blocking=True)
            span_sub_end = batch["span_sub_end"].to(device, non_blocking=True)
            span_width = batch["span_width"].to(device, non_blocking=True)
            span_labels = batch["span_labels"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    span_batch_idx=span_batch_idx,
                    span_sub_start=span_sub_start,
                    span_sub_end=span_sub_end,
                    span_width=span_width,
                    span_labels=span_labels,
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(1, step_count)
        epoch_sec = time.time() - t0_epoch

        # Evaluate on Internal Dev Partition
        t0_eval = time.time()
        dev_res = evaluate_span_model_on_dev(
            model=model,
            dev_dataset=dev_dataset,
            collate_fn=collate_fn,
            device=device,
            conf_thresh=CONF_THRESH,
        )
        eval_sec = time.time() - t0_eval

        print(
            f"Epoch {epoch}/{EPOCHS} [{epoch_sec:.1f}s train, {eval_sec:.1f}s eval] | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Dev Official Macro F0.2: {dev_res['official_score']:.6f} "
            f"(Cat 1: {dev_res['cat1_score']:.6f}, Cat 2: {dev_res['cat2_score']:.6f}) | "
            f"TP={dev_res['tp']}, FP={dev_res['fp']}, FN={dev_res['fn']}"
        )
        sys.stdout.flush()

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_time_sec": epoch_sec,
            "eval_time_sec": eval_sec,
            "metrics": dev_res,
        }
        history.append(epoch_record)

        if dev_res["official_score"] > best_dev_f02:
            best_dev_f02 = dev_res["official_score"]
            best_epoch = epoch
            best_metrics = dev_res

            print(f"  >>> New Best Checkpoint! Saving to {SAVE_DIR}...")
            sys.stdout.flush()
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

    total_training_sec = time.time() - t0_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    final_free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Total Runtime:     {total_training_sec / 60:.2f} minutes ({total_training_sec:.1f} s)")
    print(f"Best Epoch:        {best_epoch}")
    print(f"Best Dev Macro F0.2: {best_dev_f02:.6f}")
    print(f"Peak VRAM:         {peak_vram_mb:.1f} MB")
    print(f"Final Free Disk:   {final_free_gb:.2f} GB")
    print("=" * 80)

    # Save training metadata
    meta = {
        "phase": "50",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_width": MAX_WIDTH,
        "best_epoch": best_epoch,
        "best_dev_f02": best_dev_f02,
        "best_metrics": best_metrics,
        "history": history,
        "total_runtime_sec": total_training_sec,
        "peak_vram_mb": peak_vram_mb,
        "final_free_gb": final_free_gb,
    }
    with open(SAVE_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    train_phase50_span_model()
