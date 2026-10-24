"""
Phase 22: Full-Data Champion Training Pipeline.

Trains the 4 champion models on 100% of labeled training data (5,000 titles, 56,812 tokens)
from data/raw/Tagged_Titles_Train.tsv.gz without train/validation split.

Frozen Configurations:
1. GBERT:        deepset/gbert-base            seed=42  epochs=3  lr=2e-5  batch_size=4
2. XLM-R seed42: FacebookAI/xlm-roberta-base   seed=42  epochs=2  lr=2e-5  batch_size=4
3. XLM-R seed7:  FacebookAI/xlm-roberta-base   seed=7   epochs=3  lr=2e-5  batch_size=4
4. XLM-R seed21: FacebookAI/xlm-roberta-base   seed=21  epochs=2  lr=2e-5  batch_size=4

Safety & Preflight:
- Preflight check requires at least 10.0 GB free disk space.
- Sequential GPU loading (one model at a time).
- Saves only final checkpoints to checkpoints/*_fulldata_*_final.
- Never overwrites original validation checkpoints.
"""

import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.modeling.prepare_datasets import prepare_split

MIN_REQUIRED_FREE_DISK_GB = 10.0
MIN_SINGLE_MODEL_FREE_DISK_GB = 3.5

MODEL_CONFIGS: Dict[str, dict] = {
    "gbert_seed42": {
        "model_name": "deepset/gbert-base",
        "seed": 42,
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 4,
        "checkpoint_dir": Path("checkpoints/gbert_fulldata_seed42_final"),
    },
    "xlmr_seed42": {
        "model_name": "FacebookAI/xlm-roberta-base",
        "seed": 42,
        "epochs": 2,
        "lr": 2e-5,
        "batch_size": 4,
        "checkpoint_dir": Path("checkpoints/xlmr_fulldata_seed42_final"),
    },
    "xlmr_seed7": {
        "model_name": "FacebookAI/xlm-roberta-base",
        "seed": 7,
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 4,
        "checkpoint_dir": Path("checkpoints/xlmr_fulldata_seed7_final"),
    },
    "xlmr_seed21": {
        "model_name": "FacebookAI/xlm-roberta-base",
        "seed": 21,
        "epochs": 2,
        "lr": 2e-5,
        "batch_size": 4,
        "checkpoint_dir": Path("checkpoints/xlmr_fulldata_seed21_final"),
    },
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_disk_space(required_gb: float = MIN_REQUIRED_FREE_DISK_GB) -> Tuple[bool, float]:
    """Check available free disk space on current drive in GB."""
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    return free_gb >= required_gb, free_gb


def train_single_fulldata_model(
    model_id: str,
    examples: List[dict],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
) -> dict:
    """Train a single full-data champion model on all 5,000 titles."""
    cfg = MODEL_CONFIGS[model_id]
    model_name = cfg["model_name"]
    seed = cfg["seed"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    checkpoint_dir = cfg["checkpoint_dir"]

    print("\n" + "=" * 60)
    print(f"TRAINING FULL-DATA CHAMPION MODEL: {model_id.upper()}")
    print("=" * 60)
    print(f"  Base Model:       {model_name}")
    print(f"  Random Seed:      {seed}")
    print(f"  Total Epochs:     {epochs}")
    print(f"  Learning Rate:    {lr}")
    print(f"  Batch Size:       {batch_size}")
    print(f"  Target Save Dir:  {checkpoint_dir}")
    print(f"  Training Titles:  {len(examples)} (100% full data, no split)")

    # Check if model has already been trained and verified
    weights_file = checkpoint_dir / "model.safetensors"
    config_file = checkpoint_dir / "config.json"
    if weights_file.exists() and config_file.exists() and weights_file.stat().st_size > 100_000_000:
        cp_size_mb = sum(f.stat().st_size for f in checkpoint_dir.iterdir() if f.is_file()) / (1024 ** 2)
        _, current_free = check_disk_space()
        print(f"\n[SKIP] Verified checkpoint already exists at {checkpoint_dir} ({cp_size_mb:.1f} MB). Skipping training.")
        return {
            "model_id": model_id,
            "checkpoint_dir": str(checkpoint_dir),
            "epochs_completed": epochs,
            "final_loss": None,
            "checkpoint_size_mb": round(cp_size_mb, 1),
            "peak_vram_mb": 0.0,
            "post_free_gb": round(current_free, 2),
            "runtime_seconds": 0.0,
            "skipped": True,
        }

    # 1. Preflight disk space check before training this model
    is_space_ok, free_gb = check_disk_space(MIN_SINGLE_MODEL_FREE_DISK_GB)
    print(f"  Current Free Disk: {free_gb:.2f} GB (Required >= {MIN_SINGLE_MODEL_FREE_DISK_GB:.1f} GB)")
    if not is_space_ok:
        raise RuntimeError(
            f"Insufficient disk space for {model_id}: {free_gb:.2f} GB available, "
            f"required >= {MIN_SINGLE_MODEL_FREE_DISK_GB:.1f} GB! Stopping to prevent disk exhaustion."
        )

    set_random_seed(seed)

    # 2. Tokenize and align all 5,000 examples
    print(f"\nLoading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, stats = prepare_split(examples, tokenizer, label2id)
    print(f"  Active training labels: {stats['active_labels']:,}")
    print(f"  Alignment mismatches:   {stats['alignment_mismatches']:,}")
    print(f"  Max sequence length:    {stats['max_transformer_length']:,}")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Initialize model strictly from pretrained base
    print(f"\nInitializing model from pretrained weights '{model_name}'...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # 4. Training loop
    model.train()
    total_steps = len(dataloader) * epochs
    start_time = time.time()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        step_count = 0

        for batch in dataloader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch_dev)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

        avg_loss = running_loss / max(1, step_count)
        epoch_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} ({epoch_time:.1f}s)")

    total_training_time = time.time() - start_time
    final_loss = epoch_losses[-1] if epoch_losses else 0.0

    # 5. Save final checkpoint (only the final model, no intermediate epochs)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model checkpoint to {checkpoint_dir}...")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # 6. Verify saved checkpoint
    print("Verifying saved checkpoint integrity...")
    reloaded_cfg = AutoConfig.from_pretrained(checkpoint_dir)
    reloaded_model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    assert reloaded_cfg.num_labels == len(label2id), "Label count mismatch!"
    assert len(reloaded_cfg.id2label) == len(id2label), "id2label length mismatch!"
    del reloaded_model

    # Synthetic inference check
    reloaded_tok = AutoTokenizer.from_pretrained(checkpoint_dir)
    enc = reloaded_tok(["Bremsscheibe", "vorne"], is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v.to(device) for k, v in enc.items()})
    assert out.logits.shape[-1] == len(label2id), "Logits shape mismatch!"

    # 7. GPU & Disk cleanup
    del model, optimizer, dataloader, train_dataset, tokenizer, reloaded_tok
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.empty_cache()
    else:
        peak_vram = 0.0

    _, post_free_gb = check_disk_space()
    cp_size_mb = sum(f.stat().st_size for f in checkpoint_dir.iterdir() if f.is_file()) / (1024 ** 2)

    print(f"Successfully trained and verified {model_id}!")
    print(f"  Checkpoint Size:    {cp_size_mb:.1f} MB")
    print(f"  Peak VRAM:          {peak_vram:.1f} MB")
    print(f"  Disk Space Free:    {post_free_gb:.2f} GB")
    print(f"  Training Runtime:   {total_training_time:.1f}s ({total_training_time/60:.2f} min)")

    return {
        "model_id": model_id,
        "checkpoint_dir": str(checkpoint_dir),
        "epochs_completed": epochs,
        "final_loss": round(final_loss, 4),
        "checkpoint_size_mb": round(cp_size_mb, 1),
        "peak_vram_mb": round(peak_vram, 1),
        "post_free_gb": round(post_free_gb, 2),
        "runtime_seconds": round(total_training_time, 1),
    }


def run_fulldata_pipeline(models_to_run: Optional[List[str]] = None) -> List[dict]:
    """Execute sequential full-data champion training across specified models."""
    print("==================================================")
    print("PHASE 22: FULL-DATA CHAMPION TRAINING PIPELINE")
    print("==================================================")

    if models_to_run is None:
        models_to_run = ["gbert_seed42", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]

    # If multiple models are requested (e.g. --all), run each sequentially in an isolated
    # sub-process. This ensures OS working sets and GPU paging commitments on Windows
    # are completely reclaimed after each model finishes.
    if len(models_to_run) > 1:
        # Preflight Check for entire pipeline
        is_space_ok, free_gb = check_disk_space(MIN_REQUIRED_FREE_DISK_GB)
        print(f"Preflight Disk Check (Pipeline Launch): {free_gb:.2f} GB free.")
        if not is_space_ok:
            raise RuntimeError(
                f"Preflight FAIL: Disk space {free_gb:.2f} GB is below required {MIN_REQUIRED_FREE_DISK_GB:.1f} GB! "
                f"Full-data training cannot proceed without safe cleanup."
            )

        print(f"Executing sequential isolated training for {len(models_to_run)} models: {models_to_run}")
        for idx, model_id in enumerate(models_to_run, 1):
            print(f"\n[{idx}/{len(models_to_run)}] Spawning worker process for {model_id}...")
            cmd = [sys.executable, "-u", "-m", "src.modeling.train_fulldata_champion", "--model-id", model_id]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                raise RuntimeError(f"Training worker process for {model_id} failed with exit code {ret.returncode}!")

        print("\n==================================================")
        print("ALL REQUESTED FULL-DATA MODELS COMPLETED SUCCESSFULLY")
        print("==================================================")
        return []

    # Single-model worker execution:
    model_id = models_to_run[0]
    is_space_ok, free_gb = check_disk_space(MIN_SINGLE_MODEL_FREE_DISK_GB)
    print(f"Preflight Disk Check ({model_id}): {free_gb:.2f} GB free.")
    if not is_space_ok:
        raise RuntimeError(
            f"Preflight FAIL: Disk space {free_gb:.2f} GB is below required {MIN_SINGLE_MODEL_FREE_DISK_GB:.1f} GB! "
            f"Training {model_id} stopped to prevent disk exhaustion."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")

    # Load 100% full training data (5,000 titles)
    df = load_training_data()
    if df.empty or len(df) != 56812:
        raise ValueError(f"Expected 56,812 rows in Tagged_Titles_Train.tsv.gz, got {len(df)}!")
    examples = build_examples(df)
    if len(examples) != 5000:
        raise ValueError(f"Expected exactly 5,000 examples, got {len(examples)}!")
    print(f"Loaded {len(examples)} training titles ({len(df)} token rows). 100% full dataset.")

    # Shared 48-label mapping
    label2id, id2label = create_label_mappings(examples)
    print(f"BIO Label Space: {len(label2id)} classes (shared across all models).")

    res = train_single_fulldata_model(
        model_id=model_id,
        examples=examples,
        label2id=label2id,
        id2label=id2label,
        device=device,
    )
    return [res]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 22 Full-Data Champion Training")
    parser.add_argument(
        "--model-id",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Train a specific model only (gbert_seed42, xlmr_seed42, xlmr_seed7, xlmr_seed21)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 4 champion models sequentially",
    )
    args = parser.parse_args()

    if args.model_id:
        run_fulldata_pipeline([args.model_id])
    elif args.all:
        run_fulldata_pipeline()
    else:
        # Default: print plan and preflight status
        is_ok, free_gb = check_disk_space(MIN_REQUIRED_FREE_DISK_GB)
        print("Phase 22 Full-Data Champion Training Configuration:")
        for k, v in MODEL_CONFIGS.items():
            print(f"  {k:15s}: base={v['model_name']:28s} seed={v['seed']} epochs={v['epochs']} lr={v['lr']} bs={v['batch_size']}")
        print(f"\nPreflight Disk Status: {free_gb:.2f} GB free (Required: {MIN_REQUIRED_FREE_DISK_GB:.1f} GB) -> {'PASS' if is_ok else 'BLOCKED'}")
        if not is_ok:
            print("To proceed, disk space must be at least 10 GB. See cleanup plan in documentation.")


if __name__ == "__main__":
    main()
