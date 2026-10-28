"""
Phase 33B: Stage A — Small MLM Resource Pilot
Measures exact GPU VRAM, throughput (examples/sec), loss stability, and collator behavior.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sample_dapt_corpus import sample_dapt_corpus

MODEL_NAME = "deepset/gbert-base"
PILOT_SAMPLE_COUNT = 256
MAX_SEQ_LENGTH = 64
SEED = 42


class ListingTitleTextDataset(Dataset):
    """Tokenizes raw listing title tokens for MLM."""

    def __init__(self, examples: list, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.items = []
        for ex in examples:
            tokens = ex["tokens"]
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
            self.items.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "special_tokens_mask": encoded["special_tokens_mask"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def run_pilot():
    print("=" * 72)
    print("PHASE 33B: STAGE A — SMALL MLM RESOURCE PILOT")
    print("=" * 72)

    # 1. Hardware and Disk pre-check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats(0)

    _, _, free_start_bytes = shutil.disk_usage("C:")
    print(f"Drive C: Free Space: {free_start_bytes / (1024**3):.2f} GB")

    # 2. Load pilot examples (first 256 from deterministic sampler)
    print(f"\nSampling {PILOT_SAMPLE_COUNT} pilot examples...")
    pilot_samples, _ = sample_dapt_corpus(sample_size_per_category=PILOT_SAMPLE_COUNT // 2, random_seed=SEED)
    pilot_samples = pilot_samples[:PILOT_SAMPLE_COUNT]
    print(f"Loaded {len(pilot_samples)} pilot listing titles.")

    # 3. Tokenizer & Dataset
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    dataset = ListingTitleTextDataset(pilot_samples, tokenizer, max_length=MAX_SEQ_LENGTH)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 4. Model Loading
    print(f"Loading AutoModelForMaskedLM: {MODEL_NAME}...")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # 5. Test configs: Test Batch Size 4 and Batch Size 8
    test_configs = [
        {"batch_size": 4, "grad_accum": 4, "use_fp16": True},
        {"batch_size": 8, "grad_accum": 2, "use_fp16": True},
    ]

    pilot_results = []

    for cfg in test_configs:
        bs = cfg["batch_size"]
        ga = cfg["grad_accum"]
        fp16 = cfg["use_fp16"] and torch.cuda.is_available()
        print(f"\n--- Testing Config: Microbatch = {bs}, GradAccum = {ga} (Effective Batch = {bs * ga}), FP16 = {fp16} ---")

        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=data_collator,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

        t0 = time.time()
        losses = []
        optimizer.zero_grad()

        special_token_masking_verified = True
        padding_exclusion_verified = True

        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Verification: Special tokens [CLS] (id=102) and [SEP] (id=103) should not be masked
            # In DataCollatorForLanguageModeling, unmasked positions in labels have value -100
            for row_idx in range(input_ids.shape[0]):
                cls_pos = (input_ids[row_idx] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
                sep_pos = (input_ids[row_idx] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                pad_pos = (attention_mask[row_idx] == 0).nonzero(as_tuple=True)[0]

                if len(cls_pos) > 0 and labels[row_idx, cls_pos[0]] != -100:
                    special_token_masking_verified = False
                if len(sep_pos) > 0 and labels[row_idx, sep_pos[0]] != -100:
                    special_token_masking_verified = False
                if len(pad_pos) > 0:
                    for p in pad_pos:
                        if labels[row_idx, p] != -100:
                            padding_exclusion_verified = False

            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / ga

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

            scaler.scale(loss).backward()
            losses.append(loss.item() * ga)

            if step % ga == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        elapsed = time.time() - t0
        throughput = len(dataset) / max(0.001, elapsed)

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**3)
            peak_res = torch.cuda.max_memory_reserved(0) / (1024**3)
        else:
            peak_alloc, peak_res = 0.0, 0.0

        mean_loss = sum(losses) / len(losses)
        print(f"  Throughput:           {throughput:.1f} examples/sec")
        print(f"  Wall-clock time:      {elapsed:.2f}s for {len(dataset)} examples")
        print(f"  Peak VRAM Allocated:  {peak_alloc:.2f} GB")
        print(f"  Peak VRAM Reserved:   {peak_res:.2f} GB")
        print(f"  Mean MLM Loss:        {mean_loss:.4f} (Finite: True)")
        print(f"  Special Tokens Safe:  {special_token_masking_verified}")
        print(f"  Padding Excluded:     {padding_exclusion_verified}")

        pilot_results.append({
            "batch_size": bs,
            "grad_accum": ga,
            "effective_batch_size": bs * ga,
            "fp16": fp16,
            "throughput_ex_per_sec": round(throughput, 1),
            "wall_clock_sec": round(elapsed, 2),
            "peak_vram_allocated_gb": round(peak_alloc, 2),
            "peak_vram_reserved_gb": round(peak_res, 2),
            "mean_loss": round(mean_loss, 4),
            "special_tokens_safe": special_token_masking_verified,
            "padding_excluded": padding_exclusion_verified,
        })

    # Test serialization of adapted encoder
    test_save_dir = PROJECT_ROOT / "checkpoints" / "test_pilot_encoder"
    print(f"\nTesting checkpoint serialization to {test_save_dir}...")
    model.save_pretrained(test_save_dir)
    tokenizer.save_pretrained(test_save_dir)
    saved_size_mb = sum(f.stat().st_size for f in test_save_dir.glob("*")) / (1024**2)
    print(f"Serialization successful. Saved size: {saved_size_mb:.2f} MB")

    # Clean up test save directory
    shutil.rmtree(test_save_dir)
    print("Test checkpoint cleaned up successfully.")

    _, _, free_end_bytes = shutil.disk_usage("C:")
    print(f"Drive C: Free Space Post-Pilot: {free_end_bytes / (1024**3):.2f} GB")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save pilot results
    results_path = PROJECT_ROOT / "data" / "cache" / "dapt_pilot_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(pilot_results, f, indent=2)

    print("\n" + "=" * 72)
    print(">>> STAGE A RESOURCE PILOT COMPLETED SUCCESSFULLY <<<")
    print("=" * 72)


if __name__ == "__main__":
    run_pilot()
