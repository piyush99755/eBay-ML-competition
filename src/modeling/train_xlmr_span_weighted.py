"""
Training script for Class-Weighted Span-Based XLM-RoBERTa (Experiment 17).

Follow-up to Experiment 16:
- Architecture, backbone, split, seed, hyperparameters, negative sampling,
  and overlap resolution remain identical to Experiment 16.
- ONLY change: class-weighted CrossEntropyLoss for span classification:
    weights[0]  = 0.70 (NONE class)
    weights[1:] = 1.00 (all 29 entity aspect classes)
- Best checkpoint saved to: checkpoints/xlmr_span_weighted_seed42_best
"""

import argparse
from pathlib import Path
from typing import List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.build_examples import (
    build_examples,
    load_training_data,
)
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.modeling.xlmr_span import (
    MAX_SPAN_WIDTH,
    XLMRobertaSpan,
    create_span_mappings,
)
from src.modeling.train_xlmr_span import (
    SpanDataset,
    SpanDataCollator,
    evaluate_span_model_entities,
    print_entity_report,
    print_error_analysis,
    set_random_seed,
)


TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5
EPOCHS = 3

NONE_CLASS_WEIGHT = 0.70
ENTITY_CLASS_WEIGHT = 1.00

XLMR_MODEL_NAME = "FacebookAI/xlm-roberta-base"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/xlmr_span_weighted_seed42_best")


def build_span_class_weights(num_classes: int) -> torch.Tensor:
    """
    Construct fixed class-weight tensor:
    weights[0]  = 0.70 (NONE class)
    weights[1:] = 1.00 (all entity aspect classes)
    """
    weights = torch.full((num_classes,), ENTITY_CLASS_WEIGHT, dtype=torch.float32)
    weights[0] = NONE_CLASS_WEIGHT
    return weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Class-Weighted Span-Based XLM-RoBERTa model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model initialization and training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random_seed = args.seed
    set_random_seed(random_seed)

    print(f"Random seed: {random_seed}")

    if random_seed == 42:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    else:
        checkpoint_dir = Path(f"checkpoints/xlmr_span_weighted_seed_{random_seed}_best")

    # -------------------------------------------------
    # 1. GPU CHECK
    # -------------------------------------------------
    print("--- CUDA CHECK ---")
    if not torch.cuda.is_available():
        print("CUDA is not available. Stopping.")
        return

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats(device)

    # -------------------------------------------------
    # 2. TOKENIZER
    # -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL_NAME)

    # -------------------------------------------------
    # 3. LOAD DATA
    # -------------------------------------------------
    print("\n--- LOADING DATA ---")
    df = load_training_data()
    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)
    print(f"Total labeled titles: {len(examples)}")

    # -------------------------------------------------
    # 4. LABEL & SPAN MAPPINGS
    # -------------------------------------------------
    label2id, id2label = create_label_mappings(examples)
    span2id, id2span = create_span_mappings(label2id)

    num_span_classes = len(span2id)
    class_weights = build_span_class_weights(num_span_classes)

    print(f"BIO label classes: {len(label2id)}")
    print(f"Semantic span classes (including NONE): {num_span_classes}")
    print(f"Span class weights: NONE={class_weights[0].item():.2f}, entities={class_weights[1].item():.2f}")

    # -------------------------------------------------
    # 5. TRAIN / VALIDATION SPLIT (Split-A, random_state=42)
    # -------------------------------------------------
    train_examples, val_examples = split_examples(examples)
    print(f"Training titles: {len(train_examples)}")
    print(f"Validation titles: {len(val_examples)}")

    # -------------------------------------------------
    # 6. TOKENIZE + ALIGN
    # -------------------------------------------------
    print("\n--- PREPARING DATASETS ---")
    train_dataset_base, train_stats = prepare_split(
        train_examples,
        tokenizer,
        label2id,
    )
    val_dataset, val_stats = prepare_split(
        val_examples,
        tokenizer,
        label2id,
    )

    print(f"Training alignment mismatches: {train_stats['alignment_mismatches']}")
    print(f"Validation alignment mismatches: {val_stats['alignment_mismatches']}")

    if (
        train_stats["alignment_mismatches"] > 0
        or val_stats["alignment_mismatches"] > 0
    ):
        raise ValueError("Alignment mismatches detected.")

    train_span_dataset = SpanDataset(
        tokenized_dataset=train_dataset_base,
        examples=train_examples,
        span2id=span2id,
        seed=random_seed,
        max_span_width=MAX_SPAN_WIDTH,
    )

    # -------------------------------------------------
    # 7. DATA LOADERS
    # -------------------------------------------------
    train_collator = SpanDataCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_span_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collator,
        pin_memory=True,
    )

    # -------------------------------------------------
    # 8. BUILD WEIGHTED SPAN MODEL
    # -------------------------------------------------
    print("\n--- INITIALIZING CLASS-WEIGHTED SPAN XLM-R MODEL ---")
    model = XLMRobertaSpan(
        label2id=label2id,
        id2label=id2label,
        span2id=span2id,
        id2span=id2span,
        model_name=XLMR_MODEL_NAME,
        max_span_width=MAX_SPAN_WIDTH,
        start_loss_weight=0.20,
        end_loss_weight=0.20,
        span_class_weights=class_weights.to(device),
    )
    model.to(device)

    # -------------------------------------------------
    # 9. OPTIMIZER
    # -------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------
    # 10. TRAINING CONFIG
    # -------------------------------------------------
    print("\n--- EXPERIMENT 17 CONFIG (CLASS-WEIGHTED SPAN XLM-R) ---")
    print(f"Backbone: {XLMR_MODEL_NAME}")
    print(f"MAX_SPAN_WIDTH: {MAX_SPAN_WIDTH}")
    print(f"Span class weights: NONE={NONE_CLASS_WEIGHT:.2f}, entities={ENTITY_CLASS_WEIGHT:.2f}")
    print("Multi-task loss: weighted_span_loss + 0.20 * start_loss + 0.20 * end_loss")
    print(f"Epochs: {EPOCHS}")
    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"Validation batch size: {VAL_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training steps / epoch: {len(train_loader)}")
    print(f"Target checkpoint: {checkpoint_dir}")

    # -------------------------------------------------
    # 11. TRACK BEST MODEL
    # -------------------------------------------------
    best_f_beta = -1.0
    best_epoch = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 12. TRAINING LOOP
    # -------------------------------------------------
    print("\n--- STARTING CLASS-WEIGHTED SPAN TRAINING ---")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_span_loss = 0.0
        total_start_loss = 0.0
        total_end_loss = 0.0

        print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")

        for step, batch in enumerate(train_loader, start=1):
            batch_dev = {
                key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }

            optimizer.zero_grad()
            outputs = model(**batch_dev)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            if outputs.span_loss is not None:
                total_span_loss += outputs.span_loss.item()
            if outputs.start_loss is not None:
                total_start_loss += outputs.start_loss.item()
            if outputs.end_loss is not None:
                total_end_loss += outputs.end_loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Total Loss: {loss.item():.4f} "
                    f"(Weighted Span: {outputs.span_loss.item():.4f}, "
                    f"Start: {outputs.start_loss.item():.4f}, "
                    f"End: {outputs.end_loss.item():.4f})"
                )

        num_steps = len(train_loader)
        avg_train_loss = total_train_loss / num_steps
        avg_span_loss = total_span_loss / num_steps
        avg_start_loss = total_start_loss / num_steps
        avg_end_loss = total_end_loss / num_steps

        print(
            f"\nEpoch {epoch + 1} training losses -> "
            f"Total: {avg_train_loss:.4f} | Weighted Span: {avg_span_loss:.4f} | "
            f"Start: {avg_start_loss:.4f} | End: {avg_end_loss:.4f}"
        )

        # Entity metrics with greedy span overlap resolution
        evaluation_output = evaluate_span_model_entities(
            model=model,
            dataset=val_dataset,
            examples=val_examples,
            tokenizer=tokenizer,
            device=device,
            batch_size=VAL_BATCH_SIZE,
        )

        entity_results = evaluation_output["metrics"]
        print_entity_report(entity_results)
        print_error_analysis(evaluation_output)

        diag = evaluation_output["diagnostics"]
        print(
            f"Validation Span Diagnostics: "
            f"Total candidates: {diag['total_candidates']} | "
            f"None: {diag['none_candidates']} ({(diag['none_candidates'] / max(1, diag['total_candidates'])) * 100:.1f}%) | "
            f"Eligible: {diag['eligible_candidates']} | "
            f"Retained: {diag['retained_candidates']} | "
            f"Overlap rejected: {diag['rejected_overlap_candidates']}"
        )

        current_f_beta = entity_results["final_score"]

        # Save best checkpoint
        if current_f_beta > best_f_beta:
            best_f_beta = current_f_beta
            best_epoch = epoch + 1

            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            print("\nNEW BEST MODEL")
            print(
                f"Saved epoch {best_epoch} with "
                f"F{entity_results['beta']}={best_f_beta:.6f}"
            )

    # -------------------------------------------------
    # 13. FINAL SUMMARY
    # -------------------------------------------------
    print("\n==============================")
    print("EXPERIMENT 17 TRAINING COMPLETE")
    print("==============================")
    print(f"Best epoch: {best_epoch}")
    print(f"Best local F0.2: {best_f_beta:.6f}")

    # -------------------------------------------------
    # 14. GPU MEMORY
    # -------------------------------------------------
    allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
    peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3

    print(f"Allocated GPU memory: {allocated_gb:.2f} GB")
    print(f"Reserved GPU memory: {reserved_gb:.2f} GB")
    print(f"Peak GPU memory: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
