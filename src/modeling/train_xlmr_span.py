"""
Training script for Span-Based XLM-RoBERTa (Experiment 16).

Controlled comparison against:
- Plain XLM-R seed-42 baseline: 0.869284
- Boundary-aware multi-task XLM-R (Exp 14): 0.875489

Setup:
- Backbone: FacebookAI/xlm-roberta-base
- Split: Split-A (random_state=42, 4000 train / 1000 val)
- Optimizer: AdamW, lr=2e-5
- Epochs: 3
- Batch size: 4 (train), 8 (val)
- Multi-task loss: span_loss + 0.20 * start_loss + 0.20 * end_loss
- Best checkpoint saved to checkpoints/xlmr_span_seed42_best
"""

import argparse
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

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
    sample_training_spans_for_example,
)
from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.error_analysis import (
    analyze_token_confusions,
    analyze_entity_errors,
    print_top_confusions,
    print_entity_error_summary,
)


TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5
EPOCHS = 3

XLMR_MODEL_NAME = "FacebookAI/xlm-roberta-base"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/xlmr_span_seed42_best")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SpanDataset(Dataset):
    """
    Dataset wrapper attaching deterministically sampled candidate spans and labels.
    """

    def __init__(
        self,
        tokenized_dataset,
        examples: List[dict],
        span2id: dict,
        seed: int = 42,
        max_span_width: int = MAX_SPAN_WIDTH,
    ):
        self.tokenized_dataset = tokenized_dataset
        self.examples = examples
        self.span2id = span2id
        self.seed = seed
        self.max_span_width = max_span_width

    def __len__(self) -> int:
        return len(self.tokenized_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.tokenized_dataset[idx])
        example = self.examples[idx]

        spans, labels = sample_training_spans_for_example(
            bio_tags=example["bio_tags"],
            span2id=self.span2id,
            seed=self.seed,
            ex_idx=idx,
            max_span_width=self.max_span_width,
            max_neg_ratio=3,
        )

        item["spans"] = spans
        item["span_labels"] = labels
        return item


class SpanDataCollator:
    """
    Collator that handles standard dynamic subword padding while keeping
    variable-count candidate spans as lists of tensors.
    """

    def __init__(self, tokenizer):
        self.token_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def __call__(self, features: List[dict]) -> dict:
        spans_list = [
            torch.tensor(f.pop("spans"), dtype=torch.long)
            for f in features
        ]
        span_labels_list = [
            torch.tensor(f.pop("span_labels"), dtype=torch.long)
            for f in features
        ]

        batch = self.token_collator(features)
        batch["spans"] = spans_list
        batch["span_labels"] = span_labels_list
        return batch


def evaluate_span_model_entities(
    model: XLMRobertaSpan,
    dataset,
    examples: List[dict],
    tokenizer,
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> dict:
    """
    Run span model inference over validation data, greedily resolve overlapping spans,
    reconstruct BIO labels, and calculate local competition-style entity metrics.
    """
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    model.eval()
    records = []
    true_bio_sequences = []
    predicted_bio_sequences = []

    diag_totals = {
        "total_candidates": 0,
        "none_candidates": 0,
        "eligible_candidates": 0,
        "retained_candidates": 0,
        "rejected_overlap_candidates": 0,
    }

    example_index = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            pred_bio_batch, diag = model.predict_spans(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                labels=batch_dev.get("labels"),
            )

            for k in diag_totals:
                diag_totals[k] += diag[k]

            batch_size_actual = len(pred_bio_batch)

            for row_index in range(batch_size_actual):
                example = examples[example_index]
                predicted_bio = pred_bio_batch[row_index]
                true_bio = example["bio_tags"]
                tokens = example["tokens"]

                if len(predicted_bio) != len(tokens):
                    raise ValueError(
                        f"Predicted BIO count ({len(predicted_bio)}) != token count ({len(tokens)})."
                    )

                true_bio_sequences.append(true_bio)
                predicted_bio_sequences.append(predicted_bio)

                true_entities = bio_to_entities(tokens, true_bio)
                predicted_entities = bio_to_entities(tokens, predicted_bio)

                records.append(
                    {
                        "category": example["category"],
                        "true_entities": true_entities,
                        "predicted_entities": predicted_entities,
                    }
                )
                example_index += 1

    if example_index != len(examples):
        raise ValueError("Not all validation examples were evaluated.")

    results = calculate_competition_metrics(records)

    return {
        "metrics": results,
        "records": records,
        "true_bio_sequences": true_bio_sequences,
        "predicted_bio_sequences": predicted_bio_sequences,
        "diagnostics": diag_totals,
    }


def print_entity_report(results: dict) -> None:
    """Print category-level and per-aspect entity evaluation metrics."""
    beta = results["beta"]
    categories = results["categories"]

    print("\n--- ENTITY-LEVEL VALIDATION ---")

    for category, category_result in sorted(categories.items()):
        category_score = category_result["score"]
        aspects = category_result["aspects"]

        print(f"\nCategory {category} score: {category_score:.6f}")
        print("Aspect performance (sorted by validation frequency):")

        sorted_aspects = sorted(
            aspects.items(),
            key=lambda item: item[1]["true_count"],
            reverse=True,
        )

        for aspect_name, metrics in sorted_aspects:
            true_count = metrics["true_count"]
            precision = metrics["precision"]
            recall = metrics["recall"]
            f_beta = metrics["f_beta"]
            weight = metrics["weight"]

            print(
                f"  {aspect_name}: count={true_count} | "
                f"P={precision:.4f} | R={recall:.4f} | "
                f"F{beta}={f_beta:.4f} | weight={weight:.4f}"
            )

    final_score = results["final_score"]
    print(f"\nLOCAL COMPETITION-STYLE F{beta}: {final_score:.6f}")


def print_error_analysis(evaluation_output: dict) -> None:
    """Analyze and print aggregate model errors."""
    true_bio_sequences = evaluation_output["true_bio_sequences"]
    predicted_bio_sequences = evaluation_output["predicted_bio_sequences"]
    records = evaluation_output["records"]

    confusion_counts = analyze_token_confusions(
        true_bio_sequences,
        predicted_bio_sequences,
    )
    aspect_errors = analyze_entity_errors(records)

    print_top_confusions(confusion_counts, top_n=25)
    print_entity_error_summary(aspect_errors)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Span-Based XLM-RoBERTa model."
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
        checkpoint_dir = Path(f"checkpoints/xlmr_span_seed_{random_seed}_best")

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

    print(f"BIO label classes: {len(label2id)}")
    print(f"Semantic span classes (including NONE): {len(span2id)}")

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

    # Attach sampled candidate spans to training dataset
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
    # 8. BUILD SPAN MODEL
    # -------------------------------------------------
    print("\n--- INITIALIZING SPAN-BASED XLM-R MODEL ---")
    model = XLMRobertaSpan(
        label2id=label2id,
        id2label=id2label,
        span2id=span2id,
        id2span=id2span,
        model_name=XLMR_MODEL_NAME,
        max_span_width=MAX_SPAN_WIDTH,
        start_loss_weight=0.20,
        end_loss_weight=0.20,
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
    print("\n--- EXPERIMENT 16 CONFIG (SPAN-BASED XLM-R) ---")
    print(f"Backbone: {XLMR_MODEL_NAME}")
    print(f"MAX_SPAN_WIDTH: {MAX_SPAN_WIDTH}")
    print("Multi-task loss: span_loss + 0.20 * start_loss + 0.20 * end_loss")
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
    print("\n--- STARTING SPAN-BASED TRAINING ---")

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
                    f"(Span: {outputs.span_loss.item():.4f}, "
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
            f"Total: {avg_train_loss:.4f} | Span: {avg_span_loss:.4f} | "
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
    print("EXPERIMENT 16 TRAINING COMPLETE")
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
