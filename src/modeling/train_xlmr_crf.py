"""
Training script for XLM-RoBERTa + Linear-Chain CRF (Experiment 13).

Controlled comparison against the raw XLM-R seed-42 baseline:
- Backbone: FacebookAI/xlm-roberta-base
- Split: Split-A (random_state=42, 4000 train / 1000 val)
- Optimizer: AdamW, lr=2e-5
- Epochs: 3
- Batch size: 4 (train), 8 (val)
- CRF operates strictly on original-token positions (labels != -100)
- Best checkpoint saved to checkpoints/xlmr_crf_seed42_best
"""

import argparse
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
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
from src.modeling.xlmr_crf import XLMRobertaCRF
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
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/xlmr_crf_seed42_best")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_loss(
    model: XLMRobertaCRF,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Calculate average validation CRF loss without gradients."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            outputs = model(**batch)
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                total_batches += 1

    return total_loss / total_batches if total_batches > 0 else 0.0


def evaluate_crf_model_entities(
    model: XLMRobertaCRF,
    dataset,
    examples: List[dict],
    tokenizer,
    id2label: dict,
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> dict:
    """
    Run the trained CRF model on validation data with Viterbi decoding
    and calculate local competition-style entity metrics.
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
    example_index = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            # Viterbi decoding returns List[List[int]] for the active original tokens
            decoded_batch = model.decode(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels"),
            )

            batch_size_actual = len(decoded_batch)

            for row_index in range(batch_size_actual):
                example = examples[example_index]
                predicted_ids = decoded_batch[row_index]

                predicted_bio = [id2label[lid] for lid in predicted_ids]
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
        description="Train XLM-RoBERTa + CRF model for aspect extraction."
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
        checkpoint_dir = Path(f"checkpoints/xlmr_crf_seed_{random_seed}_best")

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
    # 4. LABEL MAPPING
    # -------------------------------------------------
    label2id, id2label = create_label_mappings(examples)
    print(f"BIO label classes: {len(label2id)}")

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
    train_dataset, train_stats = prepare_split(
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

    # -------------------------------------------------
    # 7. DYNAMIC PADDING & DATA LOADERS
    # -------------------------------------------------
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    # -------------------------------------------------
    # 8. BUILD XLM-R + CRF MODEL
    # -------------------------------------------------
    print("\n--- INITIALIZING XLM-R + CRF MODEL ---")
    model = XLMRobertaCRF(
        label2id=label2id,
        id2label=id2label,
        model_name=XLMR_MODEL_NAME,
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
    print("\n--- EXPERIMENT 13 CONFIG (XLM-R + CRF) ---")
    print(f"Backbone: {XLMR_MODEL_NAME}")
    print(f"CRF transitions: {len(label2id)}x{len(label2id)} parameters")
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
    print("\n--- STARTING XLM-R + CRF TRAINING ---")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"CRF Loss: {loss.item():.4f}"
                )

        # Train loss
        average_train_loss = total_train_loss / len(train_loader)

        # Validation loss
        validation_loss = evaluate_loss(model, val_loader, device)

        print(f"\nEpoch {epoch + 1} training loss: {average_train_loss:.4f}")
        print(f"Epoch {epoch + 1} validation loss: {validation_loss:.4f}")

        # Entity metrics with Viterbi decoding
        evaluation_output = evaluate_crf_model_entities(
            model=model,
            dataset=val_dataset,
            examples=val_examples,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            batch_size=VAL_BATCH_SIZE,
        )

        entity_results = evaluation_output["metrics"]
        print_entity_report(entity_results)
        print_error_analysis(evaluation_output)

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
    print("XLM-R + CRF TRAINING COMPLETE")
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
