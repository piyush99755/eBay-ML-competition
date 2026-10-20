"""
Split-B Robustness Training: XLM-RoBERTa (seeds 42, 7, 21)
Uses random_state=123 for 80/20 train/val split.
Hyperparameters match the baseline runs:
- 4,000 train / 1,000 val
- 3 epochs
- learning rate 2e-5
- batch sizes 4 (train), 8 (val)
"""

from pathlib import Path
import random
import argparse
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
from src.data.make_splits import split_examples_robust

from src.modeling.prepare_datasets import prepare_split
from src.modeling.build_model import build_model

from src.evaluation.evaluate_model import (
    evaluate_model_entities,
)

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

SPLIT_SEED = 123
XLMR_MODEL_NAME = "FacebookAI/xlm-roberta-base"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_loss(
    model,
    dataloader,
    device,
) -> float:
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
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / total_batches


def print_entity_report(
    results: dict,
) -> None:
    beta = results["beta"]
    categories = results["categories"]

    print("\n--- ENTITY-LEVEL VALIDATION (SPLIT B) ---")

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
            print(
                f"  {aspect_name}: "
                f"count={metrics['true_count']} | "
                f"P={metrics['precision']:.4f} | "
                f"R={metrics['recall']:.4f} | "
                f"F{beta}={metrics['f_beta']:.4f} | "
                f"weight={metrics['weight']:.4f}"
            )

    print(f"\nLOCAL COMPETITION-STYLE F{beta}: {results['final_score']:.6f}")


def print_error_analysis(
    evaluation_output: dict,
) -> None:
    confusion_counts = analyze_token_confusions(
        evaluation_output["true_bio_sequences"],
        evaluation_output["predicted_bio_sequences"],
    )
    aspect_errors = analyze_entity_errors(
        evaluation_output["records"]
    )
    print_top_confusions(confusion_counts, top_n=25)
    print_entity_error_summary(aspect_errors)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XLM-R on Split B (random_state=123) with a specific seed."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model initialization and training (e.g. 42, 7, 21).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random_seed = args.seed

    set_random_seed(random_seed)

    checkpoint_dir = Path(f"checkpoints/robust_split123_xlmr_seed{random_seed}_best")

    print("=" * 70)
    print(f"ROBUSTNESS TRAINING: XLM-R SEED {random_seed} ON SPLIT B (random_state=123)")
    print("=" * 70)
    print(f"Model: {XLMR_MODEL_NAME}")
    print(f"Model seed: {random_seed}")
    print(f"Split seed: {SPLIT_SEED}")
    print(f"Target checkpoint: {checkpoint_dir}")

    # 1. GPU CHECK
    if not torch.cuda.is_available():
        print("CUDA is not available. Stopping.")
        return

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.cuda.reset_peak_memory_stats(device)

    # 2. TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL_NAME)

    # 3. LOAD DATA & SPLIT B
    print("\n--- LOADING DATA & APPLYING SPLIT B ---")
    df = load_training_data()
    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)

    train_examples, val_examples = split_examples_robust(
        examples,
        split_seed=SPLIT_SEED,
    )
    print(f"Training titles (Split B): {len(train_examples):,}")
    print(f"Validation titles (Split B): {len(val_examples):,}")

    # 4. PREPARE DATASETS
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

    if train_stats["alignment_mismatches"] > 0 or val_stats["alignment_mismatches"] > 0:
        raise ValueError("Alignment mismatches detected.")

    # 5. DATA LOADERS
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
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

    # 6. BUILD MODEL & OPTIMIZER
    model = build_model(
        label2id,
        id2label,
        model_name=XLMR_MODEL_NAME,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 7. TRAINING LOOP
    print("\n--- STARTING TRAINING (SPLIT B) ---")
    best_f_beta = -1.0
    best_epoch = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                    f"Loss: {loss.item():.4f}"
                )

        average_train_loss = total_train_loss / len(train_loader)
        validation_loss = evaluate_loss(model, val_loader, device)

        print(f"\nEpoch {epoch + 1} training loss: {average_train_loss:.4f}")
        print(f"Epoch {epoch + 1} validation loss: {validation_loss:.4f}")

        evaluation_output = evaluate_model_entities(
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

        if current_f_beta > best_f_beta:
            best_f_beta = current_f_beta
            best_epoch = epoch + 1
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(
                f"\nNEW BEST MODEL: Saved epoch {best_epoch} with "
                f"F{entity_results['beta']}={best_f_beta:.6f}"
            )

    print("\n" + "=" * 70)
    print(
        f"TRAINING COMPLETE (XLM-R seed {random_seed} Split B): "
        f"Best Epoch {best_epoch} | Best F0.2: {best_f_beta:.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
