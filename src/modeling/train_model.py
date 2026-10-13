from pathlib import Path

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

from src.modeling.tokenize_and_align import MODEL_NAME
from src.modeling.prepare_datasets import prepare_split
from src.modeling.build_model import build_model

from src.evaluation.evaluate_model import (
    evaluate_model_entities,
)


TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5
EPOCHS = 3

CHECKPOINT_DIR = Path(
    "checkpoints/gbert_baseline_best"
)


def evaluate_loss(
    model,
    dataloader,
    device,
) -> float:
    """
    Calculate average validation loss.

    No gradients or model updates are performed.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value in batch.items()
            }

            outputs = model(**batch)

            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / total_batches


def print_entity_report(
    results: dict,
) -> None:
    """
    Print category and per-aspect entity metrics.
    """

    beta = results["beta"]

    print(
        "\n--- ENTITY-LEVEL VALIDATION ---"
    )

    for category, result in sorted(
        results["categories"].items()
    ):

        print(
            f"\nCategory {category} score: "
            f"{result['score']:.6f}"
        )

        print(
            "Aspect performance "
            "(sorted by validation frequency):"
        )

        sorted_aspects = sorted(
            result["aspects"].items(),
            key=lambda item: (
                item[1]["true_count"]
            ),
            reverse=True,
        )

        for aspect, metrics in sorted_aspects:

            print(
                f"  {aspect}: "
                f"count={metrics['true_count']} | "
                f"P={metrics['precision']:.4f} | "
                f"R={metrics['recall']:.4f} | "
                f"F{beta}="
                f"{metrics['f_beta']:.4f} | "
                f"weight={metrics['weight']:.4f}"
            )

    print(
        f"\nLOCAL COMPETITION-STYLE "
        f"F{beta}: "
        f"{results['final_score']:.6f}"
    )


def main() -> None:

    # -------------------------------------------------
    # 1. GPU CHECK
    # -------------------------------------------------

    print("--- CUDA CHECK ---")

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Stopping."
        )
        return

    device = torch.device("cuda")

    print(f"Device: {device}")
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # -------------------------------------------------
    # 2. TOKENIZER
    # -------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # -------------------------------------------------
    # 3. LOAD DATA
    # -------------------------------------------------

    print("\n--- LOADING DATA ---")

    df = load_training_data()

    if df.empty:
        print(
            "No training data loaded. Stopping."
        )
        return

    examples = build_examples(df)

    print(
        f"Total labeled titles: "
        f"{len(examples)}"
    )

    # -------------------------------------------------
    # 4. LABEL MAPPING
    # -------------------------------------------------

    label2id, id2label = (
        create_label_mappings(
            examples
        )
    )

    print(
        f"BIO label classes: "
        f"{len(label2id)}"
    )

    # -------------------------------------------------
    # 5. TRAIN / VALIDATION SPLIT
    # -------------------------------------------------

    train_examples, val_examples = (
        split_examples(
            examples
        )
    )

    print(
        f"Training titles: "
        f"{len(train_examples)}"
    )

    print(
        f"Validation titles: "
        f"{len(val_examples)}"
    )

    # -------------------------------------------------
    # 6. TOKENIZE + ALIGN
    # -------------------------------------------------

    print(
        "\n--- PREPARING DATASETS ---"
    )

    train_dataset, train_stats = (
        prepare_split(
            train_examples,
            tokenizer,
            label2id,
        )
    )

    val_dataset, val_stats = (
        prepare_split(
            val_examples,
            tokenizer,
            label2id,
        )
    )

    print(
        f"Training alignment mismatches: "
        f"{train_stats['alignment_mismatches']}"
    )

    print(
        f"Validation alignment mismatches: "
        f"{val_stats['alignment_mismatches']}"
    )

    if (
        train_stats["alignment_mismatches"] > 0
        or
        val_stats["alignment_mismatches"] > 0
    ):
        raise ValueError(
            "Alignment mismatches detected."
        )

    # -------------------------------------------------
    # 7. DYNAMIC PADDING
    # -------------------------------------------------

    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer=tokenizer,
        )
    )

    # -------------------------------------------------
    # 8. DATA LOADERS
    # -------------------------------------------------

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
    # 9. BUILD MODEL
    # -------------------------------------------------

    model = build_model(
        label2id,
        id2label,
    )

    model.to(device)

    # -------------------------------------------------
    # 10. OPTIMIZER
    # -------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------
    # 11. TRAINING CONFIG
    # -------------------------------------------------

    print(
        "\n--- FULL BASELINE CONFIG ---"
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")

    print(
        f"Training batch size: "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"Validation batch size: "
        f"{VAL_BATCH_SIZE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Training steps / epoch: "
        f"{len(train_loader)}"
    )

    # -------------------------------------------------
    # 12. TRACK BEST MODEL
    # -------------------------------------------------

    best_f_beta = -1.0
    best_epoch = 0

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # 13. TRAINING LOOP
    # -------------------------------------------------

    print(
        "\n--- STARTING FULL TRAINING ---"
    )

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        print(
            f"\n===== EPOCH "
            f"{epoch + 1}/{EPOCHS} ====="
        )

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value
                in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
            )

            if step % 100 == 0:

                print(
                    f"Epoch {epoch + 1} | "
                    f"Step "
                    f"{step}/{len(train_loader)} | "
                    f"Loss: "
                    f"{loss.item():.4f}"
                )

        # ---------------------------------------------
        # TRAIN LOSS
        # ---------------------------------------------

        average_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        # ---------------------------------------------
        # VALIDATION LOSS
        # ---------------------------------------------

        validation_loss = evaluate_loss(
            model,
            val_loader,
            device,
        )

        print(
            f"\nEpoch {epoch + 1} "
            f"training loss: "
            f"{average_train_loss:.4f}"
        )

        print(
            f"Epoch {epoch + 1} "
            f"validation loss: "
            f"{validation_loss:.4f}"
        )

        # ---------------------------------------------
        # ENTITY METRICS
        # ---------------------------------------------

        entity_results = (
            evaluate_model_entities(
                model=model,
                dataset=val_dataset,
                examples=val_examples,
                tokenizer=tokenizer,
                id2label=id2label,
                device=device,
                batch_size=VAL_BATCH_SIZE,
            )
        )

        print_entity_report(
            entity_results
        )

        current_f_beta = (
            entity_results[
                "final_score"
            ]
        )

        # ---------------------------------------------
        # SAVE BEST CHECKPOINT
        # ---------------------------------------------

        if current_f_beta > best_f_beta:

            best_f_beta = (
                current_f_beta
            )

            best_epoch = epoch + 1

            model.save_pretrained(
                CHECKPOINT_DIR
            )

            tokenizer.save_pretrained(
                CHECKPOINT_DIR
            )

            print(
                "\nNEW BEST MODEL"
            )

            print(
                f"Saved epoch "
                f"{best_epoch} with "
                f"F{entity_results['beta']}="
                f"{best_f_beta:.6f}"
            )

    # -------------------------------------------------
    # 14. FINAL SUMMARY
    # -------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "FULL BASELINE TRAINING COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best local F0.2: "
        f"{best_f_beta:.6f}"
    )

    # -------------------------------------------------
    # 15. GPU MEMORY
    # -------------------------------------------------

    allocated_gb = (
        torch.cuda.memory_allocated(0)
        / 1024**3
    )

    reserved_gb = (
        torch.cuda.memory_reserved(0)
        / 1024**3
    )

    peak_gb = (
        torch.cuda.max_memory_allocated(0)
        / 1024**3
    )

    print(
        f"Allocated GPU memory: "
        f"{allocated_gb:.2f} GB"
    )

    print(
        f"Reserved GPU memory: "
        f"{reserved_gb:.2f} GB"
    )

    print(
        f"Peak GPU memory: "
        f"{peak_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()