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


TRAIN_SMOKE_SIZE = 200
VAL_SMOKE_SIZE = 50

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5
EPOCHS = 1


def evaluate(
    model,
    dataloader,
    device,
) -> float:
    """
    Calculate average validation loss.

    Validation does not update model weights.
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

            loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def main() -> None:

    # -------------------------------------------------
    # 1. CHECK CUDA / GPU
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
    # 2. LOAD TOKENIZER
    # -------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # -------------------------------------------------
    # 3. LOAD TRAINING DATA
    # -------------------------------------------------

    df = load_training_data()

    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)

    # -------------------------------------------------
    # 4. CREATE LABEL MAPPINGS
    # -------------------------------------------------

    label2id, id2label = create_label_mappings(
        examples
    )

    # -------------------------------------------------
    # 5. CREATE TRAIN / VALIDATION SPLITS
    # -------------------------------------------------

    train_examples, val_examples = split_examples(
        examples
    )

    # Smoke test uses only small subsets.
    train_examples = train_examples[
        :TRAIN_SMOKE_SIZE
    ]

    val_examples = val_examples[
        :VAL_SMOKE_SIZE
    ]

    print("\n--- SMOKE TEST DATA ---")

    print(
        f"Training examples: "
        f"{len(train_examples)}"
    )

    print(
        f"Validation examples: "
        f"{len(val_examples)}"
    )

    # -------------------------------------------------
    # 6. TOKENIZE + ALIGN LABELS
    # -------------------------------------------------

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

    print(
        f"Training alignment mismatches: "
        f"{train_stats['alignment_mismatches']}"
    )

    print(
        f"Validation alignment mismatches: "
        f"{val_stats['alignment_mismatches']}"
    )

    # -------------------------------------------------
    # 7. CREATE DYNAMIC PADDING COLLATOR
    # -------------------------------------------------

    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer=tokenizer,
        )
    )

    # -------------------------------------------------
    # 8. CREATE DATA LOADERS
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
    # 10. CREATE OPTIMIZER
    # -------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\n--- TRAINING CONFIG ---")

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

    # -------------------------------------------------
    # 11. TRAINING LOOP
    # -------------------------------------------------

    print("\n--- STARTING GPU SMOKE TEST ---")

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value in batch.items()
            }

            # Step 1:
            # Clear gradients from previous batch.
            optimizer.zero_grad()

            # Step 2:
            # Forward pass through GBERT.
            outputs = model(**batch)

            # Step 3:
            # Get training loss.
            loss = outputs.loss

            # Step 4:
            # Calculate gradients.
            loss.backward()

            # Step 5:
            # Update model weights.
            optimizer.step()

            total_train_loss += loss.item()

            # Print progress every 10 batches.
            if step % 10 == 0:

                print(
                    f"Epoch {epoch + 1} | "
                    f"Step "
                    f"{step}/{len(train_loader)} | "
                    f"Loss: "
                    f"{loss.item():.4f}"
                )

        # -------------------------------------------------
        # 12. CALCULATE EPOCH TRAINING LOSS
        # -------------------------------------------------

        average_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        # -------------------------------------------------
        # 13. CALCULATE VALIDATION LOSS
        # -------------------------------------------------

        validation_loss = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"\n--- EPOCH "
            f"{epoch + 1} COMPLETE ---"
        )

        print(
            f"Average training loss: "
            f"{average_train_loss:.4f}"
        )

        print(
            f"Average validation loss: "
            f"{validation_loss:.4f}"
        )

        # -------------------------------------------------
        # 14. ENTITY-LEVEL VALIDATION
        # -------------------------------------------------

        print(
            "\n--- ENTITY-LEVEL VALIDATION ---"
        )

        entity_results = evaluate_model_entities(
            model=model,
            dataset=val_dataset,
            examples=val_examples,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            batch_size=VAL_BATCH_SIZE,
        )

        for category, result in sorted(
            entity_results["categories"].items()
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
                key=lambda item: item[1]["true_count"],
                reverse=True,
            )

            for aspect, metrics in sorted_aspects:
                print(
                    f"  {aspect}: "
                    f"count={metrics['true_count']} | "
                    f"P={metrics['precision']:.4f} | "
                    f"R={metrics['recall']:.4f} | "
                    f"F{entity_results['beta']}="
                    f"{metrics['f_beta']:.4f} | "
                    f"weight={metrics['weight']:.4f}"
                )

        print(
            f"\nLOCAL COMPETITION-STYLE "
            f"F{entity_results['beta']}: "
            f"{entity_results['final_score']:.6f}"
        )

    # -------------------------------------------------
    # 15. GPU MEMORY REPORT
    # -------------------------------------------------

    print("\n--- GPU MEMORY ---")

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
        f"Allocated memory: "
        f"{allocated_gb:.2f} GB"
    )

    print(
        f"Reserved memory: "
        f"{reserved_gb:.2f} GB"
    )

    print(
        f"Peak allocated memory: "
        f"{peak_gb:.2f} GB"
    )

    print("\nSMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()