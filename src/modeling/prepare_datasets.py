from datasets import Dataset
from transformers import AutoTokenizer

from src.data.build_examples import (
    load_training_data,
    build_examples,
)
from src.data.make_splits import split_examples
from src.data.label_mapping import create_label_mappings
from src.modeling.tokenize_and_align import (
    tokenize_and_align_example,
    MODEL_NAME,
    IGNORE_LABEL_ID,
)


def prepare_split(
    examples: list[dict],
    tokenizer,
    label2id: dict,
) -> tuple[Dataset, dict]:
    """
    Tokenize and align every example in one split.

    Returns:
    - Hugging Face Dataset
    - aggregate validation statistics
    """

    processed_examples = []
    
    alignment_mismatches = 0
    total_active_labels = 0
    total_ignored_positions = 0
    max_transformer_length = 0
    
    for example in examples:
        encoded = tokenize_and_align_example(
            example,   
            tokenizer,
            label2id,
        )
        labels = encoded["labels"]
        
        active_labels = sum(
            label != IGNORE_LABEL_ID
            for label in labels
        )
        
        ignored_positions = sum(
            label == IGNORE_LABEL_ID
            for label in labels
        )
        
        if active_labels != len(example["tokens"]):
            alignment_mismatches += 1
            
        total_active_labels += active_labels
        total_ignored_positions += ignored_positions
        
        max_transformer_length = max(
            max_transformer_length,
            len(encoded["input_ids"]),
        )
        
        processed_examples.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "token_type_ids": encoded.get("token_type_ids", []),
                "labels": encoded["labels"],
            }
        )
        
    dataset = Dataset.from_list(processed_examples)
    
    stats = {
        "examples": len(examples),
        "alignment_mismatches": alignment_mismatches,
        "active_labels": total_active_labels,
        "ignored_positions": total_ignored_positions,
        "max_transformer_length": max_transformer_length,
    }
    
    return dataset, stats


def print_stats(name: str, stats: dict) -> None:
    print(f"\n--- {name.upper()} DATASET ---")
    print(f"Examples: {stats['examples']:,}")

    print(
        f"Alignment mismatches: "
        f"{stats['alignment_mismatches']:,}"
    )

    print(
        f"Active training labels: "
        f"{stats['active_labels']:,}"
    )

    print(
        f"Ignored positions (-100): "
        f"{stats['ignored_positions']:,}"
    )

    print(
        f"Maximum transformer sequence length: "
        f"{stats['max_transformer_length']:,}"
    )
    
def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = load_training_data()

    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)

    label2id, id2label = create_label_mappings(examples)

    train_examples, val_examples = split_examples(examples)

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

    print("--- DATASET PREPARATION SUMMARY ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Number of BIO labels: {len(label2id):,}")

    print_stats("training", train_stats)
    print_stats("validation", val_stats)

    print("\n--- HUGGING FACE DATASET CHECK ---")
    print(f"Training rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(val_dataset):,}")
    print(f"Columns: {train_dataset.column_names}")

    print(
        "All training examples aligned:",
        train_stats["alignment_mismatches"] == 0,
    )

    print(
        "All validation examples aligned:",
        val_stats["alignment_mismatches"] == 0,
    )


if __name__ == "__main__":
    main()