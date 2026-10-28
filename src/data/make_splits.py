from sklearn.model_selection import train_test_split
from src.data.build_examples import load_training_data, build_examples

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

def split_examples(examples, random_state: int = RANDOM_STATE):
    """Split title-level examples into train and validation sets.

    Stratification keeps the two categories represented
    in similar proportions in both splits."""
    
    categories = [example["category"] for example in examples]
    
    train_examples, val_examples = train_test_split(
        examples,
        test_size=VALIDATION_SIZE,
        random_state=random_state,
        stratify=categories,
    )
    
    return train_examples, val_examples


def split_examples_robust(examples, split_seed: int = 123):
    """
    Produce the 80/20 stratified train/validation split using split_seed (default 123)
    for robustness experiments.
    """
    return split_examples(examples, random_state=split_seed)

def count_categories(examples):
    """Count title-level examples per category."""
    counts = {}
    for example in examples:
        category = example["category"]
        counts[category] = counts.get(category, 0) + 1
        
    return counts

def main() -> None:
    df = load_training_data()

    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)

    

    train_examples, val_examples = split_examples(examples)

    print("--- TITLE-LEVEL SPLIT ---")
    print(f"Total examples: {len(examples):,}")
    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")

    print("\n--- TRAIN CATEGORY COUNTS ---")
    for category, count in sorted(count_categories(train_examples).items()):
        print(f"Category {category}: {count:,}")

    print("\n--- VALIDATION CATEGORY COUNTS ---")
    for category, count in sorted(count_categories(val_examples).items()):
        print(f"Category {category}: {count:,}")

    train_ids = {
        example["record_number"] for example in train_examples
    }
    
    val_ids = {
        example["record_number"] for example in val_examples
    }
    
    overlap = train_ids.intersection(val_ids)

    print("\n--- LEAKAGE CHECK ---")
    print(f"Record Numbers appearing in both splits: {len(overlap):,}")


if __name__ == "__main__":
    main()