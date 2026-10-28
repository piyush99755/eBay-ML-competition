from collections import Counter

from src.data.build_examples import (
    load_training_data,
    build_examples,
)


def analyze_bio_labels(examples: list[dict]) -> None:
    """
    Analyze BIO labels across all title-level examples.
    Only aggregate label counts are printed.
    """

    label_counts = Counter()

    for example in examples:
        bio_tags = example["bio_tags"]

        for label in bio_tags:
            label_counts[label] += 1

    print("--- BIO LABEL SPACE ---")
    print(f"Unique BIO labels: {len(label_counts):,}")
    print(f"Total BIO labels: {sum(label_counts.values()):,}")

    print("\n--- BIO LABEL FREQUENCIES ---")

    for label, count in label_counts.most_common():
        print(f"{label}: {count:,}")


def main() -> None:
    df = load_training_data()

    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)

    analyze_bio_labels(examples)


if __name__ == "__main__":
    main()