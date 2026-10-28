from src.data.build_examples import (
    load_training_data,
    build_examples,
)


def create_label_mappings(examples: list[dict]) -> tuple[dict, dict]:
    """
    Create mappings between BIO label strings and integer IDs.
    """

    unique_labels = set()

    for example in examples:
        for label in example["bio_tags"]:
            unique_labels.add(label)

    sorted_labels = sorted(unique_labels)

    label2id = {
        label: index
        for index, label in enumerate(sorted_labels)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    return label2id, id2label


def main() -> None:
    df = load_training_data()

    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)

    label2id, id2label = create_label_mappings(examples)

    print("--- LABEL MAPPING SUMMARY ---")
    print(f"Number of unique labels: {len(label2id):,}")

    print("\n--- LABEL TO ID ---")

    for label, label_id in label2id.items():
        print(f"{label_id:>2} -> {label}")

    print("\n--- MAPPING CHECK ---")

    mapping_is_consistent = True

    for label, label_id in label2id.items():
        if id2label[label_id] != label:
            mapping_is_consistent = False
            break

    print(f"Forward/reverse mapping consistent: {mapping_is_consistent}")


if __name__ == "__main__":
    main()