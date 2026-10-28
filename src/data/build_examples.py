from pathlib import Path

import pandas as pd

from src.data.label_conversion import convert_tags_to_bio


TRAIN_FILE = Path("data/raw/Tagged_Titles_Train.tsv.gz")


def load_training_data() -> pd.DataFrame:
    """
    Load the tagged eBay training data while preserving empty Tag values.
    """

    try:
        df = pd.read_csv(
            TRAIN_FILE,
            sep="\t",
            compression="infer",
            low_memory=False,
            keep_default_na=False,
            na_values=None,
        )

        return df

    except Exception as e:
        print(f"[ERROR] Could not load training data from '{TRAIN_FILE}': {e}")
        return pd.DataFrame()


def build_examples(df: pd.DataFrame) -> list[dict]:
    """
    Group token-level rows into one title-level example per Record Number.

    Each example contains:
    - record number
    - category
    - original competition tokens
    - original competition tags
    - converted BIO tags
    """

    examples = []

    grouped = df.groupby("Record Number", sort=False)

    for record_number, group in grouped:
        category = group["Category"].iloc[0]

        tokens = group["Token"].tolist()
        tags = group["Tag"].tolist()

        bio_tags = convert_tags_to_bio(tags)

        example = {
            "record_number": record_number,
            "category": category,
            "tokens": tokens,
            "tags": tags,
            "bio_tags": bio_tags,
        }

        examples.append(example)

    return examples


def validate_examples(examples: list[dict]) -> None:
    """
    Validate reconstructed title-level examples using aggregate checks only.
    """

    mismatched_lengths = 0
    bio_length_mismatches = 0

    examples_with_empty_tags = 0
    examples_with_o_tags = 0
    examples_with_real_aspect_tags = 0

    total_empty_tags = 0
    total_o_tags = 0
    total_real_aspect_tags = 0

    for example in examples:
        tokens = example["tokens"]
        tags = example["tags"]
        bio_tags = example["bio_tags"]

        # Original token/tag alignment check
        if len(tokens) != len(tags):
            mismatched_lengths += 1

        # Token/BIO alignment check
        if len(tokens) != len(bio_tags):
            bio_length_mismatches += 1

        has_empty = False
        has_o = False
        has_real_aspect = False

        for tag in tags:
            if tag == "":
                total_empty_tags += 1
                has_empty = True

            elif tag == "O":
                total_o_tags += 1
                has_o = True

            else:
                total_real_aspect_tags += 1
                has_real_aspect = True

        if has_empty:
            examples_with_empty_tags += 1

        if has_o:
            examples_with_o_tags += 1

        if has_real_aspect:
            examples_with_real_aspect_tags += 1

    print("\n--- EXAMPLE VALIDATION ---")
    print(f"Total Examples: {len(examples):,}")
    print(
        f"Examples with token/tag length mismatch: "
        f"{mismatched_lengths:,}"
    )
    print(
        f"Examples with token/BIO length mismatch: "
        f"{bio_length_mismatches:,}"
    )

    print("\n--- TAG TYPE COUNTS ---")
    print(
        f"Total continuation tags (empty string): "
        f"{total_empty_tags:,}"
    )
    print(f"Total O tags: {total_o_tags:,}")
    print(
        f"Total real aspect tags: "
        f"{total_real_aspect_tags:,}"
    )

    print("\n--- EXAMPLE-LEVEL TAG PRESENCE ---")
    print(
        f"Examples containing continuation tags: "
        f"{examples_with_empty_tags:,}"
    )
    print(
        f"Examples containing O tags: "
        f"{examples_with_o_tags:,}"
    )
    print(
        f"Examples containing real aspect tags: "
        f"{examples_with_real_aspect_tags:,}"
    )


def main() -> None:
    df = load_training_data()

    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)

    print(f"Token-level rows loaded: {len(df):,}")
    print(f"Title-level examples built: {len(examples):,}")

    validate_examples(examples)

    if examples:
        first_example = examples[0]

        print("\n--- SAFE STRUCTURE CHECK ---")
        print(f"Record Number: {first_example['record_number']}")
        print(f"Category: {first_example['category']}")
        print(f"Token count: {len(first_example['tokens'])}")
        print(f"Tag count: {len(first_example['tags'])}")
        print(f"BIO tag count: {len(first_example['bio_tags'])}")

        print(
            "Tokens and tags have same length:",
            len(first_example["tokens"])
            == len(first_example["tags"]),
        )

        print(
            "Tokens and BIO tags have same length:",
            len(first_example["tokens"])
            == len(first_example["bio_tags"]),
        )


if __name__ == "__main__":
    main()