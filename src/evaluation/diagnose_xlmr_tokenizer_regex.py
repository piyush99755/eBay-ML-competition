"""
Diagnostic script: Compares default XLM-R tokenization against fix_mistral_regex=True
across the 5,000 labeled competition examples.

Privacy & Safety:
- Aggregate statistics only.
- Does NOT print raw titles, tokens, record IDs, aspect values, or competition text.
- Does NOT run model training or model inference.
"""

from transformers import AutoTokenizer

from src.data.build_examples import (
    build_examples,
    load_training_data,
)
from src.data.label_mapping import create_label_mappings
from src.modeling.tokenize_and_align import (
    tokenize_and_align_example,
    IGNORE_LABEL_ID,
)


MODEL_NAME = "FacebookAI/xlm-roberta-base"


def main() -> None:
    print("=" * 70)
    print("DIAGNOSTIC: XLM-R TOKENIZER REGEX COMPARISON")
    print("=" * 70)

    # -------------------------------------------------
    # 1. LOAD DATASET
    # -------------------------------------------------
    df = load_training_data()
    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)
    label2id, _ = create_label_mappings(examples)

    total_examples = len(examples)
    print(f"Total labeled examples loaded: {total_examples:,}")

    # -------------------------------------------------
    # 2. LOAD BOTH TOKENIZERS
    # -------------------------------------------------
    print(f"\nLoading default tokenizer ({MODEL_NAME})...")
    tokenizer_default = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading fixed-regex tokenizer ({MODEL_NAME}, fix_mistral_regex=True)...")
    tokenizer_fixed = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        fix_mistral_regex=True,
    )

    # -------------------------------------------------
    # 3. COMPARE TOKENIZATION ACROSS ALL EXAMPLES
    # -------------------------------------------------
    print("\nComparing tokenization and alignment across all examples...")

    diff_input_ids_count = 0
    diff_length_count = 0
    max_length_diff = 0

    mismatches_default = 0
    mismatches_fixed = 0

    total_tokens_default = 0
    total_tokens_fixed = 0

    for example in examples:
        tokens = example["tokens"]
        num_original_tokens = len(tokens)

        # Tokenize with default behavior
        enc_default = tokenize_and_align_example(
            example,
            tokenizer_default,
            label2id,
        )

        # Tokenize with fix_mistral_regex=True
        enc_fixed = tokenize_and_align_example(
            example,
            tokenizer_fixed,
            label2id,
        )

        ids_default = enc_default["input_ids"]
        ids_fixed = enc_fixed["input_ids"]

        len_default = len(ids_default)
        len_fixed = len(ids_fixed)

        total_tokens_default += len_default
        total_tokens_fixed += len_fixed

        # Check input_ids difference
        if ids_default != ids_fixed:
            diff_input_ids_count += 1

        # Check tokenized length difference
        if len_default != len_fixed:
            diff_length_count += 1

        length_diff = abs(len_default - len_fixed)
        if length_diff > max_length_diff:
            max_length_diff = length_diff

        # Check original-token alignment mismatches
        active_default = sum(
            1 for label in enc_default["labels"] if label != IGNORE_LABEL_ID
        )
        if active_default != num_original_tokens:
            mismatches_default += 1

        active_fixed = sum(
            1 for label in enc_fixed["labels"] if label != IGNORE_LABEL_ID
        )
        if active_fixed != num_original_tokens:
            mismatches_fixed += 1

    # -------------------------------------------------
    # 4. REPORT AGGREGATE STATISTICS ONLY
    # -------------------------------------------------
    pct_input_ids_diff = (
        (diff_input_ids_count / total_examples) * 100 if total_examples > 0 else 0.0
    )
    pct_length_diff = (
        (diff_length_count / total_examples) * 100 if total_examples > 0 else 0.0
    )

    print("\n" + "=" * 70)
    print("                 AGGREGATE DIAGNOSTIC RESULTS")
    print("=" * 70)
    print(f"Total examples checked:                        {total_examples:,}")
    print(
        f"Examples with differing input_ids:            {diff_input_ids_count:,} "
        f"({pct_input_ids_diff:.2f}%)"
    )
    print(
        f"Examples with differing tokenized length:     {diff_length_count:,} "
        f"({pct_length_diff:.2f}%)"
    )
    print(f"Maximum tokenized-length difference:          {max_length_diff} tokens")
    print("-" * 70)
    print(f"Default tokenizer alignment mismatches:       {mismatches_default:,}")
    print(f"Fixed regex tokenizer alignment mismatches:   {mismatches_fixed:,}")
    print("-" * 70)
    avg_len_default = (
        total_tokens_default / total_examples if total_examples > 0 else 0.0
    )
    avg_len_fixed = (
        total_tokens_fixed / total_examples if total_examples > 0 else 0.0
    )
    print(f"Default tokenizer average token count:        {avg_len_default:.2f}")
    print(f"Fixed regex tokenizer average token count:    {avg_len_fixed:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
