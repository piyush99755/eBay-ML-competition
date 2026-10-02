from transformers import AutoTokenizer


MODEL_NAME = "deepset/gbert-base"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("--- TOKENIZER INFO ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Model max length: {tokenizer.model_max_length}")

    # Completely synthetic example.
    words = [
        "Bosch",
        "Bremsscheiben",
        "Herstellernummer",
    ]

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    print("\n--- SYNTHETIC TOKENIZATION EXAMPLE ---")
    print(f"Original words: {words}")
    print(f"Transformer tokens: {tokens}")
    print(f"Input IDs: {encoded['input_ids']}")

    print("\n--- WORD ID ALIGNMENT ---")
    print(encoded.word_ids())


if __name__ == "__main__":
    main()