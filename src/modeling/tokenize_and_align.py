from transformers import AutoTokenizer
from src.data.build_examples import (
    load_training_data,
    build_examples,
)
from src.data.label_mapping import create_label_mappings

MODEL_NAME = "deepset/gbert-base"
IGNORE_LABEL_ID = -100

def tokenize_and_align_example(
    example: dict,
    tokenizer,
    label2id: dict,
) -> dict:
    """
    Tokenize one title-level example with GBERT and align BIO labels
    to the resulting transformer subword tokens.

    Strategy:
    - First subword of each original competition token gets the BIO label.
    - Additional subwords get -100.
    - Special tokens like [CLS] and [SEP] get -100.
    """

    tokens = example["tokens"]
    bio_tags = example["bio_tags"]
    
    encoded = tokenizer(
        tokens,
        truncation=True,
        max_length=512,
        is_split_into_words=True,
        add_special_tokens=True,
    )
    
    word_ids = encoded.word_ids()
    
    aligned_labels = []
    
    previous_word_id = None
    
    for word_id in word_ids:
        
        # Special tokens such as [CLS] and [SEP]
        if word_id is None:
            aligned_labels.append(IGNORE_LABEL_ID)
            
        # First transformer subword for this original competition token
        elif word_id != previous_word_id:
            bio_label = bio_tags[word_id]
            label_id = label2id[bio_label]
            
            aligned_labels.append(label_id)
        else:
            aligned_labels.append(IGNORE_LABEL_ID)
        
        previous_word_id = word_id
        
    encoded["labels"] = aligned_labels
    
    return encoded

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    df = load_training_data()
    
    if df.empty:
        print("No training data loaded. Stopping.")
        return
    
    examples = build_examples(df)
    label2id, _ = create_label_mappings(examples)
    
    first_example = examples[0]
    
    encoded = tokenize_and_align_example(
        first_example,  
        tokenizer,
        label2id,
    )
    
    transformer_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    
    word_ids = encoded.word_ids()
    
    labels = encoded["labels"]
    
    print("--- TOKENIZATION / ALIGNMENT CHECK ---")
    print(f"Original competition token count: {len(first_example['tokens'])}")
    print(f"Transformer token count: {len(transformer_tokens)}")
    print(f"Aligned label count: {len(labels)}")

    print(
        "Transformer tokens and labels same length:",
        len(transformer_tokens) == len(labels),
    )

    print("\n--- ALIGNMENT SUMMARY ---")
    
    special_or_ignored = 0
    active_labels = 0
    
    for label_id in labels:
        if label_id == IGNORE_LABEL_ID:
            special_or_ignored += 1
        else:
            active_labels += 1
            
    print(f"Active training labels: {active_labels}")
    print(f"Ignored positions (-100): {special_or_ignored}")

    print(
        "Active labels equal original competition tokens:",
        active_labels == len(first_example["tokens"]),
    )


if __name__ == "__main__":
    main()