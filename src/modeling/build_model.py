import torch
from torchgen import model
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data.build_examples import (
    load_training_data,
    build_examples,
)
from src.data.label_mapping import create_label_mappings
from src.modeling.tokenize_and_align import MODEL_NAME

def build_model(
    label2id: dict,
    id2label: dict,
):
    """
        Load pretrained GBERT and attach a token-classification head
        for our BIO label space.
        """
     
    
    model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
                            num_labels=len(label2id),
                            label2id=label2id,
                            id2label=id2label,
                        )
                            
    return model
    
    
    

def count_parameters(model) -> tuple[int, int]:
    """
    Count total and trainable model parameters.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def main() -> None:
    df = load_training_data()

    if df.empty:
        print("No training data loaded. Stopping.")
        return

    examples = build_examples(df)

    label2id, id2label = create_label_mappings(examples)

    model = build_model(
        label2id,
        id2label,
    )

    total_parameters, trainable_parameters = count_parameters(model)

    print("--- MODEL SUMMARY ---")
    print(f"Base model: {MODEL_NAME}")
    print(f"Model class: {model.__class__.__name__}")

    print(f"Number of output labels: {model.config.num_labels}")

    print(
        f"Classifier output size: "
        f"{model.classifier.out_features}"
    )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    print("\n--- DEVICE CHECK ---")

    if torch.cuda.is_available():
        print("CUDA GPU available: True")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU available: False")
        print("Training will use CPU unless another accelerator is configured.")

    print("\n--- LABEL CONFIG CHECK ---")
    print(
        "Model label count matches our mapping:",
        model.config.num_labels == len(label2id),
    )

    print(
        "Model label2id matches our mapping:",
        model.config.label2id == label2id,
    )


if __name__ == "__main__":
    main()