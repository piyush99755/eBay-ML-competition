import torch

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import (
    calculate_competition_metrics,
)


IGNORE_LABEL_ID = -100

def evaluate_model_entities(
     model,
    dataset,
    examples: list[dict],
    tokenizer,
    id2label: dict,
    device,
    batch_size: int = 8,
)-> dict:
    """
    Run the trained model on validation data and calculate
    local competition-style entity metrics.

    Flow:
    validation dataset
        -> model logits
        -> predicted label IDs
        -> BIO labels
        -> entities
        -> competition metrics
    """
    
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )
    
    model.eval()
    
    records = []
    
    example_index = 0
    
    with torch.no_grad():
        for batch in dataloader:
            
            batch = {
                key:value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            
            outputs = model(**batch)
            
            logits = outputs.logits
            
            predicted_ids = torch.argmax(
                logits,
                dim=-1,
            )
            
            labels = batch["labels"]

            batch_size_actual = labels.shape[0]
            
            for row_index in range(batch_size_actual):

                example = examples[example_index]

                row_labels = labels[row_index]

                row_predictions = predicted_ids[row_index]

                # Only keep positions corresponding to the
                # original competition tokens.
                active_mask = (
                    row_labels != IGNORE_LABEL_ID
                )

                active_prediction_ids = (
                    row_predictions[active_mask]
                    .detach()
                    .cpu()
                    .tolist()
                )

                predicted_bio = [
                    id2label[label_id]
                    for label_id
                    in active_prediction_ids
                ]

                true_bio = example["bio_tags"]
                tokens = example["tokens"]

                if len(predicted_bio) != len(tokens):
                    raise ValueError(
                        "Predicted BIO label count does not "
                        "match original token count."
                    )

                true_entities = bio_to_entities(
                    tokens,
                    true_bio,
                )

                predicted_entities = bio_to_entities(
                    tokens,
                    predicted_bio,
                )

                records.append(
                    {
                        "category": example["category"],
                        "true_entities": true_entities,
                        "predicted_entities": predicted_entities,
                    }
                )

                example_index += 1

    if example_index != len(examples):
        raise ValueError(
            "Not all validation examples were evaluated."
        )

    results = calculate_competition_metrics(
        records
    )

    return results
