from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from src.data.build_examples import (
    build_examples,
    load_training_data,
)
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import calculate_competition_metrics


GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
XLMR_SEED_42_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")
XLMR_SEED_7_CHECKPOINT = Path("checkpoints/xlmr_seed_7_best")
XLMR_SEED_21_CHECKPOINT = Path("checkpoints/xlmr_seed_21_best")

VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100

CURRENT_CHAMPION_SCORE = 0.878695


def collect_active_logits(
    model,
    dataset,
    tokenizer,
    device,
) -> list[list[list[float]]]:
    """
    Run evaluation and extract logits for active original tokens only.
    """
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    model.eval()
    all_active_logits = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            batch_size_actual = labels.shape[0]

            for row_index in range(batch_size_actual):
                row_labels = labels[row_index]
                row_logits = logits[row_index]

                active_mask = row_labels != IGNORE_LABEL_ID
                active_logits = (
                    row_logits[active_mask]
                    .detach()
                    .cpu()
                    .tolist()
                )

                all_active_logits.append(active_logits)

    return all_active_logits


def ensemble_3seed_xlmr_predictions(
    seed_42_logits,
    seed_7_logits,
    seed_21_logits,
    id2label: dict,
) -> list[list[str]]:
    """
    Convert logits of the 3 XLM-R models to softmax probabilities,
    average equally (1/3 each) at original-token level, and argmax the final BIO label.
    """
    predictions = []

    num_examples = len(seed_42_logits)
    if not (len(seed_7_logits) == num_examples and len(seed_21_logits) == num_examples):
        raise ValueError("Example counts across the three XLM-R seeds do not match.")

    for example_index, (ex_42, ex_7, ex_21) in enumerate(
        zip(seed_42_logits, seed_7_logits, seed_21_logits)
    ):
        if not (len(ex_42) == len(ex_7) == len(ex_21)):
            raise ValueError(
                f"Token count mismatch at validation example {example_index}: "
                f"seed_42={len(ex_42)}, seed_7={len(ex_7)}, seed_21={len(ex_21)}"
            )

        example_predictions = []

        for t_42, t_7, t_21 in zip(ex_42, ex_7, ex_21):
            tensor_42 = torch.tensor(t_42, dtype=torch.float32)
            tensor_7 = torch.tensor(t_7, dtype=torch.float32)
            tensor_21 = torch.tensor(t_21, dtype=torch.float32)

            prob_42 = torch.softmax(tensor_42, dim=-1)
            prob_7 = torch.softmax(tensor_7, dim=-1)
            prob_21 = torch.softmax(tensor_21, dim=-1)

            combined_probabilities = (prob_42 + prob_7 + prob_21) / 3.0

            predicted_id = int(torch.argmax(combined_probabilities).item())
            example_predictions.append(id2label[predicted_id])

        predictions.append(example_predictions)

    return predictions


def ensemble_gbert_multiseed_xlmr_predictions(
    gbert_logits,
    seed_42_logits,
    seed_7_logits,
    seed_21_logits,
    id2label: dict,
    gbert_weight: float = 0.40,
    xlmr_weight: float = 0.60,
) -> list[list[str]]:
    """
    Combine GBERT and 3-seed XLM-R probabilities:
    1. Average the 3 XLM-R seed probabilities equally: (p42 + p7 + p21) / 3
    2. Combine: gbert_weight * GBERT_prob + xlmr_weight * XLMR_multiseed_prob
    3. Argmax to predict the final BIO label.
    """
    predictions = []

    num_examples = len(gbert_logits)
    if not (
        len(seed_42_logits) == num_examples
        and len(seed_7_logits) == num_examples
        and len(seed_21_logits) == num_examples
    ):
        raise ValueError("Example counts across models do not match.")

    for example_index, (ex_gbert, ex_42, ex_7, ex_21) in enumerate(
        zip(gbert_logits, seed_42_logits, seed_7_logits, seed_21_logits)
    ):
        if not (len(ex_gbert) == len(ex_42) == len(ex_7) == len(ex_21)):
            raise ValueError(
                f"Token count mismatch at validation example {example_index}: "
                f"gbert={len(ex_gbert)}, seed_42={len(ex_42)}, "
                f"seed_7={len(ex_7)}, seed_21={len(ex_21)}"
            )

        example_predictions = []

        for gbert_t, t_42, t_7, t_21 in zip(ex_gbert, ex_42, ex_7, ex_21):
            gbert_tensor = torch.tensor(gbert_t, dtype=torch.float32)
            tensor_42 = torch.tensor(t_42, dtype=torch.float32)
            tensor_7 = torch.tensor(t_7, dtype=torch.float32)
            tensor_21 = torch.tensor(t_21, dtype=torch.float32)

            gbert_prob = torch.softmax(gbert_tensor, dim=-1)
            prob_42 = torch.softmax(tensor_42, dim=-1)
            prob_7 = torch.softmax(tensor_7, dim=-1)
            prob_21 = torch.softmax(tensor_21, dim=-1)

            xlmr_multiseed_prob = (prob_42 + prob_7 + prob_21) / 3.0

            combined_probabilities = (
                gbert_weight * gbert_prob + xlmr_weight * xlmr_multiseed_prob
            )

            predicted_id = int(torch.argmax(combined_probabilities).item())
            example_predictions.append(id2label[predicted_id])

        predictions.append(example_predictions)

    return predictions


def logits_to_predictions(
    all_logits,
    id2label: dict,
) -> list[list[str]]:
    """
    Convert raw logits to BIO sequence predictions by argmax.
    """
    predictions = []

    for example_logits in all_logits:
        example_predictions = []
        for token_logits in example_logits:
            predicted_id = int(
                torch.argmax(
                    torch.tensor(token_logits, dtype=torch.float32)
                ).item()
            )
            example_predictions.append(id2label[predicted_id])
        predictions.append(example_predictions)

    return predictions


def build_records(
    examples: list[dict],
    predicted_sequences: list[list[str]],
) -> list[dict]:
    """
    Convert BIO predictions back to entities for competition evaluation.
    """
    records = []

    for example, predicted_bio in zip(examples, predicted_sequences):
        tokens = example["tokens"]
        true_bio = example["bio_tags"]

        true_entities = bio_to_entities(tokens, true_bio)
        predicted_entities = bio_to_entities(tokens, predicted_bio)

        records.append(
            {
                "category": example["category"],
                "true_entities": true_entities,
                "predicted_entities": predicted_entities,
            }
        )

    return records


def calculate_score(
    examples,
    predictions,
) -> float:
    records = build_records(examples, predictions)
    metrics = calculate_competition_metrics(records)
    return metrics["final_score"]


def main() -> None:
    print(
        "--- EXPERIMENT 10: "
        "GBERT + 3-SEED MULTISEED XLM-R ENSEMBLE ---"
    )

    if not torch.cuda.is_available():
        print("CUDA is not available. Stopping.")
        return

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------
    # LOAD COMMON VALIDATION DATA
    # -----------------------------------------
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    _, val_examples = split_examples(examples)

    print(f"Validation examples: {len(val_examples)}")

    # =========================================
    # 1. GBERT BASELINE
    # =========================================
    print("\n--- RUNNING GBERT BASELINE ---")

    gbert_tokenizer = AutoTokenizer.from_pretrained(GBERT_CHECKPOINT)
    gbert_model = AutoModelForTokenClassification.from_pretrained(
        GBERT_CHECKPOINT
    )
    gbert_model.to(device)

    gbert_dataset, gbert_stats = prepare_split(
        val_examples,
        gbert_tokenizer,
        label2id,
    )
    print(
        f"GBERT alignment mismatches: "
        f"{gbert_stats['alignment_mismatches']}"
    )

    gbert_logits = collect_active_logits(
        model=gbert_model,
        dataset=gbert_dataset,
        tokenizer=gbert_tokenizer,
        device=device,
    )

    gbert_predictions = logits_to_predictions(gbert_logits, id2label)
    gbert_score = calculate_score(val_examples, gbert_predictions)
    print(f"GBERT raw F0.2: {gbert_score:.6f}")

    del gbert_model
    del gbert_dataset
    torch.cuda.empty_cache()

    # =========================================
    # 2. XLM-R SEED 42
    # =========================================
    print("\n--- RUNNING XLM-R (SEED 42) ---")

    tokenizer_42 = AutoTokenizer.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    model_42 = AutoModelForTokenClassification.from_pretrained(
        XLMR_SEED_42_CHECKPOINT
    )
    model_42.to(device)

    dataset_42, stats_42 = prepare_split(
        val_examples,
        tokenizer_42,
        label2id,
    )
    print(
        f"Seed 42 alignment mismatches: "
        f"{stats_42['alignment_mismatches']}"
    )

    logits_42 = collect_active_logits(
        model=model_42,
        dataset=dataset_42,
        tokenizer=tokenizer_42,
        device=device,
    )

    predictions_42 = logits_to_predictions(logits_42, id2label)
    score_42 = calculate_score(val_examples, predictions_42)
    print(f"XLM-R seed 42 F0.2: {score_42:.6f}")

    del model_42
    del dataset_42
    torch.cuda.empty_cache()

    # =========================================
    # 3. XLM-R SEED 7
    # =========================================
    print("\n--- RUNNING XLM-R (SEED 7) ---")

    tokenizer_7 = AutoTokenizer.from_pretrained(XLMR_SEED_7_CHECKPOINT)
    model_7 = AutoModelForTokenClassification.from_pretrained(
        XLMR_SEED_7_CHECKPOINT
    )
    model_7.to(device)

    dataset_7, stats_7 = prepare_split(
        val_examples,
        tokenizer_7,
        label2id,
    )
    print(
        f"Seed 7 alignment mismatches: "
        f"{stats_7['alignment_mismatches']}"
    )

    logits_7 = collect_active_logits(
        model=model_7,
        dataset=dataset_7,
        tokenizer=tokenizer_7,
        device=device,
    )

    predictions_7 = logits_to_predictions(logits_7, id2label)
    score_7 = calculate_score(val_examples, predictions_7)
    print(f"XLM-R seed 7 F0.2: {score_7:.6f}")

    del model_7
    del dataset_7
    torch.cuda.empty_cache()

    # =========================================
    # 4. XLM-R SEED 21
    # =========================================
    print("\n--- RUNNING XLM-R (SEED 21) ---")

    tokenizer_21 = AutoTokenizer.from_pretrained(XLMR_SEED_21_CHECKPOINT)
    model_21 = AutoModelForTokenClassification.from_pretrained(
        XLMR_SEED_21_CHECKPOINT
    )
    model_21.to(device)

    dataset_21, stats_21 = prepare_split(
        val_examples,
        tokenizer_21,
        label2id,
    )
    print(
        f"Seed 21 alignment mismatches: "
        f"{stats_21['alignment_mismatches']}"
    )

    logits_21 = collect_active_logits(
        model=model_21,
        dataset=dataset_21,
        tokenizer=tokenizer_21,
        device=device,
    )

    predictions_21 = logits_to_predictions(logits_21, id2label)
    score_21 = calculate_score(val_examples, predictions_21)
    print(f"XLM-R seed 21 F0.2: {score_21:.6f}")

    del model_21
    del dataset_21
    torch.cuda.empty_cache()

    # =========================================
    # 5. 3-SEED XLM-R ENSEMBLE
    # =========================================
    print("\n--- 3-SEED XLM-R ENSEMBLE ---")

    xlmr_ensemble_predictions = ensemble_3seed_xlmr_predictions(
        logits_42,
        logits_7,
        logits_21,
        id2label,
    )

    xlmr_ensemble_score = calculate_score(
        val_examples,
        xlmr_ensemble_predictions,
    )
    print(f"3-seed XLM-R ensemble F0.2: {xlmr_ensemble_score:.6f}")

    # =========================================
    # 6. 40/60 GBERT + MULTI-SEED XLM-R ENSEMBLE
    # =========================================
    print("\n--- 40/60 GBERT + MULTI-SEED XLM-R ENSEMBLE ---")

    combined_ensemble_predictions = ensemble_gbert_multiseed_xlmr_predictions(
        gbert_logits,
        logits_42,
        logits_7,
        logits_21,
        id2label,
        gbert_weight=0.40,
        xlmr_weight=0.60,
    )

    combined_ensemble_score = calculate_score(
        val_examples,
        combined_ensemble_predictions,
    )
    print(f"40/60 GBERT + multi-seed XLM-R F0.2: {combined_ensemble_score:.6f}")

    # =========================================
    # 7. EXPERIMENT 10 SUMMARY
    # =========================================
    print("\n--- EXPERIMENT 10 SUMMARY ---")
    print(f"GBERT raw F0.2: {gbert_score:.6f}")
    print(f"XLM-R seed 42 F0.2: {score_42:.6f}")
    print(f"XLM-R seed 7 F0.2: {score_7:.6f}")
    print(f"XLM-R seed 21 F0.2: {score_21:.6f}")
    print(f"3-seed XLM-R ensemble F0.2: {xlmr_ensemble_score:.6f}")
    print(f"40/60 GBERT + multi-seed XLM-R F0.2: {combined_ensemble_score:.6f}")

    improvement_vs_champion = combined_ensemble_score - CURRENT_CHAMPION_SCORE
    print(
        f"Improvement vs current champion ({CURRENT_CHAMPION_SCORE:.6f}): "
        f"{improvement_vs_champion:+.6f}"
    )


if __name__ == "__main__":
    main()
