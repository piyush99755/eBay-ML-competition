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
from src.data.label_mapping import (
    create_label_mappings,
)
from src.data.make_splits import (
    split_examples,
)
from src.modeling.prepare_datasets import (
    prepare_split,
)

from src.evaluation.entity_metrics import (
    bio_to_entities,
)
from src.evaluation.competition_metrics import (
    calculate_competition_metrics,
)


CHECKPOINT_DIR = Path(
    "checkpoints/gbert_weighted_vehicle_model_best"
)

VAL_BATCH_SIZE = 8

IGNORE_LABEL_ID = -100

VEHICLE_MODEL_B = (
    "B-Kompatibles_Fahrzeug_Modell"
)

VEHICLE_MODEL_I = (
    "I-Kompatibles_Fahrzeug_Modell"
)

THRESHOLDS = [
    0.55,
]

def collect_predictions_and_logits(
    model,
    dataset,
    tokenizer,
    id2label: dict,
    device,
) -> tuple[
    list[list[str]],
    list[list[list[float]]],
]:
    """
    Run validation once and keep:

    1. predicted BIO labels
    2. logits for each original competition token

    Extra WordPiece positions and special tokens are
    ignored using the -100 label mask.
    """

    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer=tokenizer,
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    model.eval()

    all_predicted_bio = []
    all_active_logits = []

    with torch.no_grad():

        for batch in dataloader:

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
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

            for row_index in range(
                batch_size_actual
            ):

                row_labels = labels[
                    row_index
                ]

                row_predictions = (
                    predicted_ids[
                        row_index
                    ]
                )

                row_logits = logits[
                    row_index
                ]

                active_mask = (
                    row_labels
                    != IGNORE_LABEL_ID
                )

                active_prediction_ids = (
                    row_predictions[
                        active_mask
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                active_logits = (
                    row_logits[
                        active_mask
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                predicted_bio = [
                    id2label[label_id]
                    for label_id
                    in active_prediction_ids
                ]

                all_predicted_bio.append(
                    predicted_bio
                )

                all_active_logits.append(
                    active_logits
                )

    return (
        all_predicted_bio,
        all_active_logits,
    )


def repair_with_confidence_threshold(
    predicted_bio: list[str],
    token_logits: list[list[float]],
    label2id: dict,
    threshold: float,
) -> tuple[list[str], int]:
    """
    Selectively convert:

        B-VehicleModel
            ->
        I-VehicleModel

    only when:

    1. the previous token is already VehicleModel
    2. the current prediction is B-VehicleModel
    3. B and I logits are close enough

    Smaller threshold = more conservative.
    """

    repaired = predicted_bio.copy()

    changed = 0

    b_id = label2id[
        VEHICLE_MODEL_B
    ]

    i_id = label2id[
        VEHICLE_MODEL_I
    ]

    for index in range(
        1,
        len(repaired),
    ):

        current_label = repaired[index]
        previous_label = repaired[
            index - 1
        ]

        previous_is_vehicle_model = (
            previous_label
            in {
                VEHICLE_MODEL_B,
                VEHICLE_MODEL_I,
            }
        )

        if not previous_is_vehicle_model:
            continue

        if (
            current_label
            != VEHICLE_MODEL_B
        ):
            continue

        current_logits = (
            token_logits[index]
        )

        b_score = current_logits[b_id]
        i_score = current_logits[i_id]

        confidence_gap = (
            b_score - i_score
        )

        if (
            0.0
            <= confidence_gap
            <= threshold
        ):
            repaired[index] = (
                VEHICLE_MODEL_I
            )

            changed += 1

    return repaired, changed


def build_records(
    examples: list[dict],
    predicted_sequences: list[list[str]],
) -> list[dict]:
    """
    Convert BIO sequences into entity records
    for local competition scoring.
    """

    records = []

    for example, predicted_bio in zip(
        examples,
        predicted_sequences,
    ):

        tokens = example["tokens"]
        true_bio = example["bio_tags"]

        true_entities = bio_to_entities(
            tokens,
            true_bio,
        )

        predicted_entities = (
            bio_to_entities(
                tokens,
                predicted_bio,
            )
        )

        records.append(
            {
                "category": example[
                    "category"
                ],
                "true_entities": (
                    true_entities
                ),
                "predicted_entities": (
                    predicted_entities
                ),
            }
        )

    return records


def main() -> None:

    print(
        "--- EXPERIMENT 3: "
        "WEIGHTED MODEL + CONFIDENCE BOUNDARY REPAIR ---"
    )

    # ---------------------------------------------
    # 1. GPU CHECK
    # ---------------------------------------------

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. "
            "Stopping."
        )
        return

    device = torch.device("cuda")

    print(f"Device: {device}")

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # ---------------------------------------------
    # 2. LOAD CHECKPOINT
    # ---------------------------------------------

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_DIR}"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            CHECKPOINT_DIR
        )
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            CHECKPOINT_DIR
        )
    )

    model.to(device)

    print(
        f"Loaded checkpoint: "
        f"{CHECKPOINT_DIR}"
    )

    # ---------------------------------------------
    # 3. LOAD VALIDATION DATA
    # ---------------------------------------------

    df = load_training_data()

    examples = build_examples(df)

    label2id, id2label = (
        create_label_mappings(
            examples
        )
    )

    _, val_examples = (
        split_examples(
            examples
        )
    )

    val_dataset, val_stats = (
        prepare_split(
            val_examples,
            tokenizer,
            label2id,
        )
    )

    print(
        f"Validation examples: "
        f"{len(val_examples)}"
    )

    print(
        f"Alignment mismatches: "
        f"{val_stats['alignment_mismatches']}"
    )

    # ---------------------------------------------
    # 4. RUN MODEL ONCE
    # ---------------------------------------------

    (
        baseline_predictions,
        active_logits,
    ) = collect_predictions_and_logits(
        model=model,
        dataset=val_dataset,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
    )

    # ---------------------------------------------
    # 5. BASELINE SCORE
    # ---------------------------------------------

    baseline_records = build_records(
        val_examples,
        baseline_predictions,
    )

    baseline_metrics = (
        calculate_competition_metrics(
            baseline_records
        )
    )

    baseline_score = (
        baseline_metrics[
            "final_score"
        ]
    )

    print(
        f"\nBaseline F0.2: "
        f"{baseline_score:.6f}"
    )

    # ---------------------------------------------
    # 6. TEST THRESHOLDS
    # ---------------------------------------------

    print(
        "\n--- THRESHOLD RESULTS ---"
    )

    best_threshold = None
    best_score = baseline_score

    for threshold in THRESHOLDS:

        repaired_predictions = []

        total_changes = 0

        for (
            predicted_bio,
            token_logits,
        ) in zip(
            baseline_predictions,
            active_logits,
        ):

            repaired_bio, changed = (
                repair_with_confidence_threshold(
                    predicted_bio,
                    token_logits,
                    label2id,
                    threshold,
                )
            )

            total_changes += changed

            repaired_predictions.append(
                repaired_bio
            )

        repaired_records = build_records(
            val_examples,
            repaired_predictions,
        )

        repaired_metrics = (
            calculate_competition_metrics(
                repaired_records
            )
        )

        repaired_score = (
            repaired_metrics[
                "final_score"
            ]
        )

        score_change = (
            repaired_score
            - baseline_score
        )

        print(
            f"Threshold "
            f"{threshold:.2f} | "
            f"changes={total_changes} | "
            f"F0.2={repaired_score:.6f} | "
            f"change={score_change:+.6f}"
        )

        if repaired_score > best_score:

            best_score = repaired_score
            best_threshold = threshold

    # ---------------------------------------------
    # 7. SUMMARY
    # ---------------------------------------------

    print(
        "\n--- EXPERIMENT SUMMARY ---"
    )

    print(
        f"Baseline F0.2: "
        f"{baseline_score:.6f}"
    )

    if best_threshold is None:

        print(
            "REJECT: none of the confidence "
            "thresholds improved the baseline."
        )

    else:

        print(
            f"Best threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Best F0.2: "
            f"{best_score:.6f}"
        )

        print(
            f"Improvement: "
            f"{best_score - baseline_score:+.6f}"
        )


if __name__ == "__main__":
    main()