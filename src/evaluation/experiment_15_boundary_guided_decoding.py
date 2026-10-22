"""
Experiment 15: Boundary-Guided Joint Decoding.

Uses the already-trained Experiment 14 checkpoint:
checkpoints/xlmr_boundary_seed42_best

Replaces independent BIO argmax with joint boundary-guided dynamic programming (Viterbi)
decoding using:
  Score = sum_t [ log P_BIO(y_t)
                  + 0.20 * log P_START(implied_start_t)
                  + 0.20 * log P_END(implied_end_t) ]

Inference-only experiment:
- No retraining
- No threshold tuning / weight sweeping (fixed 0.20 matching Exp 14 auxiliary loss weights)
- No CRF transition parameters
- No post-processing / boundary heuristics
- Operates strictly on original competition tokens
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
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
from src.modeling.xlmr_boundary import XLMRobertaBoundary
from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.error_analysis import (
    analyze_token_confusions,
    analyze_entity_errors,
    print_top_confusions,
    print_entity_error_summary,
)


CHECKPOINT_DIR = Path("checkpoints/xlmr_boundary_seed42_best")
EXPECTED_BASELINE_SCORE = 0.875489
VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100

START_WEIGHT = 0.20
END_WEIGHT = 0.20


def build_boundary_decoding_matrices(id2label: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute label property tensors and transition termination matrix E.

    Returns:
    - is_start: [K] tensor where is_start[j] = 1 if id2label[j] starts with B-, else 0.
    - is_entity: [K] tensor where is_entity[j] = 1 if id2label[j] != 'O', else 0.
    - E: [K, K] tensor where E[i, j] = 1 if transitioning from label i to label j
         terminates an entity, else 0.
    """
    num_labels = len(id2label)
    is_start = torch.zeros(num_labels, dtype=torch.long)
    is_entity = torch.zeros(num_labels, dtype=torch.long)
    E = torch.zeros((num_labels, num_labels), dtype=torch.long)

    for i in range(num_labels):
        tag_i = id2label[i]
        if tag_i.startswith("B-"):
            is_start[i] = 1
        if tag_i != "O":
            is_entity[i] = 1

    for i in range(num_labels):
        tag_i = id2label[i]
        if tag_i == "O":
            continue

        # tag_i is B-X or I-X
        entity_X = tag_i[2:]

        for j in range(num_labels):
            tag_j = id2label[j]
            is_end = False
            if tag_j == "O":
                is_end = True
            elif tag_j.startswith("B-"):
                is_end = True
            elif tag_j.startswith("I-"):
                entity_Y = tag_j[2:]
                if entity_Y != entity_X:
                    is_end = True
            else:
                is_end = True

            if is_end:
                E[i, j] = 1

    return is_start, is_entity, E


def boundary_guided_viterbi_decode(
    bio_logits: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    is_start: torch.Tensor,
    is_entity: torch.Tensor,
    E: torch.Tensor,
    start_weight: float = START_WEIGHT,
    end_weight: float = END_WEIGHT,
) -> List[int]:
    """
    Decode the highest-scoring BIO label sequence for a single title using Viterbi DP.

    Args:
    - bio_logits: [T, K] tensor of unnormalized BIO logits
    - start_logits: [T, 2] tensor of unnormalized START logits (0=not start, 1=start)
    - end_logits: [T, 2] tensor of unnormalized END logits (0=not end, 1=end)
    - is_start: [K] tensor indicating which labels start with B-
    - is_entity: [K] tensor indicating which labels are entities (!= 'O')
    - E: [K, K] binary matrix indicating entity termination on transition from i to j
    - start_weight: fixed weight for log P_START (default 0.20)
    - end_weight: fixed weight for log P_END (default 0.20)

    Returns:
    - List of decoded label IDs of length T.
    """
    T, num_labels = bio_logits.shape
    if T == 0:
        return []

    # Log probabilities
    log_p_bio = F.log_softmax(bio_logits, dim=-1)      # [T, K]
    log_p_start = F.log_softmax(start_logits, dim=-1)  # [T, 2]
    log_p_end = F.log_softmax(end_logits, dim=-1)      # [T, 2]

    # Node scores: log P_BIO(y_t) + start_weight * log P_START(implied_start_t)
    # is_start has shape [K], indexing log_p_start[:, is_start] gives [T, K]
    node_scores = log_p_bio + start_weight * log_p_start[:, is_start]  # [T, K]

    # Terminal score for the last token T - 1 (since it is the last token, all entities end)
    terminal_scores = end_weight * log_p_end[T - 1, is_entity]  # [K]

    if T == 1:
        v_final = node_scores[0] + terminal_scores
        best_label = int(v_final.argmax().item())
        return [best_label]

    # Viterbi DP
    v = node_scores[0]  # [K]
    backpointers: List[torch.Tensor] = []

    for t in range(1, T):
        prev_t = t - 1
        # Edge score from label i at prev_t to label j at t
        # Evaluates log P_END at prev_t:
        # where E[i, j] == 1: log_p_end[prev_t, 1]; where E[i, j] == 0: log_p_end[prev_t, 0]
        end_prob_0 = log_p_end[prev_t, 0]
        end_prob_1 = log_p_end[prev_t, 1]
        edge_scores_prev = end_weight * torch.where(E == 1, end_prob_1, end_prob_0)  # [K, K]

        # v is [K], unsqueeze to [K, 1] to broadcast across columns (to_label j)
        transition_matrix = v.unsqueeze(1) + edge_scores_prev  # [K, K], row i=from, col j=to

        # Maximize over from_label i (dim 0)
        max_scores, best_prev_labels = transition_matrix.max(dim=0)  # [K], [K]
        backpointers.append(best_prev_labels)

        v = max_scores + node_scores[t]  # [K]

    # Add terminal score at the final token
    v_final = v + terminal_scores
    best_last_tag = int(v_final.argmax().item())

    # Backtracking
    best_path = [best_last_tag]
    for t in range(T - 1, 0, -1):
        prev_tag = int(backpointers[t - 1][best_path[-1]].item())
        best_path.append(prev_tag)

    best_path.reverse()
    return best_path


def evaluate_predictions(
    examples: List[dict],
    predictions: List[List[str]],
) -> dict:
    """Reconstruct entities and evaluate using competition-style F0.2 metrics."""
    records = []
    true_bio_sequences = []

    for example, pred_bio in zip(examples, predictions):
        tokens = example["tokens"]
        true_bio = example["bio_tags"]

        if len(pred_bio) != len(tokens):
            raise ValueError(
                f"Predicted BIO count ({len(pred_bio)}) != token count ({len(tokens)})."
            )

        true_bio_sequences.append(true_bio)
        true_entities = bio_to_entities(tokens, true_bio)
        pred_entities = bio_to_entities(tokens, pred_bio)

        records.append(
            {
                "category": example["category"],
                "true_entities": true_entities,
                "predicted_entities": pred_entities,
            }
        )

    results = calculate_competition_metrics(records)

    return {
        "metrics": results,
        "records": records,
        "true_bio_sequences": true_bio_sequences,
        "predicted_bio_sequences": predictions,
    }


def inspect_vehicle_model_aspect(
    evaluation_output: dict,
    aspect_name: str = "Kompatibles_Fahrzeug_Modell",
) -> None:
    """Detailed inspection of Kompatibles_Fahrzeug_Modell performance, errors, and boundary confusions."""
    print(f"\n--- DETAILED INSPECTION: {aspect_name} ---")

    metrics = evaluation_output["metrics"]
    found = False

    for category, cat_data in sorted(metrics["categories"].items()):
        aspects = cat_data.get("aspects", {})
        if aspect_name in aspects:
            found = True
            asp = aspects[aspect_name]
            print(f"Category {category}:")
            print(f"  Count:     {asp['true_count']}")
            print(f"  Precision: {asp['precision']:.4f}")
            print(f"  Recall:    {asp['recall']:.4f}")
            print(f"  F0.2:      {asp['f_beta']:.4f}")
            print(f"  Weight:    {asp['weight']:.4f}")

    if not found:
        print(f"  Aspect '{aspect_name}' not found in validation categories.")

    # Entity errors
    records = evaluation_output["records"]
    aspect_errors = analyze_entity_errors(records)
    if aspect_name in aspect_errors:
        err = aspect_errors[aspect_name]
        fp = err["false_positives"]
        fn = err["false_negatives"]
        print(f"  Entity Errors: FP={fp} | FN={fn} | Total={fp + fn}")

    # Specific B -> I and I -> B boundary confusions
    true_bio_sequences = evaluation_output["true_bio_sequences"]
    predicted_bio_sequences = evaluation_output["predicted_bio_sequences"]
    confusions = analyze_token_confusions(
        true_bio_sequences,
        predicted_bio_sequences,
    )

    b_tag = f"B-{aspect_name}"
    i_tag = f"I-{aspect_name}"

    b_to_i = confusions[(b_tag, i_tag)]
    i_to_b = confusions[(i_tag, b_tag)]

    print(f"  Boundary Confusions for {aspect_name}:")
    print(f"    B -> I confusion count: {b_to_i}")
    print(f"    I -> B confusion count: {i_to_b}")


def main() -> None:
    print("==================================================")
    print("EXPERIMENT 15: BOUNDARY-GUIDED JOINT DECODING")
    print("==================================================")

    # 1. Defensive Checkpoint Check
    weights_file = CHECKPOINT_DIR / "boundary_heads.pt"
    if not CHECKPOINT_DIR.exists() or not weights_file.exists():
        print(
            f"\n[!] Checkpoint not found at: {CHECKPOINT_DIR}\n"
            f"Please run Experiment 14 training first:\n"
            f"  python -m src.modeling.train_xlmr_boundary\n"
        )
        return

    # 2. CUDA Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 3. Load Data & Validation Split (Split-A, random_state=42)
    print("\n--- LOADING DATA & VALIDATION SPLIT ---")
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    _, val_examples = split_examples(examples)
    print(f"Validation titles: {len(val_examples)}")

    # 4. Load Model & Tokenizer
    print(f"\nLoading model from {CHECKPOINT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = XLMRobertaBoundary.from_pretrained(CHECKPOINT_DIR)
    model.to(device)
    model.eval()

    # 5. Prepare Validation Dataset
    val_dataset, val_stats = prepare_split(
        val_examples,
        tokenizer,
        label2id,
    )
    print(f"Validation alignment mismatches: {val_stats['alignment_mismatches']}")

    # 6. Collect Active Logits in One Pass
    print("\n--- COLLECTING ACTIVE ORIGINAL-TOKEN LOGITS ---")
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    collected_bio_logits: List[torch.Tensor] = []
    collected_start_logits: List[torch.Tensor] = []
    collected_end_logits: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch_dev = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**batch_dev)
            bio_logits = outputs.bio_logits.cpu()
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()

            batch_size_actual = labels.shape[0]

            for row_index in range(batch_size_actual):
                row_labels = labels[row_index]
                active_mask = row_labels != IGNORE_LABEL_ID
                active_len = int(active_mask.sum().item())

                collected_bio_logits.append(bio_logits[row_index, :active_len])
                collected_start_logits.append(start_logits[row_index, :active_len])
                collected_end_logits.append(end_logits[row_index, :active_len])

    print(f"Collected logits for {len(collected_bio_logits)} validation examples.")

    # Unload model from GPU to free VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. Internal Verification: Standard BIO Argmax
    print("\n--- INTERNAL VERIFICATION: STANDARD BIO ARGMAX ---")
    argmax_predictions: List[List[str]] = []
    for bio_l in collected_bio_logits:
        pred_ids = torch.argmax(bio_l, dim=-1).tolist()
        argmax_predictions.append([id2label[lid] for lid in pred_ids])

    eval_argmax = evaluate_predictions(val_examples, argmax_predictions)
    score_argmax = eval_argmax["metrics"]["final_score"]
    print(f"Reproduced BIO-argmax F0.2: {score_argmax:.6f} (Expected: ~{EXPECTED_BASELINE_SCORE:.6f})")

    # 8. Boundary-Guided Joint Decoding
    print("\n--- RUNNING BOUNDARY-GUIDED JOINT DECODING ---")
    is_start, is_entity, E = build_boundary_decoding_matrices(id2label)

    guided_predictions: List[List[str]] = []
    total_token_count = 0
    changed_token_count = 0

    for i in range(len(collected_bio_logits)):
        bio_l = collected_bio_logits[i]
        start_l = collected_start_logits[i]
        end_l = collected_end_logits[i]

        guided_ids = boundary_guided_viterbi_decode(
            bio_logits=bio_l,
            start_logits=start_l,
            end_logits=end_l,
            is_start=is_start,
            is_entity=is_entity,
            E=E,
            start_weight=START_WEIGHT,
            end_weight=END_WEIGHT,
        )

        guided_bio = [id2label[lid] for lid in guided_ids]
        guided_predictions.append(guided_bio)

        argmax_bio = argmax_predictions[i]
        total_token_count += len(guided_bio)
        for g_tag, a_tag in zip(guided_bio, argmax_bio):
            if g_tag != a_tag:
                changed_token_count += 1

    eval_guided = evaluate_predictions(val_examples, guided_predictions)
    score_guided = eval_guided["metrics"]["final_score"]
    delta = score_guided - EXPECTED_BASELINE_SCORE

    cat1_score = eval_guided["metrics"]["categories"].get(1, {}).get("score", 0.0)
    cat2_score = eval_guided["metrics"]["categories"].get(2, {}).get("score", 0.0)

    # 9. Results Summary
    print("\n==================================================")
    print("EXPERIMENT 15 RESULTS")
    print("==================================================")
    print(f"BIO-only baseline from same checkpoint:  {EXPECTED_BASELINE_SCORE:.6f}")
    print(f"Reproduced baseline BIO argmax:          {score_argmax:.6f}")
    print(f"Boundary-Guided Joint Decoding:          {score_guided:.6f}")
    print(f"Delta vs baseline:                       {delta:+.6f}")
    print(f"Category 1 score:                        {cat1_score:.6f}")
    print(f"Category 2 score:                        {cat2_score:.6f}")
    print(
        f"\nOriginal-token BIO label changes:       {changed_token_count} / {total_token_count} "
        f"({(changed_token_count / total_token_count) * 100:.2f}%)"
    )

    # 10. General Error Analysis & Vehicle Model Aspect Inspection
    print("\n--- GENERAL ERROR ANALYSIS ---")
    confusions = analyze_token_confusions(
        eval_guided["true_bio_sequences"],
        eval_guided["predicted_bio_sequences"],
    )
    aspect_errors = analyze_entity_errors(eval_guided["records"])
    print_top_confusions(confusions, top_n=20)
    print_entity_error_summary(aspect_errors)

    inspect_vehicle_model_aspect(
        eval_guided,
        aspect_name="Kompatibles_Fahrzeug_Modell",
    )


if __name__ == "__main__":
    main()
