"""
Experiment 18: Conservative BIO + Span Ensemble.

Objective:
Test whether Experiment 16's unweighted span-based XLM-R model can improve
Experiment 10's existing GBERT + three-seed XLM-R ensemble.

Inference-only:
No model retraining or fine-tuning.
Sequential model loading and GPU memory release (6 GB VRAM safe).

Fusion Rule:
Primary model: Experiment 10 (GBERT 0.40 + 3-seed XLM-R 0.60).
1. Preserve every Experiment 10 entity.
2. If Experiment 16 predicts the exact same entity, do not duplicate it.
3. If Experiment 16 predicts an entity overlapping any Experiment 10 entity, keep Experiment 10's prediction.
4. If Experiment 16 predicts an additional entity with no token overlap with any Experiment 10 entity,
   add it only if its non-NONE span-class softmax confidence is >= 0.98.
5. Resolve any remaining overlap among added spans deterministically, preferring higher confidence.
6. Preserve existing official whitespace-token positions and entity reconstruction semantics.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
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
from src.modeling.xlmr_span import (
    MAX_SPAN_WIDTH,
    XLMRobertaSpan,
    bio_to_spans,
    create_span_mappings,
    generate_candidate_spans,
    resolve_overlapping_spans,
    spans_to_bio,
)
from src.evaluation.entity_metrics import (
    BETA,
    bio_to_entities,
    count_matches,
    precision_recall_fbeta,
)
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.error_analysis import analyze_entity_errors


# Checkpoint paths
GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
XLMR_SEED_42_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")
XLMR_SEED_7_CHECKPOINT = Path("checkpoints/xlmr_seed_7_best")
XLMR_SEED_21_CHECKPOINT = Path("checkpoints/xlmr_seed_21_best")
XLMR_SPAN_CHECKPOINT = Path("checkpoints/xlmr_span_seed42_best")

# Reference baseline scores
EXPECTED_EXP10_SCORE = 0.879852
EXPECTED_EXP16_SCORE = 0.875627

# Fixed fusion threshold (frozen in advance, no sweeping)
CONFIDENCE_THRESHOLD = 0.98
VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100


def collect_active_logits(
    model,
    dataset,
    tokenizer,
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> list[list[list[float]]]:
    """
    Run evaluation and extract logits for active original tokens only.
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


def ensemble_gbert_multiseed_xlmr_predictions(
    gbert_logits: list[list[list[float]]],
    seed_42_logits: list[list[list[float]]],
    seed_7_logits: list[list[list[float]]],
    seed_21_logits: list[list[list[float]]],
    id2label: dict,
    gbert_weight: float = 0.40,
    xlmr_weight: float = 0.60,
) -> list[list[str]]:
    """
    Combine GBERT and 3-seed XLM-R probabilities:
    1. Average 3 XLM-R seed probabilities equally: (p42 + p7 + p21) / 3
    2. Combine: gbert_weight * GBERT_prob + xlmr_weight * XLMR_multiseed_prob
    3. Argmax to predict final BIO label.
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


def collect_span_predictions(
    model: XLMRobertaSpan,
    dataset,
    tokenizer,
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> Tuple[List[List[str]], List[List[dict]], dict]:
    """
    Run inference over validation dataset for Experiment 16 Span model.

    Returns:
    - all_pred_bio: list of BIO tag lists per example
    - all_retained_spans: list of retained non-overlapping entity spans per example
      each dict: {'start': int, 'end': int, 'aspect': str, 'score': float}
    - diagnostics: aggregate counts
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
    all_pred_bio = []
    all_retained_spans = []

    diag_totals = {
        "total_candidates": 0,
        "none_candidates": 0,
        "eligible_candidates": 0,
        "retained_candidates": 0,
        "rejected_overlap_candidates": 0,
    }

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model.roberta(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
            )
            sequence_output = outputs.last_hidden_state
            batch_size_actual = sequence_output.shape[0]

            labels = batch_dev.get("labels")
            if labels is not None:
                active_mask = labels != IGNORE_LABEL_ID
            else:
                active_mask = batch_dev["attention_mask"] == 1

            for b in range(batch_size_actual):
                active_indices = active_mask[b].nonzero(as_tuple=False).squeeze(-1)
                T = len(active_indices)
                if T == 0:
                    all_pred_bio.append([])
                    all_retained_spans.append([])
                    continue

                hidden_b = sequence_output[b, active_indices]

                candidates = generate_candidate_spans(T, max_span_width=model.max_span_width)
                diag_totals["total_candidates"] += len(candidates)

                if len(candidates) == 0:
                    all_pred_bio.append(["O"] * T)
                    all_retained_spans.append([])
                    continue

                starts_t = torch.tensor([c[0] for c in candidates], dtype=torch.long, device=device)
                ends_t = torch.tensor([c[1] for c in candidates], dtype=torch.long, device=device)

                span_repr = model.extract_span_representations(hidden_b, starts_t, ends_t)
                span_logits = model.span_classifier(span_repr)
                span_probs = torch.softmax(span_logits, dim=-1)

                pred_classes = span_probs.argmax(dim=-1).cpu().tolist()
                max_probs = span_probs.max(dim=-1).values.cpu().tolist()

                eligible_spans = []
                for idx, (s, e) in enumerate(candidates):
                    cls_id = pred_classes[idx]
                    if cls_id == 0:
                        diag_totals["none_candidates"] += 1
                        continue

                    aspect_name = model.id2span[cls_id]
                    prob = max_probs[idx]
                    eligible_spans.append({
                        "start": s,
                        "end": e,
                        "aspect": aspect_name,
                        "score": prob,
                    })

                diag_totals["eligible_candidates"] += len(eligible_spans)

                retained, rejected_count = resolve_overlapping_spans(eligible_spans, T)
                diag_totals["retained_candidates"] += len(retained)
                diag_totals["rejected_overlap_candidates"] += rejected_count

                reconstructed_spans = [(r["start"], r["end"], r["aspect"]) for r in retained]
                pred_bio = spans_to_bio(reconstructed_spans, T)

                all_pred_bio.append(pred_bio)
                all_retained_spans.append(retained)

    return all_pred_bio, all_retained_spans, diag_totals


def build_evaluation_records(
    examples: List[dict],
    predicted_sequences: List[List[str]],
) -> List[dict]:
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
    examples: List[dict],
    predictions: List[List[str]],
) -> Tuple[float, dict, List[dict]]:
    """
    Compute competition metrics and return final score, full metrics dict, and evaluation records.
    """
    records = build_evaluation_records(examples, predictions)
    metrics = calculate_competition_metrics(records)
    return metrics["final_score"], metrics, records


def apply_conservative_fusion_example(
    exp10_bio: List[str],
    exp16_retained_spans: List[dict],
    seq_len: int,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[List[str], List[Tuple[int, int, str]], dict]:
    """
    Apply conservative fusion for a single title:
    1. Preserve every Experiment 10 entity.
    2. If Experiment 16 predicts the exact same entity, do not duplicate it.
    3. If Experiment 16 predicts an entity overlapping any Experiment 10 entity, keep Experiment 10.
    4. If Experiment 16 predicts an additional entity with no token overlap with any Experiment 10 entity,
       add it only if non-NONE span softmax confidence >= confidence_threshold.
    5. Resolve any remaining overlap among added spans deterministically, preferring higher confidence.
    6. Preserve existing whitespace-token positions and entity reconstruction semantics.

    Returns:
    - final_bio: merged BIO tag sequence.
    - final_spans: list of (start, end, aspect) tuples.
    - stats: fusion audit counts for this example.
    """
    exp10_spans = bio_to_spans(exp10_bio)
    occupied_tokens: Set[int] = set()
    for s, e, _ in exp10_spans:
        occupied_tokens.update(range(s, e + 1))

    exp10_span_set = set(exp10_spans)
    candidate_additions: List[dict] = []

    exact_matches_count = 0
    overlapping_rejections_count = 0
    considered_span_only_count = 0

    for r in exp16_retained_spans:
        s, e, aspect, score = r["start"], r["end"], r["aspect"], r["score"]
        span_tokens = set(range(s, e + 1))

        # Rule 2: exact match (already preserved in exp10_spans)
        if (s, e, aspect) in exp10_span_set:
            exact_matches_count += 1
            continue

        # Rule 3: overlaps any Exp 10 entity (with diff boundary or label) -> reject Exp 16
        if not span_tokens.isdisjoint(occupied_tokens):
            overlapping_rejections_count += 1
            continue

        # Rule 4: No overlap with any Exp 10 entity -> span-only entity
        considered_span_only_count += 1
        if score >= confidence_threshold:
            candidate_additions.append({
                "start": s,
                "end": e,
                "aspect": aspect,
                "score": score,
            })

    # Rule 5: Resolve any remaining overlap among added spans deterministically
    accepted_additions = []
    if candidate_additions:
        accepted_additions, _ = resolve_overlapping_spans(candidate_additions, seq_len)

    accepted_span_only_count = len(accepted_additions)

    # Combine Exp 10 entities and accepted additions
    final_spans = list(exp10_spans) + [
        (a["start"], a["end"], a["aspect"]) for a in accepted_additions
    ]
    final_spans.sort(key=lambda x: x[0])

    # Defensive check: ensure pairwise disjoint
    occupied_final: Set[int] = set()
    for s, e, _ in final_spans:
        for idx in range(s, e + 1):
            if idx in occupied_final:
                raise ValueError(f"Fusion error: overlapping token {idx} in final spans!")
            occupied_final.add(idx)

    # Rule 6: Reconstruct valid BIO sequence
    final_bio = spans_to_bio(final_spans, seq_len)

    stats = {
        "exact_matches": exact_matches_count,
        "overlapping_rejections": overlapping_rejections_count,
        "considered_span_only": considered_span_only_count,
        "accepted_span_only": accepted_span_only_count,
        "accepted_spans": [(a["start"], a["end"], a["aspect"], a["score"]) for a in accepted_additions],
    }

    return final_bio, final_spans, stats


def compute_performance_by_span_width(records: List[dict]) -> dict:
    """Calculate Precision, Recall, and F0.2 stratified by entity span width (1, 2, 3+)."""
    buckets = {
        "width1": {"tp": 0, "fp": 0, "fn": 0},
        "width2": {"tp": 0, "fp": 0, "fn": 0},
        "width3+": {"tp": 0, "fp": 0, "fn": 0},
    }

    for rec in records:
        true_list = list(rec["true_entities"])
        pred_list = list(rec["predicted_entities"])

        true_matched = [False] * len(true_list)
        pred_matched = [False] * len(pred_list)

        for t_idx, t_ent in enumerate(true_list):
            for p_idx, p_ent in enumerate(pred_list):
                if not pred_matched[p_idx] and t_ent == p_ent:
                    true_matched[t_idx] = True
                    pred_matched[p_idx] = True
                    break

        # TPs and FNs
        for t_idx, (asp, val) in enumerate(true_list):
            w = len(val.split())
            b_key = "width1" if w == 1 else ("width2" if w == 2 else "width3+")
            if true_matched[t_idx]:
                buckets[b_key]["tp"] += 1
            else:
                buckets[b_key]["fn"] += 1

        # FPs
        for p_idx, (asp, val) in enumerate(pred_list):
            if not pred_matched[p_idx]:
                w = len(val.split())
                b_key = "width1" if w == 1 else ("width2" if w == 2 else "width3+")
                buckets[b_key]["fp"] += 1

    results = {}
    for b_key, counts in buckets.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = 0.2 ** 2
        f02 = ((1 + beta_sq) * p * r) / (beta_sq * p + r) if (beta_sq * p + r) > 0 else 0.0
        results[b_key] = {
            "gold_count": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f02": f02,
        }
    return results


def extract_vehicle_model_metrics(
    metrics: dict,
    records: List[dict],
    aspect_name: str = "Kompatibles_Fahrzeug_Modell",
) -> dict:
    """Extract precision, recall, F0.2, TP, FP, FN for Kompatibles_Fahrzeug_Modell."""
    tp_total = 0
    fp_total = 0
    fn_total = 0
    gold_total = 0

    for category, cat_data in metrics.get("categories", {}).items():
        asp_data = cat_data.get("aspects", {}).get(aspect_name)
        if asp_data:
            tp_total += asp_data["tp"]
            fp_total += asp_data["fp"]
            fn_total += asp_data["fn"]
            gold_total += asp_data["true_count"]

    p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    beta_sq = BETA ** 2
    f02 = ((1 + beta_sq) * p * r) / (beta_sq * p + r) if (beta_sq * p + r) > 0 else 0.0

    return {
        "gold_count": gold_total,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": p,
        "recall": r,
        "f02": f02,
    }


def analyze_prediction_agreement(
    val_examples: List[dict],
    exp10_predictions: List[List[str]],
    exp16_predictions: List[List[str]],
    exp16_retained_spans_list: List[List[dict]],
) -> dict:
    """
    Step 2: Measure entity-level prediction agreement between Experiment 10 and Experiment 16.
    Uses exact entity positions (token indices) and aspect types.
    """
    total_gold_entities = 0
    total_exp10_entities = 0
    total_exp16_entities = 0

    identical_entities_count = 0
    exp10_only_entities_count = 0
    exp16_only_entities_count = 0

    exp10_overlapping_diff_count = 0
    exp16_overlapping_diff_count = 0

    overlap_diff_same_boundary_diff_aspect = 0
    overlap_diff_diff_boundary_same_aspect = 0
    overlap_diff_diff_boundary_diff_aspect = 0

    exp16_only_tp = 0
    exp16_only_fp = 0

    # VehicleModel aspect specific tracking
    vm_aspect = "Kompatibles_Fahrzeug_Modell"
    vm_gold = 0
    vm_exp10_total = 0
    vm_exp16_total = 0
    vm_identical = 0
    vm_exp10_only = 0
    vm_exp16_only = 0
    vm_overlapping_diff = 0

    for i, example in enumerate(val_examples):
        gold_spans = set(bio_to_spans(example["bio_tags"]))
        total_gold_entities += len(gold_spans)

        e10_spans = bio_to_spans(exp10_predictions[i])
        total_exp10_entities += len(e10_spans)

        e16_spans = [(r["start"], r["end"], r["aspect"]) for r in exp16_retained_spans_list[i]]
        total_exp16_entities += len(e16_spans)

        e10_set = set(e10_spans)
        e16_set = set(e16_spans)

        # Vehicle model counts
        vm_gold += sum(1 for _, _, a in gold_spans if a == vm_aspect)
        vm_e10 = [e for e in e10_spans if e[2] == vm_aspect]
        vm_e16 = [e for e in e16_spans if e[2] == vm_aspect]
        vm_exp10_total += len(vm_e10)
        vm_exp16_total += len(vm_e16)

        # 1. Identical predictions
        identical_in_example = e10_set & e16_set
        identical_entities_count += len(identical_in_example)
        vm_identical += sum(1 for _, _, a in identical_in_example if a == vm_aspect)

        # 2. Analyze Exp 10 entities
        for s10, e10, a10 in e10_spans:
            if (s10, e10, a10) in identical_in_example:
                continue

            # Check overlap with any Exp 16 entity
            overlapping_e16 = [
                (s16, e16, a16) for s16, e16, a16 in e16_spans
                if max(s10, s16) <= min(e10, e16)
            ]

            if not overlapping_e16:
                exp10_only_entities_count += 1
                if a10 == vm_aspect:
                    vm_exp10_only += 1
            else:
                exp10_overlapping_diff_count += 1
                if a10 == vm_aspect:
                    vm_overlapping_diff += 1

                for s16, e16, a16 in overlapping_e16:
                    same_b = (s10 == s16 and e10 == e16)
                    same_a = (a10 == a16)
                    if same_b and not same_a:
                        overlap_diff_same_boundary_diff_aspect += 1
                    elif not same_b and same_a:
                        overlap_diff_diff_boundary_same_aspect += 1
                    elif not same_b and not same_a:
                        overlap_diff_diff_boundary_diff_aspect += 1

        # 3. Analyze Exp 16 entities
        for s16, e16, a16 in e16_spans:
            if (s16, e16, a16) in identical_in_example:
                continue

            overlapping_e10 = [
                (s10, e10, a10) for s10, e10, a10 in e10_spans
                if max(s10, s16) <= min(e10, e16)
            ]

            if not overlapping_e10:
                exp16_only_entities_count += 1
                if a16 == vm_aspect:
                    vm_exp16_only += 1

                # Check precision of Exp 16-only entities
                if (s16, e16, a16) in gold_spans:
                    exp16_only_tp += 1
                else:
                    exp16_only_fp += 1
            else:
                exp16_overlapping_diff_count += 1

    exp16_only_precision = (
        exp16_only_tp / (exp16_only_tp + exp16_only_fp)
        if (exp16_only_tp + exp16_only_fp) > 0
        else 0.0
    )

    return {
        "total_gold_entities": total_gold_entities,
        "total_exp10_entities": total_exp10_entities,
        "total_exp16_entities": total_exp16_entities,
        "identical_entities": identical_entities_count,
        "exp10_only_entities": exp10_only_entities_count,
        "exp16_only_entities": exp16_only_entities_count,
        "exp10_overlapping_diff": exp10_overlapping_diff_count,
        "exp16_overlapping_diff": exp16_overlapping_diff_count,
        "overlap_diff_same_boundary_diff_aspect": overlap_diff_same_boundary_diff_aspect,
        "overlap_diff_diff_boundary_same_aspect": overlap_diff_diff_boundary_same_aspect,
        "overlap_diff_diff_boundary_diff_aspect": overlap_diff_diff_boundary_diff_aspect,
        "exp16_only_tp": exp16_only_tp,
        "exp16_only_fp": exp16_only_fp,
        "exp16_only_precision": exp16_only_precision,
        "vehicle_model_breakdown": {
            "gold_count": vm_gold,
            "exp10_total": vm_exp10_total,
            "exp16_total": vm_exp16_total,
            "identical": vm_identical,
            "exp10_only": vm_exp10_only,
            "exp16_only": vm_exp16_only,
            "overlapping_diff": vm_overlapping_diff,
        },
    }


def compute_overall_entity_metrics(records: List[dict]) -> dict:
    """Calculate aggregate entity-level TP, FP, FN, Precision, Recall, F0.2."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for rec in records:
        true_ents = rec["true_entities"]
        pred_ents = rec["predicted_entities"]
        tp, fp, fn = count_matches(pred_ents, true_ents)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    beta_sq = BETA ** 2
    f02 = ((1 + beta_sq) * p * r) / (beta_sq * p + r) if (beta_sq * p + r) > 0 else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": p,
        "recall": r,
        "f02": f02,
    }


def run_experiment_18() -> dict:
    """
    Main execution pipeline for Experiment 18.
    """
    print("==================================================")
    print("EXPERIMENT 18: CONSERVATIVE BIO + SPAN ENSEMBLE")
    print("==================================================")

    # CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Checkpoint existence check
    checkpoints = [
        ("GBERT", GBERT_CHECKPOINT),
        ("XLM-R Seed 42", XLMR_SEED_42_CHECKPOINT),
        ("XLM-R Seed 7", XLMR_SEED_7_CHECKPOINT),
        ("XLM-R Seed 21", XLMR_SEED_21_CHECKPOINT),
        ("XLM-R Span (Exp 16)", XLMR_SPAN_CHECKPOINT),
    ]
    for name, cp in checkpoints:
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint for {name} not found at {cp}!")

    # 1. Load Data & Validation Split (Split-A: seed 42)
    print("\n--- LOADING DATA & VALIDATION SPLIT (SPLIT-A, SEED 42) ---")
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    _, val_examples = split_examples(examples)
    num_val = len(val_examples)
    print(f"Validation titles: {num_val}")

    # =========================================================================
    # PART 1: SEQUENTIAL BIO MODEL LOGITS COLLECTION (EXPERIMENT 10)
    # =========================================================================
    print("\n==================================================")
    print("PART 1: EXPERIMENT 10 INFERENCE (SEQUENTIAL VRAM SAFE)")
    print("==================================================")

    # 1.1 GBERT
    print("\n[1/4] Running GBERT Baseline...")
    gbert_tokenizer = AutoTokenizer.from_pretrained(GBERT_CHECKPOINT)
    gbert_model = AutoModelForTokenClassification.from_pretrained(GBERT_CHECKPOINT)
    gbert_model.to(device)

    gbert_dataset, gbert_stats = prepare_split(val_examples, gbert_tokenizer, label2id)
    print(f"  GBERT alignment mismatches: {gbert_stats['alignment_mismatches']}")

    gbert_logits = collect_active_logits(
        model=gbert_model,
        dataset=gbert_dataset,
        tokenizer=gbert_tokenizer,
        device=device,
    )
    del gbert_model
    del gbert_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1.2 XLM-R seed 42
    print("\n[2/4] Running XLM-R (Seed 42)...")
    tokenizer_42 = AutoTokenizer.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    model_42 = AutoModelForTokenClassification.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    model_42.to(device)

    dataset_42, stats_42 = prepare_split(val_examples, tokenizer_42, label2id)
    print(f"  Seed 42 alignment mismatches: {stats_42['alignment_mismatches']}")

    logits_42 = collect_active_logits(
        model=model_42,
        dataset=dataset_42,
        tokenizer=tokenizer_42,
        device=device,
    )
    del model_42
    del dataset_42
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1.3 XLM-R seed 7
    print("\n[3/4] Running XLM-R (Seed 7)...")
    tokenizer_7 = AutoTokenizer.from_pretrained(XLMR_SEED_7_CHECKPOINT)
    model_7 = AutoModelForTokenClassification.from_pretrained(XLMR_SEED_7_CHECKPOINT)
    model_7.to(device)

    dataset_7, stats_7 = prepare_split(val_examples, tokenizer_7, label2id)
    print(f"  Seed 7 alignment mismatches: {stats_7['alignment_mismatches']}")

    logits_7 = collect_active_logits(
        model=model_7,
        dataset=dataset_7,
        tokenizer=tokenizer_7,
        device=device,
    )
    del model_7
    del dataset_7
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1.4 XLM-R seed 21
    print("\n[4/4] Running XLM-R (Seed 21)...")
    tokenizer_21 = AutoTokenizer.from_pretrained(XLMR_SEED_21_CHECKPOINT)
    model_21 = AutoModelForTokenClassification.from_pretrained(XLMR_SEED_21_CHECKPOINT)
    model_21.to(device)

    dataset_21, stats_21 = prepare_split(val_examples, tokenizer_21, label2id)
    print(f"  Seed 21 alignment mismatches: {stats_21['alignment_mismatches']}")

    logits_21 = collect_active_logits(
        model=model_21,
        dataset=dataset_21,
        tokenizer=tokenizer_21,
        device=device,
    )
    del model_21
    del dataset_21
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1.5 Experiment 10 Ensemble (0.40 GBERT + 0.60 3-seed XLM-R)
    print("\nComputing Experiment 10 ensemble probabilities...")
    exp10_predictions = ensemble_gbert_multiseed_xlmr_predictions(
        gbert_logits=gbert_logits,
        seed_42_logits=logits_42,
        seed_7_logits=logits_7,
        seed_21_logits=logits_21,
        id2label=id2label,
        gbert_weight=0.40,
        xlmr_weight=0.60,
    )

    # Clean up logits to save memory
    del gbert_logits, logits_42, logits_7, logits_21
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # PART 2: EXPERIMENT 16 SPAN MODEL INFERENCE
    # =========================================================================
    print("\n==================================================")
    print("PART 2: EXPERIMENT 16 SPAN MODEL INFERENCE")
    print("==================================================")
    print(f"Loading span model from {XLMR_SPAN_CHECKPOINT}...")
    span_tokenizer = AutoTokenizer.from_pretrained(XLMR_SPAN_CHECKPOINT)
    span_model = XLMRobertaSpan.from_pretrained(XLMR_SPAN_CHECKPOINT)
    span_model.to(device)

    span_dataset, span_stats = prepare_split(val_examples, span_tokenizer, label2id)
    print(f"  Span model alignment mismatches: {span_stats['alignment_mismatches']}")

    exp16_predictions, exp16_retained_spans_list, span_diag = collect_span_predictions(
        model=span_model,
        dataset=span_dataset,
        tokenizer=span_tokenizer,
        device=device,
        batch_size=VAL_BATCH_SIZE,
    )

    del span_model
    del span_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # STEP 1: BASELINE REPRODUCTION VERIFICATION
    # =========================================================================
    print("\n==================================================")
    print("STEP 1: BASELINE REPRODUCTION VERIFICATION")
    print("==================================================")
    score_exp10, metrics_exp10, records_exp10 = calculate_score(val_examples, exp10_predictions)
    score_exp16, metrics_exp16, records_exp16 = calculate_score(val_examples, exp16_predictions)

    print(f"Experiment 10 Reproduced F0.2: {score_exp10:.6f} (Expected: ~{EXPECTED_EXP10_SCORE:.6f})")
    print(f"Experiment 16 Reproduced F0.2: {score_exp16:.6f} (Expected: ~{EXPECTED_EXP16_SCORE:.6f})")

    diff_exp10 = abs(score_exp10 - EXPECTED_EXP10_SCORE)
    diff_exp16 = abs(score_exp16 - EXPECTED_EXP16_SCORE)

    if diff_exp10 > 0.0005:
        raise RuntimeError(
            f"Experiment 10 baseline score mismatch! Got {score_exp10:.6f}, expected {EXPECTED_EXP10_SCORE:.6f}."
        )
    if diff_exp16 > 0.0005:
        raise RuntimeError(
            f"Experiment 16 baseline score mismatch! Got {score_exp16:.6f}, expected {EXPECTED_EXP16_SCORE:.6f}."
        )
    print("PASS: Both baseline scores successfully reproduced within tolerance!")

    # =========================================================================
    # STEP 2: PREDICTION AGREEMENT ANALYSIS
    # =========================================================================
    print("\n==================================================")
    print("STEP 2: PREDICTION AGREEMENT ANALYSIS")
    print("==================================================")
    agreement = analyze_prediction_agreement(
        val_examples=val_examples,
        exp10_predictions=exp10_predictions,
        exp16_predictions=exp16_predictions,
        exp16_retained_spans_list=exp16_retained_spans_list,
    )

    tot_g = agreement["total_gold_entities"]
    tot_10 = agreement["total_exp10_entities"]
    tot_16 = agreement["total_exp16_entities"]
    ident = agreement["identical_entities"]
    e10_only = agreement["exp10_only_entities"]
    e16_only = agreement["exp16_only_entities"]
    e10_diff = agreement["exp10_overlapping_diff"]
    e16_diff = agreement["exp16_overlapping_diff"]

    print(f"Total Gold Entities:                           {tot_g}")
    print(f"Total Experiment 10 Entities:                  {tot_10}")
    print(f"Total Experiment 16 Entities:                  {tot_16}")
    print(f"\nEntity Agreement Breakdown:")
    print(f"  Identical Predictions (exact boundary & aspect): {ident} ({(ident / tot_10) * 100:.2f}% of Exp 10, {(ident / tot_16) * 100:.2f}% of Exp 16)")
    print(f"  Predicted ONLY by Experiment 10:               {e10_only} ({(e10_only / tot_10) * 100:.2f}% of Exp 10)")
    print(f"  Predicted ONLY by Experiment 16:               {e16_only} ({(e16_only / tot_16) * 100:.2f}% of Exp 16)")
    print(f"  Overlapping Predictions with Disagreement:")
    print(f"    From Exp 10 perspective:                     {e10_diff}")
    print(f"    From Exp 16 perspective:                     {e16_diff}")
    print(f"    - Same boundary, different aspect label:     {agreement['overlap_diff_same_boundary_diff_aspect']}")
    print(f"    - Different boundary, same aspect label:     {agreement['overlap_diff_diff_boundary_same_aspect']}")
    print(f"    - Different boundary, different aspect label:{agreement['overlap_diff_diff_boundary_diff_aspect']}")

    print(f"\nPrecision of Experiment 16-Only Entities (Disjoint from Exp 10):")
    print(f"  Total Considered:                              {e16_only}")
    print(f"  True Positives (match gold exactly):           {agreement['exp16_only_tp']}")
    print(f"  False Positives:                               {agreement['exp16_only_fp']}")
    print(f"  Raw Precision of Exp 16-Only:                  {agreement['exp16_only_precision']:.4f}")

    vm_data = agreement["vehicle_model_breakdown"]
    print(f"\nBreakdown for Kompatibles_Fahrzeug_Modell:")
    print(f"  Gold Count:                                    {vm_data['gold_count']}")
    print(f"  Exp 10 Total:                                  {vm_data['exp10_total']}")
    print(f"  Exp 16 Total:                                  {vm_data['exp16_total']}")
    print(f"  Identical Predictions:                         {vm_data['identical']}")
    print(f"  Exp 10 Only:                                   {vm_data['exp10_only']}")
    print(f"  Exp 16 Only:                                   {vm_data['exp16_only']}")
    print(f"  Overlapping Disagreements:                     {vm_data['overlapping_diff']}")

    # Overall FP / FN for each model
    overall_exp10 = compute_overall_entity_metrics(records_exp10)
    overall_exp16 = compute_overall_entity_metrics(records_exp16)

    print(f"\nFalse Positives & False Negatives (Overall Entity Level):")
    print(f"  Experiment 10: TP={overall_exp10['tp']} | FP={overall_exp10['fp']} | FN={overall_exp10['fn']} | P={overall_exp10['precision']:.4f} | R={overall_exp10['recall']:.4f} | F0.2={overall_exp10['f02']:.4f}")
    print(f"  Experiment 16: TP={overall_exp16['tp']} | FP={overall_exp16['fp']} | FN={overall_exp16['fn']} | P={overall_exp16['precision']:.4f} | R={overall_exp16['recall']:.4f} | F0.2={overall_exp16['f02']:.4f}")

    # =========================================================================
    # STEP 3: APPLY CONSERVATIVE FUSION RULE (THRESHOLD = 0.98)
    # =========================================================================
    print("\n==================================================")
    print(f"STEP 3: CONSERVATIVE FUSION (CONFIDENCE THRESHOLD = {CONFIDENCE_THRESHOLD})")
    print("==================================================")

    fused_bio_predictions: List[List[str]] = []
    total_exact_matches = 0
    total_overlapping_rejections = 0
    total_considered_span_only = 0
    total_accepted_span_only = 0

    accepted_additions_tp = 0
    accepted_additions_fp = 0
    accepted_aspect_counts = Counter()

    for i, example in enumerate(val_examples):
        exp10_bio = exp10_predictions[i]
        retained_spans = exp16_retained_spans_list[i]
        seq_len = len(example["tokens"])
        gold_spans = set(bio_to_spans(example["bio_tags"]))

        final_bio, final_spans, stats = apply_conservative_fusion_example(
            exp10_bio=exp10_bio,
            exp16_retained_spans=retained_spans,
            seq_len=seq_len,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        fused_bio_predictions.append(final_bio)
        total_exact_matches += stats["exact_matches"]
        total_overlapping_rejections += stats["overlapping_rejections"]
        total_considered_span_only += stats["considered_span_only"]
        total_accepted_span_only += stats["accepted_span_only"]

        for s, e, asp, score in stats["accepted_spans"]:
            accepted_aspect_counts[asp] += 1
            if (s, e, asp) in gold_spans:
                accepted_additions_tp += 1
            else:
                accepted_additions_fp += 1

    print(f"Total Exact Matches (Preserved once):          {total_exact_matches}")
    print(f"Total Overlapping Exp 16 Rejections:           {total_overlapping_rejections}")
    print(f"Total Span-Only Entities Considered:           {total_considered_span_only}")
    print(f"Total Span-Only Entities Accepted (>= 0.98):   {total_accepted_span_only}")
    print(f"Accepted Additions TP:                         {accepted_additions_tp}")
    print(f"Accepted Additions FP:                         {accepted_additions_fp}")
    add_precision = (
        accepted_additions_tp / (accepted_additions_tp + accepted_additions_fp)
        if (accepted_additions_tp + accepted_additions_fp) > 0
        else 0.0
    )
    print(f"Accepted Additions Precision:                  {add_precision:.4f}")
    print(f"Accepted Aspects Breakdown:                    {dict(accepted_aspect_counts)}")

    # =========================================================================
    # STEP 4: EVALUATION & COMPARISON
    # =========================================================================
    print("\n==================================================")
    print("STEP 4: EVALUATION & COMPARISON")
    print("==================================================")

    score_exp18, metrics_exp18, records_exp18 = calculate_score(val_examples, fused_bio_predictions)
    delta_vs_exp10 = score_exp18 - score_exp10
    delta_vs_exp16 = score_exp18 - score_exp16

    cat1_10 = metrics_exp10["categories"].get(1, {}).get("score", 0.0)
    cat2_10 = metrics_exp10["categories"].get(2, {}).get("score", 0.0)

    cat1_16 = metrics_exp16["categories"].get(1, {}).get("score", 0.0)
    cat2_16 = metrics_exp16["categories"].get(2, {}).get("score", 0.0)

    cat1_18 = metrics_exp18["categories"].get(1, {}).get("score", 0.0)
    cat2_18 = metrics_exp18["categories"].get(2, {}).get("score", 0.0)

    overall_exp18 = compute_overall_entity_metrics(records_exp18)
    delta_tp = overall_exp18["tp"] - overall_exp10["tp"]
    delta_fp = overall_exp18["fp"] - overall_exp10["fp"]
    delta_fn = overall_exp18["fn"] - overall_exp10["fn"]

    print("\n--- COMPETITION-STYLE F0.2 SCORES ---")
    print(f"{'Model':<35} | {'Overall F0.2':<12} | {'Category 1':<12} | {'Category 2':<12} | {'Delta vs Exp 10':<15}")
    print("-" * 95)
    print(f"{'Experiment 10 (GBERT+3-XLM-R)':<35} | {score_exp10:<12.6f} | {cat1_10:<12.6f} | {cat2_10:<12.6f} | {'Baseline':<15}")
    print(f"{'Experiment 16 (Span XLM-R)':<35} | {score_exp16:<12.6f} | {cat1_16:<12.6f} | {cat2_16:<12.6f} | {score_exp16 - score_exp10:<+15.6f}")
    print(f"{'Experiment 18 (Conservative Fusion)':<35} | {score_exp18:<12.6f} | {cat1_18:<12.6f} | {cat2_18:<12.6f} | {delta_vs_exp10:<+15.6f}")

    print("\n--- OVERALL ENTITY-LEVEL ERROR METRICS ---")
    print(f"{'Model':<35} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F0.2':<10}")
    print("-" * 95)
    print(f"{'Experiment 10':<35} | {overall_exp10['tp']:<6} | {overall_exp10['fp']:<6} | {overall_exp10['fn']:<6} | {overall_exp10['precision']:<10.4f} | {overall_exp10['recall']:<10.4f} | {overall_exp10['f02']:<10.4f}")
    print(f"{'Experiment 16':<35} | {overall_exp16['tp']:<6} | {overall_exp16['fp']:<6} | {overall_exp16['fn']:<6} | {overall_exp16['precision']:<10.4f} | {overall_exp16['recall']:<10.4f} | {overall_exp16['f02']:<10.4f}")
    print(f"{'Experiment 18 (Fused)':<35} | {overall_exp18['tp']:<6} | {overall_exp18['fp']:<6} | {overall_exp18['fn']:<6} | {overall_exp18['precision']:<10.4f} | {overall_exp18['recall']:<10.4f} | {overall_exp18['f02']:<10.4f}")
    print(f"Net Changes (Exp 18 - Exp 10):       TP: {delta_tp:+d} | FP: {delta_fp:+d} | FN: {delta_fn:+d}")

    # VehicleModel aspect metrics
    vm_10 = extract_vehicle_model_metrics(metrics_exp10, records_exp10)
    vm_16 = extract_vehicle_model_metrics(metrics_exp16, records_exp16)
    vm_18 = extract_vehicle_model_metrics(metrics_exp18, records_exp18)

    print("\n--- KOMPATIBLES_FAHRZEUG_MODELL METRICS ---")
    print(f"{'Model':<35} | {'Gold':<6} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F0.2':<10}")
    print("-" * 105)
    print(f"{'Experiment 10':<35} | {vm_10['gold_count']:<6} | {vm_10['tp']:<6} | {vm_10['fp']:<6} | {vm_10['fn']:<6} | {vm_10['precision']:<10.4f} | {vm_10['recall']:<10.4f} | {vm_10['f02']:<10.4f}")
    print(f"{'Experiment 16':<35} | {vm_16['gold_count']:<6} | {vm_16['tp']:<6} | {vm_16['fp']:<6} | {vm_16['fn']:<6} | {vm_16['precision']:<10.4f} | {vm_16['recall']:<10.4f} | {vm_16['f02']:<10.4f}")
    print(f"{'Experiment 18 (Fused)':<35} | {vm_18['gold_count']:<6} | {vm_18['tp']:<6} | {vm_18['fp']:<6} | {vm_18['fn']:<6} | {vm_18['precision']:<10.4f} | {vm_18['recall']:<10.4f} | {vm_18['f02']:<10.4f}")

    # Performance by Span Width
    width_10 = compute_performance_by_span_width(records_exp10)
    width_16 = compute_performance_by_span_width(records_exp16)
    width_18 = compute_performance_by_span_width(records_exp18)

    print("\n--- PERFORMANCE BY GOLD ENTITY SPAN WIDTH ---")
    for w_key in ["width1", "width2", "width3+"]:
        w10 = width_10[w_key]
        w16 = width_16[w_key]
        w18 = width_18[w_key]
        print(f"\n{w_key.upper()} (Gold Count: {w10['gold_count']}):")
        print(f"  Exp 10: P={w10['precision']:.4f} | R={w10['recall']:.4f} | F0.2={w10['f02']:.4f} | TP={w10['tp']} | FP={w10['fp']} | FN={w10['fn']}")
        print(f"  Exp 16: P={w16['precision']:.4f} | R={w16['recall']:.4f} | F0.2={w16['f02']:.4f} | TP={w16['tp']} | FP={w16['fp']} | FN={w16['fn']}")
        print(f"  Exp 18: P={w18['precision']:.4f} | R={w18['recall']:.4f} | F0.2={w18['f02']:.4f} | TP={w18['tp']} | FP={w18['fp']} | FN={w18['fn']}")
        print(f"  Delta F0.2 (Exp 18 - Exp 10): {w18['f02'] - w10['f02']:+.6f}")

    # Decision Criteria Verdict
    print("\n==================================================")
    print("DECISION CRITERIA & HYPOTHESIS INTERPRETATION")
    print("==================================================")
    if delta_vs_exp10 >= 0.003:
        verdict = "MEANINGFUL SUCCESS (Improvement >= +0.003)"
    elif delta_vs_exp10 >= 0.001:
        verdict = "MODEST SUCCESS (Improvement +0.001 to +0.003, needs robustness confirmation)"
    elif delta_vs_exp10 >= -0.001:
        verdict = "NEUTRAL (Score within +/- 0.001 of baseline)"
    else:
        verdict = "REJECT (Drop > 0.001)"

    print(f"Final Decision Verdict: {verdict}")
    print(f"Overall Delta vs Experiment 10: {delta_vs_exp10:+.6f}")
    print(
        "\nIMPORTANT NOTICE: Validation-set improvement is not proof of better hidden-test performance."
    )

    return {
        "score_exp10": score_exp10,
        "score_exp16": score_exp16,
        "score_exp18": score_exp18,
        "delta_vs_exp10": delta_vs_exp10,
        "verdict": verdict,
        "agreement": agreement,
        "metrics_exp10": metrics_exp10,
        "metrics_exp16": metrics_exp16,
        "metrics_exp18": metrics_exp18,
        "overall_exp10": overall_exp10,
        "overall_exp16": overall_exp16,
        "overall_exp18": overall_exp18,
        "vehicle_model_exp10": vm_10,
        "vehicle_model_exp16": vm_16,
        "vehicle_model_exp18": vm_18,
        "width_exp10": width_10,
        "width_exp16": width_16,
        "width_exp18": width_18,
    }


def main() -> None:
    run_experiment_18()


if __name__ == "__main__":
    main()
