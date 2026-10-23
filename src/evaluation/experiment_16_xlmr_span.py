"""
Experiment 16: Span-Based XLM-RoBERTa Evaluation.

Evaluates the checkpoint: checkpoints/xlmr_span_seed42_best
Compares directly against:
- Plain XLM-R seed-42 baseline (0.869284)
- Boundary-aware BIO Experiment 14 (0.875489)

Overlap resolution:
- Predicted candidate spans with class != NONE are sorted descending by confidence.
- Overlapping candidates are greedily rejected.
- Non-overlapping spans are converted to BIO labels for competition-style F0.2 evaluation.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
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
from src.modeling.xlmr_span import (
    MAX_SPAN_WIDTH,
    XLMRobertaSpan,
    bio_to_spans,
    create_span_mappings,
)
from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.error_analysis import (
    analyze_token_confusions,
    analyze_entity_errors,
    print_top_confusions,
    print_entity_error_summary,
)


CHECKPOINT_DIR = Path("checkpoints/xlmr_span_seed42_best")
BASELINE_XLMR_SEED42_SCORE = 0.869284
EXP14_BOUNDARY_SCORE = 0.875489
VAL_BATCH_SIZE = 8


def evaluate_span_model(
    model: XLMRobertaSpan,
    dataset,
    examples: List[dict],
    tokenizer,
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> dict:
    """Run inference over validation split and compute entity metrics and span diagnostics."""
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
    true_bio_sequences = []
    predicted_bio_sequences = []

    diag_totals = {
        "total_candidates": 0,
        "none_candidates": 0,
        "eligible_candidates": 0,
        "retained_candidates": 0,
        "rejected_overlap_candidates": 0,
    }

    example_index = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            pred_bio_batch, diag = model.predict_spans(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                labels=batch_dev.get("labels"),
            )

            for k in diag_totals:
                diag_totals[k] += diag[k]

            batch_size_actual = len(pred_bio_batch)

            for row_index in range(batch_size_actual):
                example = examples[example_index]
                predicted_bio = pred_bio_batch[row_index]
                true_bio = example["bio_tags"]
                tokens = example["tokens"]

                if len(predicted_bio) != len(tokens):
                    raise ValueError(
                        f"Predicted BIO count ({len(predicted_bio)}) != token count ({len(tokens)})."
                    )

                true_bio_sequences.append(true_bio)
                predicted_bio_sequences.append(predicted_bio)

                true_entities = bio_to_entities(tokens, true_bio)
                predicted_entities = bio_to_entities(tokens, predicted_bio)

                records.append(
                    {
                        "category": example["category"],
                        "true_entities": true_entities,
                        "predicted_entities": predicted_entities,
                    }
                )
                example_index += 1

    if example_index != len(examples):
        raise ValueError("Not all validation examples were evaluated.")

    results = calculate_competition_metrics(records)

    return {
        "metrics": results,
        "records": records,
        "true_bio_sequences": true_bio_sequences,
        "predicted_bio_sequences": predicted_bio_sequences,
        "diagnostics": diag_totals,
    }


def compute_performance_by_span_width(records: List[dict]) -> dict:
    """Calculate Precision, Recall, and F0.2 stratified by entity span width (1, 2, 3+)."""
    buckets = {"width1": {"tp": 0, "fp": 0, "fn": 0},
               "width2": {"tp": 0, "fp": 0, "fn": 0},
               "width3+": {"tp": 0, "fp": 0, "fn": 0}}

    for rec in records:
        true_list = list(rec["true_entities"])
        pred_list = list(rec["predicted_entities"])

        # Match true entities to predicted
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


def print_span_diagnostics_report(
    diagnostics: dict,
    records: List[dict],
) -> None:
    """Print comprehensive span diagnostics and performance stratified by width."""
    print("\n--- SPAN DIAGNOSTICS REPORT ---")
    tot = max(1, diagnostics["total_candidates"])
    none_cnt = diagnostics["none_candidates"]
    elig = diagnostics["eligible_candidates"]
    ret = diagnostics["retained_candidates"]
    rej = diagnostics["rejected_overlap_candidates"]

    print(f"Total candidate spans generated:               {diagnostics['total_candidates']}")
    print(f"Predicted NONE:                                {none_cnt} ({(none_cnt / tot) * 100:.2f}%)")
    print(f"Non-NONE eligible candidates before overlap:   {elig}")
    print(f"Retained candidates after overlap resolution:  {ret}")
    print(f"Rejected candidates due to overlap:            {rej} ({(rej / max(1, elig)) * 100:.2f}% of eligible)")

    # Performance stratified by span width
    width_perf = compute_performance_by_span_width(records)
    print("\nPerformance by Gold Entity Span Width:")
    for b_key in ["width1", "width2", "width3+"]:
        m = width_perf[b_key]
        print(
            f"  {b_key:8s}: Gold={m['gold_count']:4d} | "
            f"P={m['precision']:.4f} | R={m['recall']:.4f} | "
            f"F0.2={m['f02']:.4f} | FP={m['fp']} | FN={m['fn']}"
        )


def main() -> None:
    print("==================================================")
    print("EXPERIMENT 16: SPAN-BASED XLM-RoBERTa EVALUATION")
    print("==================================================")

    # 1. Defensive Checkpoint Check
    weights_file = CHECKPOINT_DIR / "span_heads.pt"
    if not CHECKPOINT_DIR.exists() or not weights_file.exists():
        print(
            f"\n[!] Checkpoint not found at: {CHECKPOINT_DIR}\n"
            f"Please run training first before evaluating Experiment 16:\n"
            f"  python -m src.modeling.train_xlmr_span\n"
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
    model = XLMRobertaSpan.from_pretrained(CHECKPOINT_DIR)
    model.to(device)

    # 5. Prepare Validation Dataset
    val_dataset, val_stats = prepare_split(
        val_examples,
        tokenizer,
        label2id,
    )
    print(f"Validation alignment mismatches: {val_stats['alignment_mismatches']}")

    # 6. Evaluate Model
    print("\nRunning span inference and overlap resolution...")
    evaluation_output = evaluate_span_model(
        model=model,
        dataset=val_dataset,
        examples=val_examples,
        tokenizer=tokenizer,
        device=device,
        batch_size=VAL_BATCH_SIZE,
    )

    metrics = evaluation_output["metrics"]
    span_score = metrics["final_score"]
    delta_baseline = span_score - BASELINE_XLMR_SEED42_SCORE
    delta_exp14 = span_score - EXP14_BOUNDARY_SCORE

    cat1_score = metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = metrics["categories"].get(2, {}).get("score", 0.0)

    # 7. Print Results
    print("\n==================================================")
    print("EXPERIMENT 16 RESULTS")
    print("==================================================")
    print(f"Plain XLM-R seed-42 baseline:         {BASELINE_XLMR_SEED42_SCORE:.6f}")
    print(f"Boundary-aware BIO Experiment 14:     {EXP14_BOUNDARY_SCORE:.6f}")
    print(f"Span-based XLM-R:                     {span_score:.6f}")
    print(f"Delta vs plain XLM-R:                 {delta_baseline:+.6f}")
    print(f"Delta vs Experiment 14:               {delta_exp14:+.6f}")
    print(f"Category 1 score:                     {cat1_score:.6f}")
    print(f"Category 2 score:                     {cat2_score:.6f}")

    # 8. Error Analysis & Vehicle Model Aspect Inspection
    print("\n--- GENERAL ERROR ANALYSIS ---")
    confusions = analyze_token_confusions(
        evaluation_output["true_bio_sequences"],
        evaluation_output["predicted_bio_sequences"],
    )
    aspect_errors = analyze_entity_errors(evaluation_output["records"])
    print_top_confusions(confusions, top_n=20)
    print_entity_error_summary(aspect_errors)

    inspect_vehicle_model_aspect(
        evaluation_output,
        aspect_name="Kompatibles_Fahrzeug_Modell",
    )

    # 9. Span Diagnostics
    print_span_diagnostics_report(
        evaluation_output["diagnostics"],
        evaluation_output["records"],
    )

    # 10. Clean up VRAM
    del model
    del val_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
