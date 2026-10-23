"""
Experiment 17: Class-Weighted Span-Based XLM-RoBERTa Evaluation.

Evaluates the checkpoint: checkpoints/xlmr_span_weighted_seed42_best
Compares directly against:
- Plain XLM-R seed-42 baseline (0.869284)
- Boundary-aware BIO Experiment 14 (0.875489)
- Unweighted Span Experiment 16 (0.875627)

Hypothesis:
Reducing NONE class loss weight to 0.70 during training improves recall
on multi-token entities (width2, width3+) without damaging precision.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from src.data.build_examples import (
    build_examples,
    load_training_data,
)
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.modeling.xlmr_span import XLMRobertaSpan
from src.evaluation.error_analysis import (
    analyze_token_confusions,
    analyze_entity_errors,
    print_top_confusions,
    print_entity_error_summary,
)
from src.evaluation.experiment_16_xlmr_span import (
    compute_performance_by_span_width,
    evaluate_span_model,
)


CHECKPOINT_DIR = Path("checkpoints/xlmr_span_weighted_seed42_best")

BASELINE_XLMR_SEED42_SCORE = 0.869284
EXP14_BOUNDARY_SCORE = 0.875489
EXP16_SPAN_UNWEIGHTED_SCORE = 0.875627

VAL_BATCH_SIZE = 8


# Reference Experiment 16 values for direct comparison table
EXP16_REF = {
    "overall_f02": 0.875627,
    "cat1": 0.882049,
    "cat2": 0.869206,
    "none_pct": 84.03,
    "vm_fp": 298,
    "vm_fn": 823,
    "vm_cat1_p": 0.7841,
    "vm_cat1_r": 0.5881,
    "vm_cat2_p": 0.7741,
    "vm_cat2_r": 0.5321,
    "width1_f02": 0.9120,
    "width2_f02": 0.7461,
    "width3_f02": 0.7168,
}


def inspect_vehicle_model_aspect(
    evaluation_output: dict,
    aspect_name: str = "Kompatibles_Fahrzeug_Modell",
) -> dict:
    """Detailed inspection of Kompatibles_Fahrzeug_Modell performance, errors, and boundary confusions."""
    print(f"\n--- DETAILED INSPECTION: {aspect_name} ---")

    metrics = evaluation_output["metrics"]
    stats = {
        "cat1_p": 0.0, "cat1_r": 0.0, "cat1_f02": 0.0,
        "cat2_p": 0.0, "cat2_r": 0.0, "cat2_f02": 0.0,
        "fp": 0, "fn": 0, "total_errors": 0,
        "b_to_i": 0, "i_to_b": 0,
    }

    for category, cat_data in sorted(metrics["categories"].items()):
        aspects = cat_data.get("aspects", {})
        if aspect_name in aspects:
            asp = aspects[aspect_name]
            p = asp["precision"]
            r = asp["recall"]
            f02 = asp["f_beta"]
            print(f"Category {category}:")
            print(f"  Count:     {asp['true_count']}")
            print(f"  Precision: {p:.4f}")
            print(f"  Recall:    {r:.4f}")
            print(f"  F0.2:      {f02:.4f}")

            if category == 1:
                stats["cat1_p"] = p
                stats["cat1_r"] = r
                stats["cat1_f02"] = f02
            elif category == 2:
                stats["cat2_p"] = p
                stats["cat2_r"] = r
                stats["cat2_f02"] = f02

    records = evaluation_output["records"]
    aspect_errors = analyze_entity_errors(records)
    if aspect_name in aspect_errors:
        err = aspect_errors[aspect_name]
        stats["fp"] = err["false_positives"]
        stats["fn"] = err["false_negatives"]
        stats["total_errors"] = stats["fp"] + stats["fn"]
        print(f"  Entity Errors: FP={stats['fp']} | FN={stats['fn']} | Total={stats['total_errors']}")

    true_bio_sequences = evaluation_output["true_bio_sequences"]
    predicted_bio_sequences = evaluation_output["predicted_bio_sequences"]
    confusions = analyze_token_confusions(
        true_bio_sequences,
        predicted_bio_sequences,
    )

    b_tag = f"B-{aspect_name}"
    i_tag = f"I-{aspect_name}"
    stats["b_to_i"] = confusions[(b_tag, i_tag)]
    stats["i_to_b"] = confusions[(i_tag, b_tag)]

    print(f"  Boundary Confusions for {aspect_name}:")
    print(f"    B -> I confusion count: {stats['b_to_i']}")
    print(f"    I -> B confusion count: {stats['i_to_b']}")

    return stats


def print_comparison_table(
    overall_f02: float,
    cat1: float,
    cat2: float,
    none_pct: float,
    vm_stats: dict,
    width_perf: dict,
) -> None:
    """Print side-by-side comparison table between Exp 16 (unweighted) and Exp 17 (weighted)."""
    print("\n==================================================")
    print("DIRECT EXPERIMENT 16 vs EXPERIMENT 17 COMPARISON")
    print("==================================================")
    headers = f"{'Metric':<30} {'Exp16 (Unweighted)':<20} {'Exp17 (Weighted)':<20} {'Delta':<10}"
    print(headers)
    print("-" * len(headers))

    rows = [
        ("Overall F0.2", EXP16_REF["overall_f02"], overall_f02, f"{overall_f02 - EXP16_REF['overall_f02']:+.6f}"),
        ("Category 1", EXP16_REF["cat1"], cat1, f"{cat1 - EXP16_REF['cat1']:+.6f}"),
        ("Category 2", EXP16_REF["cat2"], cat2, f"{cat2 - EXP16_REF['cat2']:+.6f}"),
        ("Predicted NONE %", f"{EXP16_REF['none_pct']:.2f}%", f"{none_pct:.2f}%", f"{none_pct - EXP16_REF['none_pct']:+.2f}%"),
        ("VehicleModel FP", EXP16_REF["vm_fp"], vm_stats["fp"], f"{vm_stats['fp'] - EXP16_REF['vm_fp']:+d}"),
        ("VehicleModel FN", EXP16_REF["vm_fn"], vm_stats["fn"], f"{vm_stats['fn'] - EXP16_REF['vm_fn']:+d}"),
        ("VehicleModel Cat1 Precision", f"{EXP16_REF['vm_cat1_p']:.4f}", f"{vm_stats['cat1_p']:.4f}", f"{vm_stats['cat1_p'] - EXP16_REF['vm_cat1_p']:+.4f}"),
        ("VehicleModel Cat1 Recall", f"{EXP16_REF['vm_cat1_r']:.4f}", f"{vm_stats['cat1_r']:.4f}", f"{vm_stats['cat1_r'] - EXP16_REF['vm_cat1_r']:+.4f}"),
        ("VehicleModel Cat2 Precision", f"{EXP16_REF['vm_cat2_p']:.4f}", f"{vm_stats['cat2_p']:.4f}", f"{vm_stats['cat2_p'] - EXP16_REF['vm_cat2_p']:+.4f}"),
        ("VehicleModel Cat2 Recall", f"{EXP16_REF['vm_cat2_r']:.4f}", f"{vm_stats['cat2_r']:.4f}", f"{vm_stats['cat2_r'] - EXP16_REF['vm_cat2_r']:+.4f}"),
        ("width1 F0.2", f"{EXP16_REF['width1_f02']:.4f}", f"{width_perf['width1']['f02']:.4f}", f"{width_perf['width1']['f02'] - EXP16_REF['width1_f02']:+.4f}"),
        ("width2 F0.2", f"{EXP16_REF['width2_f02']:.4f}", f"{width_perf['width2']['f02']:.4f}", f"{width_perf['width2']['f02'] - EXP16_REF['width2_f02']:+.4f}"),
        ("width3+ F0.2", f"{EXP16_REF['width3_f02']:.4f}", f"{width_perf['width3+']['f02']:.4f}", f"{width_perf['width3+']['f02'] - EXP16_REF['width3_f02']:+.4f}"),
    ]

    for name, v16, v17, delta in rows:
        v16_str = f"{v16:.6f}" if isinstance(v16, float) else str(v16)
        v17_str = f"{v17:.6f}" if isinstance(v17, float) else str(v17)
        print(f"{name:<30} {v16_str:<20} {v17_str:<20} {delta:<10}")


def print_interpretation_verdict(overall_f02: float) -> None:
    """Evaluate pre-declared interpretation rules."""
    delta = overall_f02 - EXP16_REF["overall_f02"]
    print("\n--- PRE-DECLARED INTERPRETATION RULES ---")
    print(f"Observed Delta vs Experiment 16: {delta:+.6f}")

    if delta >= 0.003:
        verdict = "STRONG SUCCESS: Overall F0.2 improved by >= +0.003 vs Exp 16."
    elif 0.001 <= delta < 0.003:
        verdict = "USEFUL BUT MODEST: Overall F0.2 improved between +0.001 and +0.003."
    elif -0.001 <= delta < 0.001:
        verdict = "EFFECTIVELY NEUTRAL: Change is between -0.001 and +0.001."
    else:
        verdict = "REJECT CHANGE: Score dropped by more than 0.001."

    print(f"Verdict: {verdict}")


def main() -> None:
    print("==================================================")
    print("EXPERIMENT 17: CLASS-WEIGHTED SPAN XLM-R")
    print("==================================================")

    # 1. Defensive Checkpoint Check
    weights_file = CHECKPOINT_DIR / "span_heads.pt"
    if not CHECKPOINT_DIR.exists() or not weights_file.exists():
        print(
            f"\n[!] Checkpoint not found at: {CHECKPOINT_DIR}\n"
            f"Please run training first before evaluating Experiment 17:\n"
            f"  python -m src.modeling.train_xlmr_span_weighted\n"
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
    print("\nRunning span inference and greedy overlap resolution...")
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
    delta_exp16 = span_score - EXP16_SPAN_UNWEIGHTED_SCORE

    cat1_score = metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = metrics["categories"].get(2, {}).get("score", 0.0)

    # 7. Print Results
    print("\n==================================================")
    print("EXPERIMENT 17 RESULTS")
    print("==================================================")
    print(f"Plain XLM-R seed-42 baseline:         {BASELINE_XLMR_SEED42_SCORE:.6f}")
    print(f"Boundary-aware BIO Experiment 14:     {EXP14_BOUNDARY_SCORE:.6f}")
    print(f"Unweighted Span Experiment 16:        {EXP16_SPAN_UNWEIGHTED_SCORE:.6f}")
    print(f"Class-Weighted Span Experiment 17:    {span_score:.6f}")
    print(f"Delta vs plain XLM-R:                 {delta_baseline:+.6f}")
    print(f"Delta vs Experiment 14:               {delta_exp14:+.6f}")
    print(f"Delta vs Experiment 16:               {delta_exp16:+.6f}")
    print(f"Category 1 score:                     {cat1_score:.6f}")
    print(f"Category 2 score:                     {cat2_score:.6f}")

    # 8. Span Diagnostics
    diag = evaluation_output["diagnostics"]
    tot = max(1, diag["total_candidates"])
    none_pct = (diag["none_candidates"] / tot) * 100
    total_predicted_entities = sum(len(rec["predicted_entities"]) for rec in evaluation_output["records"])

    print("\n--- SPAN DIAGNOSTICS ---")
    print(f"Total candidate spans generated:               {diag['total_candidates']}")
    print(f"Predicted NONE:                                {diag['none_candidates']} ({none_pct:.2f}%)")
    print(f"Non-NONE eligible candidates before overlap:   {diag['eligible_candidates']}")
    print(f"Retained candidates after overlap resolution:  {diag['retained_candidates']}")
    print(f"Rejected candidates due to overlap:            {diag['rejected_overlap_candidates']}")
    print(f"Total predicted entities across validation:    {total_predicted_entities}")

    # 9. Error Analysis & Vehicle Model Aspect Inspection
    print("\n--- GENERAL ERROR ANALYSIS ---")
    confusions = analyze_token_confusions(
        evaluation_output["true_bio_sequences"],
        evaluation_output["predicted_bio_sequences"],
    )
    aspect_errors = analyze_entity_errors(evaluation_output["records"])
    print_top_confusions(confusions, top_n=20)
    print_entity_error_summary(aspect_errors)

    vm_stats = inspect_vehicle_model_aspect(
        evaluation_output,
        aspect_name="Kompatibles_Fahrzeug_Modell",
    )

    # 10. Span Width Analysis
    width_perf = compute_performance_by_span_width(evaluation_output["records"])
    print("\n--- PERFORMANCE BY GOLD ENTITY SPAN WIDTH ---")
    for b_key in ["width1", "width2", "width3+"]:
        m = width_perf[b_key]
        print(
            f"  {b_key:8s}: Gold={m['gold_count']:4d} | "
            f"P={m['precision']:.4f} | R={m['recall']:.4f} | "
            f"F0.2={m['f02']:.4f} | FP={m['fp']} | FN={m['fn']}"
        )

    # 11. Direct Comparison Table & Interpretation Verdict
    print_comparison_table(
        overall_f02=span_score,
        cat1=cat1_score,
        cat2=cat2_score,
        none_pct=none_pct,
        vm_stats=vm_stats,
        width_perf=width_perf,
    )

    print_interpretation_verdict(overall_f02=span_score)

    # 12. Clean up VRAM
    del model
    del val_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
