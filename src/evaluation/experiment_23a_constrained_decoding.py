"""
Phase 23A: Category-Constrained BIO Decoding Experiment.

Compares two inference-time category enforcement strategies on Split-A validation (1,000 titles):
- Method A (Current Baseline):
    1. Ensemble probabilities (GBERT 0.40 + XLM-R seeds 42, 7, 21 at 0.20 each).
    2. Argmax BIO decoding.
    3. bio_to_entities reconstruction.
    4. filter_entities_by_category post-filtering.
- Method B (Category-Constrained Decoding):
    1. Ensemble probabilities.
    2. Mask forbidden BIO classes per title category before argmax (O and allowed classes preserved).
    3. Argmax over allowed classes.
    4. bio_to_entities reconstruction.
    5. Taxonomy validation.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.entity_metrics import (
    BETA,
    bio_to_entities,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
)
from src.modeling.prepare_datasets import prepare_split
from scripts.validate_submission import (
    OFFICIAL_ASPECTS,
    OFFICIAL_ASPECTS_CAT1,
    OFFICIAL_ASPECTS_CAT2,
)

# Champion Checkpoints (Validation-trained, Split A)
GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
XLMR_SEED_42_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")
XLMR_SEED_7_CHECKPOINT = Path("checkpoints/xlmr_seed_7_best")
XLMR_SEED_21_CHECKPOINT = Path("checkpoints/xlmr_seed_21_best")

VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100
EXPECTED_EXP10_RAW_SCORE = 0.879852


def get_category_allowed_label_indices(
    label2id: Dict[str, int], category: int
) -> Set[int]:
    """Return the set of allowed label indices in label2id for a given category (1 or 2)."""
    allowed_aspects = OFFICIAL_ASPECTS_CAT1 if category == 1 else OFFICIAL_ASPECTS_CAT2
    allowed_indices = set()
    for label, lid in label2id.items():
        if label == "O":
            allowed_indices.add(lid)
        elif label.startswith("B-") or label.startswith("I-"):
            aspect = label[2:]
            if aspect in allowed_aspects:
                allowed_indices.add(lid)
        else:
            raise ValueError(f"Unrecognized label format: {label}")
    return allowed_indices


def get_category_forbidden_mask(
    label2id: Dict[str, int], category: int
) -> np.ndarray:
    """Return a boolean mask of length num_labels, True where label is forbidden for this category."""
    allowed = get_category_allowed_label_indices(label2id, category)
    mask = np.zeros(len(label2id), dtype=bool)
    for lid in range(len(label2id)):
        if lid not in allowed:
            mask[lid] = True
    return mask


def collect_active_logits(
    checkpoint_path: Path,
    val_examples: List[dict],
    label2id: Dict[str, int],
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> List[List[List[float]]]:
    """Sequentially load model, run validation inference, extract active logits, and free GPU memory."""
    print(f"Loading {checkpoint_path.name}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()

    dataset, stats = prepare_split(val_examples, tokenizer, label2id)
    print(f"  Alignment mismatches: {stats['alignment_mismatches']}")

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    all_active_logits = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            for row_idx in range(labels.shape[0]):
                row_labels = labels[row_idx]
                row_logits = logits[row_idx]
                active_mask = row_labels != IGNORE_LABEL_ID
                active_logits = row_logits[active_mask].detach().cpu().tolist()
                all_active_logits.append(active_logits)

    del model, tokenizer, dataset, dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_active_logits


def compute_ensemble_probabilities(
    gbert_logits: List[List[List[float]]],
    seed_42_logits: List[List[List[float]]],
    seed_7_logits: List[List[List[float]]],
    seed_21_logits: List[List[List[float]]],
    gbert_weight: float = 0.40,
    xlmr_weight: float = 0.60,
) -> List[np.ndarray]:
    """
    Compute token-level ensemble probabilities across the 4 models.
    xlmr_multiseed_prob = (prob_42 + prob_7 + prob_21) / 3.0
    combined_prob = gbert_weight * gbert_prob + xlmr_weight * xlmr_multiseed_prob
    """
    num_examples = len(gbert_logits)
    assert len(seed_42_logits) == num_examples
    assert len(seed_7_logits) == num_examples
    assert len(seed_21_logits) == num_examples

    ensemble_probs = []
    for ex_idx in range(num_examples):
        gb_ex = gbert_logits[ex_idx]
        s42_ex = seed_42_logits[ex_idx]
        s7_ex = seed_7_logits[ex_idx]
        s21_ex = seed_21_logits[ex_idx]

        seq_len = len(gb_ex)
        assert len(s42_ex) == seq_len and len(s7_ex) == seq_len and len(s21_ex) == seq_len

        # Convert to numpy tensors
        t_gb = torch.softmax(torch.tensor(gb_ex, dtype=torch.float32), dim=-1).numpy()
        t_42 = torch.softmax(torch.tensor(s42_ex, dtype=torch.float32), dim=-1).numpy()
        t_7 = torch.softmax(torch.tensor(s7_ex, dtype=torch.float32), dim=-1).numpy()
        t_21 = torch.softmax(torch.tensor(s21_ex, dtype=torch.float32), dim=-1).numpy()

        xlmr_avg = (t_42 + t_7 + t_21) / 3.0
        combined = gbert_weight * t_gb + xlmr_weight * xlmr_avg
        ensemble_probs.append(combined)

    return ensemble_probs


def decode_unconstrained_raw(
    val_examples: List[dict],
    ensemble_probs: List[np.ndarray],
    id2label: Dict[int, str],
) -> List[List[Tuple[str, str]]]:
    """Raw unconstrained decoding (Exp 10 baseline, no category filtering)."""
    raw_entities = []
    for ex, probs in zip(val_examples, ensemble_probs):
        tokens = ex["tokens"]
        pred_ids = np.argmax(probs, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        entities = bio_to_entities(tokens, pred_bio)
        raw_entities.append(entities)
    return raw_entities


def decode_method_a(
    val_examples: List[dict],
    ensemble_probs: List[np.ndarray],
    id2label: Dict[int, str],
) -> List[List[Tuple[str, str]]]:
    """
    Method A — Current baseline:
    1. Argmax BIO decoding across all classes.
    2. bio_to_entities reconstruction.
    3. Category-aware post-filtering (discard forbidden aspect entities).
    """
    method_a_entities = []
    for ex, probs in zip(val_examples, ensemble_probs):
        tokens = ex["tokens"]
        category = ex["category"]
        pred_ids = np.argmax(probs, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        entities = bio_to_entities(tokens, pred_bio)
        filtered = filter_entities_by_category(entities, category)
        method_a_entities.append(filtered)
    return method_a_entities


def decode_method_b(
    val_examples: List[dict],
    ensemble_probs: List[np.ndarray],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> List[List[Tuple[str, str]]]:
    """
    Method B — Category-constrained decoding:
    1. Mask forbidden BIO classes for the title's category (setting prob to 0.0).
    2. Argmax over allowed classes (O and allowed aspect tags preserved).
    3. bio_to_entities reconstruction.
    """
    # Precompute masks for Category 1 and Category 2
    cat1_mask = get_category_forbidden_mask(label2id, category=1)
    cat2_mask = get_category_forbidden_mask(label2id, category=2)

    method_b_entities = []
    for ex, probs in zip(val_examples, ensemble_probs):
        tokens = ex["tokens"]
        category = ex["category"]
        mask = cat1_mask if category == 1 else cat2_mask

        # Apply mask: zero out forbidden label probabilities
        masked_probs = probs.copy()
        masked_probs[:, mask] = 0.0

        pred_ids = np.argmax(masked_probs, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        entities = bio_to_entities(tokens, pred_bio)
        method_b_entities.append(entities)
    return method_b_entities


def evaluate_predictions(
    val_examples: List[dict],
    predicted_entities_list: List[List[Tuple[str, str]]],
) -> dict:
    """Compute official competition F0.2 and micro precision/recall/F0.2."""
    records = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for ex, pred_entities in zip(val_examples, predicted_entities_list):
        tokens = ex["tokens"]
        true_bio = ex["bio_tags"]
        true_entities = bio_to_entities(tokens, true_bio)

        tp, fp, fn = count_matches(pred_entities, true_entities)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        records.append({
            "category": ex["category"],
            "true_entities": true_entities,
            "predicted_entities": pred_entities,
        })

    comp_metrics = calculate_competition_metrics(records)
    micro_metrics = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=BETA)

    cat1_score = comp_metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = comp_metrics["categories"].get(2, {}).get("score", 0.0)

    return {
        "final_f02": comp_metrics["final_score"],
        "cat1_f02": cat1_score,
        "cat2_f02": cat2_score,
        "micro_precision": micro_metrics["precision"],
        "micro_recall": micro_metrics["recall"],
        "micro_f02": micro_metrics["f_beta"],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_predicted": sum(len(p) for p in predicted_entities_list),
        "comp_details": comp_metrics,
    }


def count_invalid_predictions(
    val_examples: List[dict],
    predicted_entities_list: List[List[Tuple[str, str]]],
) -> int:
    """Count how many predicted entities violate the official category aspect restrictions."""
    invalid_count = 0
    for ex, pred_entities in zip(val_examples, predicted_entities_list):
        category = ex["category"]
        allowed = OFFICIAL_ASPECTS_CAT1 if category == 1 else OFFICIAL_ASPECTS_CAT2
        for aspect, val in pred_entities:
            if aspect not in allowed:
                invalid_count += 1
    return invalid_count


def analyze_differences(
    val_examples: List[dict],
    preds_a: List[List[Tuple[str, str]]],
    preds_b: List[List[Tuple[str, str]]],
) -> dict:
    """Analyze predictions changed, exact matches gained/lost, and per-aspect impact."""
    titles_changed = 0
    entities_changed = 0
    matches_gained = 0
    matches_lost = 0

    gained_details = []
    lost_details = []

    for ex_idx, (ex, a_ents, b_ents) in enumerate(zip(val_examples, preds_a, preds_b)):
        if a_ents != b_ents:
            titles_changed += 1

        tokens = ex["tokens"]
        true_bio = ex["bio_tags"]
        true_ents = bio_to_entities(tokens, true_bio)

        true_counter = Counter(true_ents)
        a_counter = Counter(a_ents)
        b_counter = Counter(b_ents)

        a_matches = a_counter & true_counter
        b_matches = b_counter & true_counter

        # Matches gained in B that were not in A
        diff_gained = b_matches - a_matches
        diff_lost = a_matches - b_matches

        num_gained = sum(diff_gained.values())
        num_lost = sum(diff_lost.values())

        matches_gained += num_gained
        matches_lost += num_lost

        if num_gained > 0:
            for ent, count in diff_gained.items():
                gained_details.append({
                    "example_idx": ex_idx,
                    "category": ex["category"],
                    "entity": ent,
                    "count": count,
                })

        if num_lost > 0:
            for ent, count in diff_lost.items():
                lost_details.append({
                    "example_idx": ex_idx,
                    "category": ex["category"],
                    "entity": ent,
                    "count": count,
                })

        # Count symmetric difference between A and B predictions
        sym_diff = (a_counter - b_counter) + (b_counter - a_counter)
        entities_changed += sum(sym_diff.values())

    return {
        "titles_changed": titles_changed,
        "entities_changed": entities_changed,
        "matches_gained": matches_gained,
        "matches_lost": matches_lost,
        "gained_details": gained_details,
        "lost_details": lost_details,
    }


def run_experiment_23a() -> dict:
    """Run full Phase 23A experiment and return results dictionary."""
    print("=" * 60)
    print("PHASE 23A: CATEGORY-CONSTRAINED BIO DECODING EXPERIMENT")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Split-A Validation Set (Seed 42, 1,000 titles)
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    train_examples, val_examples = split_examples(examples, random_state=42)

    assert len(val_examples) == 1000, f"Expected 1,000 validation titles, got {len(val_examples)}"
    print(f"Loaded {len(val_examples)} Split-A validation examples (48 BIO classes).")

    # 2. Sequential Inference across 4 Champion Checkpoints
    print("\n--- EXTRACTING LOGITS SEQUENTIALLY ---")
    gbert_logits = collect_active_logits(GBERT_CHECKPOINT, val_examples, label2id, device)
    s42_logits = collect_active_logits(XLMR_SEED_42_CHECKPOINT, val_examples, label2id, device)
    s7_logits = collect_active_logits(XLMR_SEED_7_CHECKPOINT, val_examples, label2id, device)
    s21_logits = collect_active_logits(XLMR_SEED_21_CHECKPOINT, val_examples, label2id, device)

    # 3. Combine Ensemble Probabilities (0.40 GBERT + 0.20 for each XLM-R seed)
    print("\nComputing ensemble token probabilities (0.40 GBERT + 0.60 3-seed XLM-R)...")
    ensemble_probs = compute_ensemble_probabilities(
        gbert_logits, s42_logits, s7_logits, s21_logits,
        gbert_weight=0.40, xlmr_weight=0.60,
    )

    # 4. Step 1: Baseline Verification (Unconstrained Raw Exp 10)
    print("\n--- EVALUATING UNCONSTRAINED RAW EXP 10 (BASELINE REPRODUCTION) ---")
    raw_preds = decode_unconstrained_raw(val_examples, ensemble_probs, id2label)
    raw_eval = evaluate_predictions(val_examples, raw_preds)
    raw_invalid = count_invalid_predictions(val_examples, raw_preds)
    print(f"Raw Exp 10 F0.2:             {raw_eval['final_f02']:.6f} (Expected: ~{EXPECTED_EXP10_RAW_SCORE:.6f})")
    print(f"Raw Category-Invalid Aspect Rows: {raw_invalid}")

    # 5. Step 2: Method A (Argmax + Category-Aware Post-Filtering)
    print("\n--- EVALUATING METHOD A (ARGMAX + CATEGORY-AWARE POST-FILTERING) ---")
    method_a_preds = decode_method_a(val_examples, ensemble_probs, id2label)
    eval_a = evaluate_predictions(val_examples, method_a_preds)
    invalid_a = count_invalid_predictions(val_examples, method_a_preds)
    print(f"Method A Final F0.2:         {eval_a['final_f02']:.6f}")
    print(f"  Category 1 F0.2:           {eval_a['cat1_f02']:.6f}")
    print(f"  Category 2 F0.2:           {eval_a['cat2_f02']:.6f}")
    print(f"  Micro Precision:           {eval_a['micro_precision']:.6f}")
    print(f"  Micro Recall:              {eval_a['micro_recall']:.6f}")
    print(f"  Micro F0.2:                {eval_a['micro_f02']:.6f}")
    print(f"  Total Entities:            {eval_a['total_predicted']}")
    print(f"  Category-Invalid Entities: {invalid_a}")

    # 6. Step 3: Method B (Category-Constrained BIO Decoding)
    print("\n--- EVALUATING METHOD B (CATEGORY-CONSTRAINED BIO DECODING) ---")
    method_b_preds = decode_method_b(val_examples, ensemble_probs, id2label, label2id)
    eval_b = evaluate_predictions(val_examples, method_b_preds)
    invalid_b = count_invalid_predictions(val_examples, method_b_preds)
    print(f"Method B Final F0.2:         {eval_b['final_f02']:.6f}")
    print(f"  Category 1 F0.2:           {eval_b['cat1_f02']:.6f}")
    print(f"  Category 2 F0.2:           {eval_b['cat2_f02']:.6f}")
    print(f"  Micro Precision:           {eval_b['micro_precision']:.6f}")
    print(f"  Micro Recall:              {eval_b['micro_recall']:.6f}")
    print(f"  Micro F0.2:                {eval_b['micro_f02']:.6f}")
    print(f"  Total Entities:            {eval_b['total_predicted']}")
    print(f"  Category-Invalid Entities: {invalid_b}")

    # 7. Comparison & Difference Analysis
    diff_analysis = analyze_differences(val_examples, method_a_preds, method_b_preds)

    delta_f02 = eval_b["final_f02"] - eval_a["final_f02"]
    delta_cat1 = eval_b["cat1_f02"] - eval_a["cat1_f02"]
    delta_cat2 = eval_b["cat2_f02"] - eval_a["cat2_f02"]

    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY: METHOD A vs METHOD B")
    print("=" * 60)
    print(f"Metric                  Method A (Post-Filter)  Method B (Constrained)  Delta (B - A)")
    print("-" * 75)
    print(f"Final F0.2 (Official)   {eval_a['final_f02']:.6f}               {eval_b['final_f02']:.6f}               {delta_f02:+.6f}")
    print(f"  Category 1 F0.2       {eval_a['cat1_f02']:.6f}               {eval_b['cat1_f02']:.6f}               {delta_cat1:+.6f}")
    print(f"  Category 2 F0.2       {eval_a['cat2_f02']:.6f}               {eval_b['cat2_f02']:.6f}               {delta_cat2:+.6f}")
    print(f"Micro Precision         {eval_a['micro_precision']:.6f}               {eval_b['micro_precision']:.6f}               {eval_b['micro_precision'] - eval_a['micro_precision']:+.6f}")
    print(f"Micro Recall            {eval_a['micro_recall']:.6f}               {eval_b['micro_recall']:.6f}               {eval_b['micro_recall'] - eval_a['micro_recall']:+.6f}")
    print(f"Micro F0.2              {eval_a['micro_f02']:.6f}               {eval_b['micro_f02']:.6f}               {eval_b['micro_f02'] - eval_a['micro_f02']:+.6f}")
    print(f"Total Predicted         {eval_a['total_predicted']}                   {eval_b['total_predicted']}                   {eval_b['total_predicted'] - eval_a['total_predicted']:+d}")
    print(f"Invalid Aspects         {invalid_a}                       {invalid_b}                       {invalid_b - invalid_a:+d}")
    print("-" * 75)
    print(f"Titles Changed:         {diff_analysis['titles_changed']} / 1,000")
    print(f"Predictions Changed:    {diff_analysis['entities_changed']}")
    print(f"Exact Matches Gained:   {diff_analysis['matches_gained']}")
    print(f"Exact Matches Lost:     {diff_analysis['matches_lost']}")

    # Per-Aspect Comparison Breakdown
    print("\n--- PER-ASPECT COMPARISON (LARGEST DIFFERENCES) ---")
    aspect_diffs = []
    # Collect all aspects across both categories
    all_aspects = set()
    for cat in [1, 2]:
        all_aspects |= set(eval_a["comp_details"]["categories"].get(cat, {}).get("aspects", {}).keys())
        all_aspects |= set(eval_b["comp_details"]["categories"].get(cat, {}).get("aspects", {}).keys())

    for aspect in sorted(all_aspects):
        for cat in [1, 2]:
            asp_a = eval_a["comp_details"]["categories"].get(cat, {}).get("aspects", {}).get(aspect)
            asp_b = eval_b["comp_details"]["categories"].get(cat, {}).get("aspects", {}).get(aspect)
            if asp_a is None and asp_b is None:
                continue
            f02_a = asp_a["f_beta"] if asp_a else 0.0
            f02_b = asp_b["f_beta"] if asp_b else 0.0
            tp_a = asp_a["tp"] if asp_a else 0
            tp_b = asp_b["tp"] if asp_b else 0
            fp_a = asp_a["fp"] if asp_a else 0
            fp_b = asp_b["fp"] if asp_b else 0
            fn_a = asp_a["fn"] if asp_a else 0
            fn_b = asp_b["fn"] if asp_b else 0
            w = asp_a["weight"] if asp_a else (asp_b["weight"] if asp_b else 0.0)

            diff = f02_b - f02_a
            if abs(diff) > 1e-6 or (tp_a != tp_b) or (fp_a != fp_b):
                aspect_diffs.append({
                    "category": cat,
                    "aspect": aspect,
                    "weight": w,
                    "f02_a": f02_a,
                    "f02_b": f02_b,
                    "delta": diff,
                    "tp_delta": tp_b - tp_a,
                    "fp_delta": fp_b - fp_a,
                    "fn_delta": fn_b - fn_a,
                })

    # Sort by absolute delta
    aspect_diffs.sort(key=lambda x: abs(x["delta"]), reverse=True)
    if aspect_diffs:
        print(f"{'Cat':<4} {'Aspect':<32} {'Weight':<8} {'F0.2 (A)':<10} {'F0.2 (B)':<10} {'Delta':<10} {'TP d':<6} {'FP d':<6}")
        print("-" * 88)
        for ad in aspect_diffs[:15]:
            print(f"{ad['category']:<4} {ad['aspect']:<32} {ad['weight']:.4f}   {ad['f02_a']:.6f}   {ad['f02_b']:.6f}   {ad['delta']:+.6f}   {ad['tp_delta']:+3d}    {ad['fp_delta']:+3d}")
    else:
        print("No aspect-level F0.2 differences observed between Method A and Method B.")

    return {
        "raw_eval": raw_eval,
        "eval_a": eval_a,
        "eval_b": eval_b,
        "diff_analysis": diff_analysis,
        "aspect_diffs": aspect_diffs,
    }


if __name__ == "__main__":
    run_experiment_23a()
