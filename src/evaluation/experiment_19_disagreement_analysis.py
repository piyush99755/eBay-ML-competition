"""
Experiment 19: BIO vs Span Disagreement Analysis.

Objective:
Investigate whether Experiment 16's unweighted span-based XLM-R model corrects
enough mistakes made by Experiment 10's four-model BIO ensemble to justify a future
selective-fusion experiment.

Strictly read-only diagnostic analysis:
No model training, fine-tuning, checkpoint modification, or rule optimization.
Sequential model loading and GPU memory freeing (6 GB VRAM safe).
Lightweight prediction caching in scratch/ for fast re-runs.

Privacy constraint:
Prints aggregate statistics only. No raw eBay titles, tokens, record IDs, or dataset content.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.modeling.xlmr_span import (
    XLMRobertaSpan,
    bio_to_spans,
    spans_to_bio,
)
from src.evaluation.entity_metrics import (
    BETA,
    bio_to_entities,
    count_matches,
)
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.experiment_18_bio_span_fusion import (
    GBERT_CHECKPOINT,
    XLMR_SEED_42_CHECKPOINT,
    XLMR_SEED_7_CHECKPOINT,
    XLMR_SEED_21_CHECKPOINT,
    XLMR_SPAN_CHECKPOINT,
    EXPECTED_EXP10_SCORE,
    EXPECTED_EXP16_SCORE,
    collect_active_logits,
    ensemble_gbert_multiseed_xlmr_predictions,
    collect_span_predictions,
    calculate_score,
)

# Cache path in scratch directory
SCRATCH_DIR = Path(r"C:\Users\piyus\.gemini\antigravity\brain\e1fbd9bc-74ac-4a17-8f5b-860cdf61f211\scratch")
PREDICTIONS_CACHE_FILE = SCRATCH_DIR / "exp18_predictions_cache.json"

EXPECTED_EXP10_ENTITIES = 7444
EXPECTED_EXP16_ENTITIES = 7105
EXPECTED_EXACT_SHARED = 6709


def load_or_generate_predictions(
    val_examples: List[dict],
    label2id: dict,
    id2label: dict,
    device: torch.device,
) -> Tuple[List[List[str]], List[List[str]], List[List[dict]]]:
    """
    Load precomputed predictions from lightweight cache if available,
    otherwise run sequential GPU inference and cache predictions.
    """
    if PREDICTIONS_CACHE_FILE.exists():
        print(f"Loading cached predictions from {PREDICTIONS_CACHE_FILE}...")
        try:
            with open(PREDICTIONS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            exp10_predictions = data["exp10_predictions"]
            exp16_predictions = data["exp16_predictions"]
            exp16_retained_spans = data["exp16_retained_spans"]
            if len(exp10_predictions) == len(val_examples) and len(exp16_predictions) == len(val_examples):
                print(f"Successfully loaded {len(val_examples)} validation predictions from cache.")
                return exp10_predictions, exp16_predictions, exp16_retained_spans
        except Exception as e:
            print(f"Cache read error: {e}. Re-running sequential inference...")

    print("Running sequential inference across models (VRAM safe)...")

    # 1. GBERT
    print("  [1/4] Running GBERT Baseline...")
    gbert_tokenizer = AutoTokenizer.from_pretrained(GBERT_CHECKPOINT)
    gbert_model = AutoModelForTokenClassification.from_pretrained(GBERT_CHECKPOINT)
    gbert_model.to(device)
    gbert_dataset, _ = prepare_split(val_examples, gbert_tokenizer, label2id)
    gbert_logits = collect_active_logits(gbert_model, gbert_dataset, gbert_tokenizer, device)
    del gbert_model, gbert_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. XLM-R seed 42
    print("  [2/4] Running XLM-R Seed 42...")
    tok_42 = AutoTokenizer.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    model_42 = AutoModelForTokenClassification.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    model_42.to(device)
    ds_42, _ = prepare_split(val_examples, tok_42, label2id)
    logits_42 = collect_active_logits(model_42, ds_42, tok_42, device)
    del model_42, ds_42
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. XLM-R seed 7
    print("  [3/4] Running XLM-R Seed 7...")
    tok_7 = AutoTokenizer.from_pretrained(XLMR_SEED_7_CHECKPOINT)
    model_7 = AutoModelForTokenClassification.from_pretrained(XLMR_SEED_7_CHECKPOINT)
    model_7.to(device)
    ds_7, _ = prepare_split(val_examples, tok_7, label2id)
    logits_7 = collect_active_logits(model_7, ds_7, tok_7, device)
    del model_7, ds_7
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. XLM-R seed 21
    print("  [4/4] Running XLM-R Seed 21...")
    tok_21 = AutoTokenizer.from_pretrained(XLMR_SEED_21_CHECKPOINT)
    model_21 = AutoModelForTokenClassification.from_pretrained(XLMR_SEED_21_CHECKPOINT)
    model_21.to(device)
    ds_21, _ = prepare_split(val_examples, tok_21, label2id)
    logits_21 = collect_active_logits(model_21, ds_21, tok_21, device)
    del model_21, ds_21
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Exp 10 ensemble
    exp10_predictions = ensemble_gbert_multiseed_xlmr_predictions(
        gbert_logits=gbert_logits,
        seed_42_logits=logits_42,
        seed_7_logits=logits_7,
        seed_21_logits=logits_21,
        id2label=id2label,
        gbert_weight=0.40,
        xlmr_weight=0.60,
    )
    del gbert_logits, logits_42, logits_7, logits_21
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Span model (Exp 16)
    print("  [5/5] Running XLM-R Span Model (Exp 16)...")
    span_tokenizer = AutoTokenizer.from_pretrained(XLMR_SPAN_CHECKPOINT)
    span_model = XLMRobertaSpan.from_pretrained(XLMR_SPAN_CHECKPOINT)
    span_model.to(device)
    span_dataset, _ = prepare_split(val_examples, span_tokenizer, label2id)
    exp16_predictions, exp16_retained_spans, _ = collect_span_predictions(
        span_model, span_dataset, span_tokenizer, device
    )
    del span_model, span_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save to scratch cache
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp10_predictions": exp10_predictions,
                "exp16_predictions": exp16_predictions,
                "exp16_retained_spans": exp16_retained_spans,
            },
            f,
        )
    print(f"Cached predictions to {PREDICTIONS_CACHE_FILE}.")

    return exp10_predictions, exp16_predictions, exp16_retained_spans


def build_connected_components_for_example(
    example_id: int,
    exp10_spans: List[Tuple[int, int, str]],
    exp16_spans: List[Tuple[int, int, str]],
    gold_spans: List[Tuple[int, int, str]],
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Build bipartite connected overlap components for a single validation title.

    Returns:
    - exact_agreements: list of component dicts with exact shared predictions (1-to-1 matching)
    - overlapping_disagreements: list of component dicts with overlapping prediction disagreements
    - exp10_disjoint: list of component dicts predicted ONLY by Exp 10 (disjoint from Exp 16)
    - exp16_disjoint: list of component dicts predicted ONLY by Exp 16 (disjoint from Exp 10)
    """
    n10 = len(exp10_spans)
    n16 = len(exp16_spans)

    # Adjacency list: nodes are ("exp10", i) and ("exp16", j)
    # Token overlap condition: max(s10, s16) <= min(e10, e16)
    adj_10 = defaultdict(list)
    adj_16 = defaultdict(list)

    for i, (s10, e10, a10) in enumerate(exp10_spans):
        for j, (s16, e16, a16) in enumerate(exp16_spans):
            if max(s10, s16) <= min(e10, e16):
                adj_10[i].append(j)
                adj_16[j].append(i)

    visited_10 = [False] * n10
    visited_16 = [False] * n16

    exact_agreements = []
    overlapping_disagreements = []
    exp10_disjoint = []
    exp16_disjoint = []

    # 1. Connected components starting from Exp 10 nodes
    for i in range(n10):
        if visited_10[i]:
            continue

        comp_10 = []
        comp_16 = []

        queue_10 = [i]
        visited_10[i] = True

        while queue_10:
            curr_10 = queue_10.pop(0)
            comp_10.append(curr_10)

            for nxt_16 in adj_10[curr_10]:
                if not visited_16[nxt_16]:
                    visited_16[nxt_16] = True
                    comp_16.append(nxt_16)
                    for nxt_10 in adj_16[nxt_16]:
                        if not visited_10[nxt_10]:
                            visited_10[nxt_10] = True
                            queue_10.append(nxt_10)

        # Build group details
        p10_entities = [exp10_spans[idx] for idx in sorted(comp_10)]
        p16_entities = [exp16_spans[idx] for idx in sorted(comp_16)]

        # Find associated gold entities overlapping any prediction in this component
        associated_gold = []
        for g_idx, (sg, eg, ag) in enumerate(gold_spans):
            overlaps = any(max(sg, s) <= min(eg, e) for s, e, _ in (p10_entities + p16_entities))
            if overlaps:
                associated_gold.append((sg, eg, ag))

        group_data = {
            "example_id": example_id,
            "p10_entities": p10_entities,
            "p16_entities": p16_entities,
            "gold_entities": associated_gold,
        }

        if len(p10_entities) > 0 and len(p16_entities) == 0:
            exp10_disjoint.append(group_data)
        elif len(p10_entities) == 1 and len(p16_entities) == 1 and p10_entities[0] == p16_entities[0]:
            exact_agreements.append(group_data)
        else:
            overlapping_disagreements.append(group_data)

    # 2. Remaining isolated Exp 16 nodes (disjoint from Exp 10)
    for j in range(n16):
        if not visited_16[j]:
            visited_16[j] = True
            p16_entities = [exp16_spans[j]]
            associated_gold = []
            for sg, eg, ag in gold_spans:
                s16, e16, _ = exp16_spans[j]
                if max(sg, s16) <= min(eg, e16):
                    associated_gold.append((sg, eg, ag))

            exp16_disjoint.append({
                "example_id": example_id,
                "p10_entities": [],
                "p16_entities": p16_entities,
                "gold_entities": associated_gold,
            })

    return exact_agreements, overlapping_disagreements, exp10_disjoint, exp16_disjoint


def classify_disagreement_group(group: dict) -> dict:
    """
    Classify a single overlapping disagreement group into one of the 5 mutually exclusive categories.

    Categories:
    1. Exp 10 fully correct, Exp 16 incorrect.
    2. Exp 16 fully correct, Exp 10 incorrect.
    3. Both fully correct (if applicable).
    4. Both incorrect (tied performance, neither has advantage).
    5. Mixed or partial cases where neither model is entirely correct:
       - 5a: Exp 10 partial advantage
       - 5b: Exp 16 partial advantage
       - 5c: Tied partial
    """
    p10 = group["p10_entities"]
    p16 = group["p16_entities"]
    gold = group["gold_entities"]

    gold_set = set(gold)

    # Exact entity match counts
    tp_10 = sum(1 for e in p10 if e in gold_set)
    fp_10 = len(p10) - tp_10
    fn_10 = len(gold) - tp_10

    tp_16 = sum(1 for e in p16 if e in gold_set)
    fp_16 = len(p16) - tp_16
    fn_16 = len(gold) - tp_16

    p10_fully_correct = (fp_10 == 0 and fn_10 == 0)
    p16_fully_correct = (fp_16 == 0 and fn_16 == 0)

    # Determine structural taxonomy
    # 1-to-1 same boundary diff aspect, diff boundary same aspect, diff boundary diff aspect, or N-to-M
    if len(p10) == 1 and len(p16) == 1:
        s10, e10, a10 = p10[0]
        s16, e16, a16 = p16[0]
        same_b = (s10 == s16 and e10 == e16)
        same_a = (a10 == a16)
        if same_b and not same_a:
            structure = "same_boundary_diff_aspect"
        elif not same_b and same_a:
            structure = "diff_boundary_same_aspect"
        else:
            structure = "diff_boundary_diff_aspect"
    else:
        structure = "complex_n_to_m"

    # Determine primary category
    if p10_fully_correct and not p16_fully_correct:
        category = "1_exp10_fully_correct"
        advantage = "exp10_clean_win"
    elif p16_fully_correct and not p10_fully_correct:
        category = "2_exp16_fully_correct"
        advantage = "exp16_clean_win"
    elif p10_fully_correct and p16_fully_correct:
        category = "3_both_fully_correct"
        advantage = "tied_both_correct"
    else:
        # Neither model is fully correct
        # Compare TP and FP
        if tp_10 > tp_16 or (tp_10 == tp_16 and fp_10 < fp_16):
            category = "5_mixed_partial"
            advantage = "exp10_partial_advantage"
        elif tp_16 > tp_10 or (tp_16 == tp_10 and fp_16 < fp_10):
            category = "5_mixed_partial"
            advantage = "exp16_partial_advantage"
        else:
            # Identical TP and identical FP
            if tp_10 == 0:
                category = "4_both_incorrect"
                advantage = "both_incorrect_tied"
            else:
                category = "5_mixed_partial"
                advantage = "tied_partial"

    # Counterfactual delta: effect of replacing Exp 10 with Exp 16 on this group
    delta_tp = tp_16 - tp_10
    delta_fp = fp_16 - fp_10
    delta_fn = fn_16 - fn_10

    # Oracle choice: select Exp 16 if it has a strict advantage, otherwise keep Exp 10
    # Advantage means: (tp_16 > tp_10) or (tp_16 == tp_10 and fp_16 < fp_10)
    oracle_chooses_exp16 = (advantage in ["exp16_clean_win", "exp16_partial_advantage"])

    return {
        "example_id": group["example_id"],
        "category": category,
        "advantage": advantage,
        "structure": structure,
        "p10_count": len(p10),
        "p16_count": len(p16),
        "gold_count": len(gold),
        "tp_10": tp_10,
        "fp_10": fp_10,
        "fn_10": fn_10,
        "tp_16": tp_16,
        "fp_16": fp_16,
        "fn_16": fn_16,
        "delta_tp": delta_tp,
        "delta_fp": delta_fp,
        "delta_fn": delta_fn,
        "oracle_chooses_exp16": oracle_chooses_exp16,
        "involves_vehicle_model": any(a == "Kompatibles_Fahrzeug_Modell" for _, _, a in (p10 + p16 + gold)),
        "gold_widths": [e - s + 1 for s, e, _ in gold],
    }


def run_experiment_19() -> dict:
    print("==================================================")
    print("EXPERIMENT 19: BIO VS SPAN DISAGREEMENT ANALYSIS")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    _, val_examples = split_examples(examples)
    print(f"Validation titles: {len(val_examples)}")

    # Load / generate predictions
    exp10_bio, exp16_bio, exp16_retained = load_or_generate_predictions(
        val_examples=val_examples,
        label2id=label2id,
        id2label=id2label,
        device=device,
    )

    # =========================================================================
    # STEP 1: BASELINE CONFIRMATION
    # =========================================================================
    print("\n==================================================")
    print("STEP 1: BASELINE CONFIRMATION")
    print("==================================================")
    score_10, _, _ = calculate_score(val_examples, exp10_bio)
    score_16, _, _ = calculate_score(val_examples, exp16_bio)

    all_p10 = []
    all_p16 = []
    all_gold = []

    for i, ex in enumerate(val_examples):
        all_p10.append(bio_to_spans(exp10_bio[i]))
        all_p16.append(bio_to_spans(exp16_bio[i]))
        all_gold.append(bio_to_spans(ex["bio_tags"]))

    tot_p10 = sum(len(sp) for sp in all_p10)
    tot_p16 = sum(len(sp) for sp in all_p16)
    tot_gold = sum(len(sp) for sp in all_gold)

    exact_shared_count = 0
    for i in range(len(val_examples)):
        s10 = set(all_p10[i])
        s16 = set(all_p16[i])
        exact_shared_count += len(s10 & s16)

    print(f"Experiment 10 F0.2:             {score_10:.6f} (Target: ~{EXPECTED_EXP10_SCORE:.6f})")
    print(f"Experiment 16 F0.2:             {score_16:.6f} (Target: ~{EXPECTED_EXP16_SCORE:.6f})")
    print(f"Experiment 10 Total Entities:   {tot_p10} (Target: {EXPECTED_EXP10_ENTITIES})")
    print(f"Experiment 16 Total Entities:   {tot_p16} (Target: {EXPECTED_EXP16_ENTITIES})")
    print(f"Exact Shared Predictions:       {exact_shared_count} (Target: {EXPECTED_EXACT_SHARED})")

    if abs(score_10 - EXPECTED_EXP10_SCORE) > 0.0005:
        raise RuntimeError(f"Exp 10 score mismatch: got {score_10:.6f}, expected {EXPECTED_EXP10_SCORE:.6f}")
    if abs(score_16 - EXPECTED_EXP16_SCORE) > 0.0005:
        raise RuntimeError(f"Exp 16 score mismatch: got {score_16:.6f}, expected {EXPECTED_EXP16_SCORE:.6f}")
    if tot_p10 != EXPECTED_EXP10_ENTITIES:
        raise RuntimeError(f"Exp 10 entity count mismatch: got {tot_p10}, expected {EXPECTED_EXP10_ENTITIES}")
    if tot_p16 != EXPECTED_EXP16_ENTITIES:
        raise RuntimeError(f"Exp 16 entity count mismatch: got {tot_p16}, expected {EXPECTED_EXP16_ENTITIES}")
    if exact_shared_count != EXPECTED_EXACT_SHARED:
        raise RuntimeError(f"Exact shared predictions mismatch: got {exact_shared_count}, expected {EXPECTED_EXACT_SHARED}")

    print("PASS: Baseline reproduction and exact entity counts confirmed!")

    # =========================================================================
    # STEP 2: BUILD OVERLAPPING DISAGREEMENT GROUPS
    # =========================================================================
    print("\n==================================================")
    print("STEP 2: OVERLAPPING DISAGREEMENT GROUP CONSTRUCTION")
    print("==================================================")

    all_exact_agreements = []
    all_overlapping_disagreements = []
    all_exp10_disjoint = []
    all_exp16_disjoint = []

    for i in range(len(val_examples)):
        exact_ag, overlap_dis, exp10_dis, exp16_dis = build_connected_components_for_example(
            example_id=i,
            exp10_spans=all_p10[i],
            exp16_spans=all_p16[i],
            gold_spans=all_gold[i],
        )
        all_exact_agreements.extend(exact_ag)
        all_overlapping_disagreements.extend(overlap_dis)
        all_exp10_disjoint.extend(exp10_dis)
        all_exp16_disjoint.extend(exp16_dis)

    print(f"Total Exact Shared Components:         {len(all_exact_agreements)}")
    print(f"Total Overlapping Disagreement Groups: {len(all_overlapping_disagreements)}")
    print(f"Total Exp 10 Disjoint Groups:          {len(all_exp10_disjoint)}")
    print(f"Total Exp 16 Disjoint Groups:          {len(all_exp16_disjoint)}")

    # Verify duplicate-safe entity counting
    sum_p10 = (
        sum(len(g["p10_entities"]) for g in all_exact_agreements)
        + sum(len(g["p10_entities"]) for g in all_overlapping_disagreements)
        + sum(len(g["p10_entities"]) for g in all_exp10_disjoint)
    )
    sum_p16 = (
        sum(len(g["p16_entities"]) for g in all_exact_agreements)
        + sum(len(g["p16_entities"]) for g in all_overlapping_disagreements)
        + sum(len(g["p16_entities"]) for g in all_exp16_disjoint)
    )
    print(f"\nDuplicate-Safe Invariant Check:")
    print(f"  Exp 10 Entities: {sum_p10} (Expected {EXPECTED_EXP10_ENTITIES}) -> {'MATCH' if sum_p10 == EXPECTED_EXP10_ENTITIES else 'FAIL'}")
    print(f"  Exp 16 Entities: {sum_p16} (Expected {EXPECTED_EXP16_ENTITIES}) -> {'MATCH' if sum_p16 == EXPECTED_EXP16_ENTITIES else 'FAIL'}")

    # =========================================================================
    # STEP 3: QUANTIFY DISAGREEMENT PATTERNS & CLASSIFICATION
    # =========================================================================
    print("\n==================================================")
    print("STEP 3: QUANTIFY DISAGREEMENT PATTERNS")
    print("==================================================")

    classified_disagreements = [
        classify_disagreement_group(g) for g in all_overlapping_disagreements
    ]

    cat_counts = Counter(cd["category"] for cd in classified_disagreements)
    adv_counts = Counter(cd["advantage"] for cd in classified_disagreements)
    struct_counts = Counter(cd["structure"] for cd in classified_disagreements)

    tot_groups = len(classified_disagreements)
    p10_in_groups = sum(cd["p10_count"] for cd in classified_disagreements)
    p16_in_groups = sum(cd["p16_count"] for cd in classified_disagreements)
    gold_in_groups = sum(cd["gold_count"] for cd in classified_disagreements)

    print(f"Total Overlapping Disagreement Groups: {tot_groups}")
    print(f"Total Predictions Involved:")
    print(f"  From Experiment 10:                 {p10_in_groups}")
    print(f"  From Experiment 16:                 {p16_in_groups}")
    print(f"  Associated Gold Entities:           {gold_in_groups}")

    print(f"\nGroup-Level Classification Outcomes:")
    print(f"  1. Exp 10 Fully Correct (Exp 10 clean win): {cat_counts['1_exp10_fully_correct']:4d} ({(cat_counts['1_exp10_fully_correct'] / tot_groups) * 100:.2f}%)")
    print(f"  2. Exp 16 Fully Correct (Exp 16 clean win): {cat_counts['2_exp16_fully_correct']:4d} ({(cat_counts['2_exp16_fully_correct'] / tot_groups) * 100:.2f}%)")
    print(f"  3. Both Fully Correct:                      {cat_counts['3_both_fully_correct']:4d} ({(cat_counts['3_both_fully_correct'] / tot_groups) * 100:.2f}%)")
    print(f"  4. Both Incorrect (tied):                   {cat_counts['4_both_incorrect']:4d} ({(cat_counts['4_both_incorrect'] / tot_groups) * 100:.2f}%)")
    print(f"  5. Mixed / Partial Outcomes:                {cat_counts['5_mixed_partial']:4d} ({(cat_counts['5_mixed_partial'] / tot_groups) * 100:.2f}%)")
    print(f"     - Exp 10 partial advantage:              {adv_counts['exp10_partial_advantage']:4d}")
    print(f"     - Exp 16 partial advantage:              {adv_counts['exp16_partial_advantage']:4d}")
    print(f"     - Tied partial advantage:                {adv_counts['tied_partial']:4d}")

    total_exp10_wins = cat_counts['1_exp10_fully_correct'] + adv_counts['exp10_partial_advantage']
    total_exp16_wins = cat_counts['2_exp16_fully_correct'] + adv_counts['exp16_partial_advantage']
    total_ties = cat_counts['3_both_fully_correct'] + cat_counts['4_both_incorrect'] + adv_counts['tied_partial']

    print(f"\nOverall Head-to-Head Win/Loss Summary:")
    print(f"  Total Experiment 10 Wins:                   {total_exp10_wins:4d} ({(total_exp10_wins / tot_groups) * 100:.2f}%)")
    print(f"  Total Experiment 16 Wins:                   {total_exp16_wins:4d} ({(total_exp16_wins / tot_groups) * 100:.2f}%)")
    print(f"  Total Ties / Neutral:                       {total_ties:4d} ({(total_ties / tot_groups) * 100:.2f}%)")

    print(f"\nDisagreement Structure Taxonomy:")
    print(f"  Same boundary, different aspect:            {struct_counts['same_boundary_diff_aspect']:4d}")
    print(f"  Different boundary, same aspect:            {struct_counts['diff_boundary_same_aspect']:4d}")
    print(f"  Different boundary, different aspect:       {struct_counts['diff_boundary_diff_aspect']:4d}")
    print(f"  Complex N-to-M overlap groups:              {struct_counts['complex_n_to_m']:4d}")

    # Counterfactual replacement analysis
    cf_delta_tp = sum(cd["delta_tp"] for cd in classified_disagreements)
    cf_delta_fp = sum(cd["delta_fp"] for cd in classified_disagreements)
    cf_delta_fn = sum(cd["delta_fn"] for cd in classified_disagreements)

    print(f"\nCounterfactual Replacement Diagnostic:")
    print(f"(Effect of replacing Exp 10 with Exp 16 across ALL {tot_groups} disagreement groups):")
    print(f"  Net Change in TP:                           {cf_delta_tp:+d}")
    print(f"  Net Change in FP:                           {cf_delta_fp:+d}")
    print(f"  Net Change in FN:                           {cf_delta_fn:+d}")

    # Build counterfactual predictions
    cf_bio_predictions = []
    for i in range(len(val_examples)):
        cf_spans = []
        # Add exact shared
        exact_in_ex = [g["p10_entities"][0] for g in all_exact_agreements if g["example_id"] == i]
        cf_spans.extend(exact_in_ex)
        # Add Exp 10 disjoint
        exp10_dis_in_ex = [ent for g in all_exp10_disjoint if g["example_id"] == i for ent in g["p10_entities"]]
        cf_spans.extend(exp10_dis_in_ex)
        # For disagreement groups: use Exp 16 entities!
        dis_in_ex = [ent for g in all_overlapping_disagreements if g["example_id"] == i for ent in g["p16_entities"]]
        cf_spans.extend(dis_in_ex)

        cf_spans.sort(key=lambda x: x[0])
        cf_bio = spans_to_bio(cf_spans, len(val_examples[i]["tokens"]))
        cf_bio_predictions.append(cf_bio)

    cf_score, _, _ = calculate_score(val_examples, cf_bio_predictions)
    print(f"  Counterfactual F0.2 Score:                  {cf_score:.6f} (Delta vs Exp 10: {cf_score - score_10:+.6f})")

    # =========================================================================
    # STEP 4: PRECISION OPPORTUNITIES & ORACLE UPPER BOUND
    # =========================================================================
    print("\n==================================================")
    print("STEP 4: PRECISION OPPORTUNITIES & ORACLE UPPER BOUND")
    print("==================================================")

    # 4.1 Breakdown by Gold Span Width
    width_group_wins = {
        "width1": {"exp10_wins": 0, "exp16_wins": 0, "ties": 0, "total": 0},
        "width2": {"exp10_wins": 0, "exp16_wins": 0, "ties": 0, "total": 0},
        "width3+": {"exp10_wins": 0, "exp16_wins": 0, "ties": 0, "total": 0},
    }

    for cd in classified_disagreements:
        widths = cd["gold_widths"]
        if not widths:
            continue
        max_w = max(widths)
        w_key = "width1" if max_w == 1 else ("width2" if max_w == 2 else "width3+")
        width_group_wins[w_key]["total"] += 1
        if cd["advantage"] in ["exp10_clean_win", "exp10_partial_advantage"]:
            width_group_wins[w_key]["exp10_wins"] += 1
        elif cd["advantage"] in ["exp16_clean_win", "exp16_partial_advantage"]:
            width_group_wins[w_key]["exp16_wins"] += 1
        else:
            width_group_wins[w_key]["ties"] += 1

    print("Disagreement Outcomes by Gold Entity Span Width:")
    for w_key in ["width1", "width2", "width3+"]:
        data = width_group_wins[w_key]
        tot_w = max(1, data["total"])
        print(f"  {w_key.upper():8s} (Groups: {data['total']:3d}): Exp 10 Wins = {data['exp10_wins']:3d} ({(data['exp10_wins']/tot_w)*100:.1f}%) | Exp 16 Wins = {data['exp16_wins']:3d} ({(data['exp16_wins']/tot_w)*100:.1f}%) | Ties = {data['ties']:3d}")

    # 4.2 Breakdown for VehicleModel (Kompatibles_Fahrzeug_Modell)
    vm_disagreements = [cd for cd in classified_disagreements if cd["involves_vehicle_model"]]
    tot_vm = len(vm_disagreements)
    vm_exp10_wins = sum(1 for cd in vm_disagreements if cd["advantage"] in ["exp10_clean_win", "exp10_partial_advantage"])
    vm_exp16_wins = sum(1 for cd in vm_disagreements if cd["advantage"] in ["exp16_clean_win", "exp16_partial_advantage"])
    vm_ties = tot_vm - vm_exp10_wins - vm_exp16_wins

    print(f"\nDisagreements Involving Kompatibles_Fahrzeug_Modell:")
    print(f"  Total Groups:                               {tot_vm}")
    print(f"  Experiment 10 Wins:                         {vm_exp10_wins} ({(vm_exp10_wins / max(1, tot_vm)) * 100:.1f}%)")
    print(f"  Experiment 16 Wins:                         {vm_exp16_wins} ({(vm_exp16_wins / max(1, tot_vm)) * 100:.1f}%)")
    print(f"  Ties / Neutral:                             {vm_ties} ({(vm_ties / max(1, tot_vm)) * 100:.1f}%)")

    # 4.3 Structure vs Wins Breakdown
    print(f"\nDisagreement Structure vs Model Advantage:")
    for struct_name in ["same_boundary_diff_aspect", "diff_boundary_same_aspect", "diff_boundary_diff_aspect", "complex_n_to_m"]:
        struct_groups = [cd for cd in classified_disagreements if cd["structure"] == struct_name]
        s_tot = max(1, len(struct_groups))
        s_10 = sum(1 for cd in struct_groups if cd["advantage"] in ["exp10_clean_win", "exp10_partial_advantage"])
        s_16 = sum(1 for cd in struct_groups if cd["advantage"] in ["exp16_clean_win", "exp16_partial_advantage"])
        s_tie = len(struct_groups) - s_10 - s_16
        print(f"  {struct_name:<32s} (Groups: {len(struct_groups):3d}): Exp 10 = {s_10:3d} ({(s_10/s_tot)*100:.1f}%) | Exp 16 = {s_16:3d} ({(s_16/s_tot)*100:.1f}%) | Ties = {s_tie:3d}")

    # 4.4 Oracle Upper Bound Evaluation
    # For each disagreement group: if Exp 16 has an advantage, use Exp 16 entities; otherwise keep Exp 10 entities.
    oracle_bio_predictions = []
    oracle_groups_switched = 0
    oracle_delta_tp = 0
    oracle_delta_fp = 0
    oracle_delta_fn = 0

    for i in range(len(val_examples)):
        oracle_spans = []
        # Add exact shared
        exact_in_ex = [g["p10_entities"][0] for g in all_exact_agreements if g["example_id"] == i]
        oracle_spans.extend(exact_in_ex)
        # Add Exp 10 disjoint
        exp10_dis_in_ex = [ent for g in all_exp10_disjoint if g["example_id"] == i for ent in g["p10_entities"]]
        oracle_spans.extend(exp10_dis_in_ex)

        # For disagreement groups: check oracle choice
        dis_groups_in_ex = [
            (g, cd) for g, cd in zip(all_overlapping_disagreements, classified_disagreements)
            if g["example_id"] == i
        ]

        for g, cd in dis_groups_in_ex:
            if cd["oracle_chooses_exp16"]:
                oracle_spans.extend(g["p16_entities"])
                oracle_groups_switched += 1
                oracle_delta_tp += cd["delta_tp"]
                oracle_delta_fp += cd["delta_fp"]
                oracle_delta_fn += cd["delta_fn"]
            else:
                oracle_spans.extend(g["p10_entities"])

        oracle_spans.sort(key=lambda x: x[0])
        oracle_bio = spans_to_bio(oracle_spans, len(val_examples[i]["tokens"]))
        oracle_bio_predictions.append(oracle_bio)

    oracle_score, oracle_metrics, _ = calculate_score(val_examples, oracle_bio_predictions)
    oracle_improvement = oracle_score - score_10

    print(f"\n==================================================")
    print("ORACLE SELECTOR UPPER BOUND (DIAGNOSTIC ONLY)")
    print("==================================================")
    print(f"Groups where Oracle selected Exp 16:          {oracle_groups_switched} / {tot_groups} ({(oracle_groups_switched / tot_groups) * 100:.2f}%)")
    print(f"Oracle Net Change in TP:                      {oracle_delta_tp:+d}")
    print(f"Oracle Net Change in FP:                      {oracle_delta_fp:+d}")
    print(f"Oracle Net Change in FN:                      {oracle_delta_fn:+d}")
    print(f"Experiment 10 Baseline F0.2:                  {score_10:.6f}")
    print(f"Oracle Upper Bound F0.2:                      {oracle_score:.6f}")
    print(f"Maximum Theoretical Improvement (Oracle Delta): {oracle_improvement:+.6f}")
    print(
        "\nIMPORTANT NOTICE: The Oracle Upper Bound uses ground truth labels to select the superior "
        "prediction per group. It represents the theoretical upper bound of selective fusion, NOT a deployable model."
    )

    # 4.5 Recommendation Synthesis
    print(f"\n==================================================")
    print("SYNTHESIS & RECOMMENDATION")
    print("==================================================")
    if oracle_improvement >= 0.003 and total_exp16_wins >= (0.35 * tot_groups):
        recommendation = "PURSUE SELECTIVE FUSION (Substantial oracle potential and clean win density)"
    elif oracle_improvement >= 0.001 and total_exp16_wins >= (0.25 * tot_groups):
        recommendation = "CONDITIONAL / MODEST POTENTIAL (Selective fusion has upside, but risk of FP bleed)"
    else:
        recommendation = "STOP SELECTIVE FUSION (Exp 10 dominates disagreements; oracle headroom is narrow or Exp 16 win density is too low)"

    print(f"Recommendation: {recommendation}")
    print(f"Exp 10 win rate: {(total_exp10_wins / tot_groups) * 100:.1f}% vs Exp 16 win rate: {(total_exp16_wins / tot_groups) * 100:.1f}%")

    return {
        "score_exp10": score_10,
        "score_exp16": score_16,
        "exact_shared_count": exact_shared_count,
        "tot_groups": tot_groups,
        "p10_in_groups": p10_in_groups,
        "p16_in_groups": p16_in_groups,
        "gold_in_groups": gold_in_groups,
        "cat_counts": dict(cat_counts),
        "adv_counts": dict(adv_counts),
        "struct_counts": dict(struct_counts),
        "total_exp10_wins": total_exp10_wins,
        "total_exp16_wins": total_exp16_wins,
        "total_ties": total_ties,
        "cf_score": cf_score,
        "cf_delta_tp": cf_delta_tp,
        "cf_delta_fp": cf_delta_fp,
        "cf_delta_fn": cf_delta_fn,
        "oracle_score": oracle_score,
        "oracle_improvement": oracle_improvement,
        "oracle_groups_switched": oracle_groups_switched,
        "recommendation": recommendation,
        "width_group_wins": width_group_wins,
        "vm_disagreements": {
            "total": tot_vm,
            "exp10_wins": vm_exp10_wins,
            "exp16_wins": vm_exp16_wins,
            "ties": vm_ties,
        },
    }


def main() -> None:
    run_experiment_19()


if __name__ == "__main__":
    main()
