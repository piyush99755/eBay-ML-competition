"""
Phase 27A: Entity-Level Agreement Research on Frozen Exp 10 Champion.

Investigates whether exact entity-level agreement among the four frozen models
can improve precision-weighted F0.2 beyond MIN 0.60.

Four Frozen Checkpoints:
1. GBERT (0.40)
2. XLM-R Seed 42 (0.20)
3. XLM-R Seed 7 (0.20)
4. XLM-R Seed 21 (0.20)

Predefined Voting Rules:
- Baseline: MIN 0.60 champion (F0.2 = 0.899802)
- Rule A: Retain entities supported by at least 2 of 4 models (>= 2/4)
- Rule B: Retain entities supported by at least 3 of 4 models (>= 3/4)
- Rule C: Retain entities supported by all 4 models (4/4)
- Rule D: Retain entities supported by GBERT and at least one XLM-R model (GBERT + >= 1 XLM-R)
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.entity_metrics import (
    bio_to_entities,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
)
from src.evaluation.competition_metrics import calculate_competition_metrics

# Frozen Champion Checkpoints
CHAMPION_CHECKPOINTS = {
    "gbert": Path("checkpoints/gbert_baseline_best"),
    "xlmr42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr21": Path("checkpoints/xlmr_seed_21_best"),
}

VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100
EXPECTED_MIN060_F02 = 0.899802
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42


def collect_active_probabilities(
    checkpoint_path: Path,
    val_examples: List[dict],
    label2id: Dict[str, int],
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> List[np.ndarray]:
    """Load model, run validation inference, extract softmax probabilities for active tokens, and free VRAM."""
    print(f"Loading {checkpoint_path.name}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()

    dataset, stats = prepare_split(val_examples, tokenizer, label2id)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    all_example_probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch_dev)
            logits = outputs.logits
            labels = batch_dev["labels"]

            for row_idx in range(labels.shape[0]):
                row_labels = labels[row_idx]
                row_logits = logits[row_idx]
                active_mask = row_labels != IGNORE_LABEL_ID
                active_logits = row_logits[active_mask]
                active_probs = torch.softmax(active_logits, dim=-1).cpu().numpy().astype(np.float32)
                all_example_probs.append(active_probs)

    del model, tokenizer, dataset, dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_example_probs


def bio_to_entities_with_spans_and_conf(
    tokens: List[str],
    bio_tags: List[str],
    token_probs: List[float] = None,
) -> List[dict]:
    """
    Extract entities with exact start and end token indices, value, and confidences.
    Returns:
    [
        {
            "aspect": aspect_name,
            "value": aspect_value,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "mean_conf": mean_conf,
            "min_conf": min_conf,
        },
        ...
    ]
    """
    if len(tokens) != len(bio_tags):
        raise ValueError("Lengths of tokens and tags must match.")
    if token_probs is not None and len(tokens) != len(token_probs):
        raise ValueError("Lengths of tokens and token_probs must match.")

    entities = []
    current_aspect = None
    current_tokens = []
    current_probs = []
    start_idx = None

    def close_current_entity(last_idx: int) -> None:
        nonlocal current_aspect, current_tokens, current_probs, start_idx
        if current_aspect is not None and current_tokens:
            value = " ".join(current_tokens)
            mean_c = float(np.mean(current_probs)) if current_probs else 1.0
            min_c = float(np.min(current_probs)) if current_probs else 1.0
            entities.append({
                "aspect": current_aspect,
                "value": value,
                "start_idx": start_idx,
                "end_idx": last_idx,
                "mean_conf": mean_c,
                "min_conf": min_c,
            })
        current_aspect = None
        current_tokens = []
        current_probs = []
        start_idx = None

    for idx, (token, tag) in enumerate(zip(tokens, bio_tags)):
        prob = token_probs[idx] if token_probs is not None else 1.0

        if tag == "O":
            close_current_entity(idx - 1)
            continue

        if tag.startswith("B-"):
            close_current_entity(idx - 1)
            current_aspect = tag[2:]
            current_tokens = [token]
            current_probs = [prob]
            start_idx = idx
            continue

        if tag.startswith("I-"):
            aspect = tag[2:]
            if current_aspect == aspect:
                current_tokens.append(token)
                current_probs.append(prob)
            else:
                close_current_entity(idx - 1)
                current_aspect = aspect
                current_tokens = [token]
                current_probs = [prob]
                start_idx = idx
            continue

        raise ValueError(f"Unknown BIO tag: {tag}")

    close_current_entity(len(tokens) - 1)
    return entities


def evaluate_records(records: List[dict]) -> dict:
    """Evaluate records and return official F0.2, category scores, and micro P/R."""
    metrics = calculate_competition_metrics(records)
    final_f02 = metrics["final_score"]
    cat1_score = metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = metrics["categories"].get(2, {}).get("score", 0.0)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    aspect_counts = Counter()

    for r in records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for asp, _ in r["predicted_entities"]:
            aspect_counts[asp] += 1

    pr_stats = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=0.2)
    return {
        "f02": final_f02,
        "cat1_f02": cat1_score,
        "cat2_f02": cat2_score,
        "precision": pr_stats["precision"],
        "recall": pr_stats["recall"],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_preds": sum(len(r["predicted_entities"]) for r in records),
        "aspect_counts": aspect_counts,
    }


def run_paired_bootstrap(
    base_records: List[dict],
    candidate_records: List[dict],
    num_rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Run paired bootstrap resampling over validation titles to estimate Delta F0.2 CI."""
    np.random.seed(seed)
    n = len(base_records)
    assert len(candidate_records) == n

    deltas = []
    for _ in range(num_rounds):
        indices = np.random.randint(0, n, size=n)
        resampled_base = [base_records[i] for i in indices]
        resampled_cand = [candidate_records[i] for i in indices]

        score_b = calculate_competition_metrics(resampled_base)["final_score"]
        score_c = calculate_competition_metrics(resampled_cand)["final_score"]
        deltas.append(score_c - score_b)

    deltas = np.array(deltas)
    return {
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
        "ci_95": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
        "prob_improved": float(np.mean(deltas > 0)),
    }


def main():
    print("=" * 80)
    print("PHASE 27A: ENTITY-LEVEL AGREEMENT RESEARCH")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Label Mapping from Reference Champion
    ref_cfg = AutoConfig.from_pretrained(CHAMPION_CHECKPOINTS["gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # 2. Load Split A Validation Set
    df = load_training_data()
    all_examples = build_examples(df)
    _, val_examples = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(val_examples)} Split-A validation titles.\n")

    # 3. Extract Probabilities Sequentially
    print("--- [1/4] Extracting GBERT probabilities ---")
    p_gb = collect_active_probabilities(CHAMPION_CHECKPOINTS["gbert"], val_examples, label2id, device)

    print("--- [2/4] Extracting XLM-R Seed 42 probabilities ---")
    p_42 = collect_active_probabilities(CHAMPION_CHECKPOINTS["xlmr42"], val_examples, label2id, device)

    print("--- [3/4] Extracting XLM-R Seed 7 probabilities ---")
    p_7 = collect_active_probabilities(CHAMPION_CHECKPOINTS["xlmr7"], val_examples, label2id, device)

    print("--- [4/4] Extracting XLM-R Seed 21 probabilities ---")
    p_21 = collect_active_probabilities(CHAMPION_CHECKPOINTS["xlmr21"], val_examples, label2id, device)

    num_ex = len(val_examples)

    # 4. Extract Independent Single-Model Predictions
    print("\nExtracting independent model predictions and building entity span sets...")
    # For each model, for each title: set of (aspect, start_idx, end_idx)
    model_entity_spans = {
        "gbert": [],
        "xlmr42": [],
        "xlmr7": [],
        "xlmr21": [],
    }

    raw_probs_by_model = {
        "gbert": p_gb,
        "xlmr42": p_42,
        "xlmr7": p_7,
        "xlmr21": p_21,
    }

    for m_name in ["gbert", "xlmr42", "xlmr7", "xlmr21"]:
        probs_list = raw_probs_by_model[m_name]
        for ex_idx, ex in enumerate(val_examples):
            tokens = ex["tokens"]
            cat_id = ex["category"]
            m_probs = probs_list[ex_idx]

            pred_ids = np.argmax(m_probs, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]

            ents = bio_to_entities_with_spans_and_conf(tokens, pred_bio)
            # Category-aware filtering
            valid_spans = set()
            for e in ents:
                if filter_entities_by_category([(e["aspect"], e["value"])], cat_id):
                    # Exact aspect type and original-token span boundaries
                    valid_spans.add((e["aspect"], e["start_idx"], e["end_idx"]))

            model_entity_spans[m_name].append(valid_spans)

    # 5. Extract Fused MIN 0.60 Champion Entities
    print("\nComputing fused probabilities and extracting MIN 0.60 champion predictions...")
    p_fused = [
        0.40 * p_gb[i] + 0.20 * p_42[i] + 0.20 * p_7[i] + 0.20 * p_21[i]
        for i in range(num_ex)
    ]

    min060_candidate_entities = []  # per title: list of dicts with agreement stats
    base_min060_records = []

    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]

        f_probs = p_fused[ex_idx]
        pred_ids = np.argmax(f_probs, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(f_probs[t, pred_ids[t]]) for t in range(len(tokens))]

        true_entities = bio_to_entities(tokens, true_bio)

        fused_ents = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)

        # Apply category filter + MIN 0.60 confidence filter
        title_min060_ents = []
        title_base_preds = []

        for e in fused_ents:
            if not filter_entities_by_category([(e["aspect"], e["value"])], cat_id):
                continue
            if e["min_conf"] < 0.60:
                continue

            asp = e["aspect"]
            val = e["value"]
            s_idx = e["start_idx"]
            e_idx = e["end_idx"]
            span_key = (asp, s_idx, e_idx)

            # Check independent model support
            has_gb = span_key in model_entity_spans["gbert"][ex_idx]
            has_42 = span_key in model_entity_spans["xlmr42"][ex_idx]
            has_7 = span_key in model_entity_spans["xlmr7"][ex_idx]
            has_21 = span_key in model_entity_spans["xlmr21"][ex_idx]

            support_count = int(has_gb) + int(has_42) + int(has_7) + int(has_21)
            rule_d_match = has_gb and (has_42 or has_7 or has_21)

            e_entry = {
                "aspect": asp,
                "value": val,
                "start_idx": s_idx,
                "end_idx": e_idx,
                "min_conf": e["min_conf"],
                "has_gb": has_gb,
                "has_42": has_42,
                "has_7": has_7,
                "has_21": has_21,
                "support_count": support_count,
                "rule_d_match": rule_d_match,
            }
            title_min060_ents.append(e_entry)
            title_base_preds.append((asp, val))

        min060_candidate_entities.append(title_min060_ents)
        base_min060_records.append({
            "category": cat_id,
            "true_entities": true_entities,
            "predicted_entities": title_base_preds,
        })

    # 6. Step 1: Reproduce Champion Score
    base_metrics = evaluate_records(base_min060_records)
    print("\n" + "=" * 50)
    print("STEP 1: REPRODUCE MIN 0.60 CHAMPION")
    print("=" * 50)
    print(f"Reproduced Local F0.2: {base_metrics['f02']:.6f}")
    print(f"Expected Local F0.2:   {EXPECTED_MIN060_F02:.6f}")
    print(f"Category 1 F0.2:      {base_metrics['cat1_f02']:.6f}")
    print(f"Category 2 F0.2:      {base_metrics['cat2_f02']:.6f}")
    print(f"Micro Precision:      {base_metrics['precision']:.4f}")
    print(f"Micro Recall:         {base_metrics['recall']:.4f}")
    print(f"Total True Positives: {base_metrics['tp']}")
    print(f"Total False Positives:{base_metrics['fp']}")
    print(f"Total Predicted Rows: {base_metrics['total_preds']}")

    assert abs(base_metrics["f02"] - EXPECTED_MIN060_F02) < 1e-5, (
        f"Champion F0.2 mismatch! Got {base_metrics['f02']:.6f}, expected {EXPECTED_MIN060_F02:.6f}"
    )
    print(">>> EXACT CHAMPION BASELINE REPRODUCED (0.899802) <<<\n")

    # 7. Step 2: Aggregate Agreement Distribution (True Positives vs False Positives)
    print("=" * 60)
    print("STEP 2: AGREEMENT DISTRIBUTION ON MIN 0.60 EMISSIONS")
    print("=" * 60)

    tp_support_counts = Counter()
    fp_support_counts = Counter()
    total_support_counts = Counter()

    for ex_idx, ex in enumerate(val_examples):
        ents = min060_candidate_entities[ex_idx]
        true_counter = Counter(base_min060_records[ex_idx]["true_entities"])
        claimed_counter = Counter()

        for e in ents:
            ent_tuple = (e["aspect"], e["value"])
            supp = e["support_count"]
            total_support_counts[supp] += 1

            # Check if this prediction matches an unclaimed ground-truth entity
            if claimed_counter[ent_tuple] < true_counter[ent_tuple]:
                claimed_counter[ent_tuple] += 1
                tp_support_counts[supp] += 1
            else:
                fp_support_counts[supp] += 1

    print(f"{'Support':<10}{'Total':<10}{'TP':<10}{'FP':<10}{'Precision':<12}")
    print("-" * 52)
    for s in range(5):
        tot = total_support_counts[s]
        tp = tp_support_counts[s]
        fp = fp_support_counts[s]
        prec = (tp / tot * 100) if tot > 0 else 0.0
        print(f"{s:<10}{tot:<10}{tp:<10}{fp:<10}{prec:<8.2f}%")
    print("-" * 52)
    print(
        f"{'Total':<10}{sum(total_support_counts.values()):<10}"
        f"{sum(tp_support_counts.values()):<10}{sum(fp_support_counts.values()):<10}"
        f"{(sum(tp_support_counts.values())/sum(total_support_counts.values())*100):<8.2f}%"
    )

    # 8. Step 3 & 4: Evaluate Predefined Voting Rules
    rules = [
        ("Baseline (MIN 0.60 Champion)", lambda e: True),
        ("Rule A (Support >= 2/4)", lambda e: e["support_count"] >= 2),
        ("Rule B (Support >= 3/4)", lambda e: e["support_count"] >= 3),
        ("Rule C (Support == 4/4)", lambda e: e["support_count"] == 4),
        ("Rule D (GBERT + >= 1 XLM-R)", lambda e: e["rule_d_match"]),
    ]

    all_rule_records = {}
    rule_results = []

    for r_name, r_filter in rules:
        r_records = []
        for ex_idx in range(num_ex):
            ex = val_examples[ex_idx]
            cand_ents = min060_candidate_entities[ex_idx]
            kept_preds = [(e["aspect"], e["value"]) for e in cand_ents if r_filter(e)]
            r_records.append({
                "category": ex["category"],
                "true_entities": base_min060_records[ex_idx]["true_entities"],
                "predicted_entities": kept_preds,
            })

        m = evaluate_records(r_records)
        all_rule_records[r_name] = r_records

        delta_f02 = m["f02"] - base_metrics["f02"]
        removed_tot = base_metrics["total_preds"] - m["total_preds"]
        removed_tp = base_metrics["tp"] - m["tp"]
        removed_fp = base_metrics["fp"] - m["fp"]

        # Check extinct aspects (in official 29 aspect list)
        extinct_count = sum(1 for asp_count in m["aspect_counts"].values() if asp_count == 0)
        emitted_aspects = len(m["aspect_counts"])

        rule_results.append({
            "name": r_name,
            "f02": m["f02"],
            "delta_f02": delta_f02,
            "cat1_f02": m["cat1_f02"],
            "cat2_f02": m["cat2_f02"],
            "precision": m["precision"],
            "recall": m["recall"],
            "total_preds": m["total_preds"],
            "removed_tot": removed_tot,
            "removed_tp": removed_tp,
            "removed_fp": removed_fp,
            "emitted_aspects": emitted_aspects,
            "aspect_counts": m["aspect_counts"],
        })

    # Display Predefined Rule Results Table
    print("\n" + "=" * 115)
    print("STEP 3 & 4: PREDEFINED VOTING RULES EVALUATION TABLE")
    print("=" * 115)
    print(
        f"{'Variant / Rule':<35}{'F0.2':<11}{'Delta':<11}{'Cat1':<10}{'Cat2':<10}"
        f"{'Prec':<9}{'Rec':<9}{'Preds':<8}{'RemTot':<8}{'RemTP':<7}{'RemFP':<7}{'Aspects':<8}"
    )
    print("-" * 115)

    for r in rule_results:
        d_str = f"{r['delta_f02']:+.6f}"
        print(
            f"{r['name']:<35}{r['f02']:<11.6f}{d_str:<11}"
            f"{r['cat1_f02']:<10.6f}{r['cat2_f02']:<10.6f}{r['precision']:<9.4f}{r['recall']:<9.4f}"
            f"{r['total_preds']:<8}{r['removed_tot']:<8}{r['removed_tp']:<7}{r['removed_fp']:<7}{r['emitted_aspects']:<8}"
        )
    print("-" * 115)

    # 9. Paired Bootstrap Resampling for All Candidate Rules vs MIN 0.60
    print("\n" + "=" * 80)
    print(f"PAIRED BOOTSTRAP RESAMPLING ({BOOTSTRAP_ROUNDS} ROUNDS VS MIN 0.60)")
    print("=" * 80)
    print(f"{'Variant':<35}{'Mean Delta':<14}{'Std Error':<12}{'95% Confidence Interval':<26}{'P(Improve)':<12}")
    print("-" * 95)

    bootstrap_stats = {}
    for r in rule_results[1:]:
        r_name = r["name"]
        b_res = run_paired_bootstrap(base_min060_records, all_rule_records[r_name])
        bootstrap_stats[r_name] = b_res
        m_delta_str = f"{b_res['mean_delta']:+.6f}"
        s_delta_str = f"{b_res['std_delta']:.6f}"
        ci_str = f"[{b_res['ci_95'][0]:+.6f}, {b_res['ci_95'][1]:+.6f}]"
        print(
            f"{r_name:<35}{m_delta_str:<14}{s_delta_str:<12}"
            f"{ci_str:<26}{b_res['prob_improved']*100:<10.1f}%"
        )
    print("-" * 95)

    # 10. Per-Aspect Impact Audit
    print("\n" + "=" * 80)
    print("PER-ASPECT PREDICTION COUNTS ACROSS VOTING RULES")
    print("=" * 80)
    base_asp = rule_results[0]["aspect_counts"]
    print(f"{'Aspect Name':<35}{'MIN 0.60':<12}{'Rule A (>=2)':<14}{'Rule B (>=3)':<14}{'Rule C (4/4)':<14}{'Rule D (GB+X)':<14}")
    print("-" * 95)

    for asp, count in base_asp.most_common():
        cnt_a = rule_results[1]["aspect_counts"].get(asp, 0)
        cnt_b = rule_results[2]["aspect_counts"].get(asp, 0)
        cnt_c = rule_results[3]["aspect_counts"].get(asp, 0)
        cnt_d = rule_results[4]["aspect_counts"].get(asp, 0)
        print(f"{asp:<35}{count:<12}{cnt_a:<14}{cnt_b:<14}{cnt_c:<14}{cnt_d:<14}")
    print("-" * 95)

    # 11. Decision Criteria Summary
    best_rule = max(rule_results[1:], key=lambda x: x["f02"])
    print("\n" + "=" * 80)
    print("PHASE 27A DECISION CRITERIA & SUMMARY")
    print("=" * 80)
    print(f"Current Verified Champion: MIN 0.60 (F0.2 = {base_metrics['f02']:.6f})")
    print(f"Highest-Scoring Rule:     {best_rule['name']} (F0.2 = {best_rule['f02']:.6f}, Delta = {best_rule['delta_f02']:+.6f})")

    if best_rule["f02"] > base_metrics["f02"]:
        print("  >>> Preliminary result: Rule shows higher local score on Split A. <<<")
    else:
        print("  >>> Result: No voting rule improves upon MIN 0.60. <<<")

    return {
        "base_metrics": base_metrics,
        "rule_results": rule_results,
        "bootstrap_stats": bootstrap_stats,
    }


if __name__ == "__main__":
    main()
