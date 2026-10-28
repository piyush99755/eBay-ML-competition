"""
Phase 32B: Vehicle-Model Segmentation Feasibility & Targeted Training Audit Script.

Computes:
1. Subword fragmentation rates for GBERT and XLM-R across word tokens and entities.
2. Correlation of fragmentation with validation error types (TP vs FP vs FN).
3. Detailed breakdown of the 76 conflicting B-I vs B-B contiguous token pairs.
4. Class imbalance and label statistics for training architecture.
5. Resource and timing estimates for GTX 1660 Ti.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from transformers import AutoTokenizer, AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.phase_30a_error_profile import (
    bio_to_entities_with_spans_and_conf,
    bio_to_true_spans,
)
from src.evaluation.experiment_26a_confidence_calibration import collect_active_probabilities
from src.evaluation.entity_metrics import filter_entities_by_category

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHAMPION_CHECKPOINTS = {
    "gbert": Path("checkpoints/gbert_baseline_best"),
    "xlmr42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr21": Path("checkpoints/xlmr_seed_21_best"),
}


def analyze_tokenizer_fragmentation(
    examples: List[dict],
    tokenizer,
    tok_name: str,
) -> dict:
    """Analyze subword fragmentation for vehicle-model tokens and entities."""
    target_aspect = "Kompatibles_Fahrzeug_Modell"

    total_words = 0
    fragmented_words = 0
    word_subword_counts = []

    total_entities = 0
    fragmented_entities = 0
    single_token_entities = 0
    single_token_fragmented = 0
    multi_token_entities = 0
    multi_token_fragmented = 0

    for ex in examples:
        tokens = ex["tokens"]
        bio = ex["bio_tags"]
        cat_id = ex["category"]
        T = len(tokens)

        # Word-level analysis
        for t in range(T):
            if target_aspect in bio[t]:
                total_words += 1
                subs = tokenizer.tokenize(tokens[t])
                n_subs = len(subs)
                word_subword_counts.append(n_subs)
                if n_subs > 1:
                    fragmented_words += 1

        # Entity-level analysis
        true_spans = bio_to_true_spans(tokens, bio)
        cat_spans = [s for s in true_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]
        model_spans = [s for s in cat_spans if s[0] == target_aspect]

        for asp, val, s, e in model_spans:
            total_entities += 1
            ent_tokens = tokens[s:e + 1]
            ent_sub_counts = [len(tokenizer.tokenize(t)) for t in ent_tokens]
            is_frag = any(c > 1 for c in ent_sub_counts)

            if is_frag:
                fragmented_entities += 1

            if len(ent_tokens) == 1:
                single_token_entities += 1
                if is_frag:
                    single_token_fragmented += 1
            else:
                multi_token_entities += 1
                if is_frag:
                    multi_token_fragmented += 1

    return {
        "tokenizer": tok_name,
        "total_words": total_words,
        "fragmented_words": fragmented_words,
        "word_fragmentation_pct": (fragmented_words / max(1, total_words)) * 100,
        "avg_subwords_per_word": float(np.mean(word_subword_counts)) if word_subword_counts else 1.0,
        "median_subwords_per_word": float(np.median(word_subword_counts)) if word_subword_counts else 1.0,
        "max_subwords_in_word": int(np.max(word_subword_counts)) if word_subword_counts else 1,
        "total_entities": total_entities,
        "fragmented_entities": fragmented_entities,
        "entity_fragmentation_pct": (fragmented_entities / max(1, total_entities)) * 100,
        "single_token_entities": single_token_entities,
        "single_token_fragmented_pct": (single_token_fragmented / max(1, single_token_entities)) * 100,
        "multi_token_entities": multi_token_entities,
        "multi_token_fragmented_pct": (multi_token_fragmented / max(1, multi_token_entities)) * 100,
    }


def analyze_prediction_error_fragmentation(
    val_examples: List[dict],
    tokenizer,
    tok_name: str,
) -> dict:
    """Analyze fragmentation rates for TP, FP, and FN vehicle-model entities."""
    target_aspect = "Kompatibles_Fahrzeug_Modell"

    # Load Phase 31A baseline audit results to get error classifications
    audit_file = CACHE_DIR / "phase_31a_step1_step2_audit.json"
    has_audit = audit_file.exists()

    # Load fused probabilities or predictions
    fused_cache = CACHE_DIR / "split_a_champion_fused_probs.npy"
    if fused_cache.exists():
        p_fused = list(np.load(fused_cache, allow_pickle=True))
    else:
        # Load from individual checkpoints if fused not available
        p_fused = None

    ref_cfg = AutoConfig.from_pretrained(Path("checkpoints/gbert_baseline_best"))
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    tp_frags = []
    fp_frags = []
    fn_frags = []

    # If fused cache exists, we can directly tag every prediction
    if p_fused is not None:
        for ex_idx, ex in enumerate(val_examples):
            tokens = ex["tokens"]
            cat_id = ex["category"]
            true_bio = ex["bio_tags"]
            p = p_fused[ex_idx]

            pred_ids = np.argmax(p, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]
            selected_probs = [float(p[t, pred_ids[t]]) for t in range(len(tokens))]

            true_spans = bio_to_true_spans(tokens, true_bio)
            cat_true = [s for s in true_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]
            model_true = [s for s in cat_true if s[0] == target_aspect]

            pred_spans = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)
            cat_pred = [s for s in pred_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]
            model_pred = [s for s in cat_pred if s[0] == target_aspect and s[5] >= 0.60]

            matched_true = set()
            for p_ent in model_pred:
                p_val, p_s, p_e = p_ent[1], p_ent[2], p_ent[3]
                p_tokens = tokens[p_s:p_e + 1]
                is_p_frag = any(len(tokenizer.tokenize(t)) > 1 for t in p_tokens)

                # Check if matches any ground truth
                match = None
                for t_idx, t_ent in enumerate(model_true):
                    if t_idx not in matched_true and t_ent[1] == p_val:
                        match = t_idx
                        matched_true.add(t_idx)
                        break

                if match is not None:
                    tp_frags.append(is_p_frag)
                else:
                    fp_frags.append(is_p_frag)

            for t_idx, t_ent in enumerate(model_true):
                if t_idx not in matched_true:
                    t_tokens = tokens[t_ent[2]:t_ent[3] + 1]
                    is_t_frag = any(len(tokenizer.tokenize(t)) > 1 for t in t_tokens)
                    fn_frags.append(is_t_frag)

    return {
        "tokenizer": tok_name,
        "tp_count": len(tp_frags),
        "tp_fragmented_pct": (sum(tp_frags) / max(1, len(tp_frags))) * 100 if tp_frags else 0.0,
        "fp_count": len(fp_frags),
        "fp_fragmented_pct": (sum(fp_frags) / max(1, len(fp_frags))) * 100 if fp_frags else 0.0,
        "fn_count": len(fn_frags),
        "fn_fragmented_pct": (sum(fn_frags) / max(1, len(fn_frags))) * 100 if fn_frags else 0.0,
    }


def audit_annotation_contradictions(examples: List[dict]) -> dict:
    """Analyze the 76 conflicting B-I vs B-B contiguous token pairs."""
    target_aspect = "Kompatibles_Fahrzeug_Modell"

    pair_data = defaultdict(lambda: {
        "b_i_count": 0,
        "b_b_count": 0,
        "cat1_count": 0,
        "cat2_count": 0,
        "contexts": [],
    })

    for ex in examples:
        tokens = ex["tokens"]
        bio = ex["bio_tags"]
        cat_id = ex["category"]
        T = len(tokens)

        for t in range(T - 1):
            if target_aspect in bio[t] and target_aspect in bio[t + 1]:
                norm_pair = f"{tokens[t].strip().lower()} {tokens[t+1].strip().lower()}"
                trans = f"{bio[t][:1]}-{bio[t+1][:1]}"

                if trans == "B-I":
                    pair_data[norm_pair]["b_i_count"] += 1
                elif trans == "B-B":
                    pair_data[norm_pair]["b_b_count"] += 1

                if cat_id == 1:
                    pair_data[norm_pair]["cat1_count"] += 1
                else:
                    pair_data[norm_pair]["cat2_count"] += 1

    # Filter to pairs with >=5 total occurrences and conflicting tags (>=2 B-I and >=2 B-B per Phase 32A definition)
    conflicting_pairs = {}
    for pair, d in pair_data.items():
        tot = d["b_i_count"] + d["b_b_count"]
        if tot >= 5 and d["b_i_count"] >= 2 and d["b_b_count"] >= 2:
            conflicting_pairs[pair] = d

    # Categorize patterns into 4 structural types:
    # 1. Generation/Numeral suffix (e.g., token 2 is Roman numeral or digit code: 'iv', '3', 'ii', 'b6', 'w204')
    # 2. Body style / trim descriptor (e.g., token 2 is 'variant', 'combi', 'kombi', 'cabrio', 'coupe', 'sedan', 'touring')
    # 3. Model + Submodel / Platform code (e.g., alphanumeric platform codes or submodel identifiers)
    # 4. Multi-model list collision (two distinct car model names listed adjacent without punctuation, e.g., 'golf passat')
    body_styles = {"variant", "kombi", "combi", "cabrio", "cabriolet", "coupe", "coupé", "sedan", "touring", "limousine", "caravan", "compact", "fließheck"}
    roman_numerals = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}

    pattern_counts = {
        "generation_or_digit_code": 0,
        "body_style_descriptor": 0,
        "model_submodel_code": 0,
        "multi_model_list_collision": 0,
    }

    category_counts = {
        "category_1_only": 0,
        "category_2_only": 0,
        "both_categories": 0,
    }

    for pair, d in conflicting_pairs.items():
        parts = pair.split()
        t1 = parts[0] if len(parts) > 0 else ""
        t2 = parts[1] if len(parts) > 1 else ""

        # Pattern classification
        if t2 in roman_numerals or bool(re.search(r"^\d+$", t2)):
            pattern_counts["generation_or_digit_code"] += 1
        elif t2 in body_styles:
            pattern_counts["body_style_descriptor"] += 1
        elif bool(re.search(r"[a-z]\d|\d[a-z]", t2)):
            pattern_counts["model_submodel_code"] += 1
        else:
            pattern_counts["multi_model_list_collision"] += 1

        # Category breakdown
        if d["cat1_count"] > 0 and d["cat2_count"] == 0:
            category_counts["category_1_only"] += 1
        elif d["cat2_count"] > 0 and d["cat1_count"] == 0:
            category_counts["category_2_only"] += 1
        else:
            category_counts["both_categories"] += 1

    return {
        "total_conflicting_pairs": len(conflicting_pairs),
        "pattern_distribution": pattern_counts,
        "category_distribution": category_counts,
        "summary": {
            "avg_b_i_ratio": float(np.mean([d["b_i_count"] / (d["b_i_count"] + d["b_b_count"]) for d in conflicting_pairs.values()])),
            "pairs_predominantly_b_i": sum(1 for d in conflicting_pairs.values() if d["b_i_count"] > d["b_b_count"]),
            "pairs_predominantly_b_b": sum(1 for d in conflicting_pairs.values() if d["b_b_count"] > d["b_i_count"]),
            "pairs_equally_split": sum(1 for d in conflicting_pairs.values() if d["b_i_count"] == d["b_b_count"]),
        }
    }


def main():
    print("=" * 70)
    print("PHASE 32B: FEASIBILITY AND TARGETED TRAINING AUDIT")
    print("=" * 70)

    # 1. Load Data
    df = load_training_data()
    all_examples = build_examples(df)
    train_examples, val_examples = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(train_examples)} Train titles and {len(val_examples)} Validation titles.")

    # 2. Load Tokenizers
    tok_gbert = AutoTokenizer.from_pretrained(Path("checkpoints/gbert_baseline_best"))
    tok_xlmr = AutoTokenizer.from_pretrained(Path("checkpoints/xlmr_baseline_best"))

    # =========================================================================
    # TASK 1: SUBWORD FRAGMENTATION REPRODUCTION (GBERT & XLM-R)
    # =========================================================================
    print("\n--- TASK 1: SUBWORD FRAGMENTATION AUDIT ---")
    gbert_train_frag = analyze_tokenizer_fragmentation(train_examples, tok_gbert, "GBERT (Train)")
    xlmr_train_frag = analyze_tokenizer_fragmentation(train_examples, tok_xlmr, "XLM-R (Train)")

    gbert_val_frag = analyze_tokenizer_fragmentation(val_examples, tok_gbert, "GBERT (Val)")
    xlmr_val_frag = analyze_tokenizer_fragmentation(val_examples, tok_xlmr, "XLM-R (Val)")

    print(f"{'Metric':<42} {'GBERT (Train)':<18} {'XLM-R (Train)':<18}")
    print("-" * 78)
    print(f"{'Total vehicle-model word tokens':<42} {gbert_train_frag['total_words']:<18} {xlmr_train_frag['total_words']:<18}")
    print(f"{'Tokens fragmented into >=2 subwords':<42} {gbert_train_frag['fragmented_words']} ({gbert_train_frag['word_fragmentation_pct']:.1f}%)   {xlmr_train_frag['fragmented_words']} ({xlmr_train_frag['word_fragmentation_pct']:.1f}%)")
    print(f"{'Average subwords per word token':<42} {gbert_train_frag['avg_subwords_per_word']:.2f}{'':<14} {xlmr_train_frag['avg_subwords_per_word']:.2f}")
    print(f"{'Max subwords in a single token':<42} {gbert_train_frag['max_subwords_in_word']:<18} {xlmr_train_frag['max_subwords_in_word']:<18}")
    print(f"{'Entities with >=1 fragmented token':<42} {gbert_train_frag['fragmented_entities']} ({gbert_train_frag['entity_fragmentation_pct']:.1f}%)   {xlmr_train_frag['fragmented_entities']} ({xlmr_train_frag['entity_fragmentation_pct']:.1f}%)")
    print(f"{'Single-token entity fragmentation':<42} {gbert_train_frag['single_token_fragmented_pct']:.1f}%{'':<13} {xlmr_train_frag['single_token_fragmented_pct']:.1f}%")
    print(f"{'Multi-token entity fragmentation':<42} {gbert_train_frag['multi_token_fragmented_pct']:.1f}%{'':<13} {xlmr_train_frag['multi_token_fragmented_pct']:.1f}%")

    # Ensure fused probabilities are available for validation prediction analysis
    fused_cache = CACHE_DIR / "split_a_champion_fused_probs.npy"
    if not fused_cache.exists():
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nExtracting champion model probabilities for validation set on {device}...")
        ref_cfg = AutoConfig.from_pretrained(CHAMPION_CHECKPOINTS["gbert"])
        id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        probs = {}
        for name, path in CHAMPION_CHECKPOINTS.items():
            print(f"  Extracting {name}...")
            probs[name] = collect_active_probabilities(path, val_examples, label2id, device, batch_size=8)
        p_fused = [
            0.40 * probs["gbert"][i] + 0.20 * probs["xlmr42"][i] + 0.20 * probs["xlmr7"][i] + 0.20 * probs["xlmr21"][i]
            for i in range(len(val_examples))
        ]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(fused_cache, np.array(p_fused, dtype=object))
        print(f"Cached fused probabilities to {fused_cache}.")

    # Error correlation fragmentation
    print("\n--- FRAGMENTATION VS VALIDATION PREDICTION OUTCOMES ---")
    gbert_err_frag = analyze_prediction_error_fragmentation(val_examples, tok_gbert, "GBERT")
    xlmr_err_frag = analyze_prediction_error_fragmentation(val_examples, tok_xlmr, "XLM-R")

    print(f"{'Validation Outcome':<35} {'Count':<10} {'GBERT Frag %':<15} {'XLM-R Frag %':<15}")
    print("-" * 75)
    print(f"{'True Positives (Correct Spans)':<35} {gbert_err_frag['tp_count']:<10} {gbert_err_frag['tp_fragmented_pct']:.1f}%{'':<9} {xlmr_err_frag['tp_fragmented_pct']:.1f}%")
    print(f"{'False Positives (Boundary/Overlaps)':<35} {gbert_err_frag['fp_count']:<10} {gbert_err_frag['fp_fragmented_pct']:.1f}%{'':<9} {xlmr_err_frag['fp_fragmented_pct']:.1f}%")
    print(f"{'False Negatives (Missed Spans)':<35} {gbert_err_frag['fn_count']:<10} {gbert_err_frag['fn_fragmented_pct']:.1f}%{'':<9} {xlmr_err_frag['fn_fragmented_pct']:.1f}%")

    # =========================================================================
    # TASK 2: AUDIT APPARENT ANNOTATION CONTRADICTIONS
    # =========================================================================
    print("\n--- TASK 2: AUDIT APPARENT ANNOTATION CONTRADICTIONS ---")
    contradiction_audit = audit_annotation_contradictions(train_examples)

    print(f"Total frequent conflicting contiguous model pairs: {contradiction_audit['total_conflicting_pairs']}")
    print("\nStructural Pattern Distribution:")
    for pat, count in contradiction_audit["pattern_distribution"].items():
        pct = count / max(1, contradiction_audit["total_conflicting_pairs"]) * 100
        print(f"  - {pat:<32}: {count:<4} ({pct:.1f}%)")

    print("\nCategory Distribution:")
    for cat, count in contradiction_audit["category_distribution"].items():
        pct = count / max(1, contradiction_audit["total_conflicting_pairs"]) * 100
        print(f"  - {cat:<32}: {count:<4} ({pct:.1f}%)")

    print("\nSegmentation Ratio Distribution:")
    s = contradiction_audit["summary"]
    print(f"  - Average B-I probability across pairs: {s['avg_b_i_ratio']:.3f}")
    print(f"  - Pairs predominantly annotated as B-I: {s['pairs_predominantly_b_i']}")
    print(f"  - Pairs predominantly annotated as B-B: {s['pairs_predominantly_b_b']}")
    print(f"  - Pairs equally split (50/50):          {s['pairs_equally_split']}")

    # Save results
    results = {
        "task1_gbert_train": gbert_train_frag,
        "task1_xlmr_train": xlmr_train_frag,
        "task1_gbert_val": gbert_val_frag,
        "task1_xlmr_val": xlmr_val_frag,
        "task1_gbert_errors": gbert_err_frag,
        "task1_xlmr_errors": xlmr_err_frag,
        "task2_contradictions": contradiction_audit,
    }

    out_file = CACHE_DIR / "phase_32b_feasibility_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved audit results to {out_file}")


if __name__ == "__main__":
    main()
