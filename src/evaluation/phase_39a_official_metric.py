"""
Phase 39A: Official Competition Metric Evaluation Utility.

Implements the verified official eBay 2026 Machine Learning Competition evaluation metric:
1. Exact entity matching on (Aspect Name, Aspect Value) tuples via multiset intersection.
2. Aspect-level Precision, Recall, and F0.2 calculation (beta = 0.2, precision weighted 25x over recall).
3. Category-level ground-truth frequency-weighted aspect aggregation:
   Score_c = sum_a (w_{c, a} * F0.2_{c, a}), where w_{c, a} = N_{true, c, a} / N_{true, c}.
4. Final competition score: unweighted macro average over categories:
   Official Score = (Score_{Cat 1} + Score_{Cat 2}) / 2.

Also provides the Global Micro F0.2 baseline for direct comparative audit.
Does NOT overwrite existing evaluation modules.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

BETA: float = 0.2
BETA_SQ: float = 0.04  # 0.2 ** 2


def compute_fbeta(precision: float, recall: float, beta: float = BETA) -> float:
    """
    Compute F-beta score from precision and recall.
    Formula: (1 + beta^2) * (P * R) / (beta^2 * P + R).
    For beta = 0.2: 1.04 * (P * R) / (0.04 * P + R).
    """
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denom)


def count_entity_matches(
    predicted_entities: List[Tuple[str, str]],
    gold_entities: List[Tuple[str, str]],
) -> Tuple[int, int, int]:
    """
    Count True Positives, False Positives, and False Negatives using multiset intersection.
    
    Arguments:
        predicted_entities: List of (aspect_name, aspect_value) tuples predicted by model.
        gold_entities: List of (aspect_name, aspect_value) tuples from ground truth.
        
    Returns:
        (true_positives, false_positives, false_negatives)
        
    Invariant:
        true_positives + false_negatives == len(gold_entities)
        true_positives + false_positives == len(predicted_entities)
    """
    pred_counter = Counter(predicted_entities)
    gold_counter = Counter(gold_entities)
    
    matched = pred_counter & gold_counter
    tp = sum(matched.values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp
    
    return tp, fp, fn


def calculate_official_competition_metrics(
    records: List[Dict[str, Any]],
    beta: float = BETA,
) -> Dict[str, Any]:
    """
    Calculate the official competition-style hierarchical evaluation metric.
    
    Each record must be a dict:
    {
        "category": int,  # 1 or 2
        "predicted_entities": list[tuple[str, str]],
        "gold_entities": list[tuple[str, str]],
    }
    
    Algorithm:
    1. Group predictions and gold entities by Category and Aspect.
    2. For each (Category, Aspect):
       - Count TP, FP, FN via multiset intersection.
       - Compute Aspect Precision: TP / (TP + FP) if (TP + FP) > 0 else 0.0
       - Compute Aspect Recall: TP / (TP + FN) if (TP + FN) > 0 else 0.0
       - Compute Aspect F0.2: (1 + 0.04) * (P * R) / (0.04 * P + R)
    3. For each Category:
       - Total true entities = sum_{aspect} N_{true, aspect}
       - For each aspect, weight w_{c, a} = N_{true, c, a} / Total true entities
       - Category Score = sum_{aspect} (w_{c, a} * F0.2_{c, a})
    4. Final Competition Score = (Category 1 Score + Category 2 Score) / 2.0
    
    Returns structured results with full intermediate statistics.
    """
    # Bookkeeping: category -> aspect -> counts
    cat_aspect_counts = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gold_count": 0, "pred_count": 0}))
    
    for rec in records:
        cat = rec["category"]
        preds = rec["predicted_entities"]
        golds = rec["gold_entities"]
        
        preds_by_asp = defaultdict(list)
        golds_by_asp = defaultdict(list)
        
        for asp, val in preds:
            preds_by_asp[asp].append((asp, val))
        for asp, val in golds:
            golds_by_asp[asp].append((asp, val))
            
        all_asps = set(preds_by_asp.keys()) | set(golds_by_asp.keys())
        
        for asp in all_asps:
            p_list = preds_by_asp[asp]
            g_list = golds_by_asp[asp]
            
            tp, fp, fn = count_entity_matches(p_list, g_list)
            
            entry = cat_aspect_counts[cat][asp]
            entry["tp"] += tp
            entry["fp"] += fp
            entry["fn"] += fn
            entry["gold_count"] += len(g_list)
            entry["pred_count"] += len(p_list)
            
    category_results = {}
    
    for cat in sorted(cat_aspect_counts.keys()):
        aspect_dict = cat_aspect_counts[cat]
        total_gold = sum(c["gold_count"] for c in aspect_dict.values())
        total_pred = sum(c["pred_count"] for c in aspect_dict.values())
        total_tp = sum(c["tp"] for c in aspect_dict.values())
        total_fp = sum(c["fp"] for c in aspect_dict.values())
        total_fn = sum(c["fn"] for c in aspect_dict.values())
        
        weighted_score = 0.0
        aspect_results = {}
        
        for asp in sorted(aspect_dict.keys()):
            c = aspect_dict[asp]
            tp = c["tp"]
            fp = c["fp"]
            fn = c["fn"]
            gold_cnt = c["gold_count"]
            pred_cnt = c["pred_count"]
            
            p_denom = tp + fp
            r_denom = tp + fn
            prec = float(tp / p_denom) if p_denom > 0 else 0.0
            rec = float(tp / r_denom) if r_denom > 0 else 0.0
            f_score = compute_fbeta(prec, rec, beta=beta)
            
            weight = float(gold_cnt / total_gold) if total_gold > 0 else 0.0
            weighted_f = float(weight * f_score)
            weighted_score += weighted_f
            
            aspect_results[asp] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gold_count": gold_cnt,
                "pred_count": pred_cnt,
                "precision": prec,
                "recall": rec,
                "f_beta": f_score,
                "weight": weight,
                "weighted_f_beta": weighted_f,
            }
            
        micro_p_denom = total_tp + total_fp
        micro_r_denom = total_tp + total_fn
        micro_prec = float(total_tp / micro_p_denom) if micro_p_denom > 0 else 0.0
        micro_rec = float(total_tp / micro_r_denom) if micro_r_denom > 0 else 0.0
        micro_f = compute_fbeta(micro_prec, micro_rec, beta=beta)
        
        category_results[cat] = {
            "score": weighted_score,  # Official frequency-weighted aspect score
            "micro_f_beta": micro_f,
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "gold_count": total_gold,
            "pred_count": total_pred,
            "aspects": aspect_results,
        }
        
    # Macro-average over categories
    if category_results:
        final_official_score = sum(res["score"] for res in category_results.values()) / len(category_results)
    else:
        final_official_score = 0.0
        
    return {
        "metric_type": "official_hierarchical_f02",
        "beta": beta,
        "official_score": final_official_score,
        "categories": category_results,
    }


def calculate_global_micro_metrics(
    records: List[Dict[str, Any]],
    beta: float = BETA,
) -> Dict[str, Any]:
    """
    Calculate unweighted global micro F-beta score.
    Pools all predictions and gold entities across all categories and aspects into a single TP, FP, FN.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gold = 0
    total_pred = 0
    
    cat_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "pred": 0})
    
    for rec in records:
        cat = rec["category"]
        preds = rec["predicted_entities"]
        golds = rec["gold_entities"]
        
        tp, fp, fn = count_entity_matches(preds, golds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gold += len(golds)
        total_pred += len(preds)
        
        cat_counts[cat]["tp"] += tp
        cat_counts[cat]["fp"] += fp
        cat_counts[cat]["fn"] += fn
        cat_counts[cat]["gold"] += len(golds)
        cat_counts[cat]["pred"] += len(preds)
        
    p_denom = total_tp + total_fp
    r_denom = total_tp + total_fn
    prec = float(total_tp / p_denom) if p_denom > 0 else 0.0
    rec = float(total_tp / r_denom) if r_denom > 0 else 0.0
    f_score = compute_fbeta(prec, rec, beta=beta)
    
    cat_results = {}
    for cat in sorted(cat_counts.keys()):
        c = cat_counts[cat]
        c_p = float(c["tp"] / (c["tp"] + c["fp"])) if (c["tp"] + c["fp"]) > 0 else 0.0
        c_r = float(c["tp"] / (c["tp"] + c["fn"])) if (c["tp"] + c["fn"]) > 0 else 0.0
        cat_results[cat] = {
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "gold_count": c["gold"],
            "pred_count": c["pred"],
            "precision": c_p,
            "recall": c_r,
            "f_beta": compute_fbeta(c_p, c_r, beta=beta),
        }
        
    return {
        "metric_type": "global_micro_f02",
        "beta": beta,
        "global_f_beta": f_score,
        "global_precision": prec,
        "global_recall": rec,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "gold_count": total_gold,
        "pred_count": total_pred,
        "categories": cat_results,
    }
