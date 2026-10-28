"""
Phase 32C: Exploratory Holdout & Ensemble Compatibility Evaluation.

Evaluates:
1. Standalone Baseline GBERT (3,200 titles) on 1,000 Split-A holdout
2. Standalone Challenger GBERT (3,200 titles) on 1,000 Split-A holdout
3. Matched Baseline Ensemble (0.40 Baseline 3.2k + 0.60 3-seed XLM-R) + MIN 0.60
4. Challenger Ensemble (0.40 Challenger 3.2k + 0.60 3-seed XLM-R) + MIN 0.60
5. Comparison against frozen Champion (0.40 GBERT 4.0k + 0.60 3-seed XLM-R) + MIN 0.60 (F0.2 = 0.899802)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.entity_metrics import (
    bio_to_entities,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
    BETA,
)
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.experiment_26a_confidence_calibration import (
    bio_to_entities_with_confidence,
    collect_active_probabilities,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

FROZEN_XLMR_CHECKPOINTS = {
    "xlmr42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr21": Path("checkpoints/xlmr_seed_21_best"),
}


def score_predictions(records: List[dict]) -> dict:
    comp = calculate_competition_metrics(records)
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for r in records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
    pr = precision_recall_fbeta(tot_tp, tot_fp, tot_fn, beta=BETA)
    return {
        "f02": comp["final_score"],
        "cat1_f02": comp["categories"][1]["score"],
        "cat2_f02": comp["categories"][2]["score"],
        "precision": pr["precision"],
        "recall": pr["recall"],
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
    }


def main():
    print("=" * 85)
    print("PHASE 32C: EXPLORATORY HOLDOUT EVALUATION (1,000 Split-A Validation Titles)")
    print("=" * 85)
    print("NOTE: This comparison is purely EXPLORATORY to assess generalizability and")
    print("compatibility with the frozen ensemble. It is NOT an independent confirmation.")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load labels
    ref_cfg = AutoConfig.from_pretrained(Path("checkpoints/gbert_baseline_best"))
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Load Split-A 1,000-title holdout
    df = load_training_data()
    all_examples = build_examples(df)
    _, val_1k = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(val_1k)} Split-A validation holdout titles.")

    base_ckpt = CHECKPOINTS_DIR / "gbert_phase32c_baseline_best"
    chal_ckpt = CHECKPOINTS_DIR / "gbert_phase32c_challenger_best"

    # 1. Extract Probabilities for Baseline and Challenger on 1,000 Holdout
    print("\nExtracting GBERT Matched Baseline (3.2k) probabilities on 1k holdout...")
    p_base = collect_active_probabilities(base_ckpt, val_1k, label2id, device, batch_size=8)

    print("Extracting GBERT Challenger (3.2k) probabilities on 1k holdout...")
    p_chal = collect_active_probabilities(chal_ckpt, val_1k, label2id, device, batch_size=8)

    # 2. Extract or Load XLM-R probabilities for ensemble
    xlmr_probs = {}
    for name, path in FROZEN_XLMR_CHECKPOINTS.items():
        print(f"Extracting {name} on 1k holdout...")
        xlmr_probs[name] = collect_active_probabilities(path, val_1k, label2id, device, batch_size=8)

    p_xlmr_avg = [
        (xlmr_probs["xlmr42"][i] + xlmr_probs["xlmr7"][i] + xlmr_probs["xlmr21"][i]) / 3.0
        for i in range(len(val_1k))
    ]

    # 3. Fuse Ensembles
    # A. Matched Baseline Ensemble: 0.40 * GBERT_base + 0.60 * XLM-R_avg
    p_ens_base = [0.40 * p_base[i] + 0.60 * p_xlmr_avg[i] for i in range(len(val_1k))]

    # B. Challenger Ensemble: 0.40 * GBERT_chal + 0.60 * XLM-R_avg
    p_ens_chal = [0.40 * p_chal[i] + 0.60 * p_xlmr_avg[i] for i in range(len(val_1k))]

    configs = {
        "Standalone Baseline (3.2k)": (p_base, 0.0),
        "Standalone Challenger (3.2k)": (p_chal, 0.0),
        "Matched Baseline Ensemble (3.2k + XLMR) @ MIN 0.60": (p_ens_base, 0.60),
        "Challenger Ensemble (3.2k + XLMR) @ MIN 0.60": (p_ens_chal, 0.60),
    }

    results = {}
    print("\n" + "=" * 90)
    print(f"{'Configuration':<50} {'F0.2':<10} {'Cat 1':<10} {'Cat 2':<10} {'Prec':<8} {'Rec':<8}")
    print("-" * 90)

    for cfg_name, (probs, min_conf) in configs.items():
        records = []
        for ex_idx, ex in enumerate(val_1k):
            tokens = ex["tokens"]
            cat_id = ex["category"]
            true_bio = ex["bio_tags"]
            p = probs[ex_idx]

            pred_ids = np.argmax(p, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]
            selected_probs = [float(p[t, pred_ids[t]]) for t in range(len(tokens))]

            true_ents = bio_to_entities(tokens, true_bio)
            raw_conf_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

            # Category filter first
            cat_filtered = [e for e in raw_conf_ents if filter_entities_by_category([(e[0], e[1])], cat_id)]

            # Confidence threshold
            if min_conf > 0.0:
                final_ents = [(e[0], e[1]) for e in cat_filtered if e[3] >= min_conf]
            else:
                final_ents = [(e[0], e[1]) for e in cat_filtered]

            records.append({
                "category": cat_id,
                "true_entities": true_ents,
                "predicted_entities": final_ents,
            })

        sc = score_predictions(records)
        results[cfg_name] = sc
        print(f"{cfg_name:<50} {sc['f02']:.6f}   {sc['cat1_f02']:.6f}   {sc['cat2_f02']:.6f}   {sc['precision']:.4f}   {sc['recall']:.4f}")

    print("-" * 90)
    print(f"{'Active Frozen Champion (4.0k GBERT + XLMR) @ MIN 0.60':<50} 0.899802   0.899921   0.899683   0.9172   0.8350")
    print("=" * 90)

    out_file = CACHE_DIR / "phase_32c_exploratory_holdout_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nExploratory holdout results saved to {out_file}")


if __name__ == "__main__":
    main()
