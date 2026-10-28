"""
Diagnostic script to audit Phase 26A/27A vs Phase 29B scorer implementations
and reproduce canonical champion metrics.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

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
)
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.experiment_26a_confidence_calibration import (
    bio_to_entities_with_confidence,
    collect_active_probabilities,
)

CHAMPION_CHECKPOINTS = {
    "gbert": Path("checkpoints/gbert_baseline_best"),
    "xlmr42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr21": Path("checkpoints/xlmr_seed_21_best"),
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Label Mapping from Reference Champion
    ref_cfg = AutoConfig.from_pretrained(CHAMPION_CHECKPOINTS["gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # 2. Load Split A Validation Set
    df = load_training_data()
    all_examples = build_examples(df)
    train_examples, val_examples = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(train_examples)} train examples, {len(val_examples)} val examples.")

    # 3. Collect probabilities
    probs = {}
    for name, path in CHAMPION_CHECKPOINTS.items():
        print(f"Extracting {name} from {path}...")
        probs[name] = collect_active_probabilities(path, val_examples, label2id, device, batch_size=8)

    num_ex = len(val_examples)
    p_fused = [
        0.40 * probs["gbert"][i] + 0.20 * probs["xlmr42"][i] + 0.20 * probs["xlmr7"][i] + 0.20 * probs["xlmr21"][i]
        for i in range(num_ex)
    ]

    # Evaluate across thresholds
    for thresh in [0.0, 0.60]:
        records = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_preds = 0

        # Method A: filter category first, then filter confidence
        for ex_idx, ex in enumerate(val_examples):
            tokens = ex["tokens"]
            cat_id = ex["category"]
            true_bio = ex["bio_tags"]

            p = p_fused[ex_idx]
            pred_ids = np.argmax(p, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]
            selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

            true_entities = bio_to_entities(tokens, true_bio)
            raw_conf_entities = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

            # Option A: category filter first, then min confidence
            cat_filtered_conf = []
            for asp, val, mean_c, min_c in raw_conf_entities:
                if filter_entities_by_category([(asp, val)], cat_id):
                    cat_filtered_conf.append((asp, val, mean_c, min_c))

            thresh_entities = [(asp, val) for asp, val, mean_c, min_c in cat_filtered_conf if min_c >= thresh]

            tp, fp, fn = count_matches(thresh_entities, true_entities)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_preds += len(thresh_entities)

            records.append({
                "category": cat_id,
                "true_entities": true_entities,
                "predicted_entities": thresh_entities,
            })

        comp_metrics = calculate_competition_metrics(records)
        pr = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=0.2)
        print(f"\n--- Threshold {thresh:.2f} ---")
        print(f"Official F0.2: {comp_metrics['final_score']:.6f}")
        print(f"Category 1:    {comp_metrics['categories'][1]['score']:.6f}")
        print(f"Category 2:    {comp_metrics['categories'][2]['score']:.6f}")
        print(f"Micro Prec:    {pr['precision']:.4f} ({total_tp} / {total_tp + total_fp})")
        print(f"Micro Rec:     {pr['recall']:.4f} ({total_tp} / {total_tp + total_fn})")
        print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, Total Preds: {total_preds}")

if __name__ == "__main__":
    main()
