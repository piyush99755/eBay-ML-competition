"""
Deterministic Span Decoder for Phase 50 Boundary-Aware Span Learning.

Implements:
1. Probabilistic scoring over candidate spans
2. Non-overlapping greedy selection sorted by confidence
3. Deterministic tie-breaking: (-score, -width, start_tok, end_tok, aspect)
4. Exact whitespace-token span reconstruction
5. Official competition category filtering
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from src.evaluation.entity_metrics import filter_entities_by_category
from src.modeling.span_dataset import ID2ASPECT, LABEL_NONE


def decode_spans_for_title(
    candidate_spans_meta: List[Dict[str, Any]],
    span_probs: np.ndarray,
    tokens: List[str],
    category: int,
    conf_thresh: float = 0.50,
    aspect_thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Deterministically decode non-overlapping entity spans for a single title.
    
    Arguments:
        candidate_spans_meta: list of candidate span metadata dicts
        span_probs: [N_spans, num_classes] softmax probabilities
        tokens: original whitespace tokens
        category: eBay competition category (1 or 2)
        conf_thresh: default confidence threshold
        aspect_thresholds: optional aspect-specific thresholds
        
    Returns:
        List of decoded entity dicts with aspect, value, start, end, score.
    """
    if len(candidate_spans_meta) == 0:
        return []

    if aspect_thresholds is None:
        aspect_thresholds = {}

    predicted_candidates = []

    for i, meta in enumerate(candidate_spans_meta):
        probs = span_probs[i]
        p_none = probs[LABEL_NONE]

        # Best non-NONE aspect class
        non_none_probs = probs.copy()
        non_none_probs[LABEL_NONE] = -1.0
        best_class = int(np.argmax(non_none_probs))
        score = float(probs[best_class])

        aspect = ID2ASPECT.get(best_class, "NONE")
        if aspect == "NONE":
            continue

        thresh = aspect_thresholds.get(aspect, conf_thresh)

        # Gating condition: must exceed threshold and beat NONE
        if score >= thresh and score > p_none:
            s_tok = meta["start_tok"]
            e_tok = meta["end_tok"]
            width = meta["width"]
            val = " ".join(tokens[s_tok : e_tok + 1])

            predicted_candidates.append({
                "start": s_tok,
                "end": e_tok + 1,  # exclusive end to match gold format
                "start_tok": s_tok,
                "end_tok": e_tok,
                "width": width,
                "aspect": aspect,
                "value": val,
                "score": score,
            })

    if not predicted_candidates:
        return []

    # Deterministic sorting: highest score first, then widest span, then earliest start
    predicted_candidates.sort(
        key=lambda c: (-c["score"], -c["width"], c["start_tok"], c["end_tok"], c["aspect"])
    )

    # Greedy non-overlapping span selection
    occupied_tokens = set()
    accepted_entities = []

    for cand in predicted_candidates:
        span_toks = set(range(cand["start_tok"], cand["end_tok"] + 1))
        if not span_toks.intersection(occupied_tokens):
            # No overlap -> accept
            occupied_tokens.update(span_toks)
            accepted_entities.append(cand)

    # Sort final accepted entities by start token
    accepted_entities.sort(key=lambda c: (c["start"], c["end"]))

    # Apply official category filtering
    entity_tuples = [(e["aspect"], e["value"]) for e in accepted_entities]
    valid_tuples = set(filter_entities_by_category(entity_tuples, category))

    filtered_entities = [e for e in accepted_entities if (e["aspect"], e["value"]) in valid_tuples]
    return filtered_entities
