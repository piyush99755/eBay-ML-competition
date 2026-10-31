"""
Span Dataset and Candidate Span Builder for Phase 50 Boundary-Aware Span NER.

Builds whitespace-token candidate spans (start, end) up to W_max:
- Training: Positives (all gold entities) + Boundary Negatives + Background Negatives
- Inference: All candidate spans up to W_max (complete coverage)
- Subword mapping: Maps whitespace token bounds to encoder subwords
"""

from collections import defaultdict
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.data.mine_hard_examples import bio_to_spanned_entities

# 29 Competition aspects
ASPECT_LIST = [
    "Anwendung",
    "Anzahl_Der_Einheiten",
    "Besonderheiten",
    "Breite",
    "Bremsscheiben-Aussendurchmesser",
    "Bremsscheibenart",
    "Einbauposition",
    "Farbe",
    "Größe",
    "Hersteller",
    "Herstellernummer",
    "Herstellungsland_Und_-Region",
    "Im_Lieferumfang_Enthalten",
    "Kompatible_Fahrzeug_Marke",
    "Kompatibles_Fahrzeug_Jahr",
    "Kompatibles_Fahrzeug_Modell",
    "Länge",
    "Material",
    "Maßeinheit",
    "Menge",
    "Modell",
    "Oberflächenbeschaffenheit",
    "Oe/Oem_Referenznummer(N)",
    "Produktart",
    "Produktlinie",
    "SAE_Viskosität",
    "Stärke",
    "Technologie",
    "Zähnezahl",
]

# Label mappings (Class 0 = NONE)
LABEL_NONE = 0
LABEL_NONE_STR = "NONE"

ASPECT2ID = {asp: idx + 1 for idx, asp in enumerate(sorted(ASPECT_LIST))}
ASPECT2ID[LABEL_NONE_STR] = LABEL_NONE

ID2ASPECT = {v: k for k, v in ASPECT2ID.items()}
NUM_CLASSES = len(ASPECT2ID)  # 30


def build_candidate_spans(
    tokens: List[str],
    gold_spans: List[Dict[str, Any]],
    token_to_subwords: Dict[int, List[int]],
    max_width: int = 8,
    is_training: bool = True,
    rng: Optional[random.Random] = None,
    neg_to_pos_ratio: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate candidate spans for a single title.
    
    Arguments:
        tokens: whitespace tokens
        gold_spans: list of dicts with 'aspect', 'start', 'end' (token indices)
        token_to_subwords: mapping from token index to subword indices
        max_width: maximum span width in whitespace tokens
        is_training: if True, applies negative sampling; if False, returns all spans
        rng: random generator for negative sampling
        neg_to_pos_ratio: ratio of negative spans to positive spans
        
    Returns:
        List of candidate span dicts.
    """
    n_tokens = len(tokens)
    if rng is None:
        rng = random.Random(42)

    # Gold spans indexed by (start_tok, end_tok) inclusive
    gold_map = {}
    for g in gold_spans:
        s_tok = g["start"]
        e_tok = g["end"] - 1  # convert to inclusive token index
        gold_map[(s_tok, e_tok)] = g["aspect"]

    all_candidate_spans = []
    positive_spans = []
    hard_boundary_negatives = []
    background_negatives = []

    # Identify all possible candidate spans up to max_width
    for s_tok in range(n_tokens):
        for e_tok in range(s_tok, min(n_tokens, s_tok + max_width)):
            width = e_tok - s_tok + 1
            if s_tok not in token_to_subwords or e_tok not in token_to_subwords:
                continue

            sub_start = token_to_subwords[s_tok][0]
            sub_end = token_to_subwords[e_tok][-1]

            span_info = {
                "start_tok": s_tok,
                "end_tok": e_tok,
                "width": width,
                "sub_start": sub_start,
                "sub_end": sub_end,
            }

            if (s_tok, e_tok) in gold_map:
                aspect = gold_map[(s_tok, e_tok)]
                label_id = ASPECT2ID.get(aspect, LABEL_NONE)
                span_info["label"] = label_id
                span_info["aspect"] = aspect
                positive_spans.append(span_info)
            else:
                span_info["label"] = LABEL_NONE
                span_info["aspect"] = LABEL_NONE_STR

                # Check if boundary adjacent to any gold span
                is_boundary = False
                for g_s, g_e in gold_map.keys():
                    if s_tok == g_s or e_tok == g_e:
                        is_boundary = True
                        break
                    if max(s_tok, g_s) <= min(e_tok, g_e):
                        is_boundary = True
                        break

                if is_boundary:
                    hard_boundary_negatives.append(span_info)
                else:
                    background_negatives.append(span_info)

    if not is_training:
        # For evaluation: return ALL candidate spans
        return positive_spans + hard_boundary_negatives + background_negatives

    # Training negative sampling policy:
    # 1. 100% of positive gold spans
    selected_spans = list(positive_spans)

    # 2. All hard boundary negatives (up to a reasonable cap)
    target_negatives = max(10, neg_to_pos_ratio * max(1, len(positive_spans)))

    if len(hard_boundary_negatives) <= target_negatives:
        selected_spans.extend(hard_boundary_negatives)
        remaining_slots = target_negatives - len(hard_boundary_negatives)
        if remaining_slots > 0 and len(background_negatives) > 0:
            k = min(remaining_slots, len(background_negatives))
            selected_spans.extend(rng.sample(background_negatives, k=k))
    else:
        # Sample from hard boundary negatives
        selected_spans.extend(rng.sample(hard_boundary_negatives, k=target_negatives))

    return selected_spans


class SpanNERDataset(Dataset):
    """PyTorch Dataset for Span NER with precomputed in-memory caching for maximal throughput."""

    def __init__(
        self,
        examples: List[dict],
        tokenizer: PreTrainedTokenizerFast,
        max_width: int = 8,
        is_training: bool = True,
        seed: int = 42,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_width = max_width
        self.is_training = is_training
        self.seed = seed

        # Precompute in-memory representations once
        self.cached_items = []
        for idx, ex in enumerate(examples):
            tokens = ex["tokens"]
            gold_spans = bio_to_spanned_entities(tokens, ex["bio_tags"]) if "bio_tags" in ex else []

            enc = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=128,
                return_tensors=None,
            )

            word_ids = enc.word_ids()
            token_to_subwords = defaultdict(list)
            for sub_idx, w_id in enumerate(word_ids):
                if w_id is not None:
                    token_to_subwords[w_id].append(sub_idx)

            rng = random.Random(self.seed + idx) if self.is_training else None

            candidates = build_candidate_spans(
                tokens=tokens,
                gold_spans=gold_spans,
                token_to_subwords=token_to_subwords,
                max_width=self.max_width,
                is_training=self.is_training,
                rng=rng,
            )

            self.cached_items.append({
                "record_number": ex["record_number"],
                "category": ex["category"],
                "tokens": tokens,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                "candidates": candidates,
                "bio_tags": ex.get("bio_tags", []),
            })

    def __len__(self) -> int:
        return len(self.cached_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.cached_items[idx]


class SpanCollateFn:
    """Collate function that vectorizes tokens and span indices for batched GPU execution."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)

        # 1. Pad token sequences
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        token_type_ids = [torch.tensor(ex["token_type_ids"], dtype=torch.long) for ex in batch]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0)
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)

        # 2. Flatten and vectorize candidate spans
        batch_indices = []
        sub_starts = []
        sub_ends = []
        widths = []
        labels = []

        # Metadata for decoding/evaluation
        span_meta = []

        for b_idx, ex in enumerate(batch):
            for cand in ex["candidates"]:
                batch_indices.append(b_idx)
                sub_starts.append(cand["sub_start"])
                sub_ends.append(cand["sub_end"])
                widths.append(cand["width"])
                labels.append(cand["label"])

                span_meta.append({
                    "batch_idx": b_idx,
                    "record_number": ex["record_number"],
                    "category": ex["category"],
                    "start_tok": cand["start_tok"],
                    "end_tok": cand["end_tok"],
                    "width": cand["width"],
                    "label": cand["label"],
                    "aspect": cand["aspect"],
                })

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "token_type_ids": padded_token_type_ids,
            "span_batch_idx": torch.tensor(batch_indices, dtype=torch.long),
            "span_sub_start": torch.tensor(sub_starts, dtype=torch.long),
            "span_sub_end": torch.tensor(sub_ends, dtype=torch.long),
            "span_width": torch.tensor(widths, dtype=torch.long),
            "span_labels": torch.tensor(labels, dtype=torch.long),
            "span_meta": span_meta,
            "batch_examples": batch,
        }
