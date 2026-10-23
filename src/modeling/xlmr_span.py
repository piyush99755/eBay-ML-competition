"""
Span-Based XLM-RoBERTa Model with Auxiliary Boundary Supervision (Experiment 16).

Architecture:
- XLMRobertaModel backbone
- Dropout
- Hidden states gathered strictly at original competition-token positions (labels != -100)
- Auxiliary START head (hidden_size -> 2)
- Auxiliary END head (hidden_size -> 2)
- Candidate span representation: [start_hidden ; end_hidden ; mean_span_hidden] (3 * hidden_size)
- Span classifier: Linear(3 * hidden_size, num_span_classes)
  where class 0 = NONE, 1..N = semantic aspect types

Multi-task training loss:
  total_loss = span_loss + 0.20 * start_loss + 0.20 * end_loss
"""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.modeling.xlmr_boundary import generate_boundary_targets_from_ids


MAX_SPAN_WIDTH = 5


@dataclass
class SpanOutput:
    loss: Optional[torch.Tensor] = None
    span_loss: Optional[torch.Tensor] = None
    start_loss: Optional[torch.Tensor] = None
    end_loss: Optional[torch.Tensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None


def create_span_mappings(label2id: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Derive semantic aspect span classes from existing BIO label mappings.

    Class 0 = NONE
    Class 1..N = semantic aspect types (sorted alphabetically)
    """
    aspects = sorted(list(set(tag[2:] for tag in label2id if tag != "O")))
    span2id = {"NONE": 0}
    for i, asp in enumerate(aspects, start=1):
        span2id[asp] = i
    id2span = {v: k for k, v in span2id.items()}
    return span2id, id2span


def bio_to_spans(bio_tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert a sequence of BIO tags into exact entity spans:
    [(start_idx, end_idx, aspect_name), ...]
    where start_idx and end_idx are 0-indexed inclusive.

    Follows the exact entity semantics of src/evaluation/entity_metrics.py:
    - O closes current entity.
    - B-X closes current entity and starts a new X entity.
    - I-X continues current entity if it is X, otherwise closes and starts new X entity.
    - End of sequence closes any open entity.
    """
    spans: List[Tuple[int, int, str]] = []
    current_aspect: Optional[str] = None
    current_start: Optional[int] = None

    for i, tag in enumerate(bio_tags):
        if tag == "O":
            if current_aspect is not None and current_start is not None:
                spans.append((current_start, i - 1, current_aspect))
                current_aspect, current_start = None, None
            continue

        if tag.startswith("B-"):
            if current_aspect is not None and current_start is not None:
                spans.append((current_start, i - 1, current_aspect))
            current_aspect = tag[2:]
            current_start = i
            continue

        if tag.startswith("I-"):
            aspect = tag[2:]
            if current_aspect == aspect:
                # Normal continuation
                pass
            else:
                # Invalid start or aspect transition: start new entity
                if current_aspect is not None and current_start is not None:
                    spans.append((current_start, i - 1, current_aspect))
                current_aspect = aspect
                current_start = i
            continue

        raise ValueError(f"Unknown BIO tag: {tag}")

    if current_aspect is not None and current_start is not None:
        spans.append((current_start, len(bio_tags) - 1, current_aspect))

    return spans


def spans_to_bio(
    spans: List[Tuple[int, int, str]],
    seq_len: int,
) -> List[str]:
    """
    Convert a list of non-overlapping entity spans back into a BIO sequence of length seq_len.
    """
    bio_tags = ["O"] * seq_len
    for start, end, aspect in spans:
        if 0 <= start <= end < seq_len:
            bio_tags[start] = f"B-{aspect}"
            for i in range(start + 1, end + 1):
                bio_tags[i] = f"I-{aspect}"
    return bio_tags


def generate_candidate_spans(
    seq_len: int,
    max_span_width: int = MAX_SPAN_WIDTH,
) -> List[Tuple[int, int]]:
    """
    Generate all valid candidate spans (start, end) up to max_span_width.
    Both start and end are 0-indexed inclusive.
    """
    candidates = []
    for s in range(seq_len):
        max_e = min(seq_len, s + max_span_width)
        for e in range(s, max_e):
            candidates.append((s, e))
    return candidates


def resolve_overlapping_spans(
    candidate_spans: List[dict],
    seq_len: int,
) -> Tuple[List[dict], int]:
    """
    Deterministically resolve overlapping candidate spans.

    Each candidate dict contains:
      {'start': int, 'end': int, 'aspect': str, 'score': float}

    Algorithm:
    - Sort candidates descending by score (tie-break: longer span first, earlier start first).
    - Greedily retain candidate if it does not overlap any already-retained span.

    Returns:
    - retained_spans: list of non-overlapping candidate dicts sorted by start position.
    - rejected_count: number of non-NONE candidates rejected due to overlap.
    """
    # Sort descending by score
    sorted_candidates = sorted(
        candidate_spans,
        key=lambda c: (c["score"], c["end"] - c["start"] + 1, -c["start"]),
        reverse=True,
    )

    retained: List[dict] = []
    occupied = [False] * seq_len
    rejected_count = 0

    for cand in sorted_candidates:
        s, e = cand["start"], cand["end"]
        # Check if span overlaps any already-occupied position
        if any(occupied[i] for i in range(s, e + 1)):
            rejected_count += 1
            continue

        retained.append(cand)
        for i in range(s, e + 1):
            occupied[i] = True

    # Sort retained spans by start position
    retained.sort(key=lambda c: c["start"])
    return retained, rejected_count


def sample_training_spans_for_example(
    bio_tags: List[str],
    span2id: Dict[str, int],
    seed: int,
    ex_idx: int,
    max_span_width: int = MAX_SPAN_WIDTH,
    max_neg_ratio: int = 3,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Deterministic negative span sampling for a single training example.

    Returns:
    - spans: list of (start, end)
    - span_labels: list of class IDs (0 for NONE, 1..N for semantic aspect types)
    """
    seq_len = len(bio_tags)
    gold_spans = bio_to_spans(bio_tags)
    gold_map = {(s, e): asp for s, e, asp in gold_spans}

    # Positives within MAX_SPAN_WIDTH
    positives = [
        (s, e, asp) for s, e, asp in gold_spans
        if (e - s + 1) <= max_span_width and asp in span2id
    ]

    all_candidates = generate_candidate_spans(seq_len, max_span_width=max_span_width)
    negatives = [(s, e) for s, e in all_candidates if (s, e) not in gold_map]

    # Sample at most 3 negatives per positive; if 0 positives, sample up to 3
    num_pos = len(positives)
    target_neg = max(3, num_pos * max_neg_ratio) if num_pos > 0 else 3
    num_to_sample = min(len(negatives), target_neg)

    rng = random.Random(seed + ex_idx * 10007)
    sampled_negatives = rng.sample(negatives, num_to_sample) if num_to_sample < len(negatives) else list(negatives)

    spans: List[Tuple[int, int]] = []
    labels: List[int] = []

    for s, e, asp in positives:
        spans.append((s, e))
        labels.append(span2id[asp])

    for s, e in sampled_negatives:
        spans.append((s, e))
        labels.append(span2id["NONE"])

    return spans, labels


class XLMRobertaSpan(nn.Module):
    """
    Span-based XLM-RoBERTa for entity aspect classification.
    """

    def __init__(
        self,
        label2id: dict,
        id2label: dict,
        span2id: dict,
        id2span: dict,
        model_name: str = "FacebookAI/xlm-roberta-base",
        classifier_dropout: float = 0.1,
        max_span_width: int = MAX_SPAN_WIDTH,
        start_loss_weight: float = 0.20,
        end_loss_weight: float = 0.20,
        span_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.label2id = label2id
        self.id2label = id2label
        self.span2id = span2id
        self.id2span = id2span
        self.num_span_classes = len(span2id)
        self.model_name = model_name
        self.max_span_width = max_span_width
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_class_weights = span_class_weights

        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.dropout = nn.Dropout(classifier_dropout)

        # Auxiliary boundary heads (operating on single-token hidden states)
        self.start_classifier = nn.Linear(hidden_size, 2)
        self.end_classifier = nn.Linear(hidden_size, 2)

        # Span classifier: [start_hidden ; end_hidden ; mean_span_hidden]
        self.span_classifier = nn.Linear(3 * hidden_size, self.num_span_classes)

    def extract_span_representations(
        self,
        hidden: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct span representations using:
        [hidden[start] ; hidden[end] ; mean_span_hidden]
        via vectorized prefix sums.

        hidden: [T, H]
        starts: [N_spans]
        ends:   [N_spans]
        Returns: [N_spans, 3 * H]
        """
        T, H = hidden.shape
        start_hidden = hidden[starts]  # [N, H]
        end_hidden = hidden[ends]      # [N, H]

        # Vectorized mean pooling via cumsum
        zeros = torch.zeros(1, H, device=hidden.device, dtype=hidden.dtype)
        cumsum = torch.cat([zeros, torch.cumsum(hidden, dim=0)], dim=0)  # [T + 1, H]
        span_sums = cumsum[ends + 1] - cumsum[starts]  # [N, H]
        span_lens = (ends - starts + 1).unsqueeze(-1).to(hidden.dtype)  # [N, 1]
        mean_span_hidden = span_sums / span_lens  # [N, H]

        return torch.cat([start_hidden, end_hidden, mean_span_hidden], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        spans: Optional[List[torch.Tensor]] = None,
        span_labels: Optional[List[torch.Tensor]] = None,
        active_mask: Optional[torch.Tensor] = None,
        span_class_weights: Optional[torch.Tensor] = None,
    ) -> SpanOutput:
        # 1. Full subword representations from XLM-R backbone
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state  # [B, S, H]
        batch_size, _, hidden_size = sequence_output.shape
        device = sequence_output.device

        # 2. Identify active original-token positions
        if active_mask is None:
            if labels is not None:
                active_mask = labels != -100
            else:
                active_mask = attention_mask == 1

        active_counts = active_mask.sum(dim=1)
        max_active_len = int(active_counts.max().item())
        if max_active_len == 0:
            max_active_len = 1

        # 3. Gather active hidden states for boundary heads
        padded_hidden = torch.zeros(
            batch_size,
            max_active_len,
            hidden_size,
            device=device,
            dtype=sequence_output.dtype,
        )
        padded_start_labels = torch.full(
            (batch_size, max_active_len),
            -100,
            device=device,
            dtype=torch.long,
        )
        padded_end_labels = torch.full(
            (batch_size, max_active_len),
            -100,
            device=device,
            dtype=torch.long,
        )

        active_hidden_per_example = []

        for b in range(batch_size):
            active_indices = active_mask[b].nonzero(as_tuple=False).squeeze(-1)
            k = len(active_indices)
            if k > 0:
                h_b = sequence_output[b, active_indices]
                padded_hidden[b, :k] = h_b
                active_hidden_per_example.append(h_b)

                if labels is not None:
                    active_lids = labels[b, active_indices]
                    starts_b, ends_b = generate_boundary_targets_from_ids(
                        active_lids,
                        self.id2label,
                        device=device,
                    )
                    padded_start_labels[b, :k] = starts_b
                    padded_end_labels[b, :k] = ends_b
            else:
                active_hidden_per_example.append(
                    torch.zeros(0, hidden_size, device=device, dtype=sequence_output.dtype)
                )

        # 4. Boundary heads and losses
        feat_boundary = self.dropout(padded_hidden)
        start_logits = self.start_classifier(feat_boundary)
        end_logits = self.end_classifier(feat_boundary)

        start_loss = None
        end_loss = None
        span_loss = None
        total_loss = None

        if labels is not None:
            start_loss = F.cross_entropy(
                start_logits.view(-1, 2),
                padded_start_labels.view(-1),
                ignore_index=-100,
            )
            end_loss = F.cross_entropy(
                end_logits.view(-1, 2),
                padded_end_labels.view(-1),
                ignore_index=-100,
            )

        # 5. Span classification and loss (if spans provided)
        if spans is not None and span_labels is not None:
            span_repr_list = []
            span_label_list = []

            for b in range(batch_size):
                spans_b = spans[b]
                labels_b = span_labels[b]
                if len(spans_b) > 0 and len(active_hidden_per_example[b]) > 0:
                    h_b = active_hidden_per_example[b]
                    s_idx = spans_b[:, 0].to(device)
                    e_idx = spans_b[:, 1].to(device)
                    repr_b = self.extract_span_representations(h_b, s_idx, e_idx)
                    span_repr_list.append(repr_b)
                    span_label_list.append(labels_b.to(device))

            if len(span_repr_list) > 0:
                all_span_repr = torch.cat(span_repr_list, dim=0)
                all_span_labels = torch.cat(span_label_list, dim=0)

                span_logits = self.span_classifier(self.dropout(all_span_repr))

                weights = span_class_weights if span_class_weights is not None else self.span_class_weights
                if weights is not None and weights.device.type != device.type:
                    weights = weights.to(device)

                span_loss = F.cross_entropy(span_logits, all_span_labels, weight=weights)

                if start_loss is not None and end_loss is not None:
                    total_loss = (
                        span_loss
                        + self.start_loss_weight * start_loss
                        + self.end_loss_weight * end_loss
                    )
                else:
                    total_loss = span_loss

        return SpanOutput(
            loss=total_loss,
            span_loss=span_loss,
            start_loss=start_loss,
            end_loss=end_loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )

    def predict_spans(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[str]], dict]:
        """
        Run inference over a batch:
        1. Forward backbone and gather active token hidden states.
        2. Generate all candidate spans up to max_span_width.
        3. Classify spans and filter NONE predictions.
        4. Greedily resolve overlapping candidate spans.
        5. Reconstruct BIO sequences.

        Returns:
        - predicted_bio_batch: list of predicted BIO tag sequences.
        - diagnostics: dictionary of aggregate span diagnostics.
        """
        self.eval()
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        batch_size = sequence_output.shape[0]
        device = sequence_output.device

        if active_mask is None:
            if labels is not None:
                active_mask = labels != -100
            else:
                active_mask = attention_mask == 1

        batch_predicted_bio: List[List[str]] = []

        total_candidates_count = 0
        none_candidates_count = 0
        eligible_candidates_count = 0
        retained_candidates_count = 0
        rejected_candidates_count = 0

        with torch.no_grad():
            for b in range(batch_size):
                active_indices = active_mask[b].nonzero(as_tuple=False).squeeze(-1)
                T = len(active_indices)
                if T == 0:
                    batch_predicted_bio.append([])
                    continue

                hidden_b = sequence_output[b, active_indices]  # [T, H]

                candidates = generate_candidate_spans(T, max_span_width=self.max_span_width)
                total_candidates_count += len(candidates)

                if len(candidates) == 0:
                    batch_predicted_bio.append(["O"] * T)
                    continue

                starts_t = torch.tensor([c[0] for c in candidates], dtype=torch.long, device=device)
                ends_t = torch.tensor([c[1] for c in candidates], dtype=torch.long, device=device)

                span_repr = self.extract_span_representations(hidden_b, starts_t, ends_t)
                span_logits = self.span_classifier(span_repr)  # [N, num_classes]
                span_probs = F.softmax(span_logits, dim=-1)     # [N, num_classes]

                pred_classes = span_probs.argmax(dim=-1).cpu().tolist()
                max_probs = span_probs.max(dim=-1).values.cpu().tolist()

                eligible_spans = []
                for idx, (s, e) in enumerate(candidates):
                    cls_id = pred_classes[idx]
                    if cls_id == 0:  # NONE
                        none_candidates_count += 1
                        continue

                    aspect_name = self.id2span[cls_id]
                    prob = max_probs[idx]
                    eligible_spans.append({
                        "start": s,
                        "end": e,
                        "aspect": aspect_name,
                        "score": prob,
                    })

                eligible_candidates_count += len(eligible_spans)

                # Overlap resolution
                retained, rejected_count = resolve_overlapping_spans(eligible_spans, T)
                retained_candidates_count += len(retained)
                rejected_candidates_count += rejected_count

                # Convert non-overlapping spans back to BIO
                reconstructed_spans = [(r["start"], r["end"], r["aspect"]) for r in retained]
                pred_bio = spans_to_bio(reconstructed_spans, T)
                batch_predicted_bio.append(pred_bio)

        diagnostics = {
            "total_candidates": total_candidates_count,
            "none_candidates": none_candidates_count,
            "eligible_candidates": eligible_candidates_count,
            "retained_candidates": retained_candidates_count,
            "rejected_overlap_candidates": rejected_candidates_count,
        }

        return batch_predicted_bio, diagnostics

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Saves transformer backbone and span classification weights."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.roberta.save_pretrained(save_dir)

        custom_weights = {
            "span_classifier": self.span_classifier.state_dict(),
            "start_classifier": self.start_classifier.state_dict(),
            "end_classifier": self.end_classifier.state_dict(),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "span2id": self.span2id,
            "id2span": self.id2span,
            "num_span_classes": self.num_span_classes,
            "max_span_width": self.max_span_width,
            "model_name": self.model_name,
            "start_loss_weight": self.start_loss_weight,
            "end_loss_weight": self.end_loss_weight,
            "span_class_weights": self.span_class_weights.cpu() if self.span_class_weights is not None else None,
        }
        torch.save(custom_weights, save_dir / "span_heads.pt")

    @classmethod
    def from_pretrained(
        cls,
        save_directory: Union[str, Path],
        **kwargs,
    ) -> "XLMRobertaSpan":
        """Loads complete span model from checkpoint directory."""
        save_dir = Path(save_directory)
        weights_path = save_dir / "span_heads.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Span weights not found at: {weights_path}")

        custom_weights = torch.load(weights_path, map_location="cpu")
        label2id = custom_weights["label2id"]
        id2label = custom_weights["id2label"]
        span2id = custom_weights["span2id"]
        id2span = custom_weights["id2span"]
        max_span_width = custom_weights.get("max_span_width", MAX_SPAN_WIDTH)
        start_loss_weight = custom_weights.get("start_loss_weight", 0.20)
        end_loss_weight = custom_weights.get("end_loss_weight", 0.20)
        span_class_weights = custom_weights.get("span_class_weights", None)

        model = cls(
            label2id=label2id,
            id2label=id2label,
            span2id=span2id,
            id2span=id2span,
            model_name=str(save_dir),
            max_span_width=max_span_width,
            start_loss_weight=start_loss_weight,
            end_loss_weight=end_loss_weight,
            span_class_weights=span_class_weights,
            **kwargs,
        )
        model.span_classifier.load_state_dict(custom_weights["span_classifier"])
        model.start_classifier.load_state_dict(custom_weights["start_classifier"])
        model.end_classifier.load_state_dict(custom_weights["end_classifier"])
        return model
