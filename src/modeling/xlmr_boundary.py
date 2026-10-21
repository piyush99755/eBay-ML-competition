"""
Boundary-Aware Multi-Task XLM-RoBERTa Model (Experiment 14).

Architecture:
- XLMRobertaModel backbone
- Dropout
- Three prediction heads operating strictly on original competition-token representations:
  1. BIO classification head (hidden_size -> 48 labels)
  2. Entity-start head (hidden_size -> 2 classes: 0 = not start, 1 = entity start)
  3. Entity-end head (hidden_size -> 2 classes: 0 = not end, 1 = entity end)

Multi-task training loss:
  total_loss = bio_loss + 0.20 * start_loss + 0.20 * end_loss
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class BoundaryOutput:
    loss: Optional[torch.Tensor] = None
    bio_loss: Optional[torch.Tensor] = None
    start_loss: Optional[torch.Tensor] = None
    end_loss: Optional[torch.Tensor] = None
    bio_logits: Optional[torch.Tensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    active_mask: Optional[torch.Tensor] = None


def generate_boundary_targets(bio_tags: List[str]) -> Tuple[List[int], List[int]]:
    """
    Generate ground-truth START and END boundary targets from a sequence of BIO tags.

    START target:
    - 1 when BIO label begins with B-
    - 0 otherwise

    END target:
    - 1 if current token is part of an entity (B-X or I-X) AND the entity ends at this token.
      An entity ends when:
      - this is the last active token, OR
      - the next label is O, OR
      - the next label begins with B-, OR
      - the next label begins with I- and its entity type differs from current entity type
    - 0 otherwise
    """
    n = len(bio_tags)
    start_targets = [0] * n
    end_targets = [0] * n

    for i in range(n):
        tag = bio_tags[i]
        if tag == "O":
            continue

        prefix = tag[:2]
        entity_type = tag[2:]

        if prefix == "B-":
            start_targets[i] = 1

        is_end = False
        if i == n - 1:
            is_end = True
        else:
            next_tag = bio_tags[i + 1]
            if next_tag == "O":
                is_end = True
            elif next_tag.startswith("B-"):
                is_end = True
            elif next_tag.startswith("I-"):
                next_entity_type = next_tag[2:]
                if next_entity_type != entity_type:
                    is_end = True
            else:
                is_end = True

        if is_end:
            end_targets[i] = 1

    return start_targets, end_targets


def generate_boundary_targets_from_ids(
    label_ids: torch.Tensor,
    id2label: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a 1D tensor of active token label IDs into START and END target tensors.
    """
    bio_tags = [id2label[lid.item()] for lid in label_ids]
    start_targets, end_targets = generate_boundary_targets(bio_tags)
    start_tensor = torch.tensor(start_targets, dtype=torch.long, device=device)
    end_tensor = torch.tensor(end_targets, dtype=torch.long, device=device)
    return start_tensor, end_tensor


class XLMRobertaBoundary(nn.Module):
    """
    XLM-RoBERTa with auxiliary entity-boundary supervision.

    Pulls out hidden states for active competition tokens (labels != -100),
    passes them through dropout and three parallel classification heads:
    1. BIO head (48 classes)
    2. Entity-start head (2 classes)
    3. Entity-end head (2 classes)
    """

    def __init__(
        self,
        label2id: dict,
        id2label: dict,
        model_name: str = "FacebookAI/xlm-roberta-base",
        classifier_dropout: float = 0.1,
        start_loss_weight: float = 0.20,
        end_loss_weight: float = 0.20,
    ):
        super().__init__()
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(label2id)
        self.model_name = model_name
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight

        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.dropout = nn.Dropout(classifier_dropout)
        self.bio_classifier = nn.Linear(hidden_size, self.num_labels)
        self.start_classifier = nn.Linear(hidden_size, 2)
        self.end_classifier = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> BoundaryOutput:
        # 1. Full subword representations from XLM-R backbone
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state  # [B, S, H]
        batch_size, _, hidden_size = sequence_output.shape
        device = sequence_output.device

        # 2. Identify active original-token positions (labels != -100 or active_mask)
        if active_mask is None:
            if labels is not None:
                active_mask = labels != -100
            else:
                active_mask = attention_mask == 1

        active_counts = active_mask.sum(dim=1)
        max_active_len = int(active_counts.max().item())
        if max_active_len == 0:
            max_active_len = 1

        # 3. Gather active hidden states and labels in original-token order
        padded_hidden = torch.zeros(
            batch_size,
            max_active_len,
            hidden_size,
            device=device,
            dtype=sequence_output.dtype,
        )
        padded_bio_labels = torch.full(
            (batch_size, max_active_len),
            -100,
            device=device,
            dtype=torch.long,
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

        for b in range(batch_size):
            active_indices = active_mask[b].nonzero(as_tuple=False).squeeze(-1)
            k = len(active_indices)
            if k > 0:
                padded_hidden[b, :k] = sequence_output[b, active_indices]
                if labels is not None:
                    active_lids = labels[b, active_indices]
                    padded_bio_labels[b, :k] = active_lids
                    starts, ends = generate_boundary_targets_from_ids(
                        active_lids,
                        self.id2label,
                        device=device,
                    )
                    padded_start_labels[b, :k] = starts
                    padded_end_labels[b, :k] = ends

        # 4. Forward through dropout and all 3 heads
        feat = self.dropout(padded_hidden)
        bio_logits = self.bio_classifier(feat)        # [B, max_active_len, 48]
        start_logits = self.start_classifier(feat)    # [B, max_active_len, 2]
        end_logits = self.end_classifier(feat)        # [B, max_active_len, 2]

        # 5. Losses
        bio_loss = None
        start_loss = None
        end_loss = None
        total_loss = None

        if labels is not None:
            bio_loss = F.cross_entropy(
                bio_logits.view(-1, self.num_labels),
                padded_bio_labels.view(-1),
                ignore_index=-100,
            )
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
            total_loss = (
                bio_loss
                + self.start_loss_weight * start_loss
                + self.end_loss_weight * end_loss
            )

        return BoundaryOutput(
            loss=total_loss,
            bio_loss=bio_loss,
            start_loss=start_loss,
            end_loss=end_loss,
            bio_logits=bio_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            active_mask=active_mask,
        )

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Saves the complete model:
        1. XLM-R backbone via transformers save_pretrained
        2. BIO, START, and END classifier weights and config via boundary_heads.pt
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.roberta.save_pretrained(save_dir)

        custom_weights = {
            "bio_classifier": self.bio_classifier.state_dict(),
            "start_classifier": self.start_classifier.state_dict(),
            "end_classifier": self.end_classifier.state_dict(),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "num_labels": self.num_labels,
            "model_name": self.model_name,
            "start_loss_weight": self.start_loss_weight,
            "end_loss_weight": self.end_loss_weight,
        }
        torch.save(custom_weights, save_dir / "boundary_heads.pt")

    @classmethod
    def from_pretrained(
        cls,
        save_directory: Union[str, Path],
        **kwargs,
    ) -> "XLMRobertaBoundary":
        """
        Loads the complete model from checkpoint directory.
        """
        save_dir = Path(save_directory)
        custom_weights_path = save_dir / "boundary_heads.pt"
        if not custom_weights_path.exists():
            raise FileNotFoundError(
                f"Boundary head weights not found at: {custom_weights_path}"
            )

        custom_weights = torch.load(custom_weights_path, map_location="cpu")
        label2id = custom_weights["label2id"]
        id2label = custom_weights["id2label"]
        start_loss_weight = custom_weights.get("start_loss_weight", 0.20)
        end_loss_weight = custom_weights.get("end_loss_weight", 0.20)

        model = cls(
            label2id=label2id,
            id2label=id2label,
            model_name=str(save_dir),
            start_loss_weight=start_loss_weight,
            end_loss_weight=end_loss_weight,
            **kwargs,
        )
        model.bio_classifier.load_state_dict(custom_weights["bio_classifier"])
        model.start_classifier.load_state_dict(custom_weights["start_classifier"])
        model.end_classifier.load_state_dict(custom_weights["end_classifier"])
        return model
