"""
XLM-RoBERTa + Linear-Chain CRF for Structured Aspect Extraction / NER.

The CRF operates strictly over original competition-token positions,
skipping continuation subwords and special tokens (where label == -100).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class CRFOutput:
    loss: Optional[torch.Tensor] = None
    emissions: Optional[torch.Tensor] = None
    crf_mask: Optional[torch.Tensor] = None
    decoded: Optional[List[List[int]]] = None


class LinearChainCRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF) supporting:
    - Trainable transition matrix (num_labels x num_labels)
    - Trainable start transitions (num_labels)
    - Trainable end transitions (num_labels)
    - Negative log-likelihood sequence loss with mask
    - Viterbi decoding with mask
    """

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

        # transitions[i, j] = score of transitioning FROM state i TO state j
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))
        # start_transitions[j] = score of transitioning from START to state j
        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        # end_transitions[i] = score of transitioning from state i to END
        self.end_transitions = nn.Parameter(torch.empty(num_labels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _forward_alg(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-partition function log(Z) using the forward algorithm.
        emissions: [batch_size, seq_len, num_labels]
        mask: [batch_size, seq_len] (bool)
        Returns: [batch_size] log-sum-exp over all valid paths.
        """
        batch_size, seq_len, num_labels = emissions.shape

        # Step 0: start transition + emission at timestep 0
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [B, N]

        # Transitions: [1, N, N], representing from_state i -> to_state j
        trans = self.transitions.unsqueeze(0)

        for t in range(1, seq_len):
            # alpha: [B, N, 1] (from_state i)
            # trans: [1, N, N] (from_state i -> to_state j)
            # emit:  [B, 1, N] (to_state j)
            emit = emissions[:, t].unsqueeze(1)
            score = alpha.unsqueeze(2) + trans + emit  # [B, N, N]

            # Sum over from_state i
            next_alpha = torch.logsumexp(score, dim=1)  # [B, N]

            # If timestep t is masked out, keep previous alpha
            mask_t = mask[:, t].unsqueeze(1)  # [B, 1]
            alpha = torch.where(mask_t, next_alpha, alpha)

        # Terminal transitions
        terminal_score = alpha + self.end_transitions.unsqueeze(0)  # [B, N]
        return torch.logsumexp(terminal_score, dim=1)  # [B]

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score for gold label sequences.
        emissions: [batch_size, seq_len, num_labels]
        tags: [batch_size, seq_len] (long)
        mask: [batch_size, seq_len] (bool)
        Returns: [batch_size]
        """
        batch_size, seq_len, _ = emissions.shape
        batch_indices = torch.arange(batch_size, device=emissions.device)

        # Start transition + emission at timestep 0
        first_tags = tags[:, 0]
        score = self.start_transitions[first_tags] + emissions[batch_indices, 0, first_tags]

        for t in range(1, seq_len):
            prev_tags = tags[:, t - 1]
            curr_tags = tags[:, t]

            trans_score = self.transitions[prev_tags, curr_tags]
            emit_score = emissions[batch_indices, t, curr_tags]
            step_score = trans_score + emit_score

            mask_t = mask[:, t].float()
            score = score + step_score * mask_t

        # End transition for the last valid token of each sequence
        seq_lengths = mask.long().sum(dim=1)
        last_indices = (seq_lengths - 1).clamp(min=0)
        last_tags = tags.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]

        return score

    def loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss: -(gold_score - log(Z)).
        """
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_alg(emissions, mask)
        log_likelihood = gold_score - forward_score

        seq_lengths = mask.long().sum(dim=1)
        valid_seqs = seq_lengths > 0
        if valid_seqs.sum() > 0:
            return -log_likelihood[valid_seqs].mean()
        return -log_likelihood.mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """
        Viterbi decoding to find highest-scoring label sequences.
        Returns: List of decoded label sequences (one list of ints per batch item).
        """
        batch_size, seq_len, num_labels = emissions.shape
        seq_lengths = mask.long().sum(dim=1).tolist()

        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [B, N]
        backpointers = []

        trans = self.transitions.unsqueeze(0)  # [1, N, N]

        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)  # [B, 1, N]
            score = viterbi.unsqueeze(2) + trans + emit  # [B, N, N]

            max_score, max_indices = score.max(dim=1)  # [B, N], [B, N]
            backpointers.append(max_indices)

            mask_t = mask[:, t].unsqueeze(1)
            viterbi = torch.where(mask_t, max_score, viterbi)

        # Backtrack optimal paths for each batch item
        decoded_batch: List[List[int]] = []

        for b in range(batch_size):
            length = seq_lengths[b]
            if length == 0:
                decoded_batch.append([])
                continue

            # viterbi[b] at end of loop contains scores at step (length - 1)
            terminal_scores = viterbi[b] + self.end_transitions
            best_last_tag = int(terminal_scores.argmax().item())

            best_path = [best_last_tag]
            for t in range(length - 1, 0, -1):
                best_tag = backpointers[t - 1][b, best_path[-1]].item()
                best_path.append(best_tag)

            best_path.reverse()
            decoded_batch.append(best_path)

        return decoded_batch


class XLMRobertaCRF(nn.Module):
    """
    XLM-RoBERTa + Linear-Chain CRF.

    Pulls out hidden states for active competition tokens (labels != -100),
    passes them through dropout and a linear classifier to produce emissions,
    and applies a linear-chain CRF over the original token sequence.
    """

    def __init__(
        self,
        label2id: dict,
        id2label: dict,
        model_name: str = "FacebookAI/xlm-roberta-base",
        classifier_dropout: float = 0.1,
    ):
        super().__init__()
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(label2id)
        self.model_name = model_name

        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.crf = LinearChainCRF(self.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        return_decoded: bool = False,
    ) -> CRFOutput:
        # 1. Full subword representations from XLM-R
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state  # [B, S, H]
        batch_size, _, hidden_size = sequence_output.shape
        device = sequence_output.device

        # 2. Identify active original-token positions (labels != -100 or provided active_mask)
        if active_mask is None:
            if labels is not None:
                active_mask = labels != -100
            else:
                active_mask = attention_mask == 1

        active_counts = active_mask.sum(dim=1)
        max_active_len = int(active_counts.max().item())
        if max_active_len == 0:
            max_active_len = 1

        # 3. Gather active hidden states in original-token order
        padded_hidden = torch.zeros(
            batch_size,
            max_active_len,
            hidden_size,
            device=device,
            dtype=sequence_output.dtype,
        )
        padded_labels = torch.zeros(
            batch_size,
            max_active_len,
            device=device,
            dtype=torch.long,
        )
        crf_mask = torch.zeros(
            batch_size,
            max_active_len,
            device=device,
            dtype=torch.bool,
        )

        for b in range(batch_size):
            active_indices = active_mask[b].nonzero(as_tuple=False).squeeze(-1)
            k = len(active_indices)
            if k > 0:
                padded_hidden[b, :k] = sequence_output[b, active_indices]
                if labels is not None:
                    padded_labels[b, :k] = labels[b, active_indices]
                crf_mask[b, :k] = True

        # 4. Emissions
        emissions = self.classifier(self.dropout(padded_hidden))

        # 5. CRF loss
        loss = None
        if labels is not None:
            loss = self.crf.loss(emissions, padded_labels, crf_mask)

        # 6. Viterbi decode if requested
        decoded = None
        if return_decoded:
            decoded = self.crf.decode(emissions, crf_mask)

        return CRFOutput(
            loss=loss,
            emissions=emissions,
            crf_mask=crf_mask,
            decoded=decoded,
        )

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Convenience method for Viterbi inference."""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            active_mask=active_mask,
            return_decoded=True,
        )
        return outputs.decoded

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Saves the complete model:
        1. XLM-R backbone via standard transformers save_pretrained
        2. Classifier and CRF parameters + label mappings via crf_head.pt
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save transformer backbone
        self.roberta.save_pretrained(save_dir)

        # Save custom head, CRF parameters, label mappings
        custom_weights = {
            "classifier": self.classifier.state_dict(),
            "crf": self.crf.state_dict(),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "num_labels": self.num_labels,
            "model_name": self.model_name,
        }
        torch.save(custom_weights, save_dir / "crf_head.pt")

    @classmethod
    def from_pretrained(
        cls,
        save_directory: Union[str, Path],
        **kwargs,
    ) -> "XLMRobertaCRF":
        """
        Loads the complete model from checkpoint directory.
        """
        save_dir = Path(save_directory)
        custom_weights_path = save_dir / "crf_head.pt"
        if not custom_weights_path.exists():
            raise FileNotFoundError(
                f"CRF head weights not found at: {custom_weights_path}"
            )

        custom_weights = torch.load(custom_weights_path, map_location="cpu")
        label2id = custom_weights["label2id"]
        id2label = custom_weights["id2label"]

        # Initialize with backbone weights loaded from save_dir
        model = cls(
            label2id=label2id,
            id2label=id2label,
            model_name=str(save_dir),
            **kwargs,
        )
        model.classifier.load_state_dict(custom_weights["classifier"])
        model.crf.load_state_dict(custom_weights["crf"])
        return model
