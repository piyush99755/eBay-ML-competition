# Phase 33A — Alternative Machine Learning Architectures & Score Improvement Research

**Project**: eBay 2026 University Machine Learning Competition  
**Date**: October 27, 2026  
**Status**: Research & Feasibility Analysis Complete (No Model Training Conducted)  
**Baseline Git Checkpoint**: `5fcb07b2bb952b37edd2eb105642c85f440132a3`  
**Current Free Disk Space**: 7.08 GB (Drive C:, strictly maintaining $\ge 4.0$ GB floor)  
**Primary Recommendation**: **PROCEED TO PHASE 33B — DOMAIN-ADAPTIVE PRETRAINING (DAPT)**

---

## 1. Executive Summary

Phase 33A investigates four alternative machine learning architectures to determine whether a scientifically defensible, computationally feasible approach can surpass our frozen Experiment 10 + Phase 26B MIN 0.60 champion on unseen competition data.

### Frozen Champion Baseline
- **Models & Weights**: GBERT seed 42 (0.40) + XLM-R seed 42 (0.20) + XLM-R seed 7 (0.20) + XLM-R seed 21 (0.20).
- **Post-Processing**: Method A category-aware filtering + entity minimum-token confidence threshold $\ge 0.60$.
- **Official EvalAI Quiz $F_{0.2}$**: **0.901245410919943** (Rank 13).
- **Local Split-A Holdout $F_{0.2}$**: **0.899802** (Precision = 0.917246, Recall = 0.834980, TP = 6,340, FP = 572, FN = 1,253).
- **Frozen Pipeline**: `src/submission/predict_test_champion.py`.

### Key Research Findings
1. **Span-Based NER (Approach A)** is theoretically capable of avoiding BIO sequence alignment errors, but enumerating candidate spans on German listing titles creates a severe class imbalance ($>95\%$ negative spans). Under the competition's $F_{0.2}$ metric, where precision is penalized $25\times$ relative to recall ($\beta^{-2} = 25$), even a tiny false-alarm rate among candidate spans devastates the score. Furthermore, span-level logits cannot be linearly averaged with the existing token probability distributions of our champion ensemble.
2. **Transformer + CRF (Approach B)** fails to address the empirical root causes of vehicle-model extraction errors. In Phase 23A and 31A, audits revealed that $<0.3\%$ of model errors are illegal BIO transitions. Vehicle-model boundary errors (e.g., adjacent models merged into one continuous entity) occur between valid BIO states (`B-Modell` vs `I-Modell`). Because a linear-chain CRF transition matrix is context-independent, it transitions `B-Modell \to I-Modell` with high probability and cannot resolve token-level lexical boundaries.
3. **Hard-Example-Aware Training (Approach C)** is an engineering-sound, low-risk data-centric method that increases exposure to multi-model and multi-token titles during training. While safe and fast, it risks shifting the model's prior towards predicting multiple entities, requiring delicate calibration to avoid precision degradation.
4. **Domain-Adaptive Pretraining (Approach D)** is our **strongest and highest-leverage recommendation**. In Phase 32B, we proved that **80.5%** of vehicle-model tokens split into multiple subwords under `deepset/gbert-base`. Because GBERT was pretrained on standard German Wikipedia and news articles, its representations of telegraphic automotive listing titles, part abbreviations, and model trims are severely under-contextualized. Continued Masked Language Modeling (MLM) on 50,000–100,000 unlabeled listing titles from the 2,000,000 raw competition pool directly adapts the token representations to automotive terminology without distorting class priors, retaining 100% architectural and ensemble drop-in compatibility.

---

## 2. Review of Prior Experimental Failures & Scientific Baseline

To prevent repeating previously rejected hypotheses, we audit the progression of experiments leading to Phase 33A:

```
+--------------------------------------------------------------------------------------------------+
| EXPERIMENTAL TIMELINE & LESSONS LEARNED                                                          |
+--------------------------------------------------------------------------------------------------+
| Phase 23A: Viterbi BIO Constraint Decoding                                                       |
|   - Finding: Enforcing legal BIO transition rules yielded zero improvement.                      |
|   - Root Cause: Base transformers already produce 99.7% structurally valid BIO transitions.       |
|                                                                                                  |
| Phase 26B: Entity Minimum-Token Confidence Calibration                                           |
|   - Finding: Filtering entities with min token confidence < 0.60 lifted Quiz F0.2 from 0.8824   |
|     to 0.901245 (+0.0188 gain, Rank 13).                                                         |
|   - Root Cause: Precision penalty (25x) means pruning low-confidence false positives is vital.   |
|                                                                                                  |
| Phase 27A: Majority Entity Voting vs Confidence Filtering                                        |
|   - Finding: Majority entity voting dropped F0.2 to 0.8920 (worse than raw ensemble).           |
|   - Root Cause: Hard binary votes destroy nuanced token confidence; linear probability averaging |
|     is mathematically superior.                                                                  |
|                                                                                                  |
| Phase 31A: Boundary-Aware Heuristic Post-Decoding                                                |
|   - Finding: Heuristic splitting of vehicle-model entities created 251 false positives, causing  |
|     F0.2 to plunge to 0.8715.                                                                    |
|   - Root Cause: Post-hoc rule heuristics cannot distinguish multi-token models from adjacent      |
|     models without contextual language understanding.                                            |
|                                                                                                  |
| Phase 32A/32B: Label Structure & Subword Fragmentation Audit                                     |
|   - Finding: 80.5% of vehicle-model tokens undergo multi-subword fragmentation. Adjacent models   |
|     exhibit identical surface structures to compound trims.                                      |
|                                                                                                  |
| Phase 32C: Controlled I-Modell Loss Weighting (1.15x)                                            |
|   - Finding: GBERT internal standalone score rose slightly (0.8676 vs 0.8669), but when placed   |
|     in the 4-model MIN 0.60 ensemble, it dropped (0.8965 vs 0.8972, Delta = -0.000635).          |
|   - Root Cause: Artificially boosting loss on subword continuation inflated confidence on        |
|     marginal boundary tokens, introducing 18 net false positives.                                |
+--------------------------------------------------------------------------------------------------+
```

### Scientific Takeaway
Methods that artificially modify decision thresholds or apply post-hoc heuristic rules consistently degrade $F_{0.2}$ due to the harsh $25\times$ precision penalty. The only interventions that successfully improve $F_{0.2}$ are:
1. Enhancing the underlying contextual representations of the encoder.
2. Sharpening the probabilistic separation between true entities and background noise.

---

## 3. Approach A: Span-Based Named Entity Recognition

### Conceptual Architecture
In span-based NER, the model classifies candidate token spans $(i, j)$ directly into one of the 29 aspect classes or $\text{NONE}$:
$$\mathbf{h}_{span}(i, j) = [\mathbf{h}_i; \, \mathbf{h}_j; \, \mathbf{e}_{length}(j - i + 1)]$$
$$\hat{y}_{i,j} = \text{Softmax}(W_{span} \mathbf{h}_{span}(i, j) + b)$$
where $\mathbf{h}_i, \mathbf{h}_j$ are the start and end subword contextual embeddings from GBERT, and $\mathbf{e}_{length}$ is a learned span-length embedding. Non-overlapping spans are resolved via Dynamic Programming or Greedy Non-Maximum Suppression (NMS).

### Feasibility & Failure Analysis
1. **Candidate Span Explosion & Negative Imbalance**:
   - For a title of $T$ tokens and maximum span length $L_{max} = 8$:
     $$S_{candidate} \approx T \cdot L_{max} - \frac{L_{max}(L_{max}-1)}{2}$$
   - For an average listing title ($T = 20$), there are $\approx 132$ candidate spans.
   - For longer titles ($T = 40$), there are $\approx 292$ candidate spans.
   - In ground truth, a title contains an average of $6.4$ true entities.
   - Consequently, **$>95\%$ of all candidate spans are true negatives ($\text{NONE}$)**.
2. **Catastrophic Precision Risk under $F_{0.2}$**:
   - The competition metric is $F_{0.2} = \frac{1.04 \cdot P \cdot R}{0.04 \cdot P + R}$. Each false positive incurs a $25\times$ heavier penalty than a false negative.
   - If a span classifier achieves $98.5\%$ negative specificity, it will still generate $\approx 2.0$ false positive spans per title.
   - In our frozen champion, the total false positive rate across all 29 aspects is only $0.572$ false positives per title (572 FP / 1,000 titles).
   - An increase of just 1.0 false positive per title drops precision from $0.917$ to $\approx 0.79$, collapsing $F_{0.2}$ from $0.90$ to $<0.82$.
3. **GPU Memory Footprint on 6 GB VRAM**:
   - Representing $B \times S$ candidate spans requires $(4 \times 150 \times 1536 \times 4) \approx 3.7 \text{ MB}$ for span embeddings, but caching intermediate backprop activations through start/end pooling and MLP heads requires $\approx 4.8\text{--}5.5\text{ GB}$ VRAM at batch size 4.
   - Training requires gradient accumulation with batch size 2, increasing training time to $>30$ minutes on a GTX 1660 Ti.
4. **Ensemble Incompatibility**:
   - Our champion ensemble relies on linear probability fusion of token BIO distributions:
     $$P_{fused}(t, c) = 0.40 \cdot P_{GBERT}(t, c) + 0.60 \cdot \sum_{k=1}^3 \frac{1}{3} P_{XLMR, k}(t, c)$$
   - A span model outputs span-level logits $S(i, j, a)$. It cannot be mathematically fused into the token probability stream of the 3 XLM-R models.

**Verdict on Approach A**: **REJECTED** due to extreme false positive risk under $F_{0.2}$, massive implementation overhead (~700 lines custom PyTorch), and ensemble incompatibility.

---

## 4. Approach B: Transformer + CRF

### Conceptual Architecture
A Linear-Chain Conditional Random Field (CRF) layer replaces the per-token independent classification head:
$$\mathcal{L}_{CRF}(\mathbf{x}, \mathbf{y}) = - \left( \sum_{t=1}^T \phi(x_t, y_t) + \sum_{t=1}^{T-1} A_{y_t, y_{t+1}} - \log Z(\mathbf{x}) \right)$$
where $A \in \mathbb{R}^{48 \times 48}$ is a trainable label transition matrix, and $Z(\mathbf{x})$ is the partition function computed via the Forward algorithm. Decoding is performed via the Viterbi algorithm.

### Feasibility & Failure Analysis
1. **The Transition Matrix Does Not Solve Lexical Boundaries**:
   - A CRF transition matrix $A_{i,j}$ models label-to-label co-occurrence $P(y_t \mid y_{t-1})$. It is **spatially static and context-independent**.
   - For vehicle models, the legal transition is $B\text{-Modell} \to I\text{-Modell}$. The CRF assigns this transition a high positive weight.
   - When presented with two adjacent vehicle models (e.g. `[Golf] [Passat]`), the CRF transition matrix sees `B-Modell` followed by a candidate vehicle model token and *favors* continuing the entity ($I\text{-Modell}$) rather than breaking it ($B\text{-Modell}$).
   - The decision to break or continue a vehicle model depends strictly on whether the token "Passat" is a new model or a trim suffix. This is a **lexical, contextual semantic decision**, not a sequence transition rule!
2. **Empirical Evidence from Phase 23A and 31A**:
   - In Phase 23A, structural Viterbi decoding produced a $+0.000000$ change in $F_{0.2}$.
   - Less than $0.3\%$ of errors in `checkpoints/gbert_baseline_best` involve illegal BIO transitions. The transformer self-attention layers already model sequential dependencies almost flawlessly.
3. **Loss of Fine-Grained Token Confidence Calibration**:
   - Viterbi decoding produces a discrete 1-best sequence $\mathbf{y}^*$.
   - To obtain token confidence for our Phase 26B MIN 0.60 filter, one must run the Forward-Backward algorithm to compute posterior marginals:
     $$P(y_t = k \mid \mathbf{x}) = \frac{\alpha_t(k) \beta_t(k)}{Z(\mathbf{x})}$$
   - These marginals smooth probability distributions across the sequence, reducing the sharpness of the boundary tokens and interfering with the optimal 0.60 threshold.

**Verdict on Approach B**: **REJECTED** because it addresses a non-existent bottleneck ($<0.3\%$ illegal transitions) and cannot resolve the semantic vehicle-model boundary problem.

---

## 5. Approach C: Hard-Example-Aware Training

### Conceptual Architecture
Rather than altering the model architecture, Approach C adjusts the training distribution to focus gradient updates on structurally complex listing titles:
1. **Multi-Model Titles**: Titles containing $\ge 2$ distinct vehicle models (compatibility lists).
2. **Multi-Token Titles**: Titles containing vehicle models of length $\ge 2$ tokens.
3. **Adjacent Boundary Titles**: Titles where an entity boundary occurs with zero intermediate $O$ tokens.

### Feasibility & Mechanism Analysis
1. **Distribution Imbalance in Training Data**:
   - In the 4,000 Split-A training titles, simple single-token models account for $>80\%$ of all vehicle-model occurrences.
   - Titles with $\ge 2$ vehicle models account for only $\approx 14\%$ of titles, but generate $>52\%$ of vehicle-model false positives.
   - In uniform random sampling, gradients from simple titles dominate, washing out the subtle cues required to detect adjacent boundaries.
2. **Implementation Simplicity**:
   - Can be implemented via a custom `WeightedRandomSampler` or targeted oversampling in `src/modeling/train_model.py` (~120 lines of code).
   - Reuses 100% of existing tokenization, BIO alignment, loss functions, and inference code.
3. **Risks under $F_{0.2}$**:
   - Phase 32C showed that artificially boosting $I\text{-Modell}$ loss weighting by 1.15x caused a slight drop in ensemble precision ($0.8972 \to 0.8965$).
   - Oversampling multi-model titles shifts the model's prior belief: it begins predicting multiple vehicle models where only one exists, generating false positives on simple titles.
   - However, unlike scalar loss weighting, whole-title resampling preserves natural token co-occurrence and grammatical context.

**Verdict on Approach C**: **VIABLE AS STRONG SECONDARY ALTERNATIVE (Rank 2)**, but carries moderate risk of prior distortion under $F_{0.2}$.

---

## 6. Approach D: Domain-Adaptive Pretraining (DAPT)

### Conceptual Architecture
Domain-Adaptive Pretraining (Gururangan et al., ACL 2020) continues self-supervised Masked Language Modeling (MLM) on domain-specific text before downstream supervised fine-tuning:

```
+-------------------------------------------------------------------------------+
| DOMAIN-ADAPTIVE PRETRAINING (DAPT) PIPELINE                                   |
+-------------------------------------------------------------------------------+
|                                                                               |
|  Unlabeled Automotive Corpus: data/raw/Listing_Titles.tsv.gz (~2,000,000)     |
|                                |                                              |
|                                v                                              |
|  Sample: 50,000 Unlabeled Listing Titles (Cleaned, Category-Stratified)       |
|                                |                                              |
|                                v                                              |
|  Pretrained deepset/gbert-base (Vocab: 31,102, Hidden: 768)                   |
|                                |                                              |
|                                v                                              |
|  AutoModelForMaskedLM: 15% Dynamic Token Masking (Whole-Word / Subword)       |
|  Epochs: 2 | Batch Size: 32 | Seq Length: 64 | LR: 5e-5 | AdamW               |
|                                |                                              |
|                                v                                              |
|  Domain-Adapted Base Checkpoint: checkpoints/gbert_dapt_base (417 MB)         |
|                                |                                              |
|                                v                                              |
|  Supervised Token Classification Fine-Tuning: 3,200 Internal Train Titles     |
|  Evaluated on 800 Internal Dev Titles against Matched Baseline                |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### Why DAPT Directly Solves the Vehicle-Model Extraction Mechanism
1. **Resolving the Subword Fragmentation Problem**:
   - In Phase 32B, we proved that **80.5%** of vehicle-model tokens split into multiple subwords under `deepset/gbert-base`.
   - `deepset/gbert-base` was trained on Wikipedia articles and news articles. It has never seen telegraphic German automotive titles, such as:
     `"Zahnriemensatz Wasserpumpe CT1028K1 für VW Golf IV 1.9 TDI Variant"`
   - Consequently, the base model assigns generic, non-automotive semantic representations to subwords like `["Pas", "##sat"]`, `["Tou", "##areg"]`, `["Sport", "##back"]`, and `["Quat", "##tro"]`.
   - Continued MLM on automotive listing titles forces the transformer attention heads to learn contextual correlations between automotive part categories (`Zahnriemensatz`, `Bremsscheiben`), vehicle brands (`VW`, `Audi`, `BMW`), and vehicle models (`Golf`, `Passat`, `A4`).
2. **Sharper Probability Distributions without Prior Distortion**:
   - Unlike loss-weighting (Phase 32C) or oversampling (Approach C), DAPT **does not alter the supervised label space or class priors**.
   - It improves the *representational quality* of the hidden states $\mathbf{h}_t$.
   - Sharper representations produce more confident, well-calibrated logits on true entities, which directly enhances the Phase 26B MIN 0.60 confidence filter.
3. **100% Architectural and Ensemble Compatibility**:
   - The fine-tuned DAPT model has the exact same architecture, weights, tokenizer, and output dimensions as `checkpoints/gbert_baseline_best`.
   - It drops directly into the 0.40 branch of the champion ensemble without requiring any modifications to `src/submission/predict_test_champion.py`.

### Resource & Feasibility Audit on GTX 1660 Ti (6 GB VRAM)
- **VRAM Utilization**: At sequence length 64 (titles average ~20 tokens) and batch size 32, `AutoModelForMaskedLM` with vocabulary projection (31,102 classes) consumes $\approx 3.6\text{--}4.1\text{ GB}$ VRAM. This safely fits within 6.0 GB.
- **Pretraining Speed**: On 50,000 listing titles, at ~110 titles/second, 2 epochs require $\approx 15\text{--}20$ minutes of GPU time.
- **Supervised Fine-Tuning Speed**: 3 epochs on 3,200 internal train titles takes $\approx 7$ minutes.
- **Total Experiment Runtime**: $\approx 25\text{--}30$ minutes.
- **Disk Space Footprint**:
  - DAPT base checkpoint: ~417 MB.
  - Fine-tuned checkpoint: ~417 MB.
  - Intermediate cache: ~50 MB.
  - Total temporary disk: $\approx 884\text{ MB}$.
  - Current free disk space: **7.08 GB**. Free space remains $>6.15\text{ GB}$, well above the strict 4.0 GB floor.

---

## 7. Comparative Decision Matrix

| Evaluation Criterion | Approach A: Span-Based NER | Approach B: Transformer + CRF | Approach C: Hard-Example Training | Approach D: Domain-Adaptive Pretraining (DAPT) |
| :--- | :---: | :---: | :---: | :---: |
| **Addresses Vehicle-Model Boundary Errors** | Moderate (High in theory, but noisy) | Low (Static transitions do not resolve words) | Moderate/High (Increases multi-model exposure) | **High** (Resolves subword fragmentation & domain context) |
| **Risk of Increasing False Positives** | **Catastrophic** (>95% negative span pool) | Low/Moderate | Moderate (Prior shift towards multi-model) | **Low** (Does not alter label priors) |
| **Implementation Complexity** | High (~700 lines custom PyTorch) | Moderate (~300 lines) | **Lowest** (~120 lines) | Moderate (~220 lines standard HF MLM) |
| **GPU Memory Requirement (6 GB VRAM)** | 4.8–5.5 GB (Borderline) | 3.0 GB (Safe) | **2.2 GB** (Very Safe) | 3.8 GB (Safe) |
| **Estimated Training Duration** | 25–35 min | 15–20 min | **6–8 min** | 25–30 min |
| **Temporary Disk Requirement** | ~450 MB | ~420 MB | **~420 MB** | ~884 MB (Well within 7.08 GB free) |
| **Compatibility with Champion Ensemble** | **Incompatible** (Span logits vs token probs) | Incompatible (Viterbi vs linear average) | **100% Drop-in** (Standard token logits) | **100% Drop-in** (Standard token logits) |
| **Fair Evaluation vs Matched Baseline** | Difficult (Different output structures) | Moderate | **Clean** (Matched 3,200/800 split) | **Clean** (Matched 3,200/800 split) |
| **Scientific Novelty in Competition** | Moderate | Low (Redundant with Phase 23A/31A) | Moderate | **High** (Exploits 2M unlabeled listing corpus) |
| **Feasibility Before Nov 1** | Poor | Moderate | **Very High** | **High** |
| **Overall Rank** | **Rank 4 (Rejected)** | **Rank 3 (Rejected)** | **Rank 2 (Viable Secondary)** | **Rank 1 (Selected Champion)** |

---

## 8. Selected Phase 33B Experiment Proposal: Domain-Adaptive Pretraining (DAPT)

### 1. Exact Experimental Hypothesis
Continuing self-supervised Masked Language Modeling on a focused corpus of 50,000 unlabeled German automotive listing titles will adapt GBERT's subword contextual embeddings to automotive shorthand, part numbers, and compound model names. When fine-tuned for token classification on the internal 3,200-title partition, the DAPT-initialized model will achieve higher exact-match vehicle-model entity precision and surpass the matched baseline's internal dev $F_{0.2} = 0.866940$, without inflating false positives under the Phase 26B MIN 0.60 confidence filter.

### 2. Error Mechanism Directly Addressed
- **Subword Fragmentation**: 80.5% of vehicle-model tokens are split into subwords. DAPT contextualizes subword co-occurrences in automotive strings.
- **Model vs Trim Ambiguity**: General GBERT cannot distinguish whether a token like "Variant" or "Cabriolet" is a product line, trim, or model. DAPT exposes the transformer to thousands of automotive listing co-occurrences.
- **Confidence Calibration**: Clean domain representations eliminate epistemic uncertainty, yielding higher confidence on genuine entities and sharper drops on non-entities.

---

## 9. Proposed Model Architecture & Pipeline

1. **Stage 1: Domain-Adaptive Masked Language Modeling**:
   - **Base Model**: `deepset/gbert-base` (local Hugging Face weights).
   - **Training Corpus**: 50,000 candidate listing titles sampled deterministically from `data/raw/Listing_Titles.tsv.gz` (disjoint from any official evaluation sets).
   - **Task**: `AutoModelForMaskedLM` with standard 15% dynamic masking probability (80% `[MASK]`, 10% random token, 10% unchanged).
   - **Hyperparameters**: Learning rate $5\text{e-}5$, AdamW optimizer, linear warmup over 500 steps, batch size 32, max sequence length 64, epochs 2.
   - **Output Checkpoint**: `checkpoints/gbert_dapt_base`.

2. **Stage 2: Supervised Token Classification Fine-Tuning**:
   - **Architecture**: `AutoModelForTokenClassification` loaded from `checkpoints/gbert_dapt_base`.
   - **Label Space**: Exact 48 BIO classes identical to the frozen champion.
   - **Training Partition**: Exact 3,200 internal train titles from `data/cache/internal_split_3200_800.json`.
   - **Hyperparameters**: Identical to Phase 32C matched baseline: LR $2\text{e-}5$, batch size 4 (train) / 8 (val), epochs 3, seed 42.
   - **Evaluation Gate**: Evaluated on 800 internal dev titles using `src/evaluation/phase_32c_evaluate_internal.py`.

---

## 10. Matched Baseline Design (3,200/800 Internal Split)

To ensure scientific rigor and prevent any data leakage, Phase 33B will use the exact matched baseline established in Phase 32C:
- **Baseline Model**: Standard `deepset/gbert-base` fine-tuned on the 3,200 internal train titles (`checkpoints/gbert_phase32c_baseline_best`).
- **Dev Gate Benchmark**:
  - Baseline Internal Dev $F_{0.2}$: **0.866940**
  - Baseline Category 1 $F_{0.2}$: **0.871032**
  - Baseline Category 2 $F_{0.2}$: **0.862413**
  - Baseline Vehicle-Model TP: 647, FP: 147, FN: 254 (Precision: 0.814862, Recall: 0.718091)
- **Phase 33B Decision Rule**:
  - The DAPT challenger must achieve Internal Dev $F_{0.2} > 0.866940$.
  - Vehicle-Model precision must not degrade below 0.814862.
  - If both criteria pass, the challenger model is evaluated inside the 4-model ensemble on the 1,000-title Split-A validation holdout to compare against the champion ensemble ($F_{0.2} = 0.899802$).

---

## 11. Resource Estimates

| Resource Dimension | Requirement | Available Margin | Safety Assessment |
| :--- | :---: | :---: | :--- |
| **GPU VRAM** | 3.8 GB peak (MLM batch 32) | 6.0 GB (GTX 1660 Ti) | **PASS** — 2.2 GB safety margin |
| **Drive C: Free Space** | ~884 MB temporary footprint | 7.08 GB free space | **PASS** — Free space remains $>6.15$ GB ($>4.0$ GB floor) |
| **Stage 1 MLM Runtime** | ~18 minutes | Expected Nov 1 release | **PASS** — Fast execution |
| **Stage 2 Fine-Tuning** | ~7 minutes | Expected Nov 1 release | **PASS** — Fast execution |
| **Total Experiment Time** | ~25–30 minutes | Expected Nov 1 release | **PASS** — Fits well before Nov 1 |
| **Package Dependencies** | Existing `transformers`, `torch` | 100% installed | **PASS** — Zero external installations required |

---

## 12. Scientific Limitations & Potential Failure Modes

1. **Risk of Catastrophic Forgetting**:
   - If MLM is run for too many epochs or with too high a learning rate, the model may forget general German syntax.
   - *Mitigation*: Limit pretraining to exactly 2 epochs with a conservative learning rate of $5\text{e-}5$ and linear decay.
2. **Subword Vocabulary Invariance**:
   - Continued MLM updates token embeddings and transformer weights, but does *not* add new tokens to the 31,102-word vocabulary. Highly fragmented words still consist of multiple subwords.
   - *Mitigation*: Transformer self-attention layers will learn contextualized multi-subword representations, mitigating the fragmentation effect without altering the token index mapping.
3. **Domain Noise in Listing Titles**:
   - Listing titles contain spelling typos, boilerplate terms ("TOP ZUSTAND", "VERSANDKOSTENFREI"), and noise.
   - *Mitigation*: Apply light heuristic title cleaning (stripping non-ASCII emojis and excessive whitespace) prior to MLM.

---

## 13. Expected Precision & Recall Risks under $F_{0.2}$

- Under $F_{0.2}$, the recall-precision trade-off is asymmetric ($25\times$ precision weight):
  $$\Delta F_{0.2} \approx \frac{\partial F_{0.2}}{\partial P} \Delta P + \frac{\partial F_{0.2}}{\partial R} \Delta R \quad \text{where} \quad \frac{\partial F_{0.2}}{\partial P} \approx 25 \cdot \frac{\partial F_{0.2}}{\partial R}$$
- Gaining 50 true positives (recall boost) while adding just 10 false positives results in a **net drop** in $F_{0.2}$!
- DAPT preserves precision because it sharpens the model's true positive representations rather than penalizing background tokens or inflating entity priors.
- Stop Condition: If the DAPT model increases false positives by $>5$ on the internal dev set, it will be immediately rejected regardless of recall gains.

---

## 14. Go/No-Go Recommendation & Stop Conditions

### Formal Recommendation:
**PROCEED TO PHASE 33B — DOMAIN-ADAPTIVE PRETRAINING (DAPT)**

### Stop Conditions for Phase 33B Implementation:
1. **Free Disk Space Floor**: If free space on Drive C: falls below 4.5 GB at any point during pretraining, training must abort immediately.
2. **Internal Dev Gate**: The DAPT model must strictly surpass the matched baseline $F_{0.2} = 0.866940$. If internal dev $F_{0.2} \le 0.866940$, terminate without ensemble testing.
3. **Vehicle-Model Precision Gate**: Vehicle-model precision must be $\ge 0.8148$. If false positives increase, retain the frozen champion.
4. **No Test/Quiz Data**: Training and model selection must use zero Test or Quiz data.
5. **No Production Mutation**: All frozen champion checkpoints (`checkpoints/gbert_baseline_best`, `checkpoints/xlmr_*_best`) and the production script `src/submission/predict_test_champion.py` must remain completely untouched.

---

## 15. Git Commit Hash & Traceability

- **Pre-Phase 33A Git Checkpoint**: `5fcb07b2bb952b37edd2eb105642c85f440132a3`
- **Audit Verification Scripts**:
  - `src/submission/validate_final_submission.py` (Independent submission validator)
  - `src/submission/merge_test_chunks.py` (Safe chunk merger & recovery)
  - `tests/test_final_submission_validator.py` (15 passing synthetic unit tests)
- **Phase 33A Deliverable Document**: `docs/phase_33a_alternative_ml_architecture_research.md`
