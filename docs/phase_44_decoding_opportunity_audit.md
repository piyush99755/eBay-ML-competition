# Phase 44 — Token-Level Decoding Opportunity & Feasibility Audit Report

## Executive Summary

Phase 44 conducted an in-depth code-level audit and feasibility study investigating whether the token-level probabilities from our frozen four-model ensemble contain recoverable information that could improve exact-match hierarchical $F_{0.2}$ beyond the active official Quiz champion (**Phase 40**, official EvalAI Quiz $F_{0.2} = \mathbf{0.9233317116970503}$).

With the competition deadline approaching on **November 6, 2026**, this phase evaluated three specific post-hoc decoding paradigms:
1. **Approach A — BIO-Constrained Sequence Decoding (Viterbi / Transition Matrix)**
2. **Approach B — Alternative Span Candidate Ranking (N-Best Span Generation & Reranking)**
3. **Approach C — Contextual Aspect Disambiguation (Local Contextual Reweighting)**

### Core Conclusions:
1. **Approach A (BIO Constraints) has negligible headroom ($< +0.0002$):** Invalid BIO transitions account for only **13 tokens out of 5,563 total tokens ($0.23\%$)** in the fused ensemble. The existing production decoder already handles orphan `I-X` tags gracefully by promoting them to new entities. Hard sequence constraints provide no meaningful score improvement.
2. **Approach B (Alternative Span Ranking) is fundamentally bottlenecked by label multimodality:** 120 of the 165 False Positives ($72.7\%$) are boundary mismatches on genuine vehicle models or parts (e.g., `Vectra B` vs `Vectra`, `5er E39` vs `5er`). As demonstrated in Phase 42B, training data annotations are inherently multimodal; attempting to rank spans without validation leakage requires 20-run cross-validation OOF generation (30+ hours on 6GB VRAM) and risks severe test set regression.
3. **Approach C (Contextual Disambiguation) carries severe precision risk under $F_{0.2}$:** Only **42 False Negatives ($5.6\%$)** involve aspect confusion (primarily 16 kit confusions between `Produktart` and `Im_Lieferumfang_Enthalten`, and 4 part number confusions between `Herstellernummer` and `Oe/Oem_Referenznummer(N)`). Because $F_{0.2}$ penalizes false positives $25\times$ more heavily than recall, any heuristic or statistical aspect reassignment risks precision collapse on high-weight aspects (e.g., `Produktart` at $18.6\%$ weight).
4. **Strategic Recommendation:** We unambiguously conclude: **RETAIN PHASE 40 — NO CONVINCING DECODING OPPORTUNITY**. Modifying decoding behavior at this stage introduces high regression risks with minimal expectation of gain. Development effort should focus exclusively on final competition test submission readiness.

---

## 1. Champion Hash Verification

The official Phase 40 champion submission file was cryptographically verified:

| Property | File / Path | Expected SHA-256 | Actual SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Active Champion** | [`submissions/submission_quiz_phase40_lieferumfang080.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase40_lieferumfang080.tsv) | `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8` | `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8` | **VERIFIED / LOCKED** |
| **Official EvalAI Quiz Score** | EvalAI Competition Leaderboard | `0.9233317116970503` | `0.9233317116970503` | **CONFIRMED** |
| **Phase 36C Fallback** | [`submissions/submission_quiz_phase36c_produktart080.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase36c_produktart080.tsv) | `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87` | `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87` | **VERIFIED** |

All submission assets remain unaltered.

---

## 2. Baseline Reproduction

Using the frozen 489-title clean holdout ([`data/cache/phase_35b_clean_holdout_manifest.json`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/data/cache/phase_35b_clean_holdout_manifest.json)) and fused 4-model ensemble probabilities ($0.40$ DAPT GBERT + $0.20$ XLM-R seed 42 + $0.20$ XLM-R seed 7 + $0.20$ XLM-R seed 21), the Phase 40 local baseline was reproduced bit-for-bit:

| Metric | Target Baseline | Reproduced Baseline | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Official-Style Score** | `0.927771` | `0.927771` | $+0.000000$ | **Exact Match** |
| **Category 1 Score** | `0.918323` | `0.918323` | $+0.000000$ | **Exact Match** |
| **Category 2 Score** | `0.937219` | `0.937219` | $+0.000000$ | **Exact Match** |
| **Global Micro TP** | `3021` | `3021` | $0$ | **Exact Match** |
| **Global Micro FP** | `165` | `165` | $0$ | **Exact Match** |
| **Global Micro FN** | `752` | `752` | $0$ | **Exact Match** |
| **Global Micro $F_{0.2}$** | `0.941539` | `0.941539` | $+0.000000$ | **Exact Match** |

---

## 3. Current Production Decoder Architecture

We performed a line-by-line inspection of the production decoding codebase across [`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py), [`src/evaluation/entity_metrics.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/entity_metrics.py), and [`src/evaluation/phase_43_part_number_calibration.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_43_part_number_calibration.py).

### Detailed Architectural Answers:

1. **How the four models' token probabilities are aligned:**
   - Candidate listing titles are split into whitespace tokens (`tokens = title.split()`).
   - For each model, subword tokenization is performed with `is_split_into_words=True`.
   - In `CandidateTitleDataset`, `encoded.word_ids()` maps subwords to original whitespace token indices: `subword_map[w_idx] = s_idx`, selecting the **first subword** representing each word.
   - Model logits at these first subwords are softmaxed into probability vectors of shape `(num_tokens, 48)`.
   - Because the target sequence length is fixed to the title's original whitespace token count, all 4 models produce perfectly co-indexed probability matrices without index drift.
2. **How probabilities are fused:**
   - Fused probability is a convex linear combination:
     $$\mathbf{P}_{\text{fused}} = 0.40 \cdot \mathbf{P}_{\text{DAPT GBERT}} + 0.20 \cdot \mathbf{P}_{\text{XLM-R, 42}} + 0.20 \cdot \mathbf{P}_{\text{XLM-R, 7}} + 0.20 \cdot \mathbf{P}_{\text{XLM-R, 21}}$$
3. **How BIO labels are selected:**
   - Token-level independent argmax:
     $$\hat{y}_t = \arg\max_{c \in \{0, \dots, 47\}} \mathbf{P}_{\text{fused}}[t, c]$$
   - There is no global transition matrix or sequence-level dynamic programming (no CRF).
4. **How invalid BIO transitions are handled:**
   - In `bio_to_entities_with_confidence` ([`src/evaluation/entity_metrics.py#L144-L154`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/entity_metrics.py#L144-L154)):
     If an `I-X` tag appears after `O` or after a different aspect $Y \ne X$, the decoder executes:
     ```python
     close_current_entity()
     current_aspect = aspect
     current_tokens = [token]
     current_probs = [prob]
     ```
     It **automatically promotes orphan `I-X` tags into a new entity of aspect $X$**, rather than dropping them or crashing. Across all 5,563 tokens in our holdout, invalid transitions occur on only **13 tokens ($0.2337\%$)**.
5. **How token spans become exact entity strings:**
   - Consecutive tokens assigned to the same aspect are appended to `current_tokens` and joined with single spaces: `val = " ".join(current_tokens)`.
6. **How confidence thresholds are applied:**
   - Each extracted span accumulates the softmax probability of the selected label across its constituent tokens.
   - For each span, minimum token confidence is computed: `min_conf = np.min(current_probs)`.
   - Threshold filtering operates on `min_conf`:
     - `Kompatibles_Fahrzeug_Modell`: $\ge 0.80$
     - `Produktart`: $\ge 0.80$
     - `Im_Lieferumfang_Enthalten`: $\ge 0.80$
     - All remaining 26 aspects: $\ge 0.60$
     Spans with `min_conf < threshold` are completely discarded.
7. **How overlapping entities are handled:**
   - Because BIO sequence labeling assigns exactly one label per token from a single argmax distribution, overlapping entities are strictly prohibited by construction.
8. **Confidence aggregation function:**
   - Minimum constituent token confidence (`min_conf = float(np.min(cur_probs))`).
   - Mean confidence is tracked as a diagnostic, but threshold decisions are governed by minimum confidence to guard against multi-token span boundary corruption.
9. **Can a high-probability alternative span be discarded by the current decoder?**
   - **Yes.** Because argmax is greedy and token-independent:
     If token 1 has $P(\text{B-Modell}) = 0.51$ and $P(\text{B-VM}) = 0.49$, while token 2 has $P(\text{I-VM}) = 0.90$ and $P(\text{O}) = 0.55$, greedy argmax selects `B-Modell` at token 1 and `O` at token 2, completely discarding the coherent vehicle model entity `["B-VM", "I-VM"]` (joint probability $0.441$).

---

## 4. Quantitative Error Decomposition & Recoverable Information

We analyzed all 752 False Negatives and 165 False Positives on the 489-title clean holdout using cached fused ensemble probabilities.

### False Negative Breakdown (Total: 752)

| Bucket | Definition | FN Count | % of FNs | Theoretical Max Score | Max Score Delta | Primary Mechanism / Aspects |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Bucket A** | Strong token support ($\ge 0.50$ per token) | **431** | **$57.3\%$** | `0.953479` | $+0.025708$ | **394 (91.4%) pruned by confidence thresholds**; 37 boundary errors |
| **Bucket B** | Partial support ($0.20 \le p < 0.50$ conflicting with argmax) | **154** | **$20.5\%$** | `0.937306` | $+0.009535$ | Token competition between model and generation code |
| **Bucket C** | Low support ($p < 0.20$ across all models) | **125** | **$16.6\%$** | `0.939849` | $+0.012078$ | Genuine model blindness / rare vocabulary / out-of-domain terms |
| **Bucket D** | Aspect Confusion (span extracted but wrong aspect) | **42** | **$5.6\%$** | `0.932321` | $+0.004551$ | Kit ambiguity (`Produktart` vs `Lieferumfang`), Part vs OEM |
| **Bucket E** | Inconsistent annotation conventions | Included in B/D | — | — | — | Multimodal ground truth (`5er E39` vs `5er`, `INA 559015410`) |

### Critical Finding on Bucket A:
Although Bucket A appears large (431 FNs), **394 of them ($91.4\%$) are entities that were already extracted by raw argmax, but deliberately pruned by confidence thresholding** ($244$ vehicle models, $41$ product types, $30$ package contents, $32$ mounting positions).
Lowering these thresholds was already evaluated in Phases 35C, 36C, 37A, and 40: doing so recalls these 394 entities but introduces over $24,000$ false positives across the competition distribution, severely degrading $F_{0.2}$. They cannot be recovered by decoding without reopening false positive gates.

### False Positive Breakdown (Total: 165)

| FP Category | Count | % of FPs | Root Cause Analysis |
| :--- | :---: | :---: | :--- |
| **Boundary Mismatch (Same Aspect)** | **120** | **$72.7\%$** | Entity detected correctly, but span boundary differs by 1 token (e.g. `Vectra B` vs `Vectra`, `Focus 3 2.0 ST` vs `Focus`) |
| **Aspect Confusion** | **41** | **$24.8\%$** | Token span extracted correctly, but assigned to a related aspect (e.g. `Produktart` vs `Im_Lieferumfang_Enthalten`) |
| **Pure Hallucination** | **4** | **$2.4\%$** | Prediction on unannotated title tokens ($< 2.5\%$ of all false positives) |

---

## 5. Evaluation of Three Decoding Approaches

### Approach A — BIO-Constrained Sequence Decoding (Viterbi / Transition Matrix)
- **Concept:** Implement a Viterbi trellis decoder with hard BIO transition constraints (forbidding transitions such as `O -> I-X` or `B-X -> I-Y`).
- **Empirical Audit:**
  - In our 489 clean holdout titles (5,563 total tokens), the fused ensemble produced exactly **13 invalid BIO transitions ($0.2337\%$ of tokens)**:
    - 5 occurrences of `I-Produktart` following `Im_Lieferumfang_Enthalten`.
    - 2 occurrences of orphan `I-Kompatibles_Fahrzeug_Modell` after `O`.
    - 6 miscellaneous cross-aspect `I-tag` transitions.
  - Furthermore, the production decoder already promotes orphan `I-X` tags into valid entities ([`src/evaluation/entity_metrics.py#L150-L153`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/entity_metrics.py#L150-L153)).
- **Theoretical Headroom:** **$< +0.0002$**. Even if all 13 transitions were miraculously corrected, the score impact is virtually zero.
- **Verdict:** **REJECT.** The problem does not exist in any meaningful magnitude.

---

### Approach B — Alternative Span Candidate Ranking
- **Concept:** Extract top-$K$ candidate spans per title from token probabilities (or beam search), compute hand-crafted or learned features (mean token probability, length, frequency in training dictionary), and rerank spans.
- **Empirical Audit:**
  - 120 of 165 False Positives and 37 Bucket A False Negatives are boundary shifts.
  - However, in 52 of these cases, the alternative span already has high token probability ($\ge 0.30$).
  - For example:
    - Record 1298: Gold = `Vectra` ($P = 0.998$); Prediction = `Vectra B`.
    - Record 3438: Gold = `5er` ($P = 0.995$); Prediction = `5er E39`.
    - Record 4561: Gold = `VITO` ($P = 0.992$); Prediction = `VITO (W447)`.
  - **The Multimodality Bottleneck:** In Phase 42B, we evaluated training-derived boundary repair on these exact chassis codes. Because German automotive title annotators are multimodal (some annotate `Vectra B` as one entity, others split `Vectra` and `B`), trimming chassis codes destroyed 9 to 14 True Positives for every 1 False Positive eliminated. $100\%$ of bootstrap resamples regressed.
  - **Leakage & Engineering Bottleneck:** Learning a reranker without validation leakage requires generating Out-Of-Fold (OOF) token probabilities across `train_4k`. This requires fine-tuning 5 folds of 4 models (20 training runs), requiring over 30 hours of GPU computation on our 6GB GPU.
- **Verdict:** **REJECT.** High risk of test set regression due to annotation multimodality, prohibitive engineering complexity, and severe time constraints.

---

### Approach C — Contextual Aspect Disambiguation
- **Concept:** Reclassify ambiguous spans between confusable aspects using contextual keywords or token probability ratios:
  - `Herstellernummer` vs `Oe/Oem_Referenznummer(N)`
  - `Produktart` vs `Im_Lieferumfang_Enthalten`
- **Empirical Audit:**
  - Total aspect confusion False Negatives across the entire holdout: **42 entities ($5.6\%$ of all FNs)**.
  - Confusions are heavily concentrated:
    - `Produktart` $\leftrightarrow$ `Im_Lieferumfang_Enthalten`: **16 cases** (e.g., `Wasserpumpe` in kit listings).
    - `Oe/Oem_Referenznummer(N)` $\leftrightarrow$ `Herstellernummer`: **4 cases**.
    - Vehicle Make $\leftrightarrow$ Vehicle Model / Manufacturer: **11 cases**.
  - **Precision Asymmetry Risk:** `Produktart` carries an official weight of **$18.61\%$** in Category 2, whereas `Im_Lieferumfang_Enthalten` carries only **$3.85\%$**. If a contextual rule falsely reassigns genuine `Produktart` entities to `Im_Lieferumfang_Enthalten`, Category 2 score plummets because false positives on `Produktart` carry a crushing $25\times$ penalty.
  - **Sample Size:** With only 4 part-number confusions and 16 kit confusions in the holdout, any contextual rule would suffer from extreme small-sample variance.
- **Verdict:** **REJECT.** Upper-bound headroom is tiny ($+0.004551$), and risk of precision cross-contamination on high-weight aspects is catastrophic.

---

## 6. Critical Feasibility Assessment

Addressing the eight mandatory feasibility questions:

| # | Feasibility Question | Answer | Evidence / Operational Reality |
| :---: | :--- | :---: | :--- |
| **1** | Are required token probabilities already cached? | **YES** | Cached in `data/cache/phase_35a_val_1k_probs.pt` for local validation. |
| **2** | Are token offsets and model alignments reliable? | **YES** | Whitespace pre-tokenization with `word_ids` alignment ensures exact token co-indexing across all 4 models. |
| **3** | Can the approach be implemented without retraining? | **YES** | All three decoding approaches operate post-hoc on fused probabilities. |
| **4** | Can ranking/calibration parameters be learned safely without validation leakage? | **NO** | Training-only ranking requires 5-fold cross-validation OOF probability generation (20 fine-tuning runs). Fitting directly on training data causes severe overconfidence mismatch. |
| **5** | Would it require new GPU inference for Quiz/Test? | **YES** | Full sequential inference across 50,000+ candidate records would be required to generate new submissions. |
| **6** | What is the expected implementation time? | **1 to 5 days** | Approach A: 1 day; Approach B: 4–5 days; Approach C: 2–3 days. |
| **7** | What is the principal failure mode? | **Overfitting to Annotation Noise** | Boundary trimming and aspect flipping destroy genuine True Positives, triggering fatal precision penalties under $F_{0.2}$. |
| **8** | Is there sufficient evidence to justify work before Nov 6? | **NO** | Headroom is negligible ($< 0.0002$ for A; $< 0.0045$ for C), failure risks are proven high (Phase 42B), and competition deadline is 7 days away. |

---

## 7. Strategic Risk-Reward Matrix & Ranking

| Approach | Theoretical Headroom | Practical Expected Gain | Generalization Risk | Engineering Complexity | Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Approach A: BIO Constraints** | $< +0.0002$ | $+0.0000$ | Very Low | Low (1 day) | **Rank 3** (Non-existent defect) |
| **Approach B: Span Reranking** | $+0.0257$ | **Negative** (Regression) | **Critical** (Label multimodality) | **Prohibitive** (4–5 days, 20 OOF runs) | **Rank 2** (High complexity, high failure probability) |
| **Approach C: Context Disambiguation**| $+0.0045$ | $\pm 0.0005$ | **High** (Precision contamination) | Moderate (2–3 days) | **Rank 1** (Highest relative interest, but still negative expected value) |

Because all three approaches have either negligible upside or negative expected value on the official test set, **no approach is recommended for implementation**.

---

## 8. Verification, Tests, and Asset Protection

- **Champion File Integrity:** [`submissions/submission_quiz_phase40_lieferumfang080.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase40_lieferumfang080.tsv) verified: `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`.
- **Phase 44 Unit Tests:** [`tests/test_phase_44_decoding_audit.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_44_decoding_audit.py) passes 6/6 tests in $0.045\text{s}$.
- **Full Regression Test Suite:** Passes **141/141 tests** in $31.6\text{s}$ with zero failures.
- **Disk Budget:** **7.92 GiB** free disk space maintained.
- **Competition Integrity:** Zero changes made to production inference pipeline ([`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py)).

---

## 9. Final Decision & Recommendation

Our diagnostic audit proves that:
1. The four-model ensemble's argmax decoding is already remarkably well-behaved: invalid BIO transitions occur on less than a quarter of one percent ($0.23\%$) of tokens and are already repaired by the existing decoder.
2. The vast majority of False Negatives (Bucket A: 394 items) are suppressed by confidence thresholds that were intentionally elevated to protect precision-weighted $F_{0.2}$ (saving our team from 24,000+ Quiz false positives).
3. Boundary mismatches reflect dataset annotation multimodality rather than a decoder defect.
4. With the competition deadline on **November 6, 2026**, our top priority must be protecting our proven champion (**EvalAI Quiz $F_{0.2} = 0.9233317116970503$, Rank 14**) and ensuring flawless test execution.

---

RETAIN PHASE 40 — NO CONVINCING DECODING OPPORTUNITY
