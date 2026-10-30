# Phase 49: Precision-Aware Fine-Tuning & Methodology Audit Report

**Date:** October 30, 2026  
**Status:** Completed  
**Official Decision Verdict:** `RETAIN PHASE 40 — NO RELIABLE IMPROVEMENT`  
**Active Champion:** `submissions/submission_quiz_phase40_lieferumfang080.tsv`  
**Champion SHA-256:** `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`  
**Official EvalAI Quiz $F_{0.2}$ Score:** `0.9233317116970503`  

---

## Executive Summary

Phase 49 was commissioned to rigorously audit the Phase 48 methodology, uncover the underlying mechanics of false-positive generation in German automotive title NER, evaluate precision-aware training techniques on training-only evidence, establish a leak-free internal validation protocol, train one precision-aware challenger model, and perform an unbiased paired bootstrap evaluation against the production baseline.

### Key Conclusions:
1. **Critical Flaw in Phase 48 Sampling Logic Identified:** A subtle sampling discrepancy caused the standard pool to be drawn *with replacement* (`random.choices`) and the hard pool *without replacement* (`random.sample`). Consequently, standard training examples were oversampled by an average of $3.92\times$ per 3 epochs, while hard examples were sampled only $1.94\times$. Standard titles were effectively upweighted $2.02\times$ over hard titles, some individual titles were repeated up to 9 times in a single epoch, and 122 training titles were never sampled at all. Furthermore, Phase 48 selected checkpoints on the 1,000-title validation holdout (indirectly leaking the 489-title clean holdout) using global micro $F_{0.2}$ rather than official competition category-macro $F_{0.2}$.
2. **False-Positive Anatomy Unveiled:** Analysis of 2,134 training-only out-of-fold (OOF) false positives revealed that **85.4%** of competition false positives are genuine entities with boundary or label mismatches, rather than spurious hallucinations. Wrong entity boundaries (Class B) account for **64.6%** (1,378 FPs), dominated by `Kompatibles_Fahrzeug_Modell` (957 FPs, 44.8% of all errors). Pure hallucinations (Class A) account for only **13.6%** (291 FPs).
3. **Internal Validation Design Established:** To prevent evaluation contamination, the 4,000 training titles were partitioned into a 3,200 internal train / 800 internal dev split (seed 42, category-stratified), completely isolating both the 1,000 validation holdout and the 489 clean holdout.
4. **Precision-Aware Training Outcome:** Fine-tuning DAPT GBERT with cost-sensitive classification ($w_O = 1.50\times$) successfully penalized non-background transitions, but substantially degraded recall without producing compensating precision gains on entity boundaries. On the 800-title internal dev partition, official competition $F_{0.2}$ fell by $\Delta = -0.036000$ ($0.942161 \to 0.906161$, bootstrap $p = 1.0000$).
5. **Secondary Diagnostic & Ensemble Evaluation:** On the 489 clean holdout, standalone Phase 49 scored $0.903189$ ($\Delta = -0.024581$ vs Phase 40). In a predeclared 90% Phase 40 + 10% Phase 49 ensemble blend, overall $F_{0.2}$ showed zero statistically significant improvement ($0.927707$ vs $0.927771$, $\Delta = -0.000064$, bootstrap 95% CI $[-0.002713, +0.002613]$, $p = 0.5780$). While 23 titles were repaired, 20 titles regressed, and 10 new false positives were introduced.
6. **Verdict:** The official Phase 40 Quiz champion remains unassailable. No Quiz or Test submission files were modified or generated.

---

## Part A: Audit of Phase 48 Sampling Mechanics & Evaluation Integrity

### 1. Sampling Implementation Analysis
In Phase 48, the training loop was intended to construct each epoch as a 70/30 mixture of "standard" and "hard" titles out of the 4,000 available training titles (2,142 standard titles and 1,858 hard titles identified via OOF errors):
- Standard quota per epoch: 2,800
- Hard quota per epoch: 1,200
- Total examples per epoch: 4,000

Inspection of `src/modeling/train_phase48_hard_negative_gbert.py` (lines 142–154) revealed:
```python
standard_batch = rng.choices(standard_examples, k=n_standard_per_epoch)  # WITH REPLACEMENT
hard_batch = rng.sample(hard_examples, k=n_hard_per_epoch)              # WITHOUT REPLACEMENT
```

### 2. Discrepancy & Mathematical Impact
Because `standard_examples` (size 2,142) was sampled with replacement to fill 2,800 slots, individual standard titles were repeatedly drawn multiple times within the same epoch. In contrast, `hard_examples` (size 1,858) was sampled without replacement to fill 1,200 slots.

| Metric | Standard Pool | Hard Pool | Ratio / Disparity |
| :--- | :---: | :---: | :---: |
| Pool Size ($N$) | 2,142 titles | 1,858 titles | - |
| Quota per Epoch ($k$) | 2,800 | 1,200 | 70% / 30% |
| Sampling Method | `rng.choices` (with replacement) | `rng.sample` (without replacement) | Asymmetric |
| Probability of selection per draw | $1 / 2142 \approx 0.0004669$ | $1 / 1858 \approx 0.0005382$ | - |
| Expected draws per title per epoch | $2800 / 2142 = \mathbf{1.307}$ | $1200 / 1858 = \mathbf{0.646}$ | **2.02x higher for standard** |
| Expected appearances across 3 epochs | $\mathbf{3.922}$ | $\mathbf{1.938}$ | **2.02x higher for standard** |
| Max observed draws for a single title in 1 epoch | **9 draws** | **1 draw** | Severe localized memorization |
| Unsampled training titles across 3 epochs | **122 titles** | 0 titles | 122 titles completely starved |

**Conclusion:** Rather than emphasizing hard examples, Phase 48 unintentionally **starved hard examples** and oversampled standard examples by a factor of $2.02\times$. Furthermore, repeatedly exposing the model to duplicate standard titles up to 9 times within a single epoch biased the model towards memorizing common patterns while starving 122 unique training titles.

### 3. Holdout Leakage & Metric Mismatch in Phase 48
Two additional methodological flaws were documented:
- **Validation Holdout Contamination:** Phase 48 evaluated checkpoint selection on `data/splits/val_split.json` (1,000 titles). Because the competition's 489-title clean holdout is a subset of this 1,000-title validation split (accounting for 48.9% of its examples), the checkpoint was selected on the exact test distribution used for local offline validation.
- **Metric Mismatch:** The validation callback in Phase 48 tracked global micro $F_{0.2}$ rather than the official competition metric: hierarchical category-macro weighted $F_{0.2}$. Global micro ignores category balance and aspect-level weighting, favoring high-frequency aspects over low-frequency aspects.

---

## Part B: Training-Only False-Positive Root Cause Analysis

To guide precision-aware modeling, an exhaustive audit was performed on **2,134 false positives** generated by the frozen baseline on training-only 5-fold cross-validation out-of-fold predictions (`data/cache/phase_35b_dapt_gbert_oof_preds.pt`).

Every false positive was categorized into one of four mutually exclusive, exhaustive root cause classes:
- **Class A (Spurious Hallucinations):** Model predicted an entity span where ground truth contains zero overlapping annotation of any aspect.
- **Class B (Wrong Entity Boundary):** Model identified the correct aspect on a true entity, but token span boundaries were over-extended or truncated.
- **Class C (Wrong Aspect Label):** Model extracted the exact character span, but assigned the wrong aspect label.
- **Class D (Valid Unannotated Entity):** Ground-truth annotation omission where the model predicted a legitimate, recognizable automotive entity.

### Empirical Breakdown:

```
+-----------------------------------------------------------------------------------------+
| CLASS                                    | COUNT | PERCENTAGE | PRIMARY DRIVER          |
+-----------------------------------------------------------------------------------------+
| Class B: Wrong Entity Boundary           | 1,378 |     64.57% | Kompatibles_Fahrzeug... |
| Class C: Wrong Aspect Label              |   444 |     20.81% | OE/OEM vs Herstellernr  |
| Class A: Spurious Hallucinations         |   291 |     13.64% | Pure non-entity tokens  |
| Class D: Valid Entity Omitted in Gold    |    21 |      0.98% | Unannotated part terms  |
+-----------------------------------------------------------------------------------------+
| TOTAL FALSE POSITIVES                    | 2,134 |    100.00% |                         |
+-----------------------------------------------------------------------------------------+
```

### Aspect Breakdown for Class B (Boundary Errors):
- `Kompatibles_Fahrzeug_Modell`: **957 errors** (69.4% of Class B; **44.8% of all false positives**)
- `Produktart`: **211 errors** (15.3% of Class B)
- `Im_Lieferumfang_Enthalten`: **121 errors** (8.8% of Class B)
- `Einbauposition`: **34 errors** (2.5% of Class B)
- All other aspects combined: **55 errors** (4.0% of Class B)

### Key Architectural Insight:
**85.38%** of false positives in this competition are not hallucinations. The model reliably identifies entity locations, but fails exact span boundary matching, particularly on German vehicle model descriptions containing complex multi-token trim, chassis codes, and model generations (e.g., `Astra J Caravan` vs `Astra J`, `Mondeo Iii B5Y+ Kombi Bwy B4Y` vs `Mondeo Iii B5Y+`).

---

## Part C: Precision-Aware Technique Evaluation & Selection

Four precision-aware techniques were evaluated against the training evidence:

1. **Hard-Negative Mining:** Ineffective for boundary correction because the error is token delineation within active spans, not false activation in background text.
2. **Focal Loss ($\gamma = 2.0$):** Focuses training on hard, low-confidence examples. However, boundary errors in GBERT already exhibit high confidence on ambiguous boundaries, meaning focal loss can inadvertently amplify over-confident boundary extensions.
3. **Aspect-Specific Temperature Calibration:** Post-hoc thresholding shifts recall/precision trade-offs globally per aspect, but cannot alter token-level sequence transitions.
4. **Cost-Sensitive Token Classification ($w_O = 1.50\times$):** Directly addresses boundary overextensions by penalizing false non-background predictions at token boundaries. By weighting the `O` (outside) tag at $1.50\times$ in the cross-entropy loss:
   $$\mathcal{L} = -\sum_{i} w_{y_i} \log P(y_i \mid x_i), \quad w_O = 1.50, \quad w_{\text{entity}} = 1.00$$
   the model is penalized for extending entity spans into trailing non-entity tokens or neighboring terms.

**Selection:** Cost-Sensitive Token Classification ($w_O = 1.50\times$) was chosen as the most direct inductive bias against Class B boundary expansion.

---

## Part D: Leak-Free Validation Protocol

To guarantee zero data contamination:
- **Dataset:** 4,000 supervised training titles (`data/raw/train.csv`).
- **Internal Split:**
  - **3,200 Internal Train Titles:** 1,600 Category 1 + 1,600 Category 2 (`data/splits/internal_train_split.json`).
  - **800 Internal Dev Titles:** 400 Category 1 + 400 Category 2 (`data/splits/internal_dev_split.json`).
  - Generated via `src/data/make_internal_split.py` using seed 42.
- **Strictly Isolated Holdouts:**
  - `data/splits/val_split.json` (1,000 titles): Completely untouched during training and checkpoint selection.
  - `data/cache/phase_35b_clean_holdout_manifest.json` (489 clean holdout titles): Strictly reserved for secondary diagnostic evaluation.
- **Selection Criterion:** Official competition hierarchical category-macro weighted $F_{0.2}$ on the 800-title internal dev set.

---

## Part E: Challenger Training Dynamics & Resource Utilization

### 1. Training Setup
- **Base Architecture:** DAPT GBERT (`checkpoints/phase33b_dapt_encoder`).
- **Loss Function:** Cross-Entropy with $w_O = 1.50$ and $w_{\text{entity}} = 1.00$.
- **Optimizer:** AdamW, learning rate $3 \times 10^{-5}$, weight decay $0.01$, linear warmup $10\%$, max grad norm $1.0$.
- **Batch Size:** 4 per device, gradient accumulation 1 (effective batch size 4).
- **Epochs:** 3 (2,400 total training steps).
- **Hardware:** NVIDIA GeForce GTX 1660 Ti (6 GB VRAM).

### 2. Training Dynamics Across Epochs (Internal Dev Set)

| Epoch | Dev Official Macro $F_{0.2}$ | Dev Cat 1 $F_{0.2}$ | Dev Cat 2 $F_{0.2}$ | Dev TP | Dev FP | Dev FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.898629 | 0.906636 | 0.890623 | 4,929 | 382 | 1,218 |
| **2 (Best)** | **0.906161** | **0.917155** | **0.895168** | **4,962** | **356** | **1,185** |
| 3 | 0.905664 | 0.916843 | 0.894485 | 4,957 | 360 | 1,190 |

**Checkpoint Saved:** `checkpoints/gbert_phase49_precision_aware_best` (Epoch 2, size 417.3 MB).

### 3. Resource & Safety Invariants
- Total training runtime: **25.07 minutes** (1,504.1 s), well within the 2-hour budget.
- Peak VRAM allocated: **2,116.6 MB** (35.3% of 6 GB capacity).
- Disk space before training: **6.42 GB**; after evaluation: **5.99 GB** (compliant with $\ge 5.0$ GB limit).

---

## Part F: Generalization Evaluation (Primary vs Matched Baseline)

To evaluate whether cost-sensitive fine-tuning improved generalization, the checkpoint was evaluated against the strictly matched baseline model (`checkpoints/phase33b_dapt_gbert_best`) on the 800-title internal dev partition using Phase 40 confidence thresholds.

```
========================================================================================
INTERNAL DEV EVALUATION (800 TITLES — 400 CAT 1 / 400 CAT 2)
========================================================================================
Model                         Official F0.2   Cat 1 F0.2   Cat 2 F0.2    TP     FP     FN
----------------------------------------------------------------------------------------
Matched Baseline (Phase 33B)       0.942161     0.949122     0.935199  5,131    251  1,016
Phase 49 Precision-Aware           0.906161     0.917155     0.895168  4,962    356  1,185
----------------------------------------------------------------------------------------
Delta vs Matched Baseline         -0.036000    -0.031967    -0.040032   -169   +105   +169
========================================================================================
```

### Paired Category-Stratified Bootstrap (1,000 Resamples):
- **Mean Delta:** $-0.036083$
- **95% Confidence Interval:** $[-0.044742, -0.028122]$ (entirely below zero)
- **Bootstrap p-value:** $p = 1.0000$ (0 / 1,000 resamples favored Phase 49)
- **Statistical Significance:** Degraded with $p < 0.001$.

**Finding:** Weighting the `O` tag at $1.50\times$ distorted token transition dynamics. The model became overly conservative on entity heads (`B-` tags), which led to premature span truncations. Instead of reducing false positives, truncations produced mismatched partial spans (which count simultaneously as $+1$ FP and $+1$ FN in exact-span scoring). False positives increased by $+105$ while true positives decreased by $-169$.

---

## Part G: Secondary Diagnostic & Ensemble Complementarity (489 Clean Holdout)

### 1. Standalone Performance on Clean Holdout
On the 489-title clean holdout:
- Phase 40 Champion Baseline: $0.927771$ (TP=3,021, FP=165, FN=752)
- Phase 49 Standalone: $0.903189$ (TP=3,073, FP=253, FN=700)
- Standalone Delta vs Phase 40: $\Delta = -0.024581$

### 2. Predeclared Ensemble Blend: 90% Phase 40 + 10% Phase 49

```
========================================================================================
CLEAN HOLDOUT EVALUATION (489 TITLES)
========================================================================================
Configuration                 Official F0.2   Cat 1 F0.2   Cat 2 F0.2    TP     FP     FN
----------------------------------------------------------------------------------------
Phase 40 Champion Baseline         0.927771     0.918323     0.937219  3,021    165    752
Phase 40 (90%) + Phase 49 (10%)    0.927707     0.917213     0.938201  3,026    167    747
----------------------------------------------------------------------------------------
Ensemble Delta vs Phase 40        -0.000064    -0.001110    +0.000982     +5     +2     -5
========================================================================================
```

### Paired Bootstrap on Clean Holdout (1,000 Resamples):
- **Mean Delta:** $-0.000234$
- **95% Confidence Interval:** $[-0.002713, +0.002613]$
- **Bootstrap p-value:** $p = 0.5780$ (statistically indistinguishable from random noise)
- **Significant Improvement:** **False**

### 3. Title-by-Title Complementarity Audit
- **Titles Changed:** 43 out of 489 (8.79%)
- **Repaired Titles (Score Increased):** 23 titles
- **Regressed Titles (Score Decreased):** 20 titles
- **Net Title Delta:** $+3$ titles
- **New False Positives Introduced:** 10 new false positives

#### Representative Case Studies:
1. **Regressed Title (Record 259, Cat 1):**
   *Title:* `Textar Bremsscheiben Beläge vorne hinten für OPEL Astra J Caravan`  
   *Gold:* `Kompatibles_Fahrzeug_Modell: Astra J Caravan`  
   *Phase 40 Pred:* None  
   *Ensemble Pred:* Added `Kompatibles_Fahrzeug_Modell: Astra J` (truncated boundary, $+1$ FP, $\Delta F_{0.2} = -0.004357$).
2. **Repaired Title (Record 4506, Cat 1):**
   *Title:* `2x ZIMMERMANN Bremsscheiben Ø295mm + Beläge Vorne passend für MB C-Klasse (S204)`  
   *Phase 40 Pred:* Missed `Kompatible_Fahrzeug_Marke: MB`  
   *Ensemble Pred:* Correctly added `Kompatible_Fahrzeug_Marke: MB` ($-1$ FN, $\Delta F_{0.2} = +0.121945$).
3. **Regressed Title (Record 4615, Cat 1):**
   *Title:* `Brembo Bremsscheiben u. Ate Ceramic-beläge mit Wkt Mini R50/52/R53 Satz VA + HA`  
   *Gold:* `Produktart: Satz`  
   *Phase 40 Pred:* Correctly predicted `Produktart: Satz`  
   *Ensemble Pred:* Dropped `Produktart: Satz` due to conservative O-penalty ($+1$ FN, $\Delta F_{0.2} = -0.107296$).

---

## Part H: Official Decision Verdict & Production Recommendations

### Official Verdict:
$$\mathbf{RETAIN\ PHASE\ 40\ —\ NO\ RELIABLE\ IMPROVEMENT}$$

### Operational Protocol Status:
- [x] Official Quiz Champion file `submissions/submission_quiz_phase40_lieferumfang080.tsv` preserved intact (SHA-256 verified).
- [x] No Quiz TSV generated.
- [x] No Test TSV generated.
- [x] Zero EvalAI submissions performed.
- [x] 20 EvalAI Quiz submissions remaining intact.
- [x] Disk space maintained above 5.0 GB limit (current: 5.99 GB).

### Scientific Takeaway for Final Submission Preparation:
1. **Token Loss Penalties Do Not Solve German Entity Boundaries:** Penalizing non-background tokens via $w_O$ shifts span extraction thresholds indiscriminately, damaging short single-token aspects (`Produktart`, `Marke`) while failing to resolve multi-token compound boundary ambiguities (`Kompatibles_Fahrzeug_Modell`).
2. **Phase 40 Ensemble Stability:** The 4-model ensemble (DAPT GBERT 0.40 + 3x XLM-R 0.20) with aspect-specific confidence gating remains the most robust, high-precision predictor available.
3. **Recommendation for Final Test Phase:** Champion `submission_quiz_phase40_lieferumfang080.tsv` should serve as the primary submission candidate for the final competition evaluation.
