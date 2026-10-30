# Phase 48 — Hard Example Mining and Precision-Aware Fine-Tuning Report

**Author**: Senior ML Competition Researcher  
**Date**: October 30, 2026  
**Competition**: eBay 2026 University Machine Learning Competition — Canada  
**Focus**: Supervised Hard Example Mining, Mixture Sampling Fine-Tuning, and Holdout Evaluation  
**Verdict**: **RETAIN PHASE 40 — NO IMPROVEMENT**  

---

## Executive Summary

Phase 48 developed and rigorously evaluated one genuinely different ML challenger using **training-only hard example mining** from supervised data.

1. **Hard Example Mining Discovery**:
   - Out of 4,000 Split-A training titles evaluated using 3-fold out-of-fold (OOF) predictions from Phase 29B, **1,858 titles (46.5%)** were categorized as difficult (composite penalty score $\ge 6$).
   - Top failure categories: missed vehicle-model entities (1,391 titles), boundary errors (958 titles, 682 on vehicle models), false-positive hallucinations (480 titles), and aspect confusions (401 titles).
   - Strict data provenance: The 1,000-title validation holdout, the 489-title clean holdout, Quiz, and Test sets were **strictly excluded** from hard example mining.
2. **Predeclared Model & Training**:
   - GBERT token classifier initialized from the domain-adapted encoder (`checkpoints/phase33b_dapt_encoder`).
   - Trained for 3 epochs on a 70/30 mixture of standard supervised examples (2,800/epoch) and mined hard examples (1,200/epoch) using AdamW (`lr = 2e-5`, linear warmup 10%).
   - Runtime: 31.8 minutes (well under the 4-hour budget); peak VRAM: 2.1 GB; free disk space: 6.43 GB ($\ge 5.0$ GB limit).
   - Best checkpoint: Epoch 2, with validation $F_{0.2} = 0.905795$.
3. **Rigorous Holdout & Ensemble Evaluation**:
   - **Baseline Reproduction**: Phase 40 4-model champion perfectly reproduced on the 489 clean holdout: $F_{0.2} = 0.927771$ (Cat 1: 0.918323, Cat 2: 0.937219; TP=3,021, FP=165, FN=752).
   - **Standalone Phase 48 GBERT**: Achieved $F_{0.2} = 0.898963$ ($\Delta = -0.028808$ vs 4-model ensemble). Recall increased (TP=3,139 vs 3,021), but false positives increased sharply (FP=306 vs 165).
   - **Predeclared Ensemble (90% Phase 40 + 10% Phase 48)**:
     - Overall Score: $0.927785$ ($\Delta = +0.000015$, essentially zero).
     - Category 1: $0.915109$ ($\Delta = -0.003214$, **active regression**).
     - Category 2: $0.940462$ ($\Delta = +0.003243$).
     - Net False Positives: **Increased by +5** (from 165 to 170; 14 new FPs introduced).
   - **Statistical Bootstrap (1,000 Paired Resamples)**:
     - Mean Delta: $-0.000173$.
     - 95% Confidence Interval: `[-0.003212, +0.003385]` (crosses zero broadly).
     - Empirical p-value $P(\Delta \le 0) = 0.5570$ (**not statistically significant**).
4. **Decision**:
   **RETAIN PHASE 40 — NO IMPROVEMENT**.
   The Phase 40 official Quiz champion ($F_{0.2} = 0.9233317116970503$, SHA-256 `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`) remains unaltered. No Quiz or Test submission is justified.

---

## 1. Resource and Data Audit

| Resource / Parameter | Specification / Measured | Constraint | Status |
| :--- | :--- | :--- | :--- |
| **GPU / VRAM** | NVIDIA GeForce GTX 1660 Ti (6.00 GB) | $\ge 4.0$ GB | **Compliant** |
| **Peak VRAM During Training** | 2,117.9 MB (35.3% of capacity) | $\le 5.5$ GB | **Safe** |
| **Workspace Free Disk Space** | 6.43 GB | $\ge 5.0$ GB | **Compliant** |
| **Training Time Budget** | 31.79 minutes (1,907.5 seconds) | $\le 4.0$ hours | **Compliant** |
| **Checkpoint Storage** | 417.3 MB (`model.safetensors` single best) | Minimal | **Safe** |
| **Phase 40 Champion SHA-256** | `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8` | Exact Match | **Verified Intact** |

### Data Provenance & Partitioning

```
Total Supervised Data (5,000 titles)
  ├── Split-A Training Partition (4,000 titles) [100% Training Only]
  │     ├── 3-Fold Stratified & Grouped CV (Phase 29B)
  │     └── Hard Example Mining (OOF predictions only)
  │           ├── Standard / Non-Hard Titles: 2,142 (53.5%)
  │           └── Mined Hard Examples:       1,858 (46.5%)
  └── Split-A Validation Partition (1,000 titles) [Strictly Isolated]
        ├── Clean Holdout Subset: 489 titles (Phase 35B Manifest)
        └── Excluded from All Mining and Training Decisions
```

Zero leakage verified: $\text{Train 4k} \cap \text{Val 1k} = \emptyset$, $\text{Hard Examples} \cap \text{Clean Holdout} = \emptyset$.

---

## 2. Hard Example Mining Breakdown

Using out-of-fold predictions from Phase 29B, every training title was evaluated by a model that never trained on it. Titles were scored using an $F_{0.2}$-aligned penalty formula:
$$\text{Score} = 5 \cdot \text{FP} + 1 \cdot \text{FN} + 3 \cdot N_{\text{boundary}} + 4 \cdot N_{\text{aspect\_confusion}} + 4 \cdot \mathbb{I}_{\text{missed\_vehicle\_model}} + 3 \cdot \mathbb{I}_{\text{part\_number\_err}}$$

Titles with $\text{Score} \ge 6$ were classified as **Hard Examples**.

### Summary Distribution (4,000 Training Titles)
- **Perfect Titles (0 FP, 0 FN under calibrated thresholds)**: 1,360 (34.0%)
- **Imperfect Titles ($\ge 1$ Error)**: 2,640 (66.0%)
- **Mined Hard Examples ($\text{Score} \ge 6$)**: 1,858 (46.5%)
  - Category 1: 952 titles (51.2% of hard subset)
  - Category 2: 906 titles (48.8% of hard subset)

### Error Type Distribution Across Training Titles

| Error Category | Titles Exhibiting Error | Percentage of Training Partition |
| :--- | :--- | :--- |
| Missed Entities (False Negatives) | 2,153 | 53.8% |
| Missed Vehicle-Model Entities | 1,391 | 34.8% |
| Boundary Errors (Truncation/Overextension) | 958 | 23.9% |
| Vehicle-Model Boundary Errors | 682 | 17.1% |
| False-Positive Hallucinations | 480 | 12.0% |
| Wrong Aspect Confusions | 401 | 10.0% |
| Part Number / OEM Number Errors | 207 | 5.2% |
| Invalid BIO State Transitions | 116 | 2.9% |

### Top 10 Aspects by Error Frequency

| Aspect Name | Total Errors | False Positives | False Negatives | Boundary Errors | Aspect Confusions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Kompatibles_Fahrzeug_Modell` | **3,553** | 220 | 2,505 | 758 | 70 |
| `Im_Lieferumfang_Enthalten` | **688** | 81 | 452 | 79 | 76 |
| `Produktart` | **679** | 97 | 450 | 76 | 56 |
| `Einbauposition` | **264** | 46 | 146 | 70 | 2 |
| `Herstellernummer` | **262** | 19 | 134 | 44 | 65 |
| `Kompatible_Fahrzeug_Marke` | **204** | 37 | 97 | 25 | 45 |
| `Oe/Oem_Referenznummer(N)` | **184** | 16 | 121 | 5 | 42 |
| `Hersteller` | **161** | 22 | 74 | 25 | 40 |
| `Modell` | **87** | 9 | 71 | 0 | 7 |
| `Bremsscheibenart` | **61** | 8 | 40 | 1 | 12 |

---

## 3. Training Curves and Checkpoint Selection

### Hyperparameters
- **Base Encoder**: `checkpoints/phase33b_dapt_encoder`
- **Mixture Ratio**: 70% Standard (2,800 titles) / 30% Hard (1,200 titles) per epoch
- **Epochs**: 3 (3,000 optimization steps total)
- **Batch Size**: 4 (train), 8 (val)
- **Optimizer**: AdamW (`lr = 2e-5`, `weight_decay = 0.01`, linear warmup 10% = 300 steps)
- **Random Seed**: 42

### Epoch History

| Epoch | Train Loss | Val Loss | Val $F_{0.2}$ | Cat 1 $F_{0.2}$ | Cat 2 $F_{0.2}$ | Precision | Recall | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.7021 | 0.3409 | 0.902724 | 0.905184 | 0.899714 | 0.9067 | 0.8130 | Evaluated |
| **2** | **0.2202** | **0.3222** | **0.905795** | **0.905532** | **0.906123** | **0.9095** | **0.8229** | **Best Checkpoint** |
| 3 | 0.1776 | 0.3277 | 0.902768 | 0.902593 | 0.902985 | 0.9058 | 0.8323 | Slight Overfit |

Saved checkpoint: `checkpoints/gbert_phase48_hard_mined_best` (selected from Epoch 2).

---

## 4. Standalone and Ensemble Performance Evaluation

### Clean Holdout Official Metric Comparison (489 Titles)

| Configuration | Official Score | Category 1 | Category 2 | TP | FP | FN | $\Delta F_{0.2}$ vs P40 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 40 Champion Baseline** | **0.927771** | **0.918323** | **0.937219** | **3,021** | **165** | **752** | Baseline |
| **Phase 48 Standalone GBERT** | 0.898963 | 0.888144 | 0.909781 | 3,139 | 306 | 634 | $-0.028808$ |
| **Predeclared Ensemble (90/10)** | **0.927785** | **0.915109** | **0.940462** | **3,032** | **170** | **741** | **$+0.000015$** |

### Statistical Significance (1,000 Paired Category-Stratified Bootstrap Resamples)
- **Bootstrap Mean Metric Delta**: $-0.000173$
- **95% Confidence Interval**: `[-0.003212, +0.003385]`
- **Empirical p-value ($P(\Delta \le 0)$)**: `0.5570`
- **Statistically Significant**: **No (False)**

---

## 5. Complementarity and Failure Analysis

Inspection of the 47 titles modified by the 10% ensemble blend reveals the core behavioral shift:

```
Total Clean Holdout Titles: 489
  ├── Unchanged by Ensemble: 442 (90.4%)
  └── Modified by Ensemble:   47 (9.6%)
        ├── Repaired Titles (Score Up):   25 titles
        ├── Regressed Titles (Score Down): 21 titles
        └── Neutral Changes:               1 title
```

### Why Hard Example Fine-Tuning Failed to Outperform Phase 40:
1. **Aggressive Precision Erosion**:
   Fine-tuning on hard examples taught the model to be more aggressive at predicting boundaries and minority aspects. While recall improved (recovering 11 true positives and reducing FNs from 752 to 741), it caused **14 new false positive errors**, raising net FP from 165 to 170. Under the competition's $F_{0.2}$ metric ($\beta^2 = 0.04$), a single false positive penalizes the score ~25 times more than a missed entity.
2. **Category 1 Regression**:
   The ensemble exhibited a noticeable regression in Category 1 ($\Delta = -0.003214$), offsetting Category 2 gains ($\Delta = +0.003243$) and yielding an overall delta of essentially zero ($+0.000015$).
   - Example (Record 259): The model truncated `Astra J Caravan` to `Astra J` and hallucinated `vorne` as `Einbauposition`, causing a $-0.0087$ title drop.
   - Example (Record 4390): The model predicted `BMW` as `Hersteller` on a genuine original parts title, introducing an unnecessary false positive.
3. **Bootstrap Distribution Crosses Zero Broadly**:
   With over 55% of bootstrap iterations yielding a negative delta ($p = 0.5570$) and a 95% CI spanning from $-0.0032$ to $+0.0034$, the slight $+0.000015$ point estimate is pure stochastic noise.

---

## 6. Official Decision

| Criterion | Evaluation Requirement | Result | Assessment |
| :--- | :--- | :--- | :--- |
| **1. Complementary Predictions** | Distinct behavioral patterns | Yes (47 titles changed) | Pass |
| **2. Meaningful F0.2 Gain** | $\Delta \ge +0.0005$ | $\Delta = +0.000015$ | **Fail** |
| **3. Controlled False Positives** | Net FP delta $\le 0$ | Net FP $+5$ (14 new FPs) | **Fail** |
| **4. Statistical Significance** | 95% CI strictly $> 0$ | $[-0.003212, +0.003385]$ ($p=0.557$) | **Fail** |
| **5. No Category Regression** | Both categories non-negative | Cat 1 regressed ($-0.003214$) | **Fail** |
| **6. Reproducibility & Safety** | Fixed seeds, disk $> 5.0$ GB | 6.43 GB free, 177 tests passing | Pass |

### Final Verdict:
```
RETAIN PHASE 40 — NO IMPROVEMENT
```
The official Phase 40 champion remains unaltered. No submission files were generated, and no upload to EvalAI is authorized.
