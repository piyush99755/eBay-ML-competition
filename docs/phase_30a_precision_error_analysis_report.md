# Phase 30A — Error-Driven Precision Research and Scorer Integrity Audit Report

**Status:** COMPLETED & AUDITED  
**Date:** October 27, 2026  
**Active Official Champion:** Experiment 10 (GBERT 0.40 + 3x XLM-R 0.20) with MIN 0.60 Confidence Filter  
**Official EvalAI Quiz F0.2:** **0.901245410919943** (Rank 13)  
**Local Split-A Holdout F0.2:** **0.899802**  
**Production Submission Pipeline:** [`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py) (Frozen & Untouched)  
**Final Phase 30A Recommendation:** **RETAIN CHAMPION** (Do not deploy aspect or category threshold adjustments)

---

## 1. Executive Summary & Verdict

Phase 30A was designed as a research-only investigation to determine whether category-aware or aspect-aware confidence calibration could improve upon our frozen Experiment 10 + MIN 0.60 champion without retraining any models.

### Key Conclusions:
1. **Scorer & Historical Audit**:
   - **Fold 1 Checkpoint Selection in Phase 29B**: In [`src/modeling/train_gbert_cv3.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/modeling/train_gbert_cv3.py), checkpoint saving operated on a **fixed final epoch (Epoch 3)** policy rather than dynamic best-validation-metric tracking. This explains why Fold 1 Epoch 3 weights were saved despite Epoch 2 having an arithmetically higher validation $F_{0.2}$ ($0.863323$ vs $0.862406$).
   - **Phase 29B Table Discrepancy**: The Phase 29B evaluation pipeline ([`src/evaluation/experiment_29b_evaluate_cv.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/experiment_29b_evaluate_cv.py)) only evaluated the challenger (Gate D) and used a hardcoded benchmark scalar (`0.899802`). The intermediate category and precision/recall values in the Phase 29B report table were manual transcription approximations. The canonical, bit-for-bit champion metrics were re-verified and reproduced with zero drift.
2. **Diagnostic Error Taxonomy**:
   - Out of 572 total False Positives produced by the champion on Split A, **309 False Positives (54.0%)** belong to a single aspect: `Kompatibles_Fahrzeug_Modell`.
   - **97.4% of these errors (301 out of 309)** are **boundary mismatches** (where the model predicted the exact correct aspect overlapping the ground-truth span, but differed slightly in token boundaries). True wrong-aspect misclassifications and spurious hallucinations are rare ($< 2.6\%$).
3. **Challenger Policy Evaluation**:
   - **Policy A** (Raising threshold to MIN 0.70 strictly for `Kompatibles_Fahrzeug_Modell`) and **Policy B** (Raising Category 1 threshold to MIN 0.65) produce superficial aggregate $F_{0.2}$ gains ($+0.004780$ and $+0.002098$, respectively).
   - **The Metric Gaming Trap**: However, deeper title-level and entity-level audits reveal that:
     - Policy A pruned **66 False Positives vs 95 True Positives** (FP/TP ratio = $0.695$). It sacrificed more correct entities than errors.
     - Policy A improved only **24 titles** while **degrading 69 titles** (a 3:1 negative ratio).
     - Policy B pruned **28 False Positives vs 31 True Positives** (FP/TP ratio = $0.903$), improving **6 titles** while **degrading 28 titles** (a 4.7:1 negative ratio).
   - The superficial arithmetic gain in aggregate $F_{0.2}$ is solely an artifact of $\beta = 0.2$ heavily weighting precision ($96.15\%$), rewarding aggressive denominator reduction at the expense of genuine entity recall and title-level accuracy.
4. **Final Recommendation**: **RETAIN THE FROZEN EXP 10 + MIN 0.60 CHAMPION**. Do not adopt aspect-specific or category-specific threshold rules.

---

## 2. Step 1: Scorer Integrity & Phase 29B Checkpoint Audit

### A. Fold 1 Checkpoint Selection Analysis
In [`src/modeling/train_gbert_cv3.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/modeling/train_gbert_cv3.py) lines 256–260:
```python
# Save final Epoch 3 model per fixed 3-epoch pilot policy
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSaving Epoch 3 checkpoint to {checkpoint_dir}...")
model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)
```
- **Criterion Identified**: **Fixed final epoch (Epoch 3)**.
- **Root Cause**: The training script evaluated validation loss and entity $F_{0.2}$ at every epoch for logging, but did not maintain an in-memory best-weights buffer or save intermediate epoch weights to disk. As a result, the weights at the end of the final epoch (Epoch 3) were saved unconditionally into `checkpoints/gbert_cv3_fold1_best`.
- **Impact**: In Fold 1, Epoch 2 validation $F_{0.2}$ was $0.863323$ (Val Loss $0.3149$), while Epoch 3 was $0.862406$ (Val Loss $0.3156$). The score delta between Epoch 2 and Epoch 3 was tiny ($\Delta = -0.000917$).

### B. Champion Comparison Discrepancy Resolution
In the Phase 29B report table, the frozen champion row listed:
`Frozen Champion (Exp 10 + MIN 0.60)   0.899802    0.899878    0.899726     0.9168    0.8400`
Whereas Phase 26A / Phase 27A reported:
`Frozen Champion (Exp 10 + MIN 0.60)   0.899802    0.899921    0.899683     0.9172    0.8350`

- **Scorer Comparison**: An inspection of [`src/evaluation/phase_30a_reproduce_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_30a_reproduce_champion.py) confirmed that [`src/evaluation/competition_metrics.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/competition_metrics.py) and [`src/evaluation/entity_metrics.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/entity_metrics.py) are identical across all phases.
- **Discrepancy Source**: In Phase 29B, `src/evaluation/experiment_29b_evaluate_cv.py` evaluated only the challenger ensemble (Gate D) and did not re-evaluate the frozen champion on the holdout, defining only a constant `BENCHMARK_CHAMPION_F02 = 0.899802`. When the markdown report was drafted, intermediate sub-metrics were transcribed with slight manual approximation.
- **Canonical Reproduction Verification**: Executing the verified Phase 26A/27A pipeline on the 1,000 Split-A validation titles reproduced the canonical champion metrics exactly:

| Metric | Target Canonical Value | Phase 30A Reproduction | Verification Status |
| :--- | :---: | :---: | :---: |
| **Official Entity $F_{0.2}$** | **0.899802** | **0.899802** | **EXACT MATCH** |
| **Category 1 $F_{0.2}$** | 0.899921 | 0.899921 | **EXACT MATCH** |
| **Category 2 $F_{0.2}$** | 0.899683 | 0.899683 | **EXACT MATCH** |
| **Micro Precision** | 0.9172 | 0.9172 | **EXACT MATCH** |
| **Micro Recall** | 0.8350 | 0.8350 | **EXACT MATCH** |
| **True Positives ($TP$)** | 6,340 | 6,340 | **EXACT MATCH** |
| **False Positives ($FP$)** | 572 | 572 | **EXACT MATCH** |
| **False Negatives ($FN$)** | 1,253 | 1,253 | **EXACT MATCH** |
| **Total Predicted Entities** | 6,912 | 6,912 | **EXACT MATCH** |

The baseline is 100% verified and reproducible bit-for-bit.

---

## 3. Step 2: Diagnostic Error Profile & Taxonomy

Using the 1,000-title Split-A validation dataset ($N = 1,000$ titles, 11,349 tokens), every champion prediction at MIN 0.60 was audited against ground-truth annotations.

### A. Category-Level Error Profile

| Category | True Entities | Predicted Entities | True Positives ($TP$) | False Positives ($FP$) | False Negatives ($FN$) | Micro Precision | Micro Recall | $TP$ Mean Conf | $FP$ Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Category 1** | 4,182 | 3,818 | 3,498 | 320 | 684 | 0.9162 | 0.8364 | 0.9630 | 0.8363 |
| **Category 2** | 3,411 | 3,094 | 2,842 | 252 | 569 | 0.9186 | 0.8332 | 0.9575 | 0.8156 |
| **Overall** | **7,593** | **6,912** | **6,340** | **572** | **1,253** | **0.9172** | **0.8350** | **0.9605** | **0.8272** |

- **Category Symmetry**: Category 1 and Category 2 exhibit remarkably balanced precision ($91.62\%$ vs $91.86\%$) and recall ($83.64\%$ vs $83.32\%$).
- **Confidence Separation**: Correct entities ($TP$) have an average minimum-token confidence of **$0.9605$**, while incorrect entities ($FP$) average **$0.8272$**, confirming that token probability is a strong statistical signal.

### B. Aspect-Level Error Taxonomy (Sorted by Support)

Errors were classified into three exhaustive types:
1. **Boundary Error ($B$)**: Predicted span overlaps a true entity with the *same* aspect, but boundary differs (e.g. prefix/suffix discrepancy).
2. **Wrong Aspect ($W$)**: Predicted span overlaps a true entity with a *different* aspect.
3. **Spurious Hallucination ($S$)**: Predicted span has zero overlap with any ground-truth entity.

*Note: Aspects with $< 30$ true examples are marked with an asterisk (\*) as statistically unreliable.*

| Aspect Name | Cat | True Count | Pred Count | $TP$ | $FP$ | $FN$ | Precision | Recall | $F_{0.2}$ | Error Type ($B / W / S$) | $FP$ in $[0.60, 0.70)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Kompatibles_Fahrzeug_Modell` | 1 | 988 | 786 | 622 | 164 | 366 | 0.7913 | 0.6296 | 0.7836 | **158 / 5 / 1** | 37 |
| `Im_Lieferumfang_Enthalten` | 1 | 958 | 960 | 908 | 52 | 50 | 0.9458 | 0.9478 | 0.9459 | 32 / 15 / 5 | 6 |
| `Kompatible_Fahrzeug_Marke` | 1 | 578 | 577 | 561 | 16 | 17 | 0.9723 | 0.9706 | 0.9722 | 3 / 13 / 0 | 0 |
| `Einbauposition` | 1 | 549 | 474 | 455 | 19 | 94 | 0.9599 | 0.8288 | 0.9541 | 15 / 2 / 2 | 8 |
| `Hersteller` | 1 | 367 | 357 | 357 | 0 | 10 | 1.0000 | 0.9728 | 0.9989 | 0 / 0 / 0 | 0 |
| `Bremsscheiben-Aussendurchmesser`| 1 | 241 | 242 | 235 | 7 | 6 | 0.9711 | 0.9751 | 0.9712 | 4 / 1 / 2 | 2 |
| `Produktart` | 1 | 192 | 176 | 134 | 42 | 58 | 0.7614 | 0.6979 | 0.7587 | 15 / 10 / 17 | 6 |
| `Bremsscheibenart` | 1 | 67 | 55 | 54 | 1 | 13 | 0.9818 | 0.8060 | 0.9736 | 0 / 1 / 0 | 1 |
| `Maßeinheit` | 1 | 64 | 64 | 64 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 0 / 0 / 0 | 0 |
| `Anzahl_Der_Einheiten` | 1 | 40 | 40 | 38 | 2 | 2 | 0.9500 | 0.9500 | 0.9500 | 0 / 2 / 0 | 0 |
| `Kompatibles_Fahrzeug_Jahr` | 1 | 31 | 30 | 26 | 4 | 5 | 0.8667 | 0.8387 | 0.8656 | 1 / 3 / 0 | 0 |
| `Modell`\* | 1 | 27 | 5 | 3 | 2 | 24 | 0.6000 | 0.1111 | 0.5132 | 0 / 2 / 0 | 0 |
| `Produktlinie`\* | 1 | 22 | 21 | 20 | 1 | 2 | 0.9524 | 0.9091 | 0.9506 | 0 / 1 / 0 | 0 |
| `Material`\* | 1 | 15 | 13 | 12 | 1 | 3 | 0.9231 | 0.8000 | 0.9176 | 0 / 1 / 0 | 0 |
| `Oe/Oem_Referenznummer(N)`\* | 1 | 14 | 0 | 0 | 0 | 14 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Herstellernummer`\* | 1 | 11 | 13 | 4 | 9 | 7 | 0.3077 | 0.3636 | 0.3095 | 0 / 9 / 0 | 8 |
| `Besonderheiten`\* | 1 | 8 | 0 | 0 | 0 | 8 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Stärke`\* | 1 | 5 | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 0 / 0 / 0 | 0 |
| `Größe`\* | 1 | 3 | 0 | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Farbe`\* | 1 | 1 | 0 | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Oberflächenbeschaffenheit`\* | 1 | 1 | 0 | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Kompatibles_Fahrzeug_Modell` | 2 | 889 | 691 | 546 | 145 | 343 | 0.7902 | 0.6142 | 0.7815 | **143 / 2 / 0** | 29 |
| `Kompatible_Fahrzeug_Marke` | 2 | 878 | 874 | 858 | 16 | 20 | 0.9817 | 0.9772 | 0.9815 | 6 / 10 / 0 | 2 |
| `Produktart` | 2 | 515 | 489 | 462 | 27 | 53 | 0.9448 | 0.8971 | 0.9429 | 17 / 8 / 2 | 10 |
| `Hersteller` | 2 | 411 | 405 | 396 | 9 | 15 | 0.9778 | 0.9635 | 0.9772 | 7 / 2 / 0 | 1 |
| `Im_Lieferumfang_Enthalten` | 2 | 317 | 310 | 281 | 29 | 36 | 0.9065 | 0.8864 | 0.9057 | 9 / 19 / 1 | 8 |
| `Herstellernummer` | 2 | 294 | 266 | 250 | 16 | 44 | 0.9398 | 0.8503 | 0.9361 | 1 / 15 / 0 | 1 |
| `Kompatibles_Fahrzeug_Jahr` | 2 | 30 | 27 | 25 | 2 | 5 | 0.9259 | 0.8333 | 0.9220 | 0 / 2 / 0 | 0 |
| `Oe/Oem_Referenznummer(N)`\* | 2 | 26 | 2 | 1 | 1 | 25 | 0.5000 | 0.0385 | 0.3421 | 0 / 1 / 0 | 1 |
| `Anzahl_Der_Einheiten`\* | 2 | 19 | 17 | 15 | 2 | 4 | 0.8824 | 0.7895 | 0.8784 | 0 / 1 / 1 | 0 |
| `Besonderheiten`\* | 2 | 7 | 0 | 0 | 0 | 7 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Menge`\* | 2 | 6 | 6 | 4 | 2 | 2 | 0.6667 | 0.6667 | 0.6667 | 0 / 2 / 0 | 2 |
| `Einbauposition`\* | 2 | 6 | 5 | 3 | 2 | 3 | 0.6000 | 0.5000 | 0.5954 | 2 / 0 / 0 | 0 |
| `Anwendung`\* | 2 | 5 | 0 | 0 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Größe`\* | 2 | 3 | 0 | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |
| `Modell`\* | 2 | 2 | 1 | 0 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | 0 / 1 / 0 | 0 |
| `Maßeinheit`\* | 2 | 2 | 1 | 1 | 0 | 1 | 1.0000 | 0.5000 | 0.9630 | 0 / 0 / 0 | 0 |
| `Zähnezahl`\* | 2 | 1 | 0 | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 / 0 | 0 |

### C. Critical Diagnostic Findings
1. **The Vehicle Model Bottleneck**:
   - `Kompatibles_Fahrzeug_Modell` accounts for **309 out of 572 total False Positives (54.0%)** across both categories.
   - **97.4% (301 of 309)** of its false positives are pure **boundary mismatches** (e.g. predicting `"Golf VII"` when the annotation is `"Golf"`, or predicting `"A4 Avant"` when the annotation is `"A4"`).
   - Only 7 are wrong-aspect classifications, and only 1 is a spurious insertion.
2. **Confidence Clustering of False Positives**:
   - Across the entire model, 122 False Positives fall into the lowest confidence band $[0.60, 0.70)$.
   - Out of these 122 false positives, **66 (54.1%)** come from `Kompatibles_Fahrzeug_Modell`.
   - The remaining aspects have very few low-confidence false positives (e.g., `Hersteller` has 0 FP, `Kompatible_Fahrzeug_Marke` has 2 FP, `Bremsscheiben-Aussendurchmesser` has 2 FP).

---

## 4. Step 3 & Step 5: Predeclared Challenger Policies & Empirical Results

Based strictly on the diagnostic evidence, two simple, conservative challenger policies were predeclared and evaluated using [`src/evaluation/phase_30a_evaluate_policies.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_30a_evaluate_policies.py):

- **Baseline**: Frozen Champion with global MIN 0.60 threshold.
- **Policy A (Aspect-Targeted Adjustment)**: Retain global MIN 0.60 for all aspects, but raise the threshold conservatively to **MIN 0.70 strictly for `Kompatibles_Fahrzeug_Modell`**.
  - *Justification*: `Kompatibles_Fahrzeug_Modell` produces 54.0% of all false positives and has the lowest precision among high-support aspects (0.7908).
- **Policy B (Category-Targeted Adjustment)**: Retain MIN 0.60 for Category 2, but raise the threshold conservatively to **MIN 0.65 for Category 1**.
  - *Justification*: Category 1 accounts for 55.9% of all false positives (320 vs 252) and contains higher low-confidence noise in `Kompatibles_Fahrzeug_Modell`, `Produktart`, and `Herstellernummer`.

### A. Comparative Performance Matrix

| Configuration | Official $F_{0.2}$ | Score Delta ($\Delta$) | Category 1 $F_{0.2}$ | Category 2 $F_{0.2}$ | Precision | Recall | True Pos ($TP$) | False Pos ($FP$) | Total Entities Pruned | $FP$ Pruned vs $TP$ Pruned | Ratio ($FP / TP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Champion (MIN 0.60)** | **0.899802** | **--** | **0.899921** | **0.899683** | **0.9172** | **0.8350** | **6,340** | **572** | **0** | **0 / 0** | -- |
| **Policy A (Modell @ 0.70)** | 0.904582 | +0.004780 | 0.905170 | 0.903993 | 0.9250 | 0.8225 | 6,245 | 506 | 161 | 66 FP / 95 TP | **0.695** |
| **Policy B (Cat 1 @ 0.65)** | 0.901900 | +0.002098 | 0.904117 | 0.899683 | 0.9206 | 0.8309 | 6,309 | 544 | 59 | 28 FP / 31 TP | **0.903** |

### B. Paired Title-Level Bootstrap Resampling ($B = 1,000$ iterations)

| Comparison | Empirical Mean Delta | Standard Error ($SE$) | 95% Confidence Interval | $P(\Delta > 0)$ |
| :--- | :---: | :---: | :---: | :---: |
| **Policy A vs Champion** | **+0.004759** | 0.001187 | [+0.002451, +0.007109] | 100.0% |
| **Policy B vs Champion** | **+0.002080** | 0.000607 | [+0.000912, +0.003272] | 100.0% |

---

## 5. Step 4: Leakage-Resistance & The Metric Gaming Trap

### A. Title-Level Impact Distribution (The Crucial Flaw)
While aggregate bootstrap resampling produces positive confidence intervals on Split A, auditing individual titles reveals why this gain is deceptive:

- **Policy A Title Impact** ($N = 1,000$ titles):
  - **Improved**: 24 titles (2.4%)
  - **Degraded**: **69 titles (6.9%)**
  - **Unchanged**: 907 titles (90.7%)
  - *Negative Ratio*: **Nearly 3x more titles got worse than improved!**
- **Policy B Title Impact** ($N = 1,000$ titles):
  - **Improved**: 6 titles (0.6%)
  - **Degraded**: **28 titles (2.8%)**
  - **Unchanged**: 966 titles (96.6%)
  - *Negative Ratio*: **Nearly 5x more titles got worse than improved!**

### B. Explanation of the Mathematical Illusion
Why does aggregate $F_{0.2}$ increase by $+0.004780$ when almost three times as many titles suffer degradation?
- Under $\beta = 0.2$ ($\beta^2 = 0.04$), precision is weighted **25 times more heavily than recall**:
  $$F_{0.2} = \frac{1.04 \cdot P \cdot R}{0.04 \cdot P + R} \approx P$$
- When Policy A prunes 161 entities, it discards **95 True Positives and 66 False Positives**.
- The ratio of removed false positives to removed true positives is **$0.695 < 1.0$**. More correct entities are destroyed than errors!
- However, because the global precision denominator shrinks by 161, precision jumps from $0.9172$ to $0.9250$ ($+0.0078$), which arithmetically overwhelms the $0.0125$ recall drop.
- **Generalization Risk**: In an uninspected hidden test set, if listing titles have slightly different annotation density or brand/model phrasing, discarding 95 valid vehicle models will severely degrade recall and lead to leaderboard score collapse. This is the exact failure mode previously identified when MIN 0.80 was rejected.

### C. Aspect Extinction Audit
Both policies avoided rare aspect extinction:

| Rare Aspect | MIN 0.60 Baseline | Policy A | Policy B | Extinction Status |
| :--- | :---: | :---: | :---: | :---: |
| `Oe/Oem_Referenznummer(N)` | 2 | 2 | 2 | Preserved |
| `Modell` | 6 | 6 | 6 | Preserved |
| `Stärke` | 5 | 5 | 5 | Preserved |
| `Material` | 13 | 13 | 13 | Preserved |
| `Produktlinie` | 21 | 21 | 21 | Preserved |
| `Besonderheiten` | 0 | 0 | 0 | Zero in Baseline |

---

## 6. Resource Usage & Audit Safeguards

- **Hardware**: NVIDIA GeForce GTX 1660 Ti (6.00 GB VRAM).
- **VRAM Utilization**: 0 MiB / 6144 MiB (Idle).
- **Drive C: Free Space**: **5.37 GB** (strictly maintained above the $\ge 4.0$ GB limit).
- **Temporary Cache Cleanup**: The 2.2 MB fused probability array (`split_a_champion_fused_probs.npy`) was deleted immediately after script execution. Only lightweight audit JSONs remain in `data/cache`.
- **Integrity Checks**:
  - Zero model training conducted.
  - Zero Quiz or Test inference run.
  - Zero EvalAI submissions made.
  - Frozen test pipeline (`src/submission/predict_test_champion.py`) was not modified.
  - All frozen checkpoints remain intact and verified.

---

## 7. Official Recommendation

### Verdict: **RETAIN CHAMPION**

1. **Reject Policy A and Policy B**:
   - Both policies prune more True Positives than False Positives ($FP/TP < 1.0$).
   - Both policies degrade significantly more listing titles than they improve (3x to 5x negative ratio).
   - The aggregate score increase on Split A is an artifact of $\beta = 0.2$ precision-weighting rather than improved extraction accuracy.
2. **Preserve Frozen Champion**:
   - The frozen Experiment 10 + MIN 0.60 champion remains our most reliable, balanced, and verified configuration ($F_{0.2} = \mathbf{0.901245410919943}$ on EvalAI Quiz).
   - No calibration rules should be promoted to the production submission pipeline.
