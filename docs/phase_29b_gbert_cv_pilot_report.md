# Phase 29B — Three-Fold GBERT Ensemble Pilot Report

**Status:** COMPLETED & AUDITED  
**Date:** October 27, 2026  
**Active Official Champion:** Exp 10 (GBERT 0.40 + 3x XLM-R 0.20) with MIN 0.60 confidence filter  
**Official Quiz F0.2:** **0.901245410919943** (EvalAI Rank 13)  
**Local Split-A Validation F0.2:** **0.899802**  
**Pilot Recommendation:** **RETAIN FROZEN CHAMPION** (Challenger scored 0.895643, $\Delta = -0.004159$)

---

## 1. Executive Summary

In Phase 29B, we conducted a rigorous, leakage-free 3-fold cross-validation pilot using `deepset/gbert-base`. The core question was whether ensembling 3 fold-models trained on 66.7% data partitions (2,667 / 2,666 titles) could improve upon our single full Split-A GBERT model (4,000 titles) when integrated into the 4-model ensemble champion (fused with the three frozen XLM-R models and filtered at MIN 0.60 confidence).

### Key Findings
1. **Sleep Resumption**: During training of Fold 1, the host machine entered Windows sleep mode for approximately 2.4 hours. Upon wake-up, the PyTorch/CUDA training process resumed execution seamlessly without NaN loss, gradient degradation, or process termination. All 3 folds and all 3 epochs completed to 100%.
2. **Individual Fold Resilience**: Training GBERT on 66.7% data partitions maintained strong performance without collapse:
   - Fold 0 validation $F_{0.2}$: **0.873544** (Holdout: 0.868441, $+0.001241$ vs single GBERT)
   - Fold 1 validation $F_{0.2}$: **0.862406** (Holdout: 0.851769, $-0.015431$)
   - Fold 2 validation $F_{0.2}$: **0.861048** (Holdout: 0.856447, $-0.010753$)
3. **Cross-Validation Ensembling Penalty**:
   - The 3-fold GBERT-only ensemble on the 1,000-title holdout reached $F_{0.2} = \mathbf{0.863321}$, trailing the single GBERT model trained on all 4,000 titles ($F_{0.2} = \mathbf{0.867200}$) by $-0.003879$.
   - The full 6-model challenger ($0.40 \times \text{GBERT}_{\text{CV3}} + 0.60 \times \text{XLM-R}_{\text{3-seed}}$ + MIN 0.60 filter) achieved $F_{0.2} = \mathbf{0.895643}$, underperforming our frozen Exp 10 champion ($F_{0.2} = \mathbf{0.899802}$) by $\mathbf{-0.004159}$.
4. **Decisive Outcome**: Training on smaller data subsets (even when ensembled) loses critical lexical and rare aspect coverage that cannot be compensated for by bagging. The frozen Exp 10 + MIN 0.60 champion remains our highest performing, fully verified configuration.

---

## 2. Partition Verification & Mathematical Integrity

Prior to training, the 4,000 Split-A training titles were partitioned into 3 category-stratified folds using `StratifiedGroupKFold(n_splits=3, random_state=42)` in [`src/data/make_cv_folds.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/data/make_cv_folds.py) and verified via [`tests/test_cv_folds.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_cv_folds.py):

| Fold | Training Size | Validation Size | Category 1 (Val) | Category 2 (Val) | Near-Duplicate Pair |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 2,667 | 1,333 | 666 (49.96%) | 667 (50.04%) | Record 1779 & 4204 (Grouped) |
| **Fold 1** | 2,666 | 1,334 | 667 (50.00%) | 667 (50.00%) | Disjoint |
| **Fold 2** | 2,667 | 1,333 | 667 (50.04%) | 666 (49.96%) | Disjoint |
| **Total** | **4,000** | **4,000** | **2,000** | **2,000** | **Zero Cross-Fold Leakage** |

- **Holdout Integrity**: The 1,000-title Split-A validation set was completely withheld as an untouched external evaluation set.
- **Near-Duplicate Audit**: Exactly one near-duplicate pair existed in the 4,000 training titles (`Record 1779` and `Record 4204`, Category 1). Both were strictly placed into Fold 0 validation (never split across train/validation).

---

## 3. Training Execution & Windows Sleep Diagnostics

Sequential training ran under `task-2589` using [`src/modeling/train_gbert_cv3.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/modeling/train_gbert_cv3.py):

| Metric | Fold 0 | Fold 1 | Fold 2 |
| :--- | :---: | :---: | :---: |
| **Epoch 1 Val Loss / F0.2** | 0.3320 / 0.846213 | 0.3624 / 0.839044 | 0.3538 / 0.830427 |
| **Epoch 2 Val Loss / F0.2** | 0.3120 / 0.864482 | 0.3149 / 0.863323 | 0.3282 / 0.837727 |
| **Epoch 3 Val Loss / F0.2** | **0.3021 / 0.873544** | **0.3156 / 0.862406** | **0.3157 / 0.861048** |
| **Epoch 3 Precision / Recall** | 0.8749 / 0.8494 | 0.8636 / 0.8668 | 0.8647 / 0.8618 |
| **Saved Checkpoint** | `checkpoints/gbert_cv3_fold0_best` | `checkpoints/gbert_cv3_fold1_best` | `checkpoints/gbert_cv3_fold2_best` |
| **Training Duration** | 2,442.4s (40.7m) | 10,300.4s (171.7m)* | 1,035.9s (17.3m) |

*\*Note on Sleep Event: During Fold 1 Epoch 3 (between step 400 and step 600), the host laptop went to sleep for ~8,700 seconds (2.4 hours). The process resumed cleanly upon system wake-up, completed Fold 1 evaluation, and trained Fold 2 without error.*

- **Hardware & Disk Utilization**:
  - GPU: NVIDIA GeForce GTX 1660 Ti (6.00 GB VRAM), peak utilization ~3.2 GB VRAM.
  - Final C: Drive Free Space: **5.58 GB** (strictly maintained above the $\ge 4.0$ GB safety threshold).

---

## 4. Evaluation Gate Results

All gates were scored using the official-style competition metric:
$$\beta = 0.2, \quad F_{0.2, c, a} = \frac{1.04 \cdot P_{c, a} \cdot R_{c, a}}{0.04 \cdot P_{c, a} + R_{c, a}}$$
Category scores are macro-averaged across aspects within each category, then macro-averaged across Category 1 and Category 2.

```
========================================================================================
EVALUATION GATES COMPARISON TABLE
========================================================================================
Gate / Configuration                   F0.2      Cat 1 F0.2  Cat 2 F0.2   Precision  Recall
----------------------------------------------------------------------------------------
Gate A: OOF (4,000 titles)            0.864973    0.879943    0.850004     0.8681    0.8593
Gate B: Fold 0 Holdout (1,000 titles) 0.868441    0.877253    0.859628     0.8699    0.8521
Gate B: Fold 1 Holdout (1,000 titles) 0.851769    0.863171    0.840366     0.8548    0.8558
Gate B: Fold 2 Holdout (1,000 titles) 0.856447    0.866742    0.846152     0.8618    0.8574
Single GBERT Baseline (1,000 titles)  0.867200    0.873215    0.861185     0.8680    0.8639
Gate C: 3-Fold GBERT Ens (Holdout)    0.863321    0.866816    0.859825     0.8662    0.8613
----------------------------------------------------------------------------------------
Frozen Champion (Exp 10 + MIN 0.60)   0.899802    0.899878    0.899726     0.9168    0.8400
Gate D: 6-Model Challenger (+ MIN 0.60)0.895643    0.895564    0.895722     0.9135    0.8360
Delta (Challenger vs Champion)       -0.004159   -0.004314   -0.004004    -0.0033   -0.0040
========================================================================================
```

### Detailed Gate Analysis
1. **Gate A (Out-of-Fold Diagnostic)**: Across all 4,000 training titles, combining the 3 disjoint validation folds yielded an out-of-fold $F_{0.2}$ of **0.864973** ($P=0.8681, R=0.8593$, TP=26,129, FP=3,970, FN=4,277). This confirms consistent model calibration across the entire training space.
2. **Gate B (Individual Holdout Generalization)**: Only Fold 0 slightly surpassed the single baseline GBERT (+0.001241), while Fold 1 (-0.015431) and Fold 2 (-0.010753) showed substantial drops due to reduced training sample diversity (missing 1,333 titles each).
3. **Gate C (GBERT CV Ensembling)**: Averaging the predicted probabilities of the three fold-models on the 1,000-title holdout produced an $F_{0.2}$ of **0.863321**, which is **0.003879 points lower** than the single GBERT model trained on all 4,000 titles. Because each fold model only saw 2,667 examples, ensembling 3 data-deprived models could not compensate for the loss of 1,333 training titles per model.
4. **Gate D (Full 6-Model Challenger Ensemble)**: Replacing the single 4,000-title GBERT with the 3-fold GBERT average inside the 4-model champion ($0.40 \times \text{GBERT} + 0.20 \times \text{XLMR}_{42} + 0.20 \times \text{XLMR}_{7} + 0.20 \times \text{XLMR}_{21}$) reduced overall holdout $F_{0.2}$ from **0.899802 to 0.895643** (Delta: **-0.004159**). Precision decreased slightly from 0.9168 to 0.9135, and Recall decreased from 0.8400 to 0.8360.

---

## 5. Aspect Emission & Extinction Check

Under the MIN 0.60 confidence filter, the 6-model challenger generated 6,949 entity predictions:

| Aspect Name | Predictions | Status |
| :--- | :---: | :---: |
| `Kompatibles_Fahrzeug_Modell` | 1,502 | Healthy |
| `Kompatible_Fahrzeug_Marke` | 1,458 | Healthy |
| `Im_Lieferumfang_Enthalten` | 1,277 | Healthy |
| `Hersteller` | 762 | Healthy |
| `Produktart` | 661 | Healthy |
| `Einbauposition` | 500 | Healthy |
| `Herstellernummer` | 271 | Healthy |
| `Bremsscheiben-Aussendurchmesser` | 240 | Healthy |
| `Maßeinheit` | 64 | Healthy |
| `Anzahl_Der_Einheiten` | 57 | Healthy |
| `Kompatibles_Fahrzeug_Jahr` | 56 | Healthy |
| `Bremsscheibenart` | 55 | Healthy |
| `Produktlinie` | 21 | Healthy |
| `Material` | 13 | Healthy |
| `Modell` | 6 | Healthy |
| `Stärke` | 5 | Healthy |
| `Oe/Oem_Referenznummer(N)` | 1 | Preserved |

- **Extinction Result**: Exactly 0 aspects suffered extinction. The rarest aspect (`Oe/Oem_Referenznummer(N)`) retained 1 confident prediction, identical to the baseline champion.

---

## 6. Official Decision & Next Steps

1. **Retain Official Champion**:
   - The frozen Exp 10 ensemble (GBERT 0.40 + 3x XLM-R 0.20 each) with MIN 0.60 threshold remains our official champion ($F_{0.2} = \mathbf{0.901245410919943}$ on EvalAI Quiz, $\mathbf{0.899802}$ local Split-A holdout).
   - The 3-fold GBERT ensemble is formally archived and rejected for final submission.
2. **Preservation of Pipeline Integrity**:
   - The Phase 28 final test pipeline (`src/submission/predict_test_champion.py`) remains untouched and ready for final test data.
   - All four frozen champion checkpoints (`checkpoints/gbert_baseline_best`, `checkpoints/xlmr_baseline_best`, `checkpoints/xlmr_seed_7_best`, `checkpoints/xlmr_seed_21_best`) remain verified and intact.
