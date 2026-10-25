# Phase 22/23B Negative Result Analysis: Full-Data Champion vs. Split-A Baseline

**Date:** 2026-10-24  
**Author:** TeamPiyush  
**Status:** Completed Analysis & Strategic Decision  

---

## 1. Official Leaderboard Results & Score Discrepancy

Following Phase 22 (full-data training on all 5,000 labeled titles) and Phase 23B (Quiz candidate inference on records 5,001–30,000), the official EvalAI Quiz evaluation produced the following results:

| Pipeline Configuration | Training Dataset | Ensemble Composition | Official EvalAI Quiz F0.2 | Score Delta vs Baseline | Leaderboard Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Experiment 10 Baseline** | Split-A (4,000 titles) | GBERT 0.40 + 3 XLM-R seeds (0.20 each) | **0.8824567479388766** | Baseline | **Selected Champion (Rank 13)** |
| **Phase 23B Full-Data** | Full Data (5,000 titles) | GBERT 0.40 + 3 XLM-R seeds (0.20 each) | **0.8742365062442021** | **-0.008220** | Rejected (-0.82 points) |

### Key Takeaway
The original Experiment 10 ensemble trained on the 4,000-title Split-A training set **outperformed** the full-data ensemble by **+0.008220 F0.2** on the official Quiz evaluation. Consequently, the team decision is to **restore and freeze Experiment 10 as our champion pipeline** for final competition Test inference.

---

## 2. Aggregate Prediction Differences on the 25,000 Quiz Titles

Both submissions used identical whitespace tokenization, explicit original-token mapping, ensemble probability weights, argmax BIO decoding, official entity reconstruction semantics, and Method A category-aware post-filtering:

| Metric | Exp 10 (Baseline) | Exp 23B (Full-Data) | Delta | Behavioral Shift |
| :--- | :---: | :---: | :---: | :--- |
| **Total Predicted Entity Rows** | 186,164 | **189,363** | **+3,199 (+1.72%)** | Full-data model predicted more entities |
| **Category 1 Entities** | 101,843 | **103,819** | **+1,976 (+1.94%)** | Higher entity density |
| **Category 2 Entities** | 84,321 | **85,544** | **+1,223 (+1.45%)** | Higher entity density |
| **Category-Invalid Entities** | 0 | **0** | **0** | Perfect taxonomy compliance in both |
| **Titles with Identical Predictions** | — | **17,484 (69.94%)** | — | Core predictions were stable |
| **Titles with Changed Predictions** | — | **7,516 (30.06%)** | — | Predictions modified or expanded |

### Aspect Frequency Shifts
The +3,199 net row increase was heavily concentrated in high-frequency aspects:
- `Kompatibles_Fahrzeug_Modell`: **+1,942** rows (39,964 $\rightarrow$ 41,906)
- `Im_Lieferumfang_Enthalten`: **+735** rows (28,345 $\rightarrow$ 29,080)
- `Herstellernummer`: **+600** rows (15,955 $\rightarrow$ 16,555)
- `Einbauposition`: **+309** rows (11,878 $\rightarrow$ 12,187)
- `Bremsscheibenart`: **+232** rows (1,535 $\rightarrow$ 1,767)
- `Hersteller`: **-538** rows (22,182 $\rightarrow$ 21,644)
- `Produktart`: **-279** rows (22,970 $\rightarrow$ 22,691)
- `Oe/Oem_Referenznummer(N)`: **-230** rows (267 $\rightarrow$ 37)

---

## 3. Scientific Analysis of the Decline

The exact root cause of the full-data score decline is **not yet definitively proven**, as we cannot inspect unlabelled Quiz ground truth. However, comparative analysis and diagnostic evidence suggest three primary hypotheses:

### Hypothesis 1: Metric Asymmetry ($F_{0.2}$ Penalizes False Positives 25x More Than False Negatives)
The official competition evaluation metric is $F_{\beta}$ with $\beta = 0.2$:
$$F_{0.2} = \frac{(1 + 0.2^2) \cdot \text{Precision} \cdot \text{Recall}}{0.2^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.04 \cdot P \cdot R}{0.04 P + R}$$
In this formulation, **Precision is weighted 25 times more heavily than Recall**.
The full-data models produced **+3,199 additional entity predictions (+1.72%)**, primarily expanding vehicle model spans (`Kompatibles_Fahrzeug_Modell` +1,942) and accessories (`Im_Lieferumfang_Enthalten` +735). If even a modest fraction of these additional entities are false positives (or boundary over-extensions), precision degrades slightly, inflicting a disproportionately large penalty on $F_{0.2}$.

### Hypothesis 2: Absence of Validation-Guided Early Stopping During Full-Data Training
- **Exp 10 Checkpoints**: Trained on 4,000 titles and checkpointed at the exact validation epoch where local Split-A $F_{0.2}$ reached its global peak (e.g. XLM-R seed 42 at epoch 2, seed 7 at epoch 3).
- **Exp 23B Checkpoints**: Trained on all 5,000 titles for the exact same frozen epoch counts (3, 2, 3, 2). Without validation loss tracking on unseen titles, slight learning rate / step overshooting or undershooting can occur. With 25% more training batches (1,250 steps/epoch vs 1,000 steps/epoch), the model traversed 25% more parameter updates, potentially shifting the decision boundary away from the precision-optimal regime.

### Hypothesis 3: Annotation Noise in the Remaining 1,000 Titles
The 1,000 titles in Split-A were previously held out as validation data. If those titles contain subtle annotation idiosyncrasies, ambiguous boundaries, or rare edge cases, gradient updates on those specific 1,000 titles may have altered the ensemble's token probability calibration, encouraging more aggressive span boundary prediction.

---

## 4. Strategic Decision & Checkpoint Policy

1. **Restore Experiment 10**:
   The four original validation-trained champion checkpoints are restored as the team's frozen production model:
   - `checkpoints/gbert_baseline_best`
   - `checkpoints/xlmr_baseline_best`
   - `checkpoints/xlmr_seed_7_best`
   - `checkpoints/xlmr_seed_21_best`
2. **Preserve Full-Data Artifacts**:
   All Phase 22 full-data checkpoints (`checkpoints/*_fulldata_*_final`) and Phase 23B prediction files will remain preserved for archival and diagnostic purposes, but will **not** be used for final competition Test inference.
3. **Freeze Inference Semantics**:
   - Model weights: 0.40 GBERT + 0.20 for each XLM-R seed.
   - Argmax BIO decoding.
   - `bio_to_entities` reconstruction.
   - Method A category-aware post-filtering (`filter_entities_by_category`).
