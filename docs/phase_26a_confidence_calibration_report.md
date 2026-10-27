# Phase 26A: Precision Confidence Calibration Study Report

**Date**: October 2026  
**Status**: **CONSIDER FILTERED CANDIDATE (With Caveats)**  
**Active Champion Benchmark**: Experiment 10 4-Model Ensemble ($F_{0.2} = 0.879852$)  
**Evaluation Set**: Local Split-A Validation ($N = 1,000$ titles, 11,349 tokens, Seed 42)  

---

## 1. Baseline Reproduction and Validation

Before evaluating confidence filtering, the exact Experiment 10 champion ensemble was loaded and scored across the 1,000 Split-A validation titles:

| Metric | Expected Value | Observed Reproduction | Status |
| :--- | :---: | :---: | :---: |
| **Official Entity $F_{0.2}$** | **0.879852** | **0.879852** | **EXACT MATCH** |
| **Category 1 $F_{0.2}$** | 0.878888 | 0.878888 | **EXACT MATCH** |
| **Category 2 $F_{0.2}$** | 0.880815 | 0.880815 | **EXACT MATCH** |
| **Micro Precision** | 0.8846 | 0.8846 | **EXACT MATCH** |
| **Micro Recall** | 0.8669 | 0.8669 | **EXACT MATCH** |
| **Total True Positives ($TP$)** | 6,582 | 6,582 | **EXACT MATCH** |
| **Total False Positives ($FP$)** | 859 | 859 | **EXACT MATCH** |
| **Total Predicted Entities** | 7,441 | 7,441 | **EXACT MATCH** |

The baseline reproduced bit-for-bit with zero drift.

---

## 2. Confidence Calculation Methodology

1. **Probability Fusion**: For each title token $k$, the fused probability distribution across the 48 BIO classes is computed using the frozen champion weights:
   $$P_{\text{fused}}[k] = 0.40 \cdot P_{\text{GBERT}}[k] + 0.20 \cdot P_{\text{XLM42}}[k] + 0.20 \cdot P_{\text{XLM7}}[k] + 0.20 \cdot P_{\text{XLM21}}[k]$$
2. **Token-Level Selected Confidence**: Let $\hat{y}_k = \operatorname{argmax}_c P_{\text{fused}}[k, c]$ be the predicted BIO label. The token confidence is the probability assigned to that chosen label:
   $$p_k = P_{\text{fused}}[k, \hat{y}_k]$$
3. **Entity Reconstruction**: Standard champion BIO sequence decoding reconstructs entities $[t_i, \dots, t_j]$ with aspect name $A$ and whitespace-joined value.
4. **Category-Aware Post-Filtering**: Entities violating Annexure.pdf category restrictions are removed (Method A).
5. **Entity Confidence Aggregation**: For each remaining entity spanning tokens $\{i, \dots, j\}$:
   - **Mean Method**: $\text{Conf}_{\text{mean}} = \frac{1}{j - i + 1} \sum_{k=i}^j p_k$
   - **Min Method**: $\text{Conf}_{\text{min}} = \min_{k \in \{i, \dots, j\}} p_k$
6. **Threshold Filtering**: Given threshold $\tau \in \{0.00, 0.50, 0.60, 0.70, 0.80\}$, retain the entity if $\text{Conf} \ge \tau$; otherwise discard the entire entity. Never alter boundaries, labels, or token values.

---

## 3. Fixed Threshold Grid Results

| Aggregation | Threshold ($\tau$) | Official $F_{0.2}$ | Delta vs Baseline | Category 1 $F_{0.2}$ | Category 2 $F_{0.2}$ | Micro Precision | Micro Recall | Total Preds | Entities Removed | Pct Removed | $TP$ Removed | $FP$ Removed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MEAN** | **0.00 (Base)**| **0.879852** | **+0.000000** | **0.878888** | **0.880815** | **0.8846** | **0.8669** | **7,441** | **0** | **0.00%** | **0** | **0** |
| **MEAN** | 0.50 | 0.885090 | +0.005239 | 0.883022 | 0.887159 | 0.8958 | 0.8568 | 7,263 | 178 | 2.39% | 76 | 102 |
| **MEAN** | 0.60 | 0.889991 | +0.010139 | 0.888433 | 0.891548 | 0.9050 | 0.8445 | 7,085 | 356 | 4.78% | 170 | 186 |
| **MEAN** | 0.70 | 0.895998 | +0.016146 | 0.896871 | 0.895124 | 0.9159 | 0.8302 | 6,883 | 558 | 7.50% | 278 | 280 |
| **MEAN** | 0.80 | 0.904208 | +0.024357 | 0.905534 | 0.902883 | 0.9332 | 0.7982 | 6,495 | 946 | 12.71% | 521 | 425 |
| **MIN** | **0.00 (Base)**| **0.879852** | **+0.000000** | **0.878888** | **0.880815** | **0.8846** | **0.8669** | **7,441** | **0** | **0.00%** | **0** | **0** |
| **MIN** | 0.50 | 0.887552 | +0.007700 | 0.884054 | 0.891050 | 0.8985 | 0.8554 | 7,229 | 212 | 2.85% | 87 | 125 |
| **MIN** | 0.60 | 0.899802 | +0.019950 | 0.899921 | 0.899683 | 0.9172 | 0.8350 | 6,912 | 529 | 7.11% | 242 | 287 |
| **MIN** | 0.70 | 0.908450 | +0.028598 | 0.910836 | 0.906064 | 0.9321 | 0.8132 | 6,625 | 816 | 10.97% | 407 | 409 |
| **MIN** | **0.80** | **0.914848** | **+0.034996** | **0.912223** | **0.917473** | **0.9469** | **0.7793** | **6,249** | **1,192** | **16.02%** | **665** | **527** |

---

## 4. Precision/Recall and Per-Category Tradeoffs

1. **Metric Sensitivity Dynamic**:
   Under the official competition $F_{0.2}$ metric, precision is weighted 25× more heavily than recall ($\beta^2 = 0.04$).
   Filtering low-confidence entities trades recall for precision:
   - At Min $\tau = 0.80$: Precision surges from $0.8846$ to $0.9469$ ($+0.0623$), while recall drops from $0.8669$ to $0.7793$ ($-0.0876$).
   - Because $F_{0.2} \approx P$, this trade yields a substantial arithmetic score increase ($+0.034996$).
2. **Category Balance**:
   Every threshold in the grid improved **both** Category 1 and Category 2 simultaneously. Neither category experienced degradation at any tested threshold.

---

## 5. Removed True-Positive vs False-Positive Analysis

Analyzing the composition of discarded entities reveals a vital distinction across thresholds:

* **Conservative Thresholds ($\tau \le 0.60$)**:
  - At **Min $\tau = 0.50$**: Discards 212 entities ($2.85\%$). **125 False Positives vs 87 True Positives** (FP/TP ratio = $1.44$). The majority of removed entities are genuine false positives. Recall remains high ($0.8554$).
  - At **Min $\tau = 0.60$**: Discards 529 entities ($7.11\%$). **287 False Positives vs 242 True Positives** (FP/TP ratio = $1.19$).
* **Aggressive Thresholds ($\tau \ge 0.70$)**:
  - At **Min $\tau = 0.80$**: Discards 1,192 entities ($16.02\%$). **527 False Positives vs 665 True Positives** (FP/TP ratio = $0.79$).
  - *Critical Warning*: More than half of the entities pruned at $\tau = 0.80$ are actually valid True Positives ($665$ correct entities discarded). While this mechanically maximizes $F_{0.2}$ on Split A, it represents aggressive recall destruction (recall falls below $78\%$). If the test set has higher label density or unannotated titles, this could backfire on hidden evaluation.

---

## 6. Stability and Overfitting Assessment

To evaluate whether the score improvement is statistically genuine rather than sample noise on Split A, paired bootstrap resampling ($B = 1,000$ iterations) was executed for the highest-scoring configuration (Min $\tau = 0.80$ vs Baseline):

| Bootstrap Resampling Parameter | Observed Value |
| :--- | :---: |
| **Number of Resampling Rounds** | 1,000 titles $\times$ 1,000 rounds |
| **Empirical Mean Delta $\Delta F_{0.2}$** | **+0.034998** |
| **Standard Error ($SE$)** | **0.003490** |
| **95% Confidence Interval** | **[+0.028098, +0.041614]** |
| **Probability of Improvement $P(\Delta > 0)$** | **100.0%** |

The 95% confidence interval is strictly positive and bounded far from zero. The precision boost is mathematically robust across titles.

---

## 7. Final Recommendation & Risk Assessment

### Recommendation: **CONSIDER FILTERED CANDIDATE (With Strict Guardrails)**

1. **Keep Baseline as Unmodified Reference**:
   Do **not** overwrite `predict_test_champion.py` or modify existing champion weights. The baseline champion remains our verified anchor ($0.882456$ on official Quiz).
2. **Recommended Filtered Candidate for Consideration**:
   If filtering is deployed, do **not** use the extreme $\tau = 0.80$ (which discards 665 true positives). Instead, consider the conservative **Min $\tau = 0.50$** or **Min $\tau = 0.60$**:
   - **Min $\tau = 0.50$**: Removes $125$ false positives while sacrificing only $87$ true positives ($2.85\%$ prune rate). Achieves local $F_{0.2} = \mathbf{0.887552}$ ($+0.007700$ gain), preserving $> 85.5\%$ recall and improving both categories safely.
   - **Min $\tau = 0.60$**: Removes $287$ false positives while sacrificing $242$ true positives ($7.11\%$ prune rate). Achieves local $F_{0.2} = \mathbf{0.899802}$ ($+0.019950$ gain).
3. **Safety Status**:
   - No models were retrained.
   - No checkpoints or files were modified.
   - No Quiz inference was run.
   - Zero EvalAI submissions were made.
   - All code is committed cleanly to `origin/main` ([`e2e1e0a`]).
