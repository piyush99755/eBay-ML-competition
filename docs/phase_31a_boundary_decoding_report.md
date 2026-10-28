# Phase 31A — Boundary-Aware Decoding Research Report

**Status:** COMPLETED & AUDITED  
**Date:** October 27, 2026  
**Active Official Champion:** Experiment 10 (GBERT 0.40 + 3x XLM-R 0.20) with MIN 0.60 Confidence Filter  
**Official EvalAI Quiz F0.2:** **0.901245410919943** (Rank 13)  
**Local Split-A Holdout F0.2:** **0.899802**  
**Production Submission Pipeline:** [`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py) (Frozen & Untouched)  
**Unit Test Suite:** [`tests/test_experiment_31a_boundary_decoding.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_experiment_31a_boundary_decoding.py) (5/5 tests passed)  
**Final Phase 31A Recommendation:** **RETAIN CHAMPION**

---

## 1. Executive Summary & Verdict

Phase 31A investigated whether boundary-aware decoding techniques (Constrained Viterbi Decoding and Local Vehicle-Model Boundary Refinement) could resolve entity boundary errors on the primary error driver (`Kompatibles_Fahrzeug_Modell`) without retraining models or modifying the frozen champion pipeline.

### Core Discoveries:
1. **Audit of Boundary Error Taxonomy**:
   - In Phase 30A, 309 False Positives were identified for `Kompatibles_Fahrzeug_Modell`.
   - Phase 31A's rigorous overlap inspection revealed that **151 of these 309 false positives (48.9%)** actually bridge **multiple distinct ground-truth entities** (one-to-many multi-entity span collisions, common in multi-vehicle listings like `"Golf 4 / Bora / Passat"`).
   - Only **150 false positives (48.5%)** are genuine single-entity same-aspect boundary mismatches.
   - Crucially, **96.0% of these same-aspect boundary errors (144 out of 150)** are **strict sub-spans (missing tokens)** where the model predicted a truncated entity, while only **4.0% (6 errors)** contained extra tokens.
2. **Evaluation of Challenger A (BIO-Constrained Viterbi)**:
   - Dynamic programming enforcing legal BIO transitions achieved $F_{0.2} = \mathbf{0.900593}$ ($\Delta = +0.000791$).
   - However, out of 1,000 titles, only **4 titles** had vehicle-model spans altered.
   - Exact recovery audit revealed that Challenger A achieved **0 wrong-to-correct recoveries** ($0$ boundary errors fixed) and caused **3 correct-to-wrong regressions** ($3$ valid true positives destroyed), resulting in a net loss of 5 True Positives globally. The $+0.000791$ arithmetic gain was entirely a side effect of pruning 7 false positives outside vehicle model (e.g. in product type / brand transitions) rather than boundary recovery.
3. **Evaluation of Challenger B (Local Boundary Refinement)**:
   - **Variant B1 (Joint Window Likelihood)**: Produced **0 span changes**, demonstrating the mathematical theorem that pointwise argmax is already the global joint likelihood maximizer under independent token emissions.
   - **Variant B2 (Length-Normalized Likelihood)**: Produced 61 span changes, but suffered a severe regression ($F_{0.2} = \mathbf{0.898276}$, $\Delta = -0.001526$). It recovered 23 true positives but **destroyed 33 valid true positives** and degraded 29 titles (improving only 20). Length normalization mechanically penalizes multi-token entities, causing valid multi-token names (705 entities) to be aggressively truncated.
4. **Final Decision**: **RETAIN CHAMPION**. Neither challenger demonstrates genuine, leakage-free entity boundary recovery. The frozen Exp 10 + MIN 0.60 champion remains our most reliable and verified system.

---

## 2. Step 1: Baseline Reproduction & Detailed Boundary Error Taxonomy

### A. Exact Bit-for-Bit Champion Reproduction
Scoring the frozen 4-model champion on the 1,000 Split-A validation titles reproduced the canonical benchmark identically:

| Metric | Target Canonical Value | Phase 31A Reproduction | Verification Status |
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

### B. Detailed Vehicle-Model Boundary Taxonomy
The 309 False Positives for `Kompatibles_Fahrzeug_Modell` were audited against all ground-truth spans:

| Error Classification | Cat 1 | Cat 2 | Total Count | Pct of Aspect FP | Nature of Error |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Single True Span: Same-Aspect Boundary Error** | 80 | 70 | **150** | **48.5%** | True boundary mismatch against a single ground-truth entity |
| **Multi-True Span: Overlaps Multiple True Spans** | 78 | 73 | **151** | **48.9%** | Predicted span bridges across $\ge 2$ true entities (e.g. multi-car lists) |
| **Single True Span: Wrong-Aspect Overlap** | 5 | 2 | **7** | **2.3%** | Predicted span overlaps a different aspect (e.g. `Modell` or `Produktlinie`) |
| **Zero True Spans: Spurious Hallucination** | 1 | 0 | **1** | **0.3%** | Predicted span is completely disjoint from any ground-truth annotation |
| **Total False Positives for Vehicle Model** | **164** | **145** | **309** | **100.0%** | — |

### C. Boundary & Token Coverage Discrepancy Breakdown
For the **150 true single-entity same-aspect boundary mismatches**:

1. **Boundary Position**:
   - **Start-Only Errors** ($s \ne s_T, e = e_T$): **55 (36.7%)**
   - **End-Only Errors** ($s = s_T, e \ne e_T$): **78 (52.0%)**
   - **Both-Boundary Errors** ($s \ne s_T, e \ne e_T$): **17 (11.3%)**
2. **Token Coverage**:
   - **Missing Tokens (Strict Sub-Span)**: **144 (96.0%)**
   - **Extra Tokens (Strict Super-Span)**: **6 (4.0%)**
   - **Shifted / Cross Tokens**: **0 (0.0%)**
3. **Multi-Token Structure**:
   - True ground-truth entity is multi-token in **148 out of 150 cases (98.7%)**.
   - Model prediction is single-token in **110 cases (73.3%)** and multi-token in only **40 cases (26.7%)**.
   - **Conclusion**: The model overwhelmingly truncates multi-token vehicle names, extracting only the primary model root (e.g. `"Golf"`) while dropping qualifiers, generation Roman numerals, or body style designations (e.g. `"Golf IV Variant"`).

---

## 3. Step 2: Probability Distributions & BIO Transition Inconsistencies

### A. Raw BIO Transition Violations
Across the 1,000 validation titles (10,349 total token transitions), argmax sequence decoding generated **35 illegal BIO transitions (0.338%)**:
- `B-Im_Lieferumfang_Enthalten -> I-Produktart`: 9 transitions
- `O -> I-Kompatibles_Fahrzeug_Modell`: 6 transitions
- `B-Kompatibles_Fahrzeug_Jahr -> I-Kompatibles_Fahrzeug_Modell`: 3 transitions
- `B-Oe/Oem_Referenznummer(N) -> I-Kompatibles_Fahrzeug_Modell`: 2 transitions
- `B-Kompatible_Fahrzeug_Marke -> I-Kompatibles_Fahrzeug_Modell`: 2 transitions
- `B-Hersteller -> I-Kompatibles_Fahrzeug_Modell`: 2 transitions
- Scattered individual transitions: 11 transitions

*Observation*: After applying the MIN 0.60 confidence filter, 31 of these 35 illegal transitions are eliminated naturally because illegal transitions almost always carry low token confidence ($< 0.60$). Only 4 surviving entities across 1,000 titles involved an illegal transition.

### B. The Back-to-Back B-Tag Phenomenon
In **66 out of 73 end-missing errors**, the token immediately following the truncated model entity ($e+1$) had an argmax label of **`B-Kompatibles_Fahrzeug_Modell`** rather than `I-Kompatibles_Fahrzeug_Modell`!
- The model split contiguous compound model names into two consecutive `B-` entities.
- However, audit revealed that across the dataset, there are 460 instances of adjacent `B-Modell B-Modell` tokens. Merging adjacent `B-Modell` tokens would convert **47 instances to True Positives, but create 413 False Positives** by merging separate cars listed in series (e.g. `"VW Golf Passat Polo"` $\to$ `"Golf Passat Polo"`). A rule merging adjacent `B-` tags is completely unviable.

---

## 4. Step 3 & Step 4: Challenger Implementations & Comparative Evaluation

All models were evaluated using the official competition metric on Split A ($N = 1,000$ titles):

```
=================================================================================================================
PHASE 31A COMPARATIVE DECODING RESULTS
=================================================================================================================
Configuration                      Official F0.2   Delta F0.2  Cat 1 F0.2  Cat 2 F0.2   Prec    Rec    TP     FP    FN
-----------------------------------------------------------------------------------------------------------------
Frozen Champion (Argmax + MIN 0.60) 0.899802        --          0.899921    0.899683   0.9172  0.8350  6340   572   1253
Challenger A (Constrained Viterbi)  0.900593       +0.000791    0.899999    0.901187   0.9181  0.8343  6335   565   1258
Challenger B1 (Joint Window Margin) 0.899802       +0.000000    0.899921    0.899683   0.9172  0.8350  6340   572   1253
Challenger B2 (Length-Norm Margin)  0.898276       -0.001526    0.897243    0.899310   0.9158  0.8337  6330   582   1263
=================================================================================================================
```

### A. Vehicle-Model Specific Performance Breakdown

| Configuration | Aspect $F_{0.2}$ | Precision | Recall | True Positives ($TP$) | False Positives ($FP$) | False Negatives ($FN$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Champion** | **0.7826** | **0.7908** | **0.6223** | **1,168** | **309** | **709** |
| **Challenger A (Viterbi)** | 0.7837 | 0.7920 | 0.6207 | 1,165 | 306 | 712 |
| **Challenger B1 (Joint Window)** | 0.7826 | 0.7908 | 0.6223 | 1,168 | 309 | 709 |
| **Challenger B2 (Length-Norm)** | 0.7759 | 0.7840 | 0.6169 | 1,158 | 319 | 719 |

---

## 5. Exact Entity Recovery & Title-Level Impact Audit

To determine whether challengers actually recover exact-match true positives instead of merely deleting difficult entities:

```
==============================================================================================
EXACT ENTITY RECOVERY & REGRESSION AUDIT (Vehicle Model)
==============================================================================================
Challenger Metric                         Challenger A (Viterbi)    Challenger B2 (Length-Norm)
----------------------------------------------------------------------------------------------
Titles with Span Changes                           4                             55
Total Spans Refined                                4                             61
Wrong-to-Correct (FP -> TP) Recoveries             0                             23
Correct-to-Wrong (TP -> Error) Regressions         3                             33
Incorrect-to-Incorrect (FP -> FP) Changes          0                             37
Net True Positive Change                          -3                            -10
----------------------------------------------------------------------------------------------
Title-Level Impact (out of 1,000 titles):
  - Improved Titles                                1                             20
  - Regressed Titles                               2                             29
  - Unchanged Titles                             997                            951
  - Impact Ratio (Regressed / Improved)          2.0x                          1.45x
==============================================================================================
```

### Deep Failure Analysis:
1. **Challenger A (Viterbi)**:
   - Viterbi did not fix a single boundary error ($0$ wrong-to-correct).
   - In 3 titles, the transition constraint forced a valid multi-token model entity to truncate or split, causing 3 valid True Positives to be lost.
   - The $+0.000791$ arithmetic increase in $F_{0.2}$ was caused by discarding 7 false positives outside vehicle model (e.g. `Produktart` hallucinations with invalid incoming transitions).
2. **Challenger B1 (Joint Window Likelihood)**:
   - Evaluated candidate spans $[s-1, e], [s+1, e], [s, e-1], [s, e+1]$ with joint window likelihood.
   - **Mathematical Invariance Theorem**: Because champion predictions were generated by pointwise argmax, the original span $S$ is by definition the global log-likelihood maximizer over independent token distributions. For every candidate $S'$, $\Delta \mathcal{L}(S') \le 0$. Thus, with any non-negative margin $\gamma \ge 0$, exactly 0 spans changed.
3. **Challenger B2 (Length-Normalized Likelihood)**:
   - When scoring spans by average token probability, trimming boundary tokens artificially increases the score (because the root token has higher probability than boundary tokens).
   - This caused aggressive trimming of valid multi-token entities, destroying **33 True Positives** while recovering only 23, with an overall $F_{0.2}$ drop of $-0.001526$.

---

## 6. Aspect Extinction Audit

All evaluated decoders preserved rare aspects without extinction:

| Rare Aspect | Champion Count | Challenger A Count | Challenger B1 Count | Challenger B2 Count | Extinction Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Oe/Oem_Referenznummer(N)` | 2 | 2 | 2 | 2 | Preserved |
| `Modell` | 6 | 6 | 6 | 6 | Preserved |
| `Stärke` | 5 | 5 | 5 | 5 | Preserved |
| `Material` | 13 | 13 | 13 | 13 | Preserved |
| `Produktlinie` | 21 | 21 | 21 | 21 | Preserved |

---

## 7. Resource Usage & Test Suite Verification

- **Hardware**: NVIDIA GeForce GTX 1660 Ti, **0 MiB / 6144 MiB VRAM used (Idle)**.
- **Drive C: Free Space**: **5.37 GB** (strictly maintained above the $\ge 4.0$ GB limit).
- **Temporary Cache Cleanup**: All temporary probability arrays were removed after execution.
- **Unit Test Suite**: Created [`tests/test_experiment_31a_boundary_decoding.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_experiment_31a_boundary_decoding.py) covering transition legality, Viterbi dynamic programming, collision avoidance, and likelihood invariance. All **5/5 unit tests passed** in 0.001s.
- **Full Test Suite Status**: 16/16 tests across the repository pass cleanly.

---

## 8. Final Decision & Next Steps

### Verdict: **RETAIN CHAMPION**

1. **Reject Challenger A and Challenger B**:
   - Challenger A recovers zero vehicle-model boundary errors and destroys 3 valid True Positives.
   - Challenger B (joint window) is mathematically invariant to argmax, while Challenger B (length-norm) causes net regressions and destroys valid multi-token spans.
2. **Preserve Production Pipeline**:
   - The production submission pipeline ([`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py)) and all 4 frozen checkpoints remain verified and untouched.
   - No experimental decoding logic should be deployed to production.
