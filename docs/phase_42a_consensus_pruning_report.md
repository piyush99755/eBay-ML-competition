# Phase 42A — Model Consensus Audit & Controlled Precision Pruning Report

**Status:** COMPLETE  
**Phase:** 42A  
**Competition:** eBay 2026 University ML Competition — Canada  
**Active Official Champion:** Phase 40 (`submissions/submission_quiz_phase40_lieferumfang080.tsv`, EvalAI Quiz $F_{0.2} = \mathbf{0.9233317116970503}$)  
**Final Decision:** **RETAIN PHASE 40 CHAMPION**

---

## 1. Executive Summary

Phase 42A conducted a rigorous model consensus audit and evaluation of controlled precision pruning on our four-model ensemble (DAPT GBERT seed 42 weight 0.40, XLM-R seeds 42, 7, 21 weight 0.20 each).

The research was designed to test **Hypothesis 1 from Phase 41**, which conjectured that requiring $\ge 2$ model agreement on high-risk aspects would prune isolated false alarms without damaging recall.

### Core Discoveries
1. **Resolution of Phase 41 Consensus Inconsistency:**
   - In Phase 41, the false positive audit found **6 false positives** supported by only 1 model (exact span match).
   - Simultaneously, a preliminary scratch script removed **11 predictions** and yielded $+0.001849$ gain.
   - We audited this at the individual title and token level: the scratch script used **partial token overlap** across all aspects, and the 11 removed predictions were **6 False Positives AND 5 True Positives**!
   - The Phase 41 claim that 11 false positives were eliminated while preserving $100\%$ of true positives was **empirically false**. The script discarded legitimate true positives, but gained score locally because $F_{0.2}$ weights precision $25\times$ more than recall ($\beta^2 = 0.04$).
2. **Strict Evaluation of Predeclared Candidates:**
   - Under the official, exact-match definition (`(record ID, aspect name, aspect value)`):
     - **Candidate A** (applied to `Kompatibles_Fahrzeug_Modell`, `Im_Lieferumfang_Enthalten`, `Produktart`, `Kompatible_Fahrzeug_Marke`) removed 12 predictions: **6 True Positives and 6 False Positives** (50.0% precision on removed entities). Overall score changed by **$+0.001755$** ($0.927771 \to 0.929526$).
     - **Candidate B** (applied exclusively to `Kompatibles_Fahrzeug_Modell`) removed 9 predictions: **4 True Positives and 5 False Positives**. Overall score changed by **$+0.001584$** ($0.927771 \to 0.929354$).
3. **Predeclared Decision Gate Failed:**
   - Neither candidate reached the required **$+0.0020$** threshold for promoting a new challenger or submitting to EvalAI.
   - The 1,000 paired bootstrap percentile confidence intervals cross zero in key strata (Candidate B 95% CI: $[-0.000003, +0.003451]$; Candidate A Category 2 95% CI: $[-0.000751, +0.005187]$).
   - More critically, discarding unambiguous true positives (e.g. `Polo`, `E87`, `Cooper D R56`, `MG`, `Water`) represents an uncontrolled, brittle trade-off driven by sample-level artifacts of 9–12 titles on the clean holdout.
4. **Final Recommendation:** **RETAIN PHASE 40 CHAMPION**. Do not create a Quiz challenger from consensus pruning.

---

## 2. Champion Verification & Baseline Reproduction

### 2.1 Champion File Integrity
- **Submission File:** `submissions/submission_quiz_phase40_lieferumfang080.tsv`
- **SHA-256 Digest:** `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8` (exact match)
- **Official EvalAI Quiz $F_{0.2}$:** **$0.9233317116970503$**

### 2.2 Local Baseline Reproduction (489 Clean Holdout Titles)
- **Overall Official-Style Score:** **$0.927771$**
- **Category 1 Hierarchical $F_{0.2}$:** **$0.918323$** (Gold: 2,530)
- **Category 2 Hierarchical $F_{0.2}$:** **$0.937219$** (Gold: 1,243)
- **Global Micro Metrics:**
  - Micro $F_{0.2}$: **$0.941539$**
  - Micro Precision: **$0.948211$**
  - Micro Recall: **$0.800689$**
  - True Positives (TP): **3,021**
  - False Positives (FP): **165**
  - False Negatives (FN): **752**

All metrics match the frozen baseline within $10^{-6}$.

---

## 3. Resolution of Phase 41 Consensus Inconsistency

### 3.1 The Apparent Conflict
In Phase 41:
- Table 5.1 reported: `"by_consensus": {4: 92, 3: 44, 2: 23, 1: 6}` $\implies$ **6 False Positives** supported by exactly 1 model.
- Hypothesis 1 stated: *"Enforcing a minimum 2-model consensus requirement ... will eliminate 11 isolated false-positive predictions while preserving 100% of true positives."*

### 3.2 Reconciliation and Root Cause Analysis
We performed an exhaustive prediction-by-prediction audit across all 3,186 predictions in the Phase 40 holdout output:

1. **Exact-Match vs Token-Overlap Definition:**
   - In `audit_fp_details.py`, agreement was evaluated on **exact aspect name and exact token span** (`me['aspect'] == asp and me['start'] == s and me['end'] == e`). Under this definition, exactly **6 False Positives** had support == 1.
   - In `test_hypotheses.py`, agreement was evaluated on **partial token overlap** (`max(me['start'], s) < min(me['end'], e)`).
2. **Total Removed vs False Positives Conflation:**
   - In `test_hypotheses.py`, the overlap rule removed **11 total predictions**.
   - The author in Phase 41 saw `Removed 11 entities` and mistakenly assumed they were all False Positives.
   - In truth, of those 11 removed predictions, **6 were False Positives** and **5 were True Positives**!
3. **Exact Support Breakdown Across All 3,186 Predictions:**
   - Predictions with Exact Support == 1: **12 total** (6 TP, 6 FP).
   - Predictions with Exact Support == 2: **79 total** (56 TP, 23 FP).
   - Predictions with Exact Support == 3: **205 total** (161 TP, 44 FP).
   - Predictions with Exact Support == 4: **2,889 total** (2,797 TP, 92 FP).

---

## 4. Exact Consensus Definition

To ensure complete scientific reproducibility, we define the consensus rule strictly without heuristics:
- For each prediction $p = (\text{aspect}, \text{value})$ generated by the Phase 40 four-model ensemble on title $i$, its model support is:
$$\text{Support}(p) = \sum_{m \in \{\text{DAPT}, \text{XLM-R}_{42}, \text{XLM-R}_7, \text{XLM-R}_{21}\}} \mathbb{I}\left[\exists q \in \text{Preds}_m(i) : q.\text{aspect} = p.\text{aspect} \land q.\text{value} = p.\text{value}\right]$$
- Predictions are decoded using the exact frozen competition pipeline (conf_thresh = 0.60, aspect_thresholds = {VM: 0.80, PA: 0.80, Lieferumfang: 0.80}).
- Partial token overlaps (e.g. `Polo` vs `Polo Fox CGPA`) do NOT count as agreement.

---

## 5. Candidate Evaluation & Results

We evaluated exactly the two predeclared candidates:
- **Candidate A:** Remove prediction if $\text{Support}(p) < 2$, applied exclusively to:
  `Kompatibles_Fahrzeug_Modell`, `Im_Lieferumfang_Enthalten`, `Produktart`, `Kompatible_Fahrzeug_Marke`.
- **Candidate B:** Apply the same rule exclusively to `Kompatibles_Fahrzeug_Modell`.

### 5.1 Comparative Metrics Table

| Metric | Phase 40 Baseline | Candidate A | Candidate B | Delta (A vs Base) | Delta (B vs Base) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall Official $F_{0.2}$** | **0.927771** | **0.929526** | **0.929354** | **+0.001755** | **+0.001584** |
| Category 1 Hierarchical $F_{0.2}$ | 0.918323 | 0.920089 | 0.919603 | +0.001766 | +0.001281 |
| Category 2 Hierarchical $F_{0.2}$ | 0.937219 | 0.938963 | 0.939105 | +0.001744 | +0.001887 |
| **Total Predictions Removed** | 0 | 12 | 9 | — | — |
| **True Positives Removed** | 0 | **6** | **4** | -6 TP | -4 TP |
| **False Positives Removed** | 0 | **6** | **5** | -6 FP | -5 FP |
| **False Negatives Introduced** | 0 | **6** | **4** | +6 FN | +4 FN |
| Affected Titles Count | 0 | 12 | 9 | 12 / 489 (2.5%) | 9 / 489 (1.8%) |
| Global Micro $F_{0.2}$ | 0.941539 | 0.943060 | 0.942835 | +0.001521 | +0.001296 |
| Global Micro Precision | 0.948211 | 0.949905 | 0.949638 | +0.001694 | +0.001427 |
| Global Micro Recall | 0.800689 | 0.799099 | 0.799629 | -0.001590 | -0.001060 |

### 5.2 Strict Invariant Verification
- **Subset Assertion:** Both Candidate A and Candidate B are strict subsets of Phase 40 ($A \subset \text{Phase 40}$, $B \subset \text{Phase 40}$). Zero new entities were generated.
- **Support Invariant:** Every removed entity has exact model support $< 2$.
- **TP/FN Tradeoff Invariant:** Every removed True Positive increases False Negatives by exactly 1 ($\Delta \text{FN} = \Delta \text{TP}_{\text{removed}}$).

---

## 6. Comprehensive Audit of Removed Predictions

Below is the complete, exhaustive record of all 12 predictions pruned by Candidate A (Candidate B prunes items 2, 3, 5, 7, 8, 9, 10, 11, 12):

| # | Rec ID | Cat | Aspect | Predicted Value | Status | Support | Supporting Model(s) | Error / Ground Truth Context |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | 3239 | 2 | `Kompatible_Fahrzeug_Marke` | `"MG"` | **TP** | 1 | `dapt_gbert` | Gold has `"MG"`. XLM-R models grouped `"MG ROVER"`. |
| 2 | 3438 | 1 | `Kompatibles_Fahrzeug_Modell` | `"5er E39"` | **FP** | 1 | `xlmr_seed21` | Gold splits into `"5er"` and `"E39"`. Boundary error. |
| 3 | 4967 | 2 | `Kompatibles_Fahrzeug_Modell` | `"Corsa"` | **FP** | 1 | `dapt_gbert` | Gold has `"Corsa A B"`. Boundary error. |
| 4 | 4133 | 1 | `Produktart` | `"Bremsen"` | **FP** | 1 | `xlmr_seed7` | Gold labeled `"Bremsen"` as `Im_Lieferumfang_Enthalten`. |
| 5 | 4910 | 1 | `Kompatibles_Fahrzeug_Modell` | `"X1"` | **FP** | 1 | `dapt_gbert` | Title is for 3ER BMW; gold does not list `X1`. |
| 6 | 2503 | 2 | `Im_Lieferumfang_Enthalten` | `"Water"` | **TP** | 1 | `xlmr_seed7` | Gold has `"Water"` (water pump kit). |
| 7 | 2187 | 2 | `Kompatibles_Fahrzeug_Modell` | `"TWINGO I"` | **FP** | 1 | `dapt_gbert` | Gold has `"TWINGO"`. Boundary error. |
| 8 | 3192 | 1 | `Kompatibles_Fahrzeug_Modell` | `"Z3"` | **FP** | 1 | `xlmr_seed21` | Spurious trim token in BMW title. |
| 9 | 3153 | 2 | `Kompatibles_Fahrzeug_Modell` | `"Cooper D R56"` | **TP** | 1 | `dapt_gbert` | Gold has `"Cooper D R56"`. Correct compound model! |
| 10 | 2567 | 1 | `Kompatibles_Fahrzeug_Modell` | `"S80 Ii"` | **TP** | 0 | None (Ensemble average) | Gold has `"S80 Ii"`. Genuine true positive! |
| 11 | 4115 | 2 | `Kompatibles_Fahrzeug_Modell` | `"Polo"` | **TP** | 1 | `dapt_gbert` | Gold has `"Polo"`. XLM-R models concatenated engine code. |
| 12 | 3146 | 1 | `Kompatibles_Fahrzeug_Modell` | `"E87"` | **TP** | 1 | `xlmr_seed42` | Gold has `"E87"`. Genuine BMW model code! |

### The Critical Flaw
Look closely at items 1, 6, 9, 10, 11, 12:
- `"Polo"` (Volkswagen Polo) is an undisputed, prominent vehicle model.
- `"E87"` (BMW 1-Series) is an exact BMW chassis code.
- `"Cooper D R56"` (Mini Cooper Diesel) is an exact gold annotation.
- `"S80 Ii"` (Volvo S80 Generation 2) is an exact gold annotation.
- `"MG"` (MG Rover) is an exact vehicle brand.

These are **NOT hallucinations or noisy predictions**. They are valid, high-value True Positives. The consensus rule discards them simply because individual XLM-R models had boundary disagreements (e.g. predicting `Polo Fox CGPA` instead of `Polo`).

The pruning mechanism does not fix boundary problems—it simply **throws away valid entities** to exploit the mathematical ratio $\frac{1}{\beta^2} = 25$ in $F_{0.2}$.

---

## 7. Paired Bootstrap Analysis (1,000 Resamples, Seed 42)

We conducted 1,000 paired, category-stratified bootstrap resamples of the 489 clean holdout titles, recalculating the official gold-frequency aspect weights for each resample:

### Candidate A
- **Overall Score Delta ($\Delta$):**
  - Mean: **$+0.001719$**
  - Standard Deviation: $0.000941$
  - **95% Percentile Confidence Interval:** **$[+0.000074, +0.003704]$**
  - $P(\Delta > 0)$: $97.7\%$
- **Category 1 Delta:** Mean $+0.001750$, 95% CI: $[+0.000038, +0.003925]$
- **Category 2 Delta:** Mean $+0.001689$, **95% CI: $[-0.000751, +0.005187]$** (Crosses zero!)

### Candidate B
- **Overall Score Delta ($\Delta$):**
  - Mean: **$+0.001554$**
  - Standard Deviation: $0.000889$
  - **95% Percentile Confidence Interval:** **$[-0.000003, +0.003451]$** (Crosses zero!)
  - $P(\Delta > 0)$: $97.2\%$
- **Category 1 Delta:** Mean $+0.001268$, **95% CI: $[-0.000291, +0.003336]$** (Crosses zero!)
- **Category 2 Delta:** Mean $+0.001840$, **95% CI: $[-0.000551, +0.005370]$** (Crosses zero!)

### Interpretation
While $P(\Delta > 0)$ is high, the lower bound of the confidence interval is virtually zero for Candidate A ($+0.000074$) and negative for Candidate B. In Category 2, Candidate A's interval crosses zero. This proves that the observed gain is heavily dependent on a handful of sampled titles in the holdout.

---

## 8. Leakage, Privacy, & Governance Guarantees

1. **Repeated Holdout Usage:** The 489-title clean holdout has been used across Phases 35B–42A. Apparent gains of $+0.0017$ across 12 titles are well within the margin of holdout overfitting.
2. **Zero Quiz/Test Leakage:** No test or quiz data was used to optimize or select rules.
3. **Privacy Compliance:** No raw user titles or confidential seller information appear in code or logs.
4. **Asset Integrity:** Active champion `submissions/submission_quiz_phase40_lieferumfang080.tsv` remains cryptographically locked and intact.

---

## 9. Final Decision & Recommendation

### Decision Criteria Assessment
- **Gain $\ge +0.0020$:** **FAILED.** Candidate A achieves $+0.001755$; Candidate B achieves $+0.001584$.
- **Improvement Mechanism Verified:** **FAILED.** 50% of pruned entities are valid True Positives. The mechanism is unselective truncation, not precision calibration.
- **Bootstrap Robustness:** **MARGINAL/WEAK.** Confidence intervals touch or cross zero in Category 2 and overall for Candidate B.
- **Risk Profile:** **HIGH.** Discarding true vehicle models and brands on test data will hurt recall whenever test annotations are clean.

### Final Verdict:

**RETAIN PHASE 40 CHAMPION**

Do not submit Candidate A or Candidate B to EvalAI.
