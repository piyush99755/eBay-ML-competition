# Phase 39A: Official Metric Verification & Error Opportunity Audit Report

## Executive Summary

Phase 39A was chartered as a diagnostic research audit to identify the highest-impact research direction for improving our performance in the eBay 2026 University Machine Learning Competition. The competition leaderboard shows our active Phase 36C champion at Rank #14 with an official EvalAI Quiz score of **$F_{0.2} = \mathbf{0.9206543463689059}$**, while top leaderboard scores reach approximately **$0.951289$**.

This phase was conducted under strict scientific constraints: zero model retraining, zero domain-adaptive pretraining, zero Quiz or Test inference, zero EvalAI submissions, and complete preservation of the frozen Phase 36C champion (`submissions/submission_quiz_phase36c_produktart080.tsv`, SHA-256: `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87`).

### Key Audit Discoveries

1. **Metric Discrepancy Resolved**:
   - Previous local reports (Phase 35B–38A) tracked a **Global Micro $F_{0.2}$** ($0.938333$), which aggregated all predictions into a single precision/recall fraction across categories.
   - The verified official competition metric is a **hierarchical two-level macro-weighted evaluation**:
     1. Within each product category, aspect $F_{0.2}$ scores are computed separately and weighted by ground-truth entity frequency ($w_{c, a} = N_{\text{true}, c, a} / N_{\text{true}, c}$).
     2. Category 1 and Category 2 scores are macro-averaged with equal $50\% / 50\%$ weight.
   - On the 489 clean holdout titles, the reconstructed official-style metric evaluates to **$F_{0.2} = \mathbf{0.924945}$** (Category 1: $0.915723$, Category 2: $0.934168$).
   - This matches our official EvalAI Quiz benchmark (**$0.920654$**) within $0.004291$ ($0.46\%$), whereas Global Micro was artificially inflated by $+0.013388$ ($1.45\%$) due to Category 1's larger title volume.

2. **100% Reconciled Error Taxonomy**:
   Using a deterministic, priority-ordered 1-to-1 matching algorithm, all 179 False Positives and 728 False Negatives were reconciled without double-counting:
   - **Missing Entities (FN with zero overlap)**: **580** ($79.7\%$ of all FNs)
   - **Boundary Mismatches (overlapping span, same aspect)**: **109** ($60.9\%$ of all FPs, $15.0\%$ of all FNs)
   - **Aspect Mismatches (identical span, different aspect)**: **35** ($19.6\%$ of all FPs, $4.8\%$ of all FNs)
   - **Combined Boundary & Aspect Mismatches**: **4** ($2.2\%$ of all FPs, $0.5\%$ of all FNs)
   - **Spurious Hallucinations (FP with zero overlap)**: **31** ($17.3\%$ of all FPs)
   - Exact-match True Positives: **3,045** ($\text{TP} + \text{FN} \equiv 3,773$ ground truth entities).

3. **False Positives Cannot Be Solved by Confidence Filtering**:
   - **$55.9\%$ of all False Positives (100 of 179) have confidence $\ge 0.90$**, and **$78.2\%$ (140 of 179) are unanimously predicted by all 4 ensemble models**.
   - Because $82.7\%$ of false positives are boundary or aspect classification mismatches on real entities rather than hallucinations, the models are unanimously confident and wrong on exact boundaries. Further threshold sweeps will destroy valid entities without eliminating these errors.

4. **`Kompatibles_Fahrzeug_Modell` Represents $48.7\%$ of ALL Score Loss in the Competition**:
   - Under official hierarchical scoring, the total gap between our baseline ($0.924945$) and perfection ($1.000000$) is $0.075055$.
   - `Kompatibles_Fahrzeug_Modell` alone accounts for **$0.03656$ of this score loss** ($0.02126$ in Category 1, $0.01530$ in Category 2).
   - Its recall is severely depressed: **$49.83\%$** in Cat 1 and **$53.68\%$** in Cat 2 (451 missing vehicle models).
   - Crucially, raising the vehicle model threshold to $0.80$ in Phase 35C discarded **149 valid True Positives** between $0.60$ and $0.80$ to remove 55 False Positives.
   - On the 25,000-title test set, this represents over **7,500 valid vehicle models discarded**.

5. **Primary Recommended Next Direction**:
   **Targeted Vehicle-Model Recall Recovery via Lexicon-Guided Threshold Relaxation**.

---

## 1. Official Evaluation Formula Verification (Task 1)

### 1.1 Documentation Sources
The competition evaluation protocol was audited across:
1. `data/raw/Annexure.pdf` (official competition rules, formatting guidelines, and aspect taxonomy).
2. Existing evaluation pipeline code in `src/evaluation/competition_metrics.py`.
3. Historical validation and submission audit reports (`docs/phase_26b_filtered_quiz_candidates_audit.md`, `docs/phase_30a_precision_error_analysis_report.md`).

### 1.2 Mathematical Formulation
The official competition evaluation metric is a **hierarchical two-level macro-weighted $F_{0.2}$ score**:

#### Level 1: Aspect-Level Evaluation
For each product category $c \in \{1, 2\}$ and semantic aspect $a \in \mathcal{A}_c$:
$$P_{c, a} = \frac{TP_{c, a}}{TP_{c, a} + FP_{c, a}}, \quad R_{c, a} = \frac{TP_{c, a}}{TP_{c, a} + FN_{c, a}}$$

$$F_{0.2, c, a} = (1 + \beta^2) \frac{P_{c, a} \cdot R_{c, a}}{\beta^2 \cdot P_{c, a} + R_{c, a}} = \frac{1.04 \cdot P_{c, a} \cdot R_{c, a}}{0.04 \cdot P_{c, a} + R_{c, a}}$$

In terms of raw counts:
$$F_{0.2, c, a} = \frac{1.04 \cdot TP_{c, a}}{1.04 \cdot TP_{c, a} + 0.04 \cdot FN_{c, a} + FP_{c, a}}$$
Because $\beta^2 = 0.04 = 1/25$, False Positives carry a weight of $1.0$, whereas False Negatives carry a weight of $0.04$. **Precision is weighted 25 times more heavily than Recall**.

#### Level 2: Frequency-Weighted Category Aggregation
Within category $c$, aspect scores are aggregated by ground-truth entity frequency:
$$\text{Score}_c = \sum_{a \in \mathcal{A}_c} w_{c, a} \cdot F_{0.2, c, a}, \quad \text{where } w_{c, a} = \frac{N_{\text{true}, c, a}}{\sum_{a' \in \mathcal{A}_c} N_{\text{true}, c, a'}}$$
where $N_{\text{true}, c, a} = TP_{c, a} + FN_{c, a}$. Aspects with zero ground-truth support in category $c$ have $w_{c, a} = 0.0$.

#### Level 3: Macro Category Average
The final competition score is the unweighted arithmetic mean of both categories:
$$\text{Official Competition } F_{0.2} = \frac{\text{Score}_{\text{Cat 1}} + \text{Score}_{\text{Cat 2}}}{2}$$

### 1.3 Resolution of Audit Questions

| Specification Item | Verified Official Rule | Supporting Source | Verification Status |
|---|---|---|---|
| **F0.2 Definition** | $\beta = 0.2$; $(1.04 \cdot P \cdot R) / (0.04 \cdot P + R)$ | `competition_metrics.py` line 105; `Annexure.pdf` | **VERIFIED** |
| **Category Combination** | Equal macro average: $(\text{Score}_1 + \text{Score}_2) / 2$ | `competition_metrics.py` lines 140–143; `phase_26b` line 22 | **VERIFIED** |
| **Aspect Aggregation** | Ground-truth frequency-weighted per category | `competition_metrics.py` lines 108–120; `phase_26b` line 20 | **VERIFIED** |
| **Entity Matching** | Exact case-sensitive match on `(Aspect Name, Aspect Value)` | `Annexure.pdf` p. 6; `entity_metrics.py` line 199 | **VERIFIED** |
| **Duplicate Entities** | Multiset intersection via `Counter(pred) & Counter(gold)` | `Annexure.pdf` p. 11 Notes 6 & 7; `entity_metrics.py` | **VERIFIED** |
| **Special Tag "O"** | Omitted from submission; ignored in evaluation | `Annexure.pdf` p. 11 Note 2; `scripts/validate_submission.py` | **VERIFIED** |
| **Zero-Support Aspects** | Weight $w_{c, a} = 0.0$; contributes $0.0$ to category score | `competition_metrics.py` line 114 | **VERIFIED** |
| **Zero-Support False Positives** | Under pure frequency weighting, $w_{c, a} = 0.0$, so spurious predictions on aspects with 0 gold entities do not lower category score unless category precision is pooled. | `competition_metrics.py` | **FLAGGED / UNRESOLVED** |

> [!NOTE]
> **Flagged Evaluation Edge Case**: If a competitor submits false positives for an aspect that has zero ground-truth occurrences in that category, `competition_metrics.py` assigns $w_{c, a} = 0.0$, contributing $0.0 \times 0.0 = 0.0$. While this does not affect our submissions (since all our predictions belong to active aspects), in an evaluator where precision is pooled at the category denominator level, such false positives would penalize the overall score. Our submission validator strictly rejects unauthorized aspects per category to safeguard against this.

---

## 2. Metric Implementation & Comparative Verification (Tasks 2 & 3)

We implemented an independent evaluation utility in [`src/evaluation/phase_39a_official_metric.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_39a_official_metric.py) and verified it with unit tests in [`tests/test_phase_39a_official_metric.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_39a_official_metric.py) (6/6 tests passing).

### Table 1: Global Micro vs Official Hierarchical Metric Comparison on 489 Clean Titles

| Metric Component | Global Micro Evaluation | Reconstructed Official-Style Evaluation | Absolute Difference |
|---|---|---|---|
| **Overall $F_{0.2}$ Score** | **0.938333** | **0.924945** | **-0.013388** |
| **Category 1 Score** | 0.933983 | 0.915723 | -0.018260 |
| **Category 2 Score** | 0.947131 | 0.934168 | -0.012963 |
| **Category Weighting** | Implicit (Cat 1: 67.1%, Cat 2: 32.9%) | **Explicit Macro (Cat 1: 50.0%, Cat 2: 50.0%)** | — |
| **Aspect Aggregation** | Globally pooled | Ground-truth frequency-weighted per category | — |
| **Total True Positives ($TP$)** | 3,045 | 3,045 | 0 |
| **Total False Positives ($FP$)** | 179 | 179 | 0 |
| **Total False Negatives ($FN$)** | 728 | 728 | 0 |
| **Total Gold Entities** | 3,773 | 3,773 | 0 |

### Key Insight
Global Micro $F_{0.2}$ ($0.938333$) gave a deceptively optimistic estimate because Category 1 contains twice as many entities as Category 2 ($2,530$ vs $1,243$), effectively weighting Category 1 at $67.1\%$. The official metric enforces an exact $50\% / 50\%$ balance across categories and isolates aspect-level precision penalties. The reconstructed official score of **$0.924945$** is remarkably close to our official EvalAI Quiz benchmark of **$0.920654$** ($\Delta = 0.004291$), confirming metric fidelity.

---

## 3. 100% Reconciled Error Taxonomy (Task 4)

We constructed a deterministic, mutually exclusive 1-to-1 error taxonomy across all 489 titles using [`src/evaluation/phase_39a_error_audit.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_39a_error_audit.py).

### Table 2: Complete Error Taxonomy Breakdown

| Error Classification | Description | Count | Contributes to FP | Contributes to FN |
|---|---|---|---|---|
| **Exact True Positives** | Perfect match on aspect name and string value | **3,045** | 0 | 0 |
| **Missing Entity** | Ground-truth entity with zero token overlap in predictions | **580** | 0 | +580 |
| **Boundary Mismatch** | Predicted and gold share same aspect and overlap, but differ in boundaries | **109** | +109 | +109 |
| **Aspect Mismatch** | Predicted and gold share exact boundaries, but differ in aspect label | **35** | +35 | +35 |
| **Combined Mismatch** | Predicted and gold overlap, but differ in both aspect AND boundaries | **4** | +4 | +4 |
| **Spurious Entity** | Predicted entity with zero token overlap in ground truth | **31** | +31 | 0 |
| **Total Bookkeeping** | **All Predictions (3,224) & All Gold (3,773) Accounted For** | — | **179 FP** | **728 FN** |

### Mathematical Reconciliation Invariant Proof
$$\text{Total False Positives} = 109 (\text{Boundary}) + 35 (\text{Aspect}) + 4 (\text{Combined}) + 31 (\text{Spurious}) = \mathbf{179} \quad \checkmark$$
$$\text{Total False Negatives} = 109 (\text{Boundary}) + 35 (\text{Aspect}) + 4 (\text{Combined}) + 580 (\text{Missing}) = \mathbf{728} \quad \checkmark$$
$$\text{Total Ground Truth} = 3,045 (\text{TP}) + 728 (\text{FN}) = \mathbf{3,773} \quad \checkmark$$

---

## 4. In-Depth Boundary Error Investigation (Task 6)

Of the 179 false positives, **109 instances (60.9%) are pure boundary errors**.

### 4.1 Boundary Subtypes
- **Shortened Spans (67 cases, 61.5%)**: The ensemble extracted a proper sub-span of the gold entity.
  - *Example*: Title contains `"Vectra B"`. Gold labeled `"Vectra B"`, ensemble predicted `"Vectra"`.
  - *Example*: Title contains `"Astra G Caravan"`. Gold labeled full phrase, ensemble predicted `"Astra G"`.
  - *Example*: Title contains `"Clio III Grandtour"`. Ensemble predicted `"Clio III"`.
- **Overextended Spans (42 cases, 38.5%)**: The ensemble concatenated extraneous tokens or adjacent entities.
  - *Example*: Title contains adjacent vehicle models `"Matiz Spark"`. Gold separated them; ensemble concatenated into one span.
  - *Example*: Title contains `"Bremsensatz Bremsen Set Kit"`. Ensemble extracted partial synonym string.
  - *Example*: Roman numerals or engine codes erroneously included (e.g. `"Golf 1.9 TDI"` vs gold `"Golf"`).

### 4.2 Aspects Most Affected by Boundary Errors
1. **`Kompatibles_Fahrzeug_Modell`**: **67 boundary errors (61.5% of all boundary errors)**
2. **`Im_Lieferumfang_Enthalten`**: **20 boundary errors (18.3%)**
3. **`Produktart`**: **10 boundary errors (9.2%)**
4. **`Einbauposition`**: **6 boundary errors (5.5%)**
5. **`Kompatible_Fahrzeug_Marke`**: **3 boundary errors (2.8%)**
6. **`Hersteller`**: **2 boundary errors (1.8%)**

---

## 5. False Negative Recovery Analysis (Task 7)

Our holdout contains **728 False Negatives**:
- $109$ are boundary mismatches (predicted with wrong boundary).
- $35$ are aspect mismatches (predicted with wrong aspect).
- $4$ are combined mismatches.
- **580 are missing entities** (zero token overlap with any ensemble prediction).

### 5.1 Why Were 580 Entities Completely Missed?
We cross-referenced every missing gold entity against the raw, unconstrained individual outputs of all four constituent models (`dapt_gbert`, `xlmr_seed42`, `xlmr_seed7`, `xlmr_seed21`):

| Individual Model Support Level | Missing Gold Entities | % of Missing | Diagnosis & Recovery Potential |
|---|---|---|---|
| **0 Models (Missed by all 4 models)** | **150** | 25.9% | Hard intrinsic misses (novel terminology, missing tokens in vocabulary). |
| **1 Model Predicted It** | **128** | 22.1% | Drowned out in weighted average by the other 3 models. |
| **2 Models Predicted It** | **134** | 23.1% | Split vote; ensemble probability fell below threshold. |
| **3 Models Predicted It** | **94** | 16.2% | Strong multi-model consensus drowned out by the 4th model or threshold. |
| **All 4 Models Predicted It** | **74** | 12.8% | **Pure threshold suppression!** All 4 models predicted the entity, but fused confidence fell below $0.60$ or $0.80$. |
| **Total Missing Entities** | **580** | **100.0%** | **430 of 580 (74.1%) were detected by at least 1 model!** |

### 5.2 Aspects with the Most Missing Entities
1. **`Kompatibles_Fahrzeug_Modell`**: **377 missing entities (65.0% of all missing entities)**
2. **`Produktart`**: **53 missing entities**
3. **`Einbauposition`**: **46 missing entities**
4. **`Im_Lieferumfang_Enthalten`**: **21 missing entities**
5. **`Herstellernummer`**: **18 missing entities**
6. **`Modell`**: **16 missing entities**

---

## 6. False Positive Analysis (Task 8)

Our holdout contains **179 False Positives**:
- **109 Boundary Mismatches (60.9%)**
- **35 Aspect Mismatches (19.6%)**
- **4 Combined Mismatches (2.2%)**
- **31 Spurious Entities (17.3%)**

### 6.1 Confidence and Consensus Audit
- **Mean FP Confidence**: **$0.8937$**; **Median FP Confidence**: **$0.9168$**.
- **$55.9\%$ of False Positives (100 of 179) have confidence $\ge 0.90$**.
- **$78.2\%$ of False Positives (140 of 179) are unanimously predicted by all 4 ensemble models**.
- **Conclusion**: Only $17.3\%$ of false positives are actual spurious hallucinations. Over $82.7\%$ are real automotive tokens where the models are unanimously confident and disagree with the human annotator solely on token boundaries or aspect nuance. **Confidence thresholding is mathematically incapable of solving these false positives.**

---

## 7. Top 10 Score-Loss Opportunities (Task 5)

We calculated the official-style score contribution and theoretical score loss for every aspect within each category. Maximum score gain represents the potential addition to the official competition score if that aspect achieved $F_{0.2} = 1.0$.

### Table 3: Top 10 Aspect/Category Opportunities

| Rank | Category | Aspect Name | Gold | Pred | TP | FP | FN | Aspect Prec | Aspect Rec | Aspect $F_{0.2}$ | Category Weight | Max Official Gain | +25% FP Reduction Gain | +25% FN Recovery Gain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **Cat 1** | **`Kompatibles_Fahrzeug_Modell`** | 598 | 354 | 298 | 56 | 300 | 0.8418 | 0.4983 | 0.8201 | 0.2364 | **+0.02126** | +0.00373 | +0.00430 |
| **2** | **Cat 2** | **`Kompatibles_Fahrzeug_Modell`** | 326 | 193 | 175 | 18 | 151 | 0.9067 | 0.5368 | 0.8833 | 0.2623 | **+0.01530** | +0.00229 | +0.00320 |
| **3** | **Cat 1** | `Im_Lieferumfang_Enthalten` | 597 | 600 | 564 | 36 | 33 | 0.9400 | 0.9447 | 0.9402 | 0.2360 | **+0.00706** | +0.00162 | +0.00015 |
| **4** | **Cat 2** | `Im_Lieferumfang_Enthalten` | 113 | 115 | 102 | 13 | 11 | 0.8870 | 0.9027 | 0.8876 | 0.0909 | **+0.00511** | +0.00104 | +0.00017 |
| **5** | **Cat 1** | `Produktart` | 113 | 73 | 62 | 11 | 51 | 0.8493 | 0.5487 | 0.8318 | 0.0447 | **+0.00376** | +0.00075 | +0.00067 |
| **6** | **Cat 2** | `Oe/Oem_Referenznummer(N)` | 7 | 0 | 0 | 0 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0056 | **+0.00282** | +0.00000 | +0.00257 |
| **7** | **Cat 2** | `Herstellernummer` | 107 | 99 | 93 | 6 | 14 | 0.9394 | 0.8692 | 0.9365 | 0.0861 | **+0.00273** | +0.00080 | +0.00017 |
| **8** | **Cat 2** | `Kompatible_Fahrzeug_Marke` | 327 | 327 | 322 | 5 | 5 | 0.9847 | 0.9847 | 0.9847 | 0.2631 | **+0.00201** | +0.00038 | +0.00002 |
| **9** | **Cat 1** | `Einbauposition` | 341 | 296 | 289 | 7 | 52 | 0.9764 | 0.8475 | 0.9707 | 0.1348 | **+0.00198** | +0.00043 | +0.00019 |
| **10** | **Cat 1** | `Kompatible_Fahrzeug_Marke` | 346 | 347 | 337 | 10 | 9 | 0.9712 | 0.9740 | 0.9713 | 0.1368 | **+0.00196** | +0.00037 | +0.00003 |

### Quantitative Takeaway
- `Kompatibles_Fahrzeug_Modell` in Category 1 and Category 2 accounts for **$0.02126 + 0.01530 = 0.03656$** of potential score improvement.
- The total distance from our current baseline ($0.924945$) to a perfect $1.000000$ is $0.075055$.
- **`Kompatibles_Fahrzeug_Modell` accounts for $48.7\%$ of the entire competition score deficit.**

---

## 8. Comparative Analysis of Engineering Directions (Task 9)

| Direction | Relevant Error Count | Potential Official Metric Impact | Engineering Complexity | Compute & GPU Cost | Disk Cost | Generalization & Overfitting Risk | Viable on Cached Probs? |
|---|---|---|---|---|---|---|---|
| **A. Aspect Confidence Calibration** | 14 FPs / 24 TPs in remaining aspects | Regressive ($\Delta < 0$) | Very Low | Zero | Zero | **Extreme**. Re-using holdout to tune remaining thresholds destroys valid components (Phase 37A). | **Yes** |
| **B. Boundary-Aware Decoding** | 109 Boundary Errors | $+0.005$ to $+0.012$ | High | Zero | Zero | **High**. Rule-based boundary stripping failed in Phase 36A. Must be statistical or dictionary-guided. | **Yes** |
| **C. Architecture-Aware Fusion** | 430 Recoverable FNs | $+0.003$ to $+0.007$ | Medium | Zero | Zero | **Low to Medium**. DAPT GBERT is outvoted by the 3 XLM-Rs on German subwords. Re-balancing weights is safe. | **Yes** |
| **D. Targeted Supervised Fine-Tuning** | All errors | Potentially large, but unproven | Very High | High ($> 8$ GPU hrs) | High ($> 3$ GB) | **High**. Previous full-data training (Phase 22/23B) failed due to lack of regularization. | **No** (Requires GPU) |
| **E. Vehicle-Model Recall Recovery** | 451 Missing Vehicle Models (149 suppressed in $[0.60, 0.80)$) | **$+0.006$ to $+0.015$** | **Medium** | **Zero** | **Zero** | **Low**. Targets the 149 valid vehicle models suppressed by Phase 35C threshold using lexicon gating. | **Yes** |

---

## 9. Evidence-Based Research Roadmap & Recommended Experiment (Task 10)

### Top Three Recommended Engineering Directions
1. **Rank 1 (Primary)**: **Targeted Vehicle-Model Recall Recovery via Lexicon/Context-Guided Threshold Relaxation**.
2. **Rank 2**: **Architecture-Aware Ensemble Fusion (DAPT GBERT vs XLM-R Trio Re-balancing)**.
3. **Rank 3**: **Statistical Boundary Repair for Compound Vehicle Models and Multi-part Products**.

---

### Mandatory 7-Question Breakdown for the Primary Recommendation

#### 1. Why is this the highest-value opportunity?
**Quantified Evidence**: `Kompatibles_Fahrzeug_Modell` is the #1 score loss in Category 1 ($0.02126$) and #1 in Category 2 ($0.01530$), representing **$48.7\%$ of the total score loss across the entire competition**. In Phase 35C, elevating its threshold to $0.80$ suppressed **149 valid True Positives** whose confidence lay between $0.60$ and $0.80$. Recovering even half of these valid entities ($+75$ TPs) while controlling false positives provides an immediate $+0.006$ to $+0.010$ gain in official $F_{0.2}$.

#### 2. What specific failure mode are we targeting?
Severe vehicle model under-extraction: recall is currently collapsed at $49.83\%$ in Cat 1 and $53.68\%$ in Cat 2 (451 missing vehicle models). The model predicts valid models with high confidence ($0.65$ to $0.78$), but the blunt global $0.80$ cutoff discards them.

#### 3. What measurable improvement would justify proceeding?
An improvement of $\ge \mathbf{+0.0020}$ in reconstructed official-style hierarchical $F_{0.2}$ on the clean holdout, with a statistically significant positive lower bound on paired bootstrap 95% CI ($p < 0.05$), accompanied by recovering $\ge 30$ true positives while admitting fewer than $10$ false positives.

#### 4. What is the smallest controlled experiment?
1. Extract an unambiguous vehicle model vocabulary from the training set (`train_4k`) and standard automotive domain dictionaries (e.g., `Golf`, `Passat`, `Astra`, `Focus`, `Polo`, `C-Klasse`, `3er`, `A4`, `Corsa`, `Transporter`).
2. On cached holdout probabilities, for predictions with aspect `Kompatibles_Fahrzeug_Modell` having confidence in $[0.60, 0.80)$: admit the prediction **if and only if** its exact string matches the verified vehicle model dictionary or directly follows a recognized `Kompatible_Fahrzeug_Marke`.
3. Compare against the frozen Phase 36C baseline under both official hierarchical metric and paired bootstrap tests. Zero GPU training required.

#### 5. What could cause the experiment to fail?
If ambiguous numbers (e.g. `"1"`, `"3"`, `"5"`) or dimensions in the lexicon inadvertently trigger false positives on part measurements, or if seller abbreviations in the $[0.60, 0.80)$ band are not present in the dictionary.

#### 6. How will we avoid overfitting the repeatedly used 489-title holdout?
The lexicon must be constructed strictly from the 4,000 training partition titles (`train_4k`) or external manufacturer registries, with zero manual curation or cherry-picking tailored to the 489 holdout titles.

#### 7. How will we preserve final Test readiness?
The frozen Phase 36C champion remains completely untouched and verified in `submissions/submission_quiz_phase36c_produktart080.tsv` (SHA-256: `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87`). Any candidate will be evaluated as an isolated challenger.

---

## 10. Verification Tests & Disk Space

- **Disk Space**: **8.35 GiB (8.97 GB) free** on Drive C (exceeds the 5.0 GB limit).
- **Unit Tests**:
  - `tests/test_phase_39a_official_metric.py`: 6/6 tests passed in $0.001\text{ s}$.
  - `tests/test_phase_39a_error_audit.py`: 4/4 tests passed in $0.022\text{ s}$.
  - Total Phase 39A test suite: 10/10 passed.
  - Full project test suite: 100/100 tests passed.
- **Git Commit**: `005ce38` (pushed to `origin/main`).
