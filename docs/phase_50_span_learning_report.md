# Phase 50: Boundary-Aware Span Learning Report

**Date:** October 30, 2026  
**Status:** In Progress / Controlled Execution  
**Active Champion:** `submissions/submission_quiz_phase40_lieferumfang080.tsv`  
**Champion SHA-256:** `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8` (Verified Intact)  
**Official EvalAI Quiz $F_{0.2}$ Score:** `0.9233317116970503`  
**Hardware & VRAM:** NVIDIA GeForce GTX 1660 Ti (6.00 GB VRAM)  
**Free Disk Space Floor:** $\ge 5.0$ GB (Verified: 6.01 GB preflight)  

---

## Executive Summary

Phase 50 investigates whether a lightweight, boundary-aware span-based named entity recognition (Span-NER) architecture built upon the domain-adapted GBERT encoder (`checkpoints/phase33b_dapt_encoder`) can resolve exact entity boundary mismatches on German automotive eBay listing titles. 

Unlike token-level BIO sequence taggers—which make independent token decisions or linear-chain transitions that frequently overextend compound boundaries (e.g., `Astra J Caravan` vs `Astra J`)—a span-based classifier constructs explicit span-level representations for every candidate token span $(start, end)$ and predicts the exact aspect label directly.

---

## Part A: Metric Verification & False-Positive Root Cause Taxonomy

### 1. Algebraic Verification of Official $F_{0.2}$ Evaluation
The competition evaluates predictions using hierarchical precision-weighted $F_{0.2}$ with support-weighted aspect aggregation within each category, followed by equal macro-averaging of Category 1 and Category 2:
$$F_{\beta} = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}, \quad \beta = 0.2, \quad \beta^2 = 0.04$$

Substituting precision $P = \frac{TP}{TP + FP}$ and recall $R = \frac{TP}{TP + FN}$ into the entity-level formula:
$$F_{0.2} = \frac{1.04 \cdot \frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}}{0.04 \cdot \frac{TP}{TP + FP} + \frac{TP}{TP + FN}} = \frac{1.04 \cdot TP^2}{0.04 \cdot TP(TP + FN) + TP(TP + FP)}$$
Factoring out $TP > 0$:
$$F_{0.2} = \frac{1.04 \cdot TP}{0.04(TP + FN) + (TP + FP)} = \frac{1.04 \cdot TP}{1.04 \cdot TP + 0.04 \cdot FN + FP}$$

**Empirical Verification:** Direct comparison between `compute_fbeta(P, R, beta=0.2)` in `src/evaluation/phase_39a_official_metric.py` and the direct algebraic formula across test cases yielded zero discrepancy ($< 10^{-16}$).

### 2. Error Taxonomy Verification from Phase 49
Phase 49 audited **2,134 false positives** from training-only 5-fold cross-validation out-of-fold predictions (`data/cache/phase_35b_dapt_gbert_oof_preds.pt`):

```
+-----------------------------------------------------------------------------------------+
| ROOT CAUSE CLASS                         | COUNT | PERCENTAGE | PRIMARY ERROR PATTERN   |
+-----------------------------------------------------------------------------------------+
| Class B: Wrong Entity Boundary           | 1,378 |     64.57% | Boundary overextension  |
| Class C: Wrong Aspect Label              |   444 |     20.81% | OE/OEM vs Herstellernr  |
| Class A: Spurious Hallucinations         |   291 |     13.64% | Pure non-entity text    |
| Class D: Valid Entity Omitted in Gold    |    21 |      0.98% | Unannotated part terms  |
+-----------------------------------------------------------------------------------------+
| TOTAL                                    | 2,134 |    100.00% |                         |
+-----------------------------------------------------------------------------------------+
```

### 3. Aspect Breakdown of Boundary Errors (Class B):
- `Kompatibles_Fahrzeug_Modell`: **957 errors** (69.4% of Class B; **44.85% of all false positives** in the competition).
- `Produktart`: **211 errors** (15.3% of Class B).
- `Im_Lieferumfang_Enthalten`: **121 errors** (8.8% of Class B).
- `Einbauposition`: **34 errors** (2.5% of Class B).
- Remaining 25 aspects: **55 errors** (4.0% of Class B).

### 4. Categorization Reproducibility & Boundary Ambiguities:
- **Priority Resolution:** Exact matches ($TP$) are removed first via multiset intersection. Unmatched predictions are categorized strictly in order: Class C (exact span, wrong aspect) $\to$ Class B (overlapping span with gold of same aspect) $\to$ Class D (unannotated domain match) $\to$ Class A (spurious).
- **Hybrid Error Handling:** In cases where a predicted span overlaps with a gold entity of a *different* aspect with *non-exact* boundaries, the prediction falls to Class A because neither the boundary nor the aspect matches. Such hybrid errors account for $<1.8\%$ of errors.

---

## Part B: Data Provenance & Leak-Free Partitioning

To guarantee zero contamination of test distributions:
1. **Supervised Data:** 4,000 listing titles from `data/raw/train.csv`.
2. **Partitioning Design (`src/data/make_internal_split.py`):**
   - **3,200 Internal Train Titles:** 1,600 Category 1 + 1,600 Category 2.
   - **800 Internal Dev Titles:** 400 Category 1 + 400 Category 2.
   - Partitioned using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouping near-duplicate titles to prevent train/dev leakage.
3. **Strictly Isolated Holdouts:**
   - `data/splits/val_split.json` (1,000 historical validation titles): Zero exposure during training or model selection.
   - `data/cache/phase_35b_clean_holdout_manifest.json` (489 clean holdout titles): Strictly reserved for secondary diagnostics.
4. **Baseline Provenance Clarification:**
   - Checkpoint `checkpoints/phase33b_dapt_gbert_best` was fine-tuned on all 4,000 training titles (meaning the 800 dev titles were in its training distribution). Its high in-sample score on the 800 dev set ($0.942161$) reflects train-set evaluation.
   - Checkpoint `checkpoints/gbert_phase49_precision_aware_best` was fine-tuned strictly on the 3,200 internal train split, scoring $0.906161$ on the 800 dev set ($0.917155$ Cat 1 / $0.895168$ Cat 2). This serves as the genuine out-of-sample baseline.

---

## Part C: Span Model Feasibility Analysis

### 1. Empirical Span Length Statistics (3,200 Internal Train Titles)
- Total Gold Entities: **24,259 entities**.
- Mean Title Length: **11.35 whitespace tokens** (Max: 21 tokens).
- Mean Gold Entity Span Length: **1.25 tokens** (Median: 1.0 token).
- Percentiles: $p_{90} = 2.0$, $p_{95} = 3.0$, $p_{99} = 5.0$, $p_{99.5} = 6.0$, $p_{99.9} = 10.0$, $\max = 16.0$.

### 2. Gold Span Coverage by Maximum Span Width ($W_{max}$):
```
+-----------------------------------------------------------------------------------------+
| MAX SPAN WIDTH (W_max)  | COVERED SPANS | TOTAL SPANS | COVERAGE PERCENTAGE             |
+-----------------------------------------------------------------------------------------+
| W <= 1                  |        20,814 |      24,259 |                          85.80% |
| W <= 2                  |        22,961 |      24,259 |                          94.65% |
| W <= 3                  |        23,642 |      24,259 |                          97.46% |
| W <= 4                  |        23,940 |      24,259 |                          98.69% |
| W <= 5                  |        24,075 |      24,259 |                          99.24% |
| W <= 6                  |        24,151 |      24,259 |                          99.55% |
| W <= 8 (Selected)       |        24,209 |      24,259 |               99.79% (COVERED)  |
| W <= 10                 |        24,244 |      24,259 |                          99.94% |
| W <= 12                 |        24,253 |      24,259 |                          99.98% |
| W <= 15                 |        24,258 |      24,259 |                         100.00% |
+-----------------------------------------------------------------------------------------+
```
**Selection:** $W_{max} = 8$ was chosen as the optimal operational boundary. It captures **99.79%** of all gold entities while generating only an average of **62.9 candidate spans per title** (max 140).

### 3. Structural Properties of Annotations:
- **Overlapping Entities in Gold:** **0**
- **Nested Entities in Gold:** **0**
- Because competition annotations follow flat BIO sequences, overlapping or nested entities do not exist in the training labels. A greedy non-overlapping decoder strictly preserves validity without discarding nested gold annotations.

### 4. Subword Boundary Alignment:
- Total Gold Entities Checked: **24,259**.
- Unaligned Entities: **0** (**100.00% alignment success rate**).
- Every whitespace token span boundary $[start, end]$ maps onto subword ranges $[sub\_start, sub\_end]$ produced by the GBERT WordPiece tokenizer.

### 5. Computational Feasibility:
- **GPU Memory:** 2,483 MiB allocated during training (well within 6,144 MiB GTX 1660 Ti capacity).
- **Training Runtime:** Estimated at ~13–15 minutes for 3 epochs.
- **Disk Storage:** Model weights + config $\approx 417$ MB (free disk maintained above 5.5 GB).

---

## Part D: Controlled Architecture & Training Design

### 1. Model Architecture (`SpanNERModel`)
- **Base Encoder:** Domain-Adapted German BERT (`checkpoints/phase33b_dapt_encoder`, 110M params).
- **Span Width Embedding:** `nn.Embedding(num_embeddings=9, embedding_dim=64)`.
- **Span Representation:** For candidate span $[s, e]$ with whitespace width $w \in [1, 8]$:
  $$h_{span} = [ h_{sub\_start}; h_{sub\_end}; \text{WidthEmb}(w) ] \in \mathbb{R}^{768 + 768 + 64} = \mathbb{R}^{1600}$$
- **Classification Head:**
  $$\text{Linear}(1600 \to 256) \to \text{GELU} \to \text{Dropout}(0.10) \to \text{Linear}(256 \to 30)$$
  Predicts logits over 30 mutually exclusive classes (Class 0: `NONE`, Classes 1–29: competition aspects).

### 2. Negative Span Sampling Policy (Training Only)
To prevent the background non-entity class (`NONE`) from overwhelming gradient updates while focusing training on boundary delineation:
1. **Positives:** 100% of gold entity spans are included in every title.
2. **Hard Boundary Negatives:** Every span $[g_s \pm 1, g_e]$, $[g_s, g_e \pm 1]$, $[g_s \pm 2, g_e]$, $[g_s, g_e \pm 2]$ overlapping with a gold entity is included as a negative (`label = 0`).
3. **Background Negatives:** Sampled from non-overlapping spans up to a target negative-to-positive ratio of $3:1$ (min 10 negatives per title).
4. **Evaluation Mode:** Zero negative sampling. All candidate spans up to $W_{max} = 8$ are scored.

### 3. Hyperparameters
- Epochs: 3
- Batch Size: 8 titles (effective batch size 8)
- Optimizer: AdamW, learning rate $3 \times 10^{-5}$, weight decay 0.01
- Schedule: Linear warmup 10% (120 steps), linear decay over 1,200 total steps
- Precision: Mixed precision (fp16) via PyTorch AMP with gradient clipping (max norm 1.0)
- Checkpoint Selection: Evaluated after each epoch on 800 internal dev titles; selected strictly by official competition hierarchical category-macro $F_{0.2}$.

---

## Part E: Deterministic Span Decoding Pipeline

The span decoding algorithm converts dense candidate span probabilities into discrete, non-overlapping entity predictions matching the competition evaluation format:
1. **Candidate Span Probabilistic Scoring:**
   For each candidate span $[s, e]$, let $P(c \mid [s, e])$ be the softmax probability vector over 30 classes.
   The candidate aspect is chosen as:
   $$c^* = \arg\max_{c \in \{1, \dots, 29\}} P(c \mid [s, e]), \quad \text{score} = P(c^* \mid [s, e])$$
2. **Confidence Gating Policy:**
   A candidate is retained if and only if:
   $$\text{score} \ge \tau \quad \text{and} \quad \text{score} > P(\text{NONE} \mid [s, e])$$
   where default $\tau = 0.50$ (or aspect-specific threshold).
3. **Deterministic Tie-Breaking:**
   Candidate spans are sorted in strict priority:
   $$\text{Key} = (-\text{score}, -\text{width}, s, e, \text{aspect})$$
   ensuring that among equal-confidence spans, longer spans take precedence, and ties are resolved by token position.
4. **Greedy Non-Overlapping Selection:**
   Iterating through sorted candidates, a span is accepted if and only if its whitespace token indices do not intersect any already accepted span. This matches the ground-truth topology (zero overlaps in gold data).
5. **Competition Category Filtering:**
   Accepted entities are passed through `filter_entities_by_category(category)` to enforce the competition rule that category-ineligible aspects are stripped before evaluation.

---

## Part F: Generalization Evaluation (Primary 800 Internal Dev Partition)

The trained SpanNERModel (`checkpoints/gbert_phase50_span_best`, Epoch 2) was evaluated on the 800-title internal development partition using deterministic greedy span decoding with threshold $\tau = 0.50$. It was compared directly against the matched baseline BIO model trained on the exact same 3,200 training partition (`gbert_phase49_precision_aware_best`).

```
========================================================================================
INTERNAL DEV EVALUATION (800 TITLES — 400 CAT 1 / 400 CAT 2)
========================================================================================
Model                         Official F0.2   Cat 1 F0.2   Cat 2 F0.2    TP     FP     FN
----------------------------------------------------------------------------------------
Matched BIO Baseline (P49)         0.906161     0.917155     0.895168  4,962    356  1,185
Phase 50 Span Model                0.871157     0.886975     0.855339  5,288    669    859
----------------------------------------------------------------------------------------
Delta vs Matched Baseline         -0.035004    -0.030180    -0.039828   +326   +313   -326
========================================================================================
```

### 1. Metric Trade-Off Analysis:
- **Precision:** $88.77\%$ (Span model) vs $93.30\%$ (BIO baseline) $\implies \mathbf{-4.53\%}$ drop.
- **Recall:** $86.03\%$ (Span model) vs $80.72\%$ (BIO baseline) $\implies \mathbf{+5.31\%}$ gain.
- **Impact on $F_{0.2}$:** Because the competition metric heavily penalizes false positives ($\beta=0.2$, precision is weighted $25\times$ more heavily than recall: $\beta^2 = 0.04$), the $+313$ surge in false positives caused a severe $-0.035004$ degradation in official score.

### 2. Exact Boundary Error Audit (Internal Dev):
- **Total Boundary Errors:** 312 errors.
- **By Type:**
  - **Truncation:** 178 errors (57.1%) — span model selected a shorter sub-span than ground truth.
  - **Overextension:** 134 errors (42.9%) — span model extended into neighboring codes or tokens.
- **By Aspect:**
  - `Kompatibles_Fahrzeug_Modell`: **237 errors** (76.0% of all boundary errors).
  - `Produktart`: **28 errors** (9.0%).
  - `Im_Lieferumfang_Enthalten`: **16 errors** (5.1%).
  - `Einbauposition`: **15 errors** (4.8%).
  - `Kompatible_Fahrzeug_Marke`: **7 errors** (2.2%).
  - `Herstellernummer`: **5 errors** (1.6%).
  - `Hersteller`: **3 errors** (1.0%).
  - `Kompatibles_Fahrzeug_Jahr`: **1 error** (0.3%).

---

## Part G: Complementarity & Qualitative Error Case Studies

### 1. Qualitative Boundary Error Inspection:
Inspection of actual predictions revealed why span-level representations failed to conquer vehicle-model boundary confusion:
- **Chassis & Trim Overextension:**
  - *Title:* `Zimmermann Bremsenset gelocht VW Multivan,Transporter V + VI Vorne 340MM`  
    *Gold:* `Multivan,Transporter` | *Span Pred:* `Multivan,Transporter V` (included generation `V`).
  - *Title:* `DOLZ Steuerkettensatz für VW Golf VI Scirocco III AUDI A4 B6 A3 V Touran SEAT`  
    *Gold:* `A3` | *Span Pred:* `A3 V` (included trim `V`).
  - *Title:* `Zahnriemensatz + Wasserpumpe für AUDI A1(8X) A3(8V) A4 B9/8W A5(F5) CT1169WP1`  
    *Gold:* `A4` | *Span Pred:* `A4 B9/8W` (included chassis code `B9/8W`).
- **Chassis Code Truncation:**
  - *Title:* `BREMBO Bremsensatz Hinten für JEEP COMMANDER (XK,XH) GRAND CHEROKEE (WH,WK)`  
    *Gold:* `COMMANDER (XK,XH)` | *Span Pred:* `COMMANDER` (dropped chassis code).
    *Gold:* `GRAND CHEROKEE (WH,WK)` | *Span Pred:* `GRAND` (dropped model & chassis).
  - *Title:* `WASSERPUMPE + ZAHNRIEMENSATZ FÜR VW AAB 2.4L AAF/ACU/AEN 2.5L 5cyl`  
    *Gold:* `AAB 2.4L AAF/ACU/AEN 2.5L 5cyl` | *Span Pred:* `AAB 2.4L` (severely truncated complex engine spec).

### 2. Architectural Conclusion on Span vs BIO NER:
Span NER evaluates an unconstrained combinatorial hypothesis space ($\approx 63$ candidate spans per title). In German automotive titles where ground truth boundaries on vehicle models follow inconsistent conventions (sometimes including chassis codes, sometimes excluding them), span classification heads assign moderate-to-high probabilities to multiple overlapping sub-spans. Even with greedy non-overlapping decoding, this leads to an explosion of partial span false alarms (+313 FPs on dev, +278 FPs on clean holdout), which is completely fatal for precision-heavy $F_{0.2}$. BIO sequence tagging, by enforcing local Markovian token continuity (`B-` followed by `I-`), acts as a natural regularizer against spurious boundary hypotheses.

---

## Part H: Secondary Diagnostic on 489 Clean Holdout

```
========================================================================================
CLEAN HOLDOUT EVALUATION (489 TITLES — NON-INDEPENDENT DIAGNOSTIC)
========================================================================================
Model                         Official F0.2   Cat 1 F0.2   Cat 2 F0.2    TP     FP     FN
----------------------------------------------------------------------------------------
Phase 40 Champion Baseline         0.927771     0.918323     0.937219  3,021    165    752
Phase 50 Span Model (Standalone)   0.870979     0.867328     0.874630  3,280    443    493
----------------------------------------------------------------------------------------
Delta vs Phase 40 Champion        -0.056792    -0.050995    -0.062589   +259   +278   -259
========================================================================================
```
- **Boundary Error Breakdown on Clean Holdout:** 203 total boundary errors (127 truncation, 76 overextension; 137 in `Kompatibles_Fahrzeug_Modell`).
- **Ensemble Feasibility:** Given that standalone Span NER degrades performance by $-0.056792$ and introduces $+278$ false positives, blending span predictions into the Phase 40 ensemble is unjustified and would severely degrade competition precision.

---

## Part I: Resource & Champion Integrity Controls

1. **Hardware & VRAM Safety:**
   - GPU: NVIDIA GeForce GTX 1660 Ti (6,144 MiB physical capacity).
   - Peak Training Memory: **2,126.9 MiB** (34.6% utilization). Zero out-of-memory events.
2. **Disk Space Safety:**
   - Preflight Free Disk: **6.01 GB**.
   - Post-Training Free Disk: **5.62 GB** (Compliant with $\ge 5.0$ GB limit).
   - Checkpoint storage footprint: **~441 MB**.
3. **Submissions & Leaderboard Protocol:**
   - No Quiz TSV submission file was generated.
   - No Test TSV submission file was generated.
   - Zero EvalAI submissions performed (20 remaining).
   - Champion file `submissions/submission_quiz_phase40_lieferumfang080.tsv` verified intact (`BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`).

---

## Part J: Official Decision Verdict

$$\mathbf{RETAIN\ PHASE\ 40\ —\ SPAN\ MODEL\ NOT\ COMPETITIVE}$$

### Verdict Rationale:
1. **Severe Precision Regression:** The span-based architecture increased recall (+326 TPs on dev) but incurred a devastating precision penalty (+313 FPs on dev, +278 FPs on clean holdout).
2. **Failure to Resolve Compound Boundary Ambiguities:** 76.0% of span boundary errors on dev were concentrated in `Kompatibles_Fahrzeug_Modell`. The model frequently truncated or overextended chassis, engine, and trim codes.
3. **Official Baseline Superiority:** Phase 40 official champion ($0.923331$ Quiz / $0.927771$ local) remains markedly superior. No submission files were modified.


