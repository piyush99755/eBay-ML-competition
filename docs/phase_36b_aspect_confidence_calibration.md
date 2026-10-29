# Phase 36B: Aspect-Specific Confidence Calibration Research

## Executive Summary

Phase 36B investigated whether targeted confidence threshold calibration on non-vehicle aspects (`Produktart`, `Im_Lieferumfang_Enthalten`, `Kompatible_Fahrzeug_Marke`) can improve the precision-weighted $F_{0.2}$ score beyond our active official EvalAI Quiz champion (**0.9159834927416699**).

Under the frozen Phase 35C champion configuration (`Kompatibles_Fahrzeug_Modell` threshold = 0.80, all other 28 aspects = 0.60), the 489-title clean holdout yields:
$$\text{Precision} = 0.939162, \quad \text{Recall} = 0.814206, \quad F_{0.2} = \mathbf{0.933651}$$
with exact ground-truth invariants verified: $\text{TP} = 3,072$, $\text{FP} = 199$, $\text{FN} = 701$ ($\text{TP} + \text{FN} \equiv 3,773$).

Using zero-compute reusable cached probability tensors (`data/cache/phase_35a_val_1k_probs.pt`) and the clean holdout manifest (`data/cache/phase_35b_clean_holdout_manifest.json`), we evaluated isolated candidate thresholds ($0.70$ and $0.80$) across the three focal non-vehicle aspects alongside two-sample Kolmogorov-Smirnov tests, Mann-Whitney rank tests, and 1,000 paired title-level bootstrap resamples.

### Key Research Findings

1. **`Kompatible_Fahrzeug_Marke` (Brand)**:
   - Already exhibits exceptional precision under baseline: $\text{Precision} = 0.977745$ (only 15 FPs across 489 titles).
   - Raising threshold to 0.70 or 0.80 removes only 2 FPs while sacrificing 3 to 10 TPs.
   - Causes **net regression in Category 2** ($\Delta F_{0.2} = -0.000086$ at 0.70; $\Delta F_{0.2} = -0.000604$ at 0.80).
   - Bootstrap 95% CIs span zero ($p = 0.3620$ and $p = 0.3680$). **Rejected.**

2. **`Im_Lieferumfang_Enthalten` (Included in Delivery)**:
   - Contains 49 FPs and 666 TPs.
   - Raising threshold to 0.80 removes 14 FPs but sacrifices 24 TPs (net loss of 10 ground-truth matches).
   - While $F_{0.2}$ increases mathematically ($+0.003108$) due to extreme precision weighting, error inspection reveals that true core package components (`Scheiben`, `Backen`, `Radzylinder`, `Sensoren`, `Keilriemen`) are dropped because multi-item bundle listings naturally exhibit slightly lower per-token confidence on secondary items.
   - Substantially damages package recall without a compelling structural justification.

3. **`Produktart` (Product Type) @ 0.80**:
   - Contains 33 FPs under baseline, with marked confidence divergence:
     - TP Confidence: $\text{Mean} = 0.9354, \quad \text{Median} = 0.9849, \quad Q_{25} = 0.9262$
     - FP Confidence: $\text{Mean} = 0.7884, \quad \text{Median} = 0.7643, \quad Q_{75} = 0.9056$
     - Two-sample Kolmogorov-Smirnov: $D = 0.6311, \quad p = 1.34 \times 10^{-11}$
     - Mann-Whitney AUC ($\text{TP} > \text{FP}$): $0.8629, \quad p = 1.34 \times 10^{-11}$
   - At threshold 0.80, **60.6% of all false positives are eliminated** (20 of 33 FPs removed), while only **11.1% of true positives** are lost (27 of 244 TPs lost).
   - Overall Precision jumps from **0.939162 to 0.944479** ($+0.005317$).
   - Overall $F_{0.2}$ improves from **0.933651 to 0.938333** ($\Delta = \mathbf{+0.004682}$).
   - Both categories advance in lockstep:
     - Category 1 $F_{0.2}$: $0.928895 \to 0.933983$ ($\mathbf{+0.005088}$)
     - Category 2 $F_{0.2}$: $0.943340 \to 0.947131$ ($\mathbf{+0.003791}$)
   - Paired title-level bootstrap (1,000 resamples, seed 42):
     $$\Delta F_{0.2} = +0.004705, \quad 95\% \text{ CI: } [+0.002288, +0.007141], \quad p < 0.0001$$
     Every single bootstrap resample ($1,000 / 1,000$) demonstrated a positive $F_{0.2}$ gain.

### Recommendation
**IMPLEMENT ONE THRESHOLD CHALLENGER** (`Produktart` @ 0.80).
In accordance with competition protocol, this candidate is documented and held in reserve; no Quiz inference or official submission will occur without explicit user approval.

---

## 1. Verified Baseline Champion

The active official EvalAI Quiz champion is the Phase 35C Precision-Optimized Vehicle Model Ensemble:

- **Official EvalAI Quiz $F_{0.2}$**: `0.9159834927416699`
- **Submission File**: `submissions/submission_quiz_phase35c_vehicle080.tsv`
- **SHA-256**: `4B591EF6A9072C9319FF1C0134DFC9736E5A436431ED512FE47B980ECCC01382`
- **Model Checkpoints and Weights**:
  1. `checkpoints/phase33b_dapt_gbert_best` (Weight 0.40)
  2. `checkpoints/xlmr_baseline_best` (Weight 0.20)
  3. `checkpoints/xlmr_seed_7_best` (Weight 0.20)
  4. `checkpoints/xlmr_seed_21_best` (Weight 0.20)
- **Thresholds**:
  - `Kompatibles_Fahrzeug_Modell`: Minimum token confidence $\ge 0.80$
  - All other 28 aspects: Minimum token confidence $\ge 0.60$
- **Holdout Baseline Metrics (489 Clean Titles)**:
  - $\text{TP} = 3,072, \quad \text{FP} = 199, \quad \text{FN} = 701, \quad \text{Total Gold} = 3,773$
  - $\text{Precision} = 0.939162, \quad \text{Recall} = 0.814206, \quad F_{0.2} = \mathbf{0.933651}$
  - Category 1: $\text{TP} = 2,050, \text{FP} = 144, \text{FN} = 480 \implies \text{Precision} = 0.934366, \text{Recall} = 0.810277, F_{0.2} = 0.928895$
  - Category 2: $\text{TP} = 1,022, \text{FP} = 55, \text{FN} = 221 \implies \text{Precision} = 0.948932, \text{Recall} = 0.822204, F_{0.2} = 0.943340$

---

## 2. Confidence Distributions and Statistical Separation Analysis

For each target aspect, we extracted token-minimum confidence scores for True Positives vs. False Positives under the Phase 35C champion baseline:

| Aspect | Gold | Pred | TP | FP | FN | Precision | Recall | $F_{0.2}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Produktart`** | 293 | 277 | 244 | 33 | 49 | 0.880866 | 0.832765 | 0.878914 |
| **`Im_Lieferumfang_Enthalten`** | 710 | 715 | 666 | 49 | 44 | 0.931469 | 0.938028 | 0.931719 |
| **`Kompatible_Fahrzeug_Marke`** | 673 | 674 | 659 | 15 | 14 | 0.977745 | 0.979198 | 0.977801 |

### Confidence Quantiles (TP vs. FP)

| Aspect | Status | Count | Mean | Std | Min | $Q_{25}$ | Median | $Q_{75}$ | $Q_{90}$ | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Produktart`** | **TP** | 244 | 0.9354 | 0.0983 | 0.6008 | 0.9262 | 0.9849 | 0.9955 | 0.9961 | 0.9969 |
| | **FP** | 33 | 0.7884 | 0.0992 | 0.6152 | 0.7065 | 0.7643 | 0.9056 | 0.9295 | 0.9572 |
| **`Im_Lieferumfang_Enthalten`** | **TP** | 666 | 0.9782 | 0.0561 | 0.6365 | 0.9929 | 0.9968 | 0.9976 | 0.9978 | 0.9980 |
| | **FP** | 49 | 0.8716 | 0.1210 | 0.6009 | 0.7899 | 0.8950 | 0.9899 | 0.9958 | 0.9972 |
| **`Kompatible_Fahrzeug_Marke`** | **TP** | 659 | 0.9913 | 0.0367 | 0.6138 | 0.9974 | 0.9981 | 0.9985 | 0.9986 | 0.9989 |
| | **FP** | 15 | 0.9222 | 0.1224 | 0.6201 | 0.9165 | 0.9814 | 0.9910 | 0.9977 | 0.9980 |

### Statistical Separation Tests

- **`Produktart`**:
  - Two-sample Kolmogorov-Smirnov: $D = 0.6311, \quad p = 1.34 \times 10^{-11}$
  - Mann-Whitney AUC ($\text{TP} > \text{FP}$): $0.8629, \quad p = 1.34 \times 10^{-11}$
  - Interpretation: Pronounced separation. Over 60% of false positives cluster between 0.60 and 0.80, whereas 75% of true positives have confidence $\ge 0.9262$.
- **`Im_Lieferumfang_Enthalten`**:
  - Two-sample Kolmogorov-Smirnov: $D = 0.5569, \quad p = 8.71 \times 10^{-14}$
  - Mann-Whitney AUC ($\text{TP} > \text{FP}$): $0.8597, \quad p = 4.05 \times 10^{-17}$
  - Interpretation: Statistically significant separation, but 40% of false positives have confidence exceeding 0.90, overlapping with high-confidence true positives.
- **`Kompatible_Fahrzeug_Marke`**:
  - Two-sample Kolmogorov-Smirnov: $D = 0.7302, \quad p = 1.76 \times 10^{-8}$
  - Mann-Whitney AUC ($\text{TP} > \text{FP}$): $0.9012, \quad p = 1.05 \times 10^{-7}$
  - Interpretation: Strong theoretical separation, but baseline FP count is tiny ($N=15$), with median FP confidence at $0.9814$.

---

## 3. Threshold Sensitivity and Bootstrap Comparison Table

All simulations maintain `Kompatibles_Fahrzeug_Modell` fixed at 0.80 and all other aspects at 0.60. Exactly one aspect threshold is varied at a time.

| Configuration | TP | FP | FN | Precision | Recall | $F_{0.2}$ | $\Delta F_{0.2}$ | Cat 1 $F_{0.2}$ ($\Delta$) | Cat 2 $F_{0.2}$ ($\Delta$) | Bootstrap 95% CI | Boot $p$-val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **Baseline (Phase 35C)** | **3,072** | **199** | **701** | **0.939162** | **0.814206** | **0.933651** | **0.000000** | **0.928895** | **0.943340** | — | — |
| `Produktart` @ 0.70 | 3,055 | 194 | 718 | 0.940289 | 0.809701 | 0.934493 | +0.000841 | 0.930204 (+0.001309) | 0.943168 (-0.000172) | [-0.000190, +0.002170] | 0.0670 |
| `Produktart` @ 0.80 | **3,045** | **179** | **728** | **0.944479** | **0.807050** | **0.938333** | **+0.004682** | **0.933983 (+0.005088)** | **0.947131 (+0.003791)** | **[+0.002288, +0.007141]** | **0.0000** |
| `Im_Lieferumfang` @ 0.70 | 3,064 | 194 | 709 | 0.940454 | 0.812086 | 0.934771 | +0.001120 | 0.929871 (+0.000975) | 0.944763 (+0.001423) | [+0.000023, +0.002439] | 0.0230 |
| `Im_Lieferumfang` @ 0.80 | 3,048 | 185 | 725 | 0.942778 | 0.807845 | 0.936760 | +0.003108 | 0.931651 (+0.002756) | 0.947131 (+0.003791) | [+0.001058, +0.005448] | 0.0000 |
| `Fahrzeug_Marke` @ 0.70 | 3,069 | 197 | 704 | 0.939682 | 0.813411 | 0.934104 | +0.000453 | 0.929609 (+0.000714) | 0.943254 (-0.000086) | [-0.000192, +0.001663] | 0.3620 |
| `Fahrzeug_Marke` @ 0.80 | 3,062 | 197 | 711 | 0.939552 | 0.811556 | 0.933887 | +0.000236 | 0.929561 (+0.000666) | 0.942736 (-0.000604) | [-0.000480, +0.001492] | 0.3680 |

---

## 4. Entity-Level Qualitative Error Analysis

To verify whether threshold filtering addresses genuine labeling artifacts or merely exploits local data noise, we examined individual predictions removed between confidence $0.60$ and $0.80$:

### A. `Produktart` (20 FPs Removed vs. 27 TPs Lost)
- **Primary Source of Ambiguity**: The generic token `"Satz"` ("set" / "kit") and composite descriptors (e.g. `"Bremsen"`, `"komplett"`).
  - Among the 20 FPs removed:
    - 12 instances are isolated `"Satz"` predictions (e.g., *Bosch Bremsscheiben und Bremsbeläge Meyle Mercedes VITO Satz hinten*).
    - 3 instances are generic boundary fragments (*"Bremsen"*, *"komplett"*, *"SET"*).
    - 5 instances are compound parts in Category 2 (*"Timing Chain Kit"*, *"Rep.-Satz"*, *"Zahnriemen Satz"*).
  - Among the 27 TPs lost:
    - 14 instances are isolated `"Satz"` tokens where ground truth happened to include `"Satz"` as a standalone `Produktart`.
    - 13 instances are valid multi-word titles where confidence was marginally suppressed ($0.65 - 0.78$).
- **Structural Insight**: In German eBay titles, `"Satz"` represents the most inconsistently annotated aspect token in the competition dataset: annotators alternately included or omitted it from `Produktart`. Removing marginal predictions below 0.80 purges 60.6% of these noisy false alarms.

### B. `Im_Lieferumfang_Enthalten` (14 FPs Removed vs. 24 TPs Lost)
- **Inspection of TPs Lost**: The lost true positives are genuine primary package components:
  - `"Scheiben"` (disc), `"Backen"` (shoes), `"Radzylinder"` (wheel cylinders), `"Sensoren"` (sensors), `"Keilriemen"` (V-belt), `"Bremsklötze"` (brake pads).
- **Structural Insight**: When sellers list complex kits (e.g., *"Bremsscheiben Beläge Backen Sensoren vorne hinten"*), each secondary included part receives probability around $0.70 - 0.79$. Purging these parts damages genuine multi-item package recall, making threshold elevation on this aspect undesirable for general domain coverage.

---

## 5. Generalization and Risk Assessment

1. **Proportional False Positive Removal**:
   - `Produktart` @ 0.80: Removes **60.6% of FPs** (20 of 33) while losing only **11.1% of TPs** (27 of 244). Proportion ratio: $5.46 \times$ more effective against false positives than true positives.
   - `Im_Lieferumfang_Enthalten` @ 0.80: Removes 28.6% of FPs (14 of 49) while losing 3.6% of TPs (24 of 666).
   - `Kompatible_Fahrzeug_Marke` @ 0.80: Removes 13.3% of FPs (2 of 15) while losing 1.5% of TPs (10 of 659).

2. **Category Balance**:
   - `Produktart` @ 0.80 improves both categories: Category 1 advances by $+0.005088$; Category 2 advances by $+0.003791$.
   - By contrast, `Kompatible_Fahrzeug_Marke` regresses Category 2 at both thresholds.

3. **Sample Count Sensitivity**:
   - The `Produktart` gain is distributed across 37 titles (20 titles with FPs removed, 27 titles with TPs lost, with 10 titles having overlapping impacts). It does not depend on a single aberrant listing.

4. **Distribution Shift Stability**:
   - The token `"Satz"` accounts for a large fraction of the filter's impact. If Quiz or Test evaluation titles follow the exact same seller vocabulary (which is highly likely for German automotive parts), this precision gain is structurally robust.
   - However, because the 489 clean holdout was used during Phase 35B/35C development, this gain is **exploratory** and subject to mild holdout optimism.

---

## 6. Decision Gate and Actionable Recommendation

Evaluating the decision criteria established in the Phase 36B charter:

- [x] **Precision benefit is clear**: Precision increases by $+0.005317$ (from $0.939162$ to $0.944479$).
- [x] **$F_{0.2}$ gain is positive**: Local $F_{0.2}$ improves by $+0.004682$ (from $0.933651$ to $0.938333$).
- [x] **Both categories show acceptable behavior**: Cat 1 ($+0.005088$) and Cat 2 ($+0.003791$) both improve.
- [x] **Confidence analysis supports the choice**: Two-sample KS ($D = 0.6311, p < 10^{-10}$) and Mann-Whitney AUC ($0.8629, p < 10^{-10}$) confirm extreme confidence separation.
- [x] **Recall is protected**: Overall recall drops only from $0.814206$ to $0.807050$ ($-0.007156$), which is negligible compared to the $25 \times$ precision weighting in $F_{0.2}$.
- [x] **Implementation is simple and reversible**: Modifies exactly one key in `ASPECT_THRESHOLDS`:
  ```python
  ASPECT_THRESHOLDS = {
      "Kompatibles_Fahrzeug_Modell": 0.80,
      "Produktart": 0.80,
  }
  ```

### Final Recommendation
**IMPLEMENT ONE THRESHOLD CHALLENGER**

The single vetted challenger is:
- Ensemble: DAPT GBERT (0.40) + XLM-R 42 (0.20) + XLM-R 7 (0.20) + XLM-R 21 (0.20)
- Aspect Thresholds: `Kompatibles_Fahrzeug_Modell` = **0.80**, `Produktart` = **0.80**, all other 27 aspects = **0.60**.
- Projected local holdout $F_{0.2}$: **0.938333** ($\Delta = +0.004682$, 95% CI: $[+0.002288, +0.007141]$).

**Production Status**: In accordance with user directives, no submission pipeline code has been modified and no Quiz or Test inference has been executed. The Phase 35C champion remains active and untouched pending user review.
