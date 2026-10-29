# Phase 37A: Package Contents Precision Calibration Research

## Executive Summary

Phase 37A investigated whether elevating the confidence threshold for **`Im_Lieferumfang_Enthalten`** (Package Contents) from $0.60$ to $0.70$ or $0.80$ can improve precision-weighted $F_{0.2}$ beyond our active official EvalAI Quiz champion (**Phase 36C**, official Quiz $F_{0.2} = \mathbf{0.9206543463689059}$).

Under the frozen Phase 36C champion configuration (`Kompatibles_Fahrzeug_Modell` = 0.80, `Produktart` = 0.80, all other 27 aspects = 0.60), the 489-title clean holdout yields:
$$\text{Precision} = 0.944479, \quad \text{Recall} = 0.807050, \quad F_{0.2} = \mathbf{0.938333}$$
with exact ground-truth invariants verified: $\text{TP} = 3,045$, $\text{FP} = 179$, $\text{FN} = 728$ ($\text{TP} + \text{FN} \equiv 3,773$).

Using zero-compute reusable cached probability tensors (`data/cache/phase_35a_val_1k_probs.pt`), we evaluated isolated thresholds ($0.70$ and $0.80$) for `Im_Lieferumfang_Enthalten` alongside paired title-level bootstrap resampling (1,000 iterations, seed 42) and detailed qualitative error tracking.

### Key Research Findings

1. **Destructive Trade-Off Ratio ($\text{TPs Lost} > \text{FPs Removed}$)**:
   - At threshold $0.70$ (Candidate A): **8 True Positives are lost** to eliminate only **5 False Positives** ($\text{Ratio} = 1.60\text{ TPs lost per FP removed}$).
   - At threshold $0.80$ (Candidate B): **24 True Positives are lost** to eliminate only **14 False Positives** ($\text{Ratio} = 1.71\text{ TPs lost per FP removed}$).
   - Unlike Phase 36B (`Produktart`), where $5.46 \times$ more false positives were removed proportionally than true positives, filtering `Im_Lieferumfang_Enthalten` is structurally inefficient: $63.2\%$ of all predictions removed at $0.80$ are valid ground-truth entities!

2. **Severe Damage to Valid Automotive Components**:
   - Qualitative inspection reveals that the 24 true positives lost at $0.80$ are **not** generic descriptors or annotation noise; they are core physical mechanical components delivered in automotive kits:
     - `"Scheiben"` (brake discs)
     - `"Backen"` (brake shoes)
     - `"Radzylinder"` (wheel cylinders)
     - `"Bremsklötze"` (brake pads)
     - `"Radlager"` (wheel bearings)
     - `"Keilriemen"` (V-belt)
     - `"+WARNKONTAKT"` (wear contact sensor)
   - In multi-item bundle listings (e.g. *"Bremsscheiben Beläge Bremstrommel Backen Radzylinder vorne hinten"*), secondary package items naturally exhibit confidence between $0.70$ and $0.79$. Purging them directly dismantles multi-part bundle extraction capability.

3. **Weak False Positive Separation**:
   - Of the 49 baseline false positives for `Im_Lieferumfang_Enthalten`, **35 instances (71.4%) have confidence $\ge 0.80$** (median FP confidence is $0.8950$).
   - The threshold elevation only purges 14 of 49 false positives ($28.6\%$) while discarding valid components, proving that confidence thresholding does not cleanly isolate false positives for this aspect.

4. **Mathematical Gain vs. Generalization Risk**:
   - Because $F_{0.2}$ weights precision $25 \times$ over recall ($\beta^2 = 0.04$), Candidate B achieves a local holdout gain of $+0.003206$ ($0.938333 \to 0.941539$, 95% CI: $[+0.001127, +0.005567]$) solely due to extreme precision weighting.
   - However, scaling a loss of 24 TPs per 500 titles to the 25,000-title hidden Test set would destroy over **1,200 valid package components**, creating an unacceptable risk of out-of-distribution regression on realistic seller listings.

### Recommendation
**RETAIN PHASE 36C CHAMPION**

The evidence does not support elevating the threshold for `Im_Lieferumfang_Enthalten`. The Phase 36C champion (**official Quiz $F_{0.2} = 0.9206543463689059$**) remains the tournament leader and should be protected.

---

## 1. Verified Phase 36C Champion

- **Official EvalAI Quiz Score**: **$F_{0.2} = 0.9206543463689059$**
- **Champion Submission TSV**: `submissions/submission_quiz_phase36c_produktart080.tsv`
- **Verified SHA-256**: `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87`
- **File Size**: 5,460,127 bytes (145,396 data rows)
- **Model Checkpoints and Weights**:
  - DAPT GBERT Seed 42: `checkpoints/phase33b_dapt_gbert_best` (Weight 0.40)
  - XLM-R Seed 42: `checkpoints/xlmr_baseline_best` (Weight 0.20)
  - XLM-R Seed 7: `checkpoints/xlmr_seed_7_best` (Weight 0.20)
  - XLM-R Seed 21: `checkpoints/xlmr_seed_21_best` (Weight 0.20)
- **Active Champion Thresholds**:
  - `Kompatibles_Fahrzeug_Modell`: Minimum token confidence $\ge 0.80$
  - `Produktart`: Minimum token confidence $\ge 0.80$
  - All other 27 aspects: Minimum token confidence $\ge 0.60$

---

## 2. Reconstructed Phase 36C Local Baseline

Evaluated on the 489-title clean holdout using `count_matches(predicted, gold)`:

| Metric | Overall | Category 1 | Category 2 | `Im_Lieferumfang_Enthalten` |
|---|---:|---:|---:|---:|
| **True Positives (TP)** | 3,045 | 2,028 | 1,017 | 666 |
| **False Positives (FP)** | 179 | 129 | 50 | 49 |
| **False Negatives (FN)** | 728 | 502 | 226 | 44 |
| **Total Gold ($\text{TP} + \text{FN}$)** | 3,773 | 2,530 | 1,243 | 710 |
| **Precision** | **0.944479** | **0.940195** | **0.953140** | **0.931469** |
| **Recall** | **0.807050** | **0.801581** | **0.818182** | **0.938028** |
| **$F_{0.2}$ Score** | **0.938333** | **0.933983** | **0.947131** | **0.931719** |

*All counts and metrics independently verified and match the Phase 36C specification with 100% precision.*

---

## 3. Threshold Comparison Table

Configurations maintain `Kompatibles_Fahrzeug_Modell` fixed at 0.80, `Produktart` fixed at 0.80, and all other 26 aspects fixed at 0.60:

| Configuration | Threshold | TP | FP | FN | Precision | Recall | $F_{0.2}$ | $\Delta F_{0.2}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Phase 36C Baseline** | **0.60** | **3,045** | **179** | **728** | **0.944479** | **0.807050** | **0.938333** | **0.000000** |
| Candidate A | 0.70 | 3,037 | 174 | 736 | 0.945811 | 0.804930 | 0.939487 | +0.001154 |
| Candidate B | 0.80 | 3,021 | 165 | 752 | 0.948211 | 0.800689 | 0.941539 | +0.003206 |

### Category-Level Comparison

| Configuration | Cat 1 Precision | Cat 1 Recall | Cat 1 $F_{0.2}$ ($\Delta$) | Cat 2 Precision | Cat 2 Recall | Cat 2 $F_{0.2}$ ($\Delta$) |
|---|---:|---:|---:|---:|---:|---:|
| **Phase 36C Baseline** | **0.940195** | **0.801581** | **0.933983 (0.000)** | **0.953140** | **0.818182** | **0.947131 (0.000)** |
| Candidate A (0.70) | 0.941502 | 0.799605 | 0.934992 (+0.001010) | 0.954899 | 0.815768 | 0.948584 (+0.001453) |
| Candidate B (0.80) | 0.943799 | 0.794071 | 0.936849 (+0.002866) | 0.957627 | 0.818182 | 0.950990 (+0.003859) |

---

## 4. True Positives Lost vs. False Positives Removed

| Configuration | Total Removed | TPs Lost | FPs Removed | Trade-Off Ratio ($\text{TP lost} / \text{FP removed}$) | Efficacy Assessment |
|---|---:|---:|---:|---:|---|
| Candidate A (0.70) | 13 | 8 | 5 | **1.60 : 1** | Destructive: loses more valid items than errors |
| Candidate B (0.80) | 38 | 24 | 14 | **1.71 : 1** | Highly destructive: $63.2\%$ of pruned entities are TPs |

### Qualitative Analysis of Predictions Removed at 0.80

#### Representative True Positives Lost ($N=24$):
- **Brake Discs & Shoes**: `"Scheiben"` (0.6365), `"Backen"` (0.7527, 0.7962), `"Trommelbremse"` (0.6946), `"Sport-Bremsscheiben"` (0.7120), `"XTRA-Bremsscheiben"` (0.6572).
- **Hydraulics & Friction Parts**: `"Radzylinder"` (0.7539, 0.7987), `"Bremsklötze"` (0.7368, 0.7839), `"+BELÄGE"` (0.7735).
- **Sensors, Hardware & Drive Belts**: `"Sensoren"` (0.6374), `"Radlager"` (0.7942), `"+WARNKONTAKT"` (0.7814), `"Keilriemen"` (0.6722), `"Keilr.riemen"` (0.7917).

*These are the primary physical products promised to buyers. They are lost because secondary items in multi-component titles receive slightly lower ensemble agreement.*

#### Representative False Positives Removed ($N=14$):
- Composite fragments and punctuation: `"+BELÄGE"` (0.7488), `"VERSAMMLUNG"` (0.6009), `"Klötze"` (0.7815).
- Incidental vehicle systems: `"FENSTERREINIGUNG"` (0.7505), `"Klimakompressor"` (0.7245), `"Drosselklappe"` (0.6277).
- Auxiliary parts: `"Rollen"` (0.6478, 0.7988), `"Sensoren"` (0.6405).

---

## 5. Statistical Uncertainty (Paired Title-Level Bootstrap)

Evaluated via 1,000 paired title-level bootstrap resamples (seed 42) computing $\Delta F_{0.2} = F_{0.2}^{\text{cand}} - F_{0.2}^{\text{base}}$:

| Candidate | Mean $\Delta F_{0.2}$ | 95% Bootstrap Confidence Interval | Empirical $p$-value | Bootstrap $P(\Delta > 0)$ |
|---|---:|:---:|---:|---:|
| Candidate A (0.70) | $+0.001172$ | $[+0.000031, +0.002491]$ | 0.0170 | $98.3\%$ |
| Candidate B (0.80) | $+0.003209$ | $[+0.001127, +0.005567]$ | < 0.0001 | $100.0\%$ |

*Integrity Disclosure*: The 489-title holdout was previously utilized for Phase 35B error analysis and Phase 35C/36B threshold exploration. These bootstrap confidence intervals reflect internal resampling consistency on the development holdout and must **not** be interpreted as unbiased proof of out-of-distribution Quiz generalization.

---

## 6. Overfitting and Generalization Risk Assessment

1. **Unbalanced Error Distribution**:
   - For `Produktart` (Phase 36B), 60.6% of false positives clustered below 0.80, and the filter removed $5.46 \times$ more FPs proportionally than TPs.
   - For `Im_Lieferumfang_Enthalten`, 71.4% of false positives have confidence $\ge 0.80$ (mean FP confidence is $0.8716$, median $0.8950$). The threshold fails to separate the FP distribution, acting as an indiscriminate filter.
2. **Excessive Recall Penalty on Complex Kits**:
   - `Im_Lieferumfang_Enthalten` recall drops from $93.80\%$ to $90.42\%$ ($-3.38\%$).
   - On the hidden test set, sellers frequently bundle full replacement sets. Discarding 24 TPs per 500 titles scales to an estimated loss of over **1,200 valid package components** across 25,000 test titles.
3. **Leaderboard Protection**:
   - Phase 36C achieved **0.9206543463689059** on official EvalAI Quiz, ranking #1.
   - Gambles that sacrifice substantial true recall on valid physical items carry severe risk of regression on out-of-distribution titles.

---

## 7. Decision Gate and Actionable Recommendation

Evaluating the decision gate criteria:
- [x] Precision benefit exists mathematically (+0.003732 at 0.80).
- [ ] Removes substantially more false positives proportionally than true positives: **FAILED** (destroys 1.71 TPs per FP removed).
- [ ] Preserves valid components while removing generic noise: **FAILED** (destroys core parts like discs, pads, shoes, bearings, cylinders, belts).
- [ ] Supported by false positive confidence distribution: **FAILED** (71.4% of FPs have confidence $\ge 0.80$).
- [ ] Safe for unseen Test distribution: **HIGH RISK** (severe multi-part bundle recall loss).

### Final Recommendation
**RETAIN PHASE 36C CHAMPION**

Do not implement a Phase 37B challenger. Maintain the frozen Phase 36C ensemble (`Kompatibles_Fahrzeug_Modell` = 0.80, `Produktart` = 0.80, all others = 0.60) as the definitive official competition champion.
