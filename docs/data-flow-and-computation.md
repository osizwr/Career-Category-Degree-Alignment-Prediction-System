# Data Flow and Computation Reference

**An Explainable Career Category Recommendation System for Computer Science and
Information Technology Graduating Students using Ensemble and Baseline Classifiers**
Palawan State University — College of Sciences

This document traces every transformation applied to a student's questionnaire
answers, from the raw Google Forms export to the final recommendation report. For
each step it states **where** the code lives, **the formula** computed, **what it
means in plain terms**, and **the real numbers** for student R001.

---

## 0. The map — where data comes from and where it ends

```
INPUT
  data/real-data.xlsx              Google Forms export (raw survey)
        |  scripts/clean_real_data.py
  data/real-data-clean.csv         tidy, one row per respondent
        |  scripts/build_model_dataset.py   (+ rubric assigns work_category)
  data/real-cohort-model.csv       100 students x 74 columns, model schema
        |
PROCESSING                          career_system/
        |  features.py              raw -> 68 engineered features
        |  models.py                impute -> SMOTE -> XGBoost
        |  benchmark.py             XGBoost vs LogReg vs ZeroR
        |  explain.py               SHAP attribution + narration
        |  alignment.py             degree %, career tracks, gaps, employers
        |  recommender.py           assembles the student report
        v
OUTPUT
  outputs/tables/                  metrics, confusion matrices, profiles
  outputs/figures/                 comparison charts, SHAP plots
  outputs/models/                  fitted pipelines (joblib)
  outputs/reports/                 per-student reports (.txt and .json)
```

There is a second input door: `python main.py predict` runs the questionnaire in
the terminal (`interactive.py`) and produces a single-row frame with the identical
schema. Everything downstream is unchanged.

---

## 1. Notation and constants

| Symbol | Meaning | Value / source |
|---|---|---|
| `g` | a grade on the PSU transcript scale | 1.00 (best) – 5.00 (fail) |
| `p` | the same grade as points | 0–100, higher is better |
| `G_best`, `G_worst` | scale bounds | 1.00, 5.00 |
| `D` | academic domains | 10 (`config.ACADEMIC_DOMAINS`) |
| `S` | competency signals | 11 (`config.SIGNALS`) |
| `C` | work categories | 4: Technical, Clerical, Supervisory, Managerial |
| `K` | career tracks | 16 (`config.CAREER_TRACKS`) |
| `n` | students in the cohort | 100 (real), 400 (synthetic) |
| `m` | features in the `full` set | 68 |

Fixed weights used throughout:

```
Rubric (Chapter IV)      core_courses .30  gwa .20  certifications .25  skills .25
Signal evidence groups   courses .45       skills .40    certs .15
EVT value                skills .50        cert domains .30   credential types .20
Track match              competency .75    model probability .25
Split / CV               test_size 0.30    cv_folds 5     random_state 42
Skill gap                target 75.0       min_weight 0.08
```

A helper used in several places rescales any weight dictionary to sum to 1:

```
normalise_weights(w)_i = w_i / sum_j(w_j)
```

---

## 2. Step 1 — Survey ingestion and cleaning

**Code:** `scripts/clean_real_data.py` → `data/real-data-clean.csv`

The export is messy: typed grades such as `"1 .75"` and `"2..25"`, multi-select
answers split across five spreadsheet columns, and a sheet formula that divided
blank cells in as zero.

Cleaning rules applied:

- Consent (Q1–Q2): rows without "Yes" to both are excluded, then the columns are
  dropped because they become constant.
- `student_id` assigned as `R001…` in sheet order; `source_row` points back to the
  spreadsheet row.
- Grade text repaired; values otherwise kept exactly as entered, including 4.00
  (conditional), 5.00 (failed) and off-scale entries.
- Multi-select answers expanded to one 0/1 column per option. A respondent who
  skipped a whole question gets blanks, **not** zeros.

GWA is recomputed rather than trusted, as the unweighted mean of the respondent's
non-blank course grades:

```
GWA = ( sum over reported courses of g_i ) / (number of reported courses)
```

The sheet's own `=SUM(...)/17` skipped text grades and counted blanks as zero,
which understated GWA for four respondents.

**Design note.** No respondent is silently dropped. Rows that may warrant
exclusion (not 4th year, duplicate grade patterns, off-scale values) are kept and
listed in a `flags` column, so the exclusion decision stays visible in the
methodology rather than disappearing into a script.

---

## 3. Step 2 — Label derivation (the Chapter IV rubric)

**Code:** `career_system/labelling.py` — `rubric_scores`, `assign_work_categories`

The survey never asked which career category a student belongs to, so the label is
computed with the Chapter IV rubric. For each student and each candidate category
`c`, four components are formed on a 0–100 scale.

Each component is a weighted mean over the columns that category cares about, with
blanks renormalised away:

```
component(x, W) = ( sum_i W_i * x_i ) / ( sum_i W_i * [x_i is present] )
```

The four components:

```
core_courses(c)   = component( acad_<domain> ,  profile_c.domains )
gwa               = gwa_points                          (same for every category)
certifications(c) = component( cert_<domain> * 100 , profile_c.certs )
skills(c)         = component( <skill> * 100 ,  profile_c.skills )
```

The composite, with the same renormalisation applied across the four weights:

```
composite(c) = ( .30*core + .20*gwa + .25*cert + .25*skills )
               / ( .30 + .20 + .25 + .25  restricted to present terms )

work_category = argmax over c of composite(c)
label_margin  = composite(1st) - composite(2nd)
```

**R001:** Supervisory 47.89, Clerical 46.91, Managerial 43.58, Technical 42.21
→ labelled **Supervisory**, margin **0.98**.

> **Two properties to state in Chapter V.**
>
> 1. **The label is a deterministic function of the features.** A classifier
>    trained on it is measuring whether it can re-learn this formula, not whether
>    it can predict a real career. Expect high accuracy and report it as a
>    *pipeline check*. Real labels require an observed outcome: the two labelers'
>    independent manual coding, an OJT placement record, or a tracer study.
> 2. **Only two of the four criteria can change the winner.** GWA (20%) is one
>    number identical across all four categories, so it raises or lowers every
>    composite equally and never breaks a tie — it sets the level, not the
>    ranking. Certifications (25%) decide nothing for a student holding none,
>    which is most of the current batch. In practice the label is driven by core
>    course performance and ticked skills, re-weighted 55/45. This follows from
>    the rubric as written; if the panel expects GWA to carry a fifth of the
>    decision, the rubric needs a category-relative GWA term.

---

## 4. Step 3 — Grade normalisation

**Code:** `career_system/config.py` — `grade_to_points`

```
p = clip( (G_worst - g) / (G_worst - G_best) * 100 , 0 , 100 )
  = clip( (5.00 - g) / 4.00 * 100 , 0 , 100 )
```

| grade | points |
|---|---|
| 1.00 | 100.00 |
| 1.25 | 93.75 |
| 1.75 | 81.25 |
| 2.00 | 75.00 |
| 3.00 | 50.00 |
| 5.00 | 0.00 |

**Why it matters.** On the Philippine scale 1.00 is excellent — *lower is better*.
Fed to a model unchanged, every SHAP plot would read backwards, appearing to say
that high grades push a student away from Technical work. Flipping the scale once,
here, means every downstream number obeys "higher is better".

**R001:** `gwa 2.0588 → (5 − 2.0588)/4 × 100 = 73.53 points`

---

## 5. Step 4 — Folding courses into shared domains

**Code:** `career_system/features.py` — `add_grade_points`

```
acad_<d> = mean over { courses c : d in c.domains , grade present } of p_c
```

A single course may belong to more than one domain (Data Structures counts in both
`programming` and `theory_analytics`).

```
core_grade_mean = mean over the 10 acad_<d>
core_grade_std  = population standard deviation over the 10 acad_<d>   (ddof = 0)
```

**Why the raw grade columns never reach the model.** BSCS students answered 17
courses, BSIT students 24, and they are *different* courses, so every BSIT column
is structurally blank for a BSCS student. Given those 41 columns, a tree would
learn to read the degree program off the *pattern of missing values* rather than
off the grades. Folding into 10 shared domains makes a BSCS and a BSIT profile
directly comparable.

**R001 (BSCS, 17 courses):**

```
acad_foundations            = mean(87.5)                            = 87.50
acad_programming            = mean(50.0, 56.25, 50.0, 75.0)         = 57.81
acad_theory_analytics       = mean(50.0, 68.75, 75.0, 75.0, 50.0)   = 63.75
acad_data_management        = mean(68.75)                           = 68.75
acad_networking             = mean(75.0)                            = 75.00
acad_security               = mean(87.5)                            = 87.50
acad_web_design             = mean(81.25)                           = 81.25
acad_systems_engineering    = mean(56.25, 93.75)                    = 75.00
acad_infrastructure_support = mean(87.5, 81.25)                     = 84.38
acad_professional_practice  = mean(87.5)                            = 87.50

core_grade_mean = 76.84      core_grade_std = 10.11
```

The standard deviation is informative: 10.11 describes a spiky profile — strong in
security and infrastructure (87), weak in programming (58).

---

## 6. Step 5 — Aggregate counts

**Code:** `career_system/features.py` — `add_aggregate_features`, `checkbox_to_percent`

```
cert_count        = sum over the 8 cert_<domain> columns
cert_type_count   = sum over the 5 cert_type_<type> columns
tech_skill_count  = sum over the tech_<skill> columns
hard_skill_count  = sum over the hard_<skill> columns
soft_skill_count  = sum over the soft_<skill> columns

checkbox_to_percent(x) = clip(x, 0, 1) * 100
```

A ticked box is evidence the skill has been used; it is not a rating. Mapping
ticked to 100 preserves the yes/no information exactly and invents no gradations
the questionnaire never collected.

**R001:** `cert_count 0, cert_type_count 0, tech 1, hard 2, soft 2` (5 of 15 skills).

---

## 7. Step 6 — Competency signals

**Code:** `career_system/features.py` — `compute_signal_matrix`

Eleven signals, each 0–100, each blending up to three evidence groups:

```
group weights:  courses 0.45   skills 0.40   certs 0.15

courses_s = mean over s.domains of acad_<d>
skills_s  = mean over s.skills  of (checkbox * 100)
certs_s   = mean over s.certs   of (cert_<domain> * 100)

                 sum over present groups g of  W_g * value_g
sig_<s>  =  clip( -------------------------------------------- , 0 , 100 )
                 sum over present groups g of  W_g
```

**The denominator is the point.** If a student skipped a question, only that
evidence group drops out and the remaining weights renormalise. A plain weighted
sum would turn one blank cell into a NaN, which would silently corrupt every
alignment percentage for that student. The same convention is reused in the rubric
and in both alignment computations.

**R001, two signals in full:**

```
sig_programming
  courses  0.45 x acad_programming(57.81)
  skills   0.40 x mean(tech_programming=100, tech_web_mobile_dev=0) = 50.00
  certs    0.15 x 0
  = (0.45*57.81 + 0.40*50.00 + 0.15*0.00) / 1.00 = 46.02

sig_systems_engineering          <- no cert group defined for this signal
  courses  0.45 x mean(acad_systems_engineering=75.0, acad_foundations=87.5) = 81.25
  skills   0.40 x mean(hard_project_management=0, hard_research_writing=0)   = 0.00
  = (0.45*81.25 + 0.40*0.00) / 0.85 = 43.01
```

The `/0.85` is the renormalisation: the absent certificate group is removed from
the denominator instead of diluting the score toward zero.

**All 11 for R001:**

| Signal | Score |
|---|---|
| Communication & Client Service | 69.85 |
| Web, Interface & Multimedia | 56.56 |
| Data & Database Management | 50.94 |
| Algorithmic Thinking & Analytics | 48.69 |
| Programming & Software Construction | 46.02 |
| Infrastructure & Technical Support | 44.67 |
| Systems Analysis & Engineering | 43.01 |
| Documentation & Office Operations | 39.38 |
| Leadership & Project Management | 39.38 |
| Information Assurance & Security | 36.56 |
| Networking & Communications | 33.75 |

His grades average 76.84 but his signals sit in the 30s–60s, because the 40%
skills component and the 15% certificate component are near zero. The system is
reporting, correctly, a profile that is strong on paper and thin on demonstrated
practice.

---

## 8. Step 7 — Holland RIASEC archetypes

**Code:** `career_system/features.py` — `add_riasec_features`, `riasec_code`

```
riasec_<a> = mean over s in RIASEC_MAP[a] of sig_<s>
```

| Archetype | Signals aggregated |
|---|---|
| Realistic | networking, infrastructure_support |
| Investigative | theory_analytics, programming, security |
| Artistic | web_design |
| Social | communication_service |
| Enterprising | leadership_management |
| Conventional | documentation_clerical, data_management |

The summary code is the first letter of the three highest archetypes:

```
code = concat of first letters of top-3 archetypes by score
```

**R001:** Social 69.85, Artistic 56.56, Conventional 45.16, Investigative 43.76,
Realistic 39.21, Enterprising 39.38 → **SAC**.

---

## 9. Step 8 — Expectancy-Value Theory

**Code:** `career_system/features.py` — `add_evt_features`

Chapter III's formula, computed rather than described.

```
E = 0.50 * gwa_points + 0.50 * core_grade_mean

V = 0.50 * (mean of all 15 skill checkboxes * 100)
  + 0.30 * (cert_count      / 8  * 100)
  + 0.20 * (cert_type_count / 5  * 100)

Motivation(S) = E * V / 100
```

The division by 100 rescales the product back onto a readable 0–100 range.

**R001:**

```
E = 0.50 * 73.53 + 0.50 * 76.84                      = 75.19
V = 0.50 * 33.33 + 0.30 * 0.00 + 0.20 * 0.00         = 16.67
Motivation = 75.19 * 16.67 / 100                     = 12.53
```

**Why the product form matters.** Because Motivation is a *product*, a near-zero
Value collapses the result however high Expectancy runs: 75 × 17 → 12.5. R001 has
the ability but has not converted it into credentials. That multiplicative
collapse is exactly what EVT predicts, and it is visible in the data.

---

## 10. Step 9 — Feature matrix assembly

**Code:** `career_system/features.py` — `build_features`, `make_xy`, `FEATURE_SETS`

`build_features` runs the five steps above in order, then adds
`program_code = 1 if BSIT else 0`.

Three selectable feature sets:

| Set | Columns | Contents |
|---|---|---|
| `core` | 41 | `program_code`, `gwa_points`, 10 `acad_*`, `has_certification`, 5 `cert_type_*`, 8 `cert_*`, 15 skills |
| `engineered` | 28 | `program_code`, `core_grade_mean/std`, 5 counts, 11 `sig_*`, 6 `riasec_*`, 3 EVT |
| `full` | 68 | `core` + `engineered` — the default, and what the thesis describes |

`python main.py ablation` runs the benchmark on all three. If `full` does not beat
`core`, that belongs in Chapter V: an honest negative result about one's own
feature engineering is stronger than a claim the numbers do not support.

Before any of this, `validate_raw_frame` refuses to continue on a grade outside
1.00–5.00, a non-binary skill column, an unknown program or category value, or a
BSIT student carrying BSCS grades — the last catching a mis-wired column map long
before it becomes a confusion matrix.

---

## 11. Step 10 — Model pipeline and XGBoost

**Code:** `career_system/models.py` — `build_models`, `AdaptiveSMOTE`, `_boosting_estimator`

```
X (68 features) -> median imputation -> SMOTE -> XGBoost -> 4 probabilities
```

### 11.1 Imputation

Median per column, with `keep_empty_features=True`. On real data an item nobody
answered arrives as an all-NaN column; the default behaviour *drops* it, so the
matrix would no longer match the feature-name list that SHAP and the reports rely
on. Kept columns are constant and therefore inert.

### 11.2 Adaptive SMOTE

Plain SMOTE raises when a class has fewer than `k+1` members. Inside 5-fold CV the
Managerial class can fall to roughly 20 students, so `k` is chosen adaptively:

```
k = min(5, smallest_class_count - 1)
k < 1  ->  skip resampling, pass the data through untouched
```

### 11.3 Why both steps live inside the pipeline

If scaling or SMOTE is applied to the dataset up front, the test set's statistics —
and synthetic points interpolated from test-set neighbours — leak into training.
Accuracy can rise 5–15 points for free and will not reproduce on new students.
Inside an `imblearn` Pipeline, `cross_validate` refits both on each training fold
only, so the hold-out and every validation fold stay clean.

ZeroR deliberately receives **no** SMOTE: its purpose is to report the score
obtained by always answering the genuine majority class, and balancing its input
would destroy the very floor it measures.

### 11.4 The boosting objective

XGBoost minimises a regularised objective (Chen & Guestrin, 2016):

```
Obj = sum_i L( y_i , y_hat_i )  +  sum_k Omega( f_k )

Omega(f) = gamma * T  +  (1/2) * lambda * ||w||^2  +  alpha * ||w||_1
```

where `T` is the number of leaves in tree `f` and `w` its leaf weights. The
`Omega` penalty is what stops the model memorising a single College of Sciences
cohort.

Configured values:

```
n_estimators 400   max_depth 4    learning_rate 0.08
subsample 0.90     colsample_bytree 0.80   min_child_weight 2
reg_lambda 1.5     reg_alpha 0.10          gamma 0.05
objective multi:softprob    eval_metric mlogloss    tree_method hist
```

### 11.5 Prediction

Each class `c` accumulates a raw margin across the ensemble:

```
margin_c(x) = sum over the 400 trees of f_k,c(x)
```

turned into probabilities by the softmax:

```
P(c | x) = exp(margin_c) / sum_j exp(margin_j)

prediction = argmax_c P(c | x)
confidence = max_c P(c | x)
```

**R001:**

```
margins      Clerical -2.058   Managerial -2.401   Supervisory -1.877   Technical -1.666
exp          0.1277            0.0907              0.1530                0.1890
sum = 0.5604
P            22.79%            16.17%              27.31%                33.74%

PREDICTED: Technical (33.74%)        TRUE LABEL: Supervisory
```

---

## 12. Step 11 — Benchmark and evaluation metrics

**Code:** `career_system/benchmark.py` — `run_benchmark`, `_score_row`, `mcnemar_test`

Evaluation plan: a 70–30 stratified split, then stratified 5-fold cross-validation
**on the training portion only**, so the hold-out is touched exactly once.

### 12.1 ZeroR null accuracy floor

```
Accuracy_ZeroR = (1/n) * sum_i  I( y_i == C_majority )
```

### 12.2 Classification metrics

Per class `c`, from the confusion matrix:

```
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
Accuracy    = sum_c TP_c / n
```

Aggregated two ways:

```
macro:     (1/|C|) * sum_c  metric_c                 (every class counts equally)
weighted:  sum_c (n_c / n) * metric_c                (large classes dominate)
```

Macro is the honest figure on an imbalanced cohort — it exposes a model that
ignores Managerial entirely.

### 12.3 Improvement over the floor

```
Gain (pp)            = (Accuracy_model - Accuracy_ZeroR) * 100
Relative Improvement = Accuracy_model / Accuracy_ZeroR - 1
```

### 12.4 Exact McNemar test

Counts only cases where exactly one model was right:

```
only_a     = #{ i : model A correct , model B wrong }
only_b     = #{ i : model A wrong   , model B correct }
discordant = only_a + only_b

p = two-sided exact binomial test ( only_a ; discordant , 0.5 )
significant if p < 0.05
```

Under the null hypothesis that two models are equally accurate, the discordant
wins split 50/50. This is what converts "XGBoost is better" from an assertion into
a measured claim.

### 12.5 Current saved results

> **Provenance:** the tables in `outputs/tables/` were produced from a **400-student
> run** (`n_total 400`, train 280 / test 120), i.e. the synthetic cohort — not the
> 100-student real cohort in `data/real-cohort-model.csv`. Re-run
> `python main.py train --data data/real-cohort-model.csv` before quoting these in
> Chapter V.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Gain over ZeroR |
|---|---|---|---|---|---|
| ZeroR | 0.5417 | 0.1354 | 0.2500 | 0.1757 | — |
| Logistic Regression | 0.6583 | 0.5105 | 0.5541 | 0.5230 | +11.67 pp |
| XGBoost | 0.7500 | 0.6528 | 0.6387 | 0.6446 | +20.83 pp |

Stratified 5-fold CV on the training set: ZeroR 0.5429, Logistic Regression 0.6750,
XGBoost 0.7143 (std 0.0608).

McNemar: XGBoost vs ZeroR `p = 7.03e-05` (significant); XGBoost vs Logistic
Regression `p = 0.0433` (significant).

Note the gap between ZeroR's accuracy (0.5417) and its macro F1 (0.1757). That is
the imbalance made visible: always answering Technical is right 54% of the time
while being useless for the other three categories.

---

## 13. Step 12 — SHAP attribution

**Code:** `career_system/explain.py` — `transform_for_model`, `explain_student`, `compute_shap`

### 13.1 The Shapley value

For feature `j`, the contribution is its average marginal effect over all orderings
of the feature set `F` (Lundberg & Lee, 2017):

```
phi_j = sum over subsets T of F\{j}  of

          |T|! * (|F| - |T| - 1)!
          -----------------------  *  [ f(T + {j}) - f(T) ]
                   |F|!
```

`TreeExplainer` computes this exactly for tree ensembles in polynomial time.

### 13.2 Feeding SHAP the right matrix

```python
for name, step in pipeline.steps[:-1]:
    if hasattr(step, "fit_resample") and not hasattr(step, "transform"):
        continue              # a sampler: inactive outside fit()
    data = step.transform(data)
```

SMOTE acts only during `fit`; at prediction time data flows straight through it.
Replaying the imputer while stepping over the sampler means SHAP sees exactly the
matrix the booster scored.

### 13.3 Units and additivity

For a multi-class model, values are contributions to the **raw margin (log-odds)**
of a class, not to a probability. `+0.28` means the feature pushed the predicted
category's log-odds up by 0.28. They satisfy:

```
base_value_c  +  sum over all m features of phi_j,c   =   margin_c(x)
```

This identity is checkable by hand and is what the narration quotes.

### 13.4 Global importance

```
importance_j,c = (1/n) * sum_i | phi_i,j,c |
Overall_j      = (1/|C|) * sum_c importance_j,c
```

Written to `outputs/tables/shap_feature_importance.csv`; this is the table used for
curriculum review.

### 13.5 Contrast with the runner-up

The single feature that most favours the prediction over the second-place category:

```
delta_j = phi_j,predicted - phi_j,runner_up
answer  = argmax_j delta_j
```

**R001 — top contributions toward Technical:**

```
base value (cohort average margin for Technical)        = +0.1841

Research & Technical Writing (Hard skill)   value  0.00   phi +0.5336
RIASEC: Realistic                           value 39.21   phi -0.5018
Networking (Technical skill)                value  0.00   phi -0.3717
Computing Foundations (Grades)              value 87.50   phi -0.3651
Data & Database Management (Signal)         value 50.94   phi -0.2774
Communication (Soft skill)                  value  1.00   phi -0.2620
Degree Program (BSCS / BSIT)                value  0.00   phi +0.2503
Social & Professional Practice (Grades)     value 87.50   phi -0.2402

ADDITIVITY:  +0.1841 + (-1.8500) = -1.6659
model's actual margin for Technical        = -1.6659      exact match
```

The contributions are readable as reasoning: *not* ticking Networking and *not*
ticking Research Writing outweigh his grades, and his 87.50 in Computing
Foundations pushes him **away** from Technical — in this cohort, high foundations
grades without technical skills indicate a Supervisory or Clerical profile. The
model learned that; SHAP makes it inspectable.

If the `shap` package is unavailable the module degrades to permutation importance
and the CLI says so loudly.

---

## 14. Step 13 — Degree alignment

**Code:** `career_system/alignment.py` — `compute_degree_alignment`

```
                 sum over s of  W_prog,s * sig_s       (present signals only)
align(prog) = clip( -------------------------------- , 0 , 100 )
                 sum over s of  W_prog,s
```

Discipline weights follow CMO No. 25 s.2015 and the two PSU prospectuses:

| Signal | BSCS | BSIT |
|---|---|---|
| programming | 0.26 | 0.06 |
| theory_analytics | 0.24 | — |
| systems_engineering | 0.16 | 0.10 |
| data_management | 0.10 | 0.14 |
| web_design | 0.08 | 0.12 |
| security | 0.06 | 0.16 |
| networking | 0.05 | 0.22 |
| infrastructure_support | 0.05 | 0.20 |

BSCS emphasises algorithmic problem solving and software design; BSIT emphasises
application, integration and management of computing technologies.

Derived properties:

```
own_program_percent    = align(enrolled program)
stronger_program       = argmax over {BSCS, BSIT} of align
is_aligned_with_enrolment = (stronger_program == enrolled program)

verdict band:  >= 75 strongly aligned | >= 60 aligned
               >= 45 moderately aligned | otherwise weakly aligned
```

**R001:**

```
BSCS = .260*46.0 + .240*48.7 + .160*43.0 + .100*50.9
     + .080*56.6 + .060*36.6 + .050*33.8 + .050*44.7   = 46.26%
BSIT = .220*33.8 + .200*44.7 + .160*36.6 + .140*50.9
     + .120*56.6 + .100*43.0 + .060*46.0               = 43.19%

enrolled BSCS -> own-program 46.3%, stronger = BSCS
"This profile is moderately aligned with BS Computer Science (46.3%)."
```

---

## 15. Step 14 — Career track matching

**Code:** `career_system/alignment.py` — `score_career_tracks`

Each of the 16 tracks carries its own signal weight profile.

```
                sum over s of  W_k,s * sig_s        (present signals only)
fit_k  =  clip( ------------------------------ , 0 , 100 )
                sum over s of  W_k,s

match_k = clip( 0.75 * fit_k  +  0.25 * P(category_k | x) * 100 , 0 , 100 )
```

Tracks are then sorted by `match_k` descending.

**Why 75/25.** The score stays dominated by measured evidence the student can see
and act on, while still letting the classifier reorder the list. The machine
learning result steers the ranking rather than decorating it. Passing
`category_probabilities=None` scores on competency fit alone, which is useful for
sanity-checking the rule layer with the model out of the loop.

**R001 — top 3:**

```
1. Multimedia / UI-UX Developer   (Technical)
   fit   = .46*56.6 + .20*46.0 + .14*69.9 + .12*43.0 + .08*39.4 = 53.3
   match = 0.75 * 53.3 + 0.25 * 33.74 = 48.4%

2. Web Developer                  (Technical)
   fit = 52.3   ->  match = 47.7%

3. Helpdesk / Customer Service    (Clerical)
   fit = 53.9   ->  match = 46.1%
```

Helpdesk has the highest raw competency fit (53.9) yet ranks third, because the
model assigned Clerical only 22.79%. That is the 25% term doing its work.

---

## 16. Step 15 — Skill gaps and suggested actions

**Code:** `career_system/alignment.py` — `analyse_skill_gaps`, `suggested_actions`

```
for each signal s with  W_k,s >= 0.08  and  sig_s < 75:

    shortfall_s = (75 - sig_s) * W_k,s

sorted by shortfall descending
```

The weight filter keeps the advice short and relevant instead of listing every
weak area; multiplying by the weight ranks gaps by how much the target role
actually cares about them.

The action list is substituted when the leading gap is severe:

```
if shortfall_1 > 6.0 and the track has certifications:
    lead with "Pursue: <first certification>"
```

**R001 — gaps for Multimedia / UI-UX Developer:**

| Competency | Current | Weight | Shortfall |
|---|---|---|---|
| Web, Interface & Multimedia | 56.6 | 0.460 | 8.48 |
| Programming & Software Construction | 46.0 | 0.200 | 5.80 |
| Systems Analysis & Engineering | 43.0 | 0.120 | 3.84 |
| Documentation & Office Operations | 39.4 | 0.080 | 2.85 |
| Communication & Client Service | 69.9 | 0.140 | 0.72 |

Documentation is his *lowest raw score* yet ranks fourth, because it barely matters
for a UI/UX role. Web Design at 56.6 is far from his worst number but leads,
because it carries 46% of the role. Since 8.48 > 6.0, the action list becomes:

```
Pursue: UI/UX Design Workshop | Build a Design Portfolio |
Learn Figma Prototyping | Study Usability Testing | Join Design Sprints
```

---

## 17. Step 16 — Employer and sector mapping

**Code:** `career_system/alignment.py` — `employers_for_track`, `sector_summary`

Each track names sector keys; those index into `config.PALAWAN_EMPLOYERS`, which
returns up to two destinations per sector.

| Sector key | Label |
|---|---|
| `it_bpm` | IT-BPM / Business Process Outsourcing |
| `government` | Government & Public Administration |
| `academe_utilities` | Academe & Public Utilities |
| `tourism_hospitality` | Tourism & Hospitality |
| `private_tech` | Private Technology Enterprises |
| `finance` | Banking & Finance |

**Provenance.** Five of the six sectors trace to Chapter II of the paper, which
names the Digital Cities 2025 programme and the City Government of Puerto Princesa,
SupportNinja, hotels and resorts, Palawan State University with local water and
electric districts, and local startups serving tourism and logistics. Only
SupportNinja and PSU are named in the paper; the remaining organisations are
specific instances of categories the paper describes generically, and
`config.py` marks the whole table as placeholders to be verified with OSAS / the
OJT coordinator.

**`finance` has no basis in the paper** — the text contains no mention of banking,
rural banks or microfinance. Either add a sourced sentence to Chapter II or remove
that sector. `data/list-of-companies.xlsx` (2,796 Puerto Princesa establishments
with addresses) is present in the repository but is not referenced by any module;
filtering it would turn this table from researcher judgement into a sourced
derivation.

---

## 18. Where the data ends

**Code:** `career_system/recommender.py` — `save_report`, `recommend_batch`

| Path | Produced by | Contents |
|---|---|---|
| `outputs/tables/holdout_metrics.csv` | `train` | accuracy, precision, recall, F1, gain over ZeroR |
| `outputs/tables/cv_metrics.csv` | `train` | per-model CV means and standard deviations |
| `outputs/tables/confusion_matrix_*.csv` | `train` | one per model |
| `outputs/tables/classification_reports.txt` | `train` | per-class scikit-learn reports |
| `outputs/tables/benchmark_metadata.json` | `train` | split sizes, class counts, McNemar results |
| `outputs/tables/shap_feature_importance.csv` | `explain` | mean abs SHAP per feature per class |
| `outputs/tables/ablation_feature_sets.csv` | `ablation` | core vs engineered vs full |
| `outputs/tables/hyperparameter_search.csv` | `tune` | randomised search results |
| `outputs/tables/cohort_recommendations.csv` | `batch` | one summary row per student |
| `outputs/tables/profile_*.csv` | `profile` | descriptive cohort statistics |
| `outputs/figures/*.png` | several | comparison charts, confusion matrices, SHAP plots |
| `outputs/models/*.joblib` | `train` | fitted pipelines, so `recommend` reloads instead of refitting |
| `outputs/reports/report_<id>.txt` | `recommend` | the formatted report |
| `outputs/reports/report_<id>.json` | `recommend` | the same data, machine-readable |

The JSON report is the API contract for any future dashboard: predicted category
and all four probabilities, degree alignment with verdict, RIASEC code and scores,
ranked career tracks with drivers, the top 8 SHAP contributions, the narrative in
both registers, the additivity check, skill gaps, suggested actions, employer
destinations, all 11 signal scores, and an academic summary.

---

## 19. Worked example — R001 end to end

```
INPUT      BSCS, gwa 2.0588, 17 course grades, 0 certificates, 5 of 15 skills

Step 3     gwa 2.0588            -> 73.53 points
           grade 1.5             -> 87.50 points
Step 4     acad_programming       = mean(50.0, 56.25, 50.0, 75.0)  = 57.81
           core_grade_mean = 76.84   core_grade_std = 10.11
Step 5     cert_count 0   cert_type_count 0   skills 1+2+2 = 5
Step 6     sig_programming = (.45*57.81 + .40*50.00 + .15*0) / 1.00 = 46.02
           sig_systems_engineering = (.45*81.25 + .40*0) / 0.85     = 43.01
Step 7     riasec_social 69.85 -> Holland code SAC
Step 8     E 75.19   V 16.67   Motivation = 75.19*16.67/100 = 12.53
Step 9     68-feature vector, 0 NaNs
Step 10    margins (-2.058, -2.401, -1.877, -1.666) -> softmax
           -> Technical 33.74%  [true label: Supervisory]
Step 12    base +0.1841 + sum(phi) -1.8500 = -1.6659 = model margin   (exact)
Step 13    BSCS 46.26% / BSIT 43.19% -> moderately aligned, correctly enrolled
Step 14    UI-UX Developer: 0.75*53.3 + 0.25*33.74 = 48.4%
Step 15    top gap Web Design (75-56.6)*0.46 = 8.48 -> "Pursue: UI/UX Design Workshop"

OUTPUT     outputs/reports/report_r001.txt and .json
           outputs/figures/shap_local_r001.png
```

Every number traces back to a grade, a certificate or a ticked checkbox. Nothing
is introduced along the way, which is the property that makes the system
defensible as explainable decision support rather than a black box.

---

## 20. Caveats to state in Chapter V

1. **Label circularity.** `work_category` is computed from the same answers the
   model consumes as predictors (Step 2). Any accuracy figure obtained on
   rubric-derived labels measures rule recovery, not career prediction. Say so
   explicitly; do not let a reader infer otherwise from a high number.

2. **The rubric's two inert criteria.** GWA (20%) cannot change which category
   wins, and certifications (25%) decide nothing for students holding none. The
   effective rubric is 55% core courses / 45% skills. This follows from the
   rubric as written in Chapter IV.

3. **Batch confidence is not accuracy.** `python main.py batch` prints a mean
   confidence across the cohort, but the saved model was fitted on 70% of that
   cohort and is unusually certain about students it has already seen. The honest
   figures are the hold-out results from `train`. The CLI already prints this
   warning; keep it.

4. **The saved benchmark tables came from a 400-student run**, i.e. the synthetic
   cohort. Re-run the benchmark on `data/real-cohort-model.csv` before quoting any
   metric.

5. **R001 is misclassified, and that is worth volunteering.** The model predicted
   Technical at 33.74% confidence against a true label of Supervisory. But the
   rubric margin between Supervisory (47.89) and Clerical (46.91) was 0.98 points:
   this is a genuinely ambiguous profile, and low confidence on an ambiguous case
   is correct calibrated behaviour. Chapter IV already routes such students to the
   adviser. Presenting this case strengthens the methodology rather than weakening
   it.

6. **Employer destinations are unverified placeholders**, and the Banking &
   Finance sector has no basis in the paper (Step 16).

---

## Appendix — command reference

```
python scripts/clean_real_data.py        Google Forms export -> clean CSV
python scripts/build_model_dataset.py    clean CSV -> model schema + rubric labels

python main.py profile                   descriptive statistics        (SOP 1)
python main.py train                     XGBoost vs LogReg vs ZeroR    (SOP 2)
python main.py explain                   SHAP global importance        (SOP 4)
python main.py recommend --student-id R001   one student's full report (SOP 3)
python main.py predict                   answer the questionnaire yourself
python main.py batch                     cohort-wide export
python main.py ablation                  core vs engineered vs full
python main.py tune                      randomised hyperparameter search

python main.py --data data/real-cohort-model.csv    run on the real cohort
```
