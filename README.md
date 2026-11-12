# PSU Explainable Career Category Recommendation System

Reference implementation for the thesis:

> **An Explainable Career Category Recommendation System for Computer Science and
> Information Technology Graduating Students using Ensemble and Baseline Classifiers**
> Cajilo, M. J. M. · Donghil, Z. B. · Mahinay, J. V. · Sibayan, G. P. U.
> College of Sciences, Palawan State University

It classifies active fourth-year BSCS/BSIT students into broad **work categories**
(Technical / Clerical / Supervisory / Managerial), computes their **percentage degree
alignment** to BSCS and BSIT, explains every prediction with **SHAP**, and maps students to
**career tracks and Palawan employers** for pre-internship guidance.

Every input it asks for comes from the approved **Student Profile and Skills
Questionnaire**: the BSCS prospectus (17 courses) or the BSIT prospectus (24 courses),
cumulative GWA, the Part III certification block, and the three Part IV skill checklists.
`career_system/config.py` mirrors that instrument item for item.

---

## Quick start

```bash
# Python 3.11 or 3.12 recommended (3.13+ may lack xgboost/shap wheels)
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                       # runs the entire pipeline
```

That single command generates a cohort, profiles it, benchmarks all three classifiers,
computes SHAP explanations, prints a full student report and exports the whole cohort.
Everything lands in `outputs/`.

### Individual steps

| Command | What it does | Thesis objective |
|---|---|---|
| `python main.py generate-data` | Build the synthetic cohort CSV | — |
| `python main.py profile` | Descriptive statistics of the cohort | SOP 1 / Obj. 1 |
| `python main.py train` | XGBoost vs Logistic Regression vs ZeroR | SOP 2 / Obj. 2 |
| `python main.py recommend` | One student's full recommendation report | SOP 3 / Obj. 3 |
| `python main.py predict` | **Answer the questionnaire yourself and get a prediction** | Obj. 3 / 5 demo |
| `python main.py explain` | SHAP global feature importance | SOP 4 / Obj. 4 |
| `python main.py batch` | Cohort-wide recommendation export | Obj. 5 input |
| `python main.py ablation` | Does the feature engineering actually help? | — |
| `python main.py tune` | Randomised hyperparameter search | — |

Useful flags: `--data your.csv`, `--n 500`, `--seed 42`, `--feature-set core`,
`--no-smote`, `--student-id PSU-0007`. Add `--help` to any command.

### Testing your own profile

After `python main.py` finishes training it offers to run you through the questionnaire
yourself. You can also start it directly:

```bash
python main.py predict                              # answer the form now
python main.py predict --save-answers me.json       # keep your answers
python main.py predict --answers me.json            # replay them later
```

It asks for your degree program first (BSCS or BSIT), then shows only that program's
prospectus, then Part III and Part IV exactly as printed on the form. You get back the
predicted work category with the model's confidence, the illustrative job roles inside it,
your BSCS/BSIT alignment, your Holland code, the SHAP breakdown of what drove the
prediction, your skill gaps and the matching Palawan employers — plus the Chapter IV
labelling rubric scored on the same answers, so you can see where the model and the rubric
agree.

Nothing you type is added to the training data.

---

## What the system actually produces

```
outputs/
├── tables/     holdout_metrics.csv, cv_metrics.csv, confusion matrices,
│               classification reports, descriptive profiles, SHAP importances,
│               ablation results, cohort_recommendations.csv
├── figures/    model_comparison.png, confusion matrices, CV stability box plot,
│               SHAP beeswarms + global + per-student explanations,
│               skill profile, competency radar, GWA distributions
├── models/     fitted pipelines (.joblib) + the label encoder
└── reports/    per-student recommendation reports (.txt and .json)
```

A sample run on 400 synthetic students:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Gain over ZeroR |
|---|---|---|---|---|---|
| ZeroR | 0.5417 | 0.1354 | 0.2500 | 0.1757 | — |
| Logistic Regression | 0.6583 | 0.5105 | 0.5541 | 0.5230 | +11.67 pp |
| **XGBoost** | **0.7500** | **0.6528** | **0.6387** | **0.6446** | **+20.83 pp** |

Reproduce exactly with `python main.py` (seed 42, 400 students).

On the **100 real respondents** collected so far the ranking flips: Logistic Regression
reaches 0.80 hold-out accuracy against XGBoost's 0.63, and McNemar cannot separate them
(p = 0.06, only 5 discordant cases in a 30-student test set). With n = 100 those numbers are
too unstable to support a claim either way. Report the interval, not the winner, until the
full AY 2026–2027 cohort is in.

---

## How it maps to your chapters

### Chapter III — the theories are code, not prose

| Theory | Where it lives | What it actually computes |
|---|---|---|
| **Expectancy-Value Theory** | `features.add_evt_features` | `Motivation(S) = E × V`, with E from GWA + domain grades and V from ticked skills, certification domains and credential types |
| **Holland's RIASEC** | `config.RIASEC_MAP`, `features.add_riasec_features` | Six vocational archetype scores + a three-letter Holland code per student |
| **Ensemble Learning** | `models._boosting_estimator` | Regularised gradient-boosted trees |
| **XGBoost** | `models._boosting_estimator` | The Ω regularisation terms (`reg_lambda`, `reg_alpha`, `gamma`) from the objective function |
| **ZeroR** | `models.ZeroRClassifier` | The null-accuracy-floor formula, implemented literally |
| **SHAP** | `explain.py` | Global + per-student attributions in log-odds space |
| **SMOTE** | `models.AdaptiveSMOTE` | Synthetic minority samples, generated **inside** each CV fold |
| **Stratified K-Fold** | `benchmark.run_benchmark` | k=5, class proportions preserved per fold |
| **Information Overload** | `alignment.score_career_tracks` | Reduces 16 career tracks to a ranked top 3 |
| **Chapter IV rubric** | `labelling.py` | 30% core courses, 20% GWA, 25% certifications, 25% skills; highest composite wins |

### Chapter IV — the methodology settings

70-30 stratified split, Stratified 5-Fold CV, `random_state=42`, median imputation,
`StandardScaler`, `LabelEncoder`, SMOTE, Accuracy/Precision/Recall/F1. All in
`config.py` so you can change them in one place.

---

## Three things you will be asked about at the defense

### 1. "Where do the labels come from?"

This is the question to prepare hardest for, because the honest answer has two parts.

**On the synthetic cohort they are manufactured.** Until data collection finishes,
`data_generator.py` produces students whose labels come from a documented random-utility
model:

```
U(category) = intercept + strength · Σ wᵢ · z(signalᵢ) + Gumbel noise
label       = argmax over categories
```

The noise is tuned so the ceiling sits near 85%, and Supervisory/Managerial deliberately
overlap — which is why they are the confusable pair in the confusion matrix, just like real
data would be.

**On the real cohort they come from the Chapter IV rubric**, implemented in
`career_system/labelling.py`: core course performance 30%, GWA 20%, certifications 25%,
self-reported skills 25%, highest composite wins. `scripts/build_model_dataset.py` applies
it and writes a per-student worksheet (`data/real-cohort-model-rubric.csv`) showing every
component, so you can answer "why was this student labelled Supervisory?" with arithmetic.

**Either way, reported accuracy is optimistic, and for the same reason.** The rubric scores a
student on exactly the variables the classifier uses as predictors, so the label is a
deterministic function of the features and the model is re-learning a formula this repository
already contains. Say this plainly in Chapter V rather than letting a panelist find it.

**The fix is in your own methodology.** Chapter IV already specifies two researchers coding
each profile independently with an adviser adjudicating. Those manual labels — or OJT
placement records, or a graduate tracer study — are what make an accuracy figure mean
something. `build_model_dataset.py` prints how many students sit within 2.0 rubric points of
a second category; start the manual coding with those.

### 2. "Why is SMOTE inside the pipeline?"

Because doing it any other way inflates your results. If you SMOTE or scale *before*
splitting, synthetic points built from test-set neighbours leak into training, and accuracy
can jump 5–15 points for free. Putting both steps inside an `imblearn.Pipeline` means
`cross_validate` refits them on each training fold only.

**ZeroR deliberately gets no SMOTE** — its entire purpose is to report what you would score by
always answering the genuine majority class.

### 3. "Is XGBoost actually better than Logistic Regression?"

Run `python main.py train` and read the McNemar section. On 400 synthetic students XGBoost
leads by ~4–5 pp but the exact McNemar test returns **p ≈ 0.31 — not significant**. The system
prints this and tells you to report it honestly.

This is a feature, not a failure. A thesis that says *"boosting led by 5 points but with
n=120 in the test set the difference was not statistically significant"* is far stronger than
one that overclaims. If your real cohort is small, expect the same result and plan for it.

---

## Plugging in your real data

Export from Google Forms + the Registrar into a CSV with these columns:

| Column | Type | Notes |
|---|---|---|
| `student_id` | string | anonymised code, never a name |
| `program` | `BSCS` or `BSIT` | |
| `gwa` | 1.00–5.00 | **1.00 is highest** |
| `grade_bscs_<course>` | 1.00–5.00 | one per `config.BSCS_COURSES`; blank for BSIT students |
| `grade_bsit_<course>` | 1.00–5.00 | one per `config.BSIT_COURSES`; blank for BSCS students |
| `has_certification` | 0 or 1 | Q7 |
| `cert_type_<type>` | 0 or 1 | one per `config.CERTIFICATION_TYPES` (Q9) |
| `cert_<domain>` | 0 or 1 | one per `config.CERTIFICATION_DOMAINS`, read from Q8's free text |
| `tech_*`, `hard_*`, `soft_*` | 0 or 1 | **checkboxes, not ratings** — Part IV asks what you have used |
| `work_category` | Technical / Clerical / Supervisory / Managerial | the label |

Two scripts do this for you:

```bash
python scripts/clean_real_data.py        # Google Forms export -> data/real-data-clean.csv
python scripts/build_model_dataset.py    # clean CSV          -> data/real-cohort-model.csv
python main.py --data data/real-cohort-model.csv
```

**The two prospectuses are different, and that is handled.** A BSCS student has blanks in
every `grade_bsit_*` column and vice versa, so the per-course grades are never fed to the
model directly — a tree would just learn to read the degree program off the pattern of
blanks. Instead every course is tagged with one of ten shared **academic domains**
(`config.ACADEMIC_DOMAINS`) and the model sees `acad_<domain>`: the student's mean grade
points in that domain. Both prospectuses cover all ten, and `config.py` refuses to import if
one of them stops doing so.

`features.validate_raw_frame` checks every column, range and category up front, so a bad
export fails immediately with a specific message instead of producing quiet nonsense.

**Grades are on the Philippine 1.00–5.00 scale where lower is better.** The system converts
them to 0–100 "grade points" internally so that higher always means better — otherwise every
model and every SHAP plot reads backwards. If your registrar exports percentages, convert
first.

### Before you collect data, edit `config.py`

Everything you need to change is in that one file:

1. **`BSCS_COURSES` / `BSIT_COURSES`** — the two prospectuses as printed on the
   questionnaire, each course tagged with its academic domain
2. **`CERT_KEYWORDS`** — how Q8's free-text credential names are read into subject domains;
   add a pattern whenever a credential you care about goes unrecognised
3. **`WORK_CATEGORY_PROFILES`** — what evidence counts toward each work category in the
   Chapter IV labelling rubric
4. **`PALAWAN_EMPLOYERS`** — the entries are drawn from Chapter II and are **placeholders**;
   verify current internship capacity with OSAS / your OJT coordinator
5. **`CAREER_TRACKS`** — role definitions and their competency weights
6. **`DISCIPLINE_PROFILES`** — the BSCS vs BSIT weights behind the alignment percentage

### About cohort size

The default is 400 students so 5-fold CV on a 70% split still leaves usable folds for the
minority Managerial class. **Your real fourth-year BSCS+BSIT population will be smaller.**
With ~150 students and 68 features you are in overfitting territory — run
`python main.py ablation` and consider `--feature-set core`, and report the smaller
test set honestly when discussing significance.

---

## Project layout

```
psu-career-alignment/
├── main.py                     entry point
├── requirements.txt
├── career_system/
│   ├── config.py               ← all domain knowledge you edit
│   ├── features.py             raw → engineered (domains, RIASEC, EVT, signals)
│   ├── labelling.py            the Chapter IV scoring rubric (30/20/25/25)
│   ├── data_generator.py       synthetic cohort
│   ├── models.py               ZeroR, LogReg, XGBoost + leak-free pipelines
│   ├── benchmark.py            split, CV, metrics, McNemar, figures
│   ├── explain.py              SHAP global + per-student
│   ├── alignment.py            degree alignment %, career match, employers
│   ├── profiling.py            descriptive statistics (SOP 1)
│   ├── recommender.py          the full per-student report
│   ├── interactive.py          the questionnaire you answer yourself
│   └── cli.py                  command line interface
├── data/
└── outputs/
```

### The competency-signal design

One idea keeps the system internally consistent. Eleven **competency signals** (0–100
composites of domain grades, certifications and ticked skills) are computed once, then
reused by four different consumers:

```
BSCS prospectus (17)   BSIT prospectus (24)
          └──────────┬──────────┘
                     ▼
     10 shared academic domains (0–100 points)
                     │
      + certifications + ticked skills
                     │
                     ▼
        11 competency signals (0–100)
                     │
      ┌──────┬───────┴───────┬──────────────┐
      ▼      ▼               ▼              ▼
   RIASEC  career-track   BSCS / BSIT   Chapter IV
 archetypes  match %      alignment %     rubric
```

This is why every number the system shows a student can be traced back to a specific grade or
skill — which is the whole point of the explainable-AI framing.

---

## Ethics

The thesis commits to RA 10173 (Data Privacy Act of 2012). What the code does:

- `student_id` is the only identifier and is expected to be pre-anonymised
- no demographic variables (age, sex, address) are used as predictors — deliberately, since
  Chapter II criticises Huerte et al. (2025) for exactly that
- every report closes with a disclaimer that output is **guidance, not a placement decision**

What the code cannot do for you: obtaining informed consent, securing storage, and getting
clearance from the PSU ethics committee.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` compiles from source forever | You are on Python 3.13+. Use 3.11 or 3.12. |
| `[!] xgboost is not installed` | The system falls back to `HistGradientBoostingClassifier`. `pip install xgboost` to match the thesis. |
| `[!] shap is not installed` | Falls back to permutation importance. `pip install shap`. |
| `Dataset is missing N required column(s)` | Compare your CSV against `data/psu_computing_students.csv`. |
| `has values outside the 1.00-5.00 grading scale` | Your export is in percentages — convert first. |
| SMOTE errors on a small real cohort | `AdaptiveSMOTE` already shrinks `k_neighbors` automatically and skips resampling if a class is too small to interpolate. |
