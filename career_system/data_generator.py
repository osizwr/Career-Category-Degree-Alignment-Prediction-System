"""
Synthetic dataset generator for the PSU Career Category Recommendation System.

WHY THIS EXISTS
---------------
The thesis collects data from active fourth-year BSCS/BSIT students during the
first semester of AY 2026-2027. Until that collection finishes there is not
enough data to train on, so this module manufactures a cohort that is *shaped*
like the real one: the two separate prospectuses, the same 1.00-5.00 grading
scale, the same binary skill checkboxes, the same yes/no certification block,
and a deliberately imbalanced work-category distribution so that SMOTE and the
ZeroR baseline have something real to do.

HOW THE LABELS ARE MADE  (read this before you defend the numbers)
------------------------------------------------------------------
Students are drawn from four latent traits -- general academic ability,
technical inclination, people/leadership inclination and detail orientation.
Those traits generate the grades, skills and certifications. The work-category
label is then sampled from a multinomial-logit over the *observed* competency
signals plus Gumbel noise, i.e. a random-utility model:

        U(category) = intercept + strength * SUM( w_i * z(signal_i) ) + Gumbel
        label       = argmax over categories

Two consequences you must state honestly in Chapter V:

  1. Because a known rule generates the labels, accuracy on this dataset is
     OPTIMISTIC. It validates that the PIPELINE works end to end; it is not
     evidence about real student career outcomes.
  2. The noise term is tuned so the ceiling sits near 85%, not 100%. The
     Supervisory and Managerial categories deliberately overlap, which is why
     they are the confusable pair in the confusion matrix -- exactly what you
     would expect from real data.

Note that this is a DIFFERENT labelling rule from the one in
`career_system/labelling.py`. That module reproduces the Chapter IV rubric and
is deterministic; this one adds noise on purpose, so the synthetic benchmark
does not report a meaningless 100%.

Replace this whole module's output with the real Google Forms export once
collection is done; nothing downstream needs to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import features as feat


# --------------------------------------------------------------------------- #
# How common each work category is in the simulated cohort
# --------------------------------------------------------------------------- #
# Deliberately imbalanced: this is what justifies SMOTE in the methodology and
# what makes the ZeroR "null accuracy floor" land around 54%.
TARGET_DISTRIBUTION = {
    "Technical": 0.54,
    "Clerical": 0.19,
    "Supervisory": 0.17,
    "Managerial": 0.10,
}


# --------------------------------------------------------------------------- #
# The latent rule that assigns a work category
# --------------------------------------------------------------------------- #
# Weights are applied to z-scored (standardised) driver columns.
CATEGORY_UTILITY: dict[str, dict[str, float]] = {
    "Technical": {
        "sig_programming": 0.90,
        "sig_networking": 0.70,
        "sig_infrastructure_support": 0.70,
        "sig_security": 0.60,
        "sig_theory_analytics": 0.50,
        "sig_web_design": 0.40,
        "sig_data_management": 0.30,
        "sig_documentation_clerical": -0.60,
        "sig_communication_service": -0.40,
        "sig_leadership_management": -0.50,
    },
    "Clerical": {
        "sig_documentation_clerical": 1.10,
        "sig_communication_service": 0.50,
        "sig_data_management": 0.30,
        "sig_programming": -0.70,
        "sig_theory_analytics": -0.60,
        "sig_security": -0.45,
        "sig_networking": -0.40,
        "gwa_points": -0.30,
    },
    "Supervisory": {
        "sig_leadership_management": 1.00,
        "sig_communication_service": 0.60,
        "sig_infrastructure_support": 0.40,
        "sig_systems_engineering": 0.40,
        "cert_type_count": 0.30,
        "sig_documentation_clerical": -0.20,
    },
    "Managerial": {
        "sig_leadership_management": 1.10,
        "sig_communication_service": 0.60,
        "sig_systems_engineering": 0.50,
        "gwa_points": 0.60,
        "evt_motivation": 0.40,
        "cert_count": 0.30,
        "sig_programming": -0.30,
    },
}

# Signal-to-noise knob. Gumbel noise has scale 1.0, so this controls how much
# of the label is explainable. 1.15 lands the practical ceiling near 85%.
UTILITY_STRENGTH = 1.15


# --------------------------------------------------------------------------- #
# How each academic domain responds to the latent traits
# --------------------------------------------------------------------------- #
# (technical, people, detail) loading per academic domain.
_DOMAIN_TRAIT_LOADING = {
    "foundations":            (0.55, 0.15, 0.40),
    "programming":            (1.00, 0.00, 0.05),
    "theory_analytics":       (0.90, 0.00, 0.15),
    "data_management":        (0.60, 0.00, 0.45),
    "networking":             (0.85, 0.00, 0.10),
    "security":               (0.80, 0.05, 0.20),
    "web_design":             (0.70, 0.15, 0.10),
    "systems_engineering":    (0.60, 0.30, 0.30),
    "infrastructure_support": (0.80, 0.05, 0.15),
    "professional_practice":  (0.05, 0.95, 0.25),
}

# Programme-specific curricular emphasis (CMO No. 25 s.2015 + the two PSU
# prospectuses): BSCS students do better in the theory-heavy domains, BSIT
# students in the applied infrastructure ones.
_PROGRAM_DOMAIN_BONUS = {
    "BSCS": {"programming": 3.5, "theory_analytics": 4.0, "systems_engineering": 1.5},
    "BSIT": {"networking": 4.0, "infrastructure_support": 3.5, "security": 3.0,
             "data_management": 2.0, "web_design": 2.0},
}

# Which latent trait drives each self-reported checkbox, and how strongly,
# together with the share of the cohort expected to tick it.
_SKILL_TRAIT_LOADING = {
    "tech_programming":          ((1.00, 0.00, 0.00), 0.72),
    "tech_web_mobile_dev":       ((0.80, 0.10, 0.00), 0.55),
    "tech_database_sql":         ((0.65, 0.00, 0.45), 0.60),
    "tech_networking":           ((0.90, 0.00, 0.10), 0.38),
    "tech_it_support":           ((0.70, 0.20, 0.15), 0.52),
    "hard_data_analysis":        ((0.80, 0.00, 0.40), 0.45),
    "hard_graphic_ui_design":    ((0.35, 0.35, 0.30), 0.48),
    "hard_office_productivity":  ((0.10, 0.20, 0.90), 0.58),
    "hard_project_management":   ((0.15, 0.90, 0.35), 0.30),
    "hard_research_writing":     ((0.25, 0.30, 0.85), 0.50),
    "soft_communication":        ((0.00, 1.00, 0.15), 0.62),
    "soft_teamwork":             ((0.05, 0.90, 0.10), 0.70),
    "soft_leadership":           ((0.00, 1.00, 0.05), 0.42),
    "soft_problem_solving":      ((0.70, 0.20, 0.20), 0.66),
    "soft_adaptability":         ((0.10, 0.65, 0.15), 0.60),
}

# Which trait mix makes each certification SUBJECT domain likely, among the
# students who hold any credential at all.
_CERT_TRAIT_LOADING = {
    "networking":            (1.00, 0.00, 0.10),
    "security":              (0.95, 0.00, 0.15),
    "programming":           (1.00, 0.00, 0.05),
    "data_analytics":        (0.85, 0.05, 0.35),
    "web_cloud":             (0.80, 0.10, 0.05),
    "database":              (0.70, 0.00, 0.45),
    "office_productivity":   (0.05, 0.25, 0.95),
    "leadership_management": (0.00, 1.00, 0.25),
}
_CERT_BASE_RATE = {
    "networking": 0.22, "security": 0.18, "programming": 0.30,
    "data_analytics": 0.20, "web_cloud": 0.16, "database": 0.15,
    "office_productivity": 0.24, "leadership_management": 0.26,
}

# Q7: has the student completed any credential at all?
_HAS_CERT_BASE_RATE = 0.55

# Q9: which KIND of credential, among students who hold one. Seminars and
# webinars dominate in an undergraduate cohort; exam-based certifications are
# comparatively rare.
_CERT_TYPE_RATE = {
    "certification_exam": 0.22,
    "technical_seminar": 0.62,
    "workshop": 0.45,
    "training_program": 0.28,
    "webinar": 0.50,
}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _round_to_grade_step(points: np.ndarray) -> np.ndarray:
    """Convert 0-100 points to a real PH grade rounded to the nearest 0.25."""
    grades = cfg.points_to_grade(points)
    grades = np.round(grades * 4.0) / 4.0
    # Anything below the 3.00 passing mark is recorded as a re-taken/passed
    # 3.00, since fourth-year students have already cleared their core courses.
    return np.clip(grades, cfg.GRADE_BEST, cfg.GRADE_PASS)


def _sample_traits(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Draw the four correlated latent traits that define a student."""
    general = rng.normal(0.0, 1.0, n)

    # Technical and detail orientation are mildly opposed; people orientation is
    # largely independent of both.
    technical = rng.normal(0.0, 1.0, n)
    detail = -0.35 * technical + np.sqrt(1 - 0.35 ** 2) * rng.normal(0.0, 1.0, n)
    people = rng.normal(0.0, 1.0, n) + 0.15 * detail

    return {
        "general": general,
        "technical": technical,
        "people": people,
        "detail": detail,
    }


def _trait_response(traits, loading, rng, n, noise=0.55):
    """Combine (technical, people, detail) loadings into a standardised score."""
    w_tech, w_people, w_detail = loading
    raw = (
        w_tech * traits["technical"]
        + w_people * traits["people"]
        + w_detail * traits["detail"]
    )
    norm = np.sqrt(w_tech ** 2 + w_people ** 2 + w_detail ** 2) or 1.0
    return raw / norm + rng.normal(0.0, noise, n)


def _logistic_draw(rng, response, traits, base_rate, n, slope=0.95):
    """Bernoulli draw whose marginal rate matches `base_rate`."""
    intercept = np.log(base_rate / (1.0 - base_rate))
    logit = intercept + slope * response + 0.35 * traits["general"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(n) < prob).astype(int)


def _generate_raw_profiles(rng: np.random.Generator, n_students: int) -> pd.DataFrame:
    traits = _sample_traits(rng, n_students)

    program = rng.choice(cfg.PROGRAMS, size=n_students, p=[0.45, 0.55])
    data: dict[str, np.ndarray] = {
        "student_id": np.array([f"PSU-{i:04d}" for i in range(1, n_students + 1)]),
        "program": program,
    }

    # ------------------------------------------------------------- grades ---
    # Every course is generated, then blanked for the students who are not in
    # that program -- exactly the shape the real two-prospectus export has.
    all_points = []
    for course in cfg.ALL_COURSES:
        loadings = [_DOMAIN_TRAIT_LOADING[d] for d in course.domains if d in _DOMAIN_TRAIT_LOADING]
        if loadings:
            loading = tuple(float(np.mean([l[i] for l in loadings])) for i in range(3))
        else:
            # IT electives carry no domain: a generic blend of all three traits.
            loading = (0.45, 0.25, 0.30)

        affinity = _trait_response(traits, loading, rng, n_students, noise=0.60)

        bonus = np.zeros(n_students)
        for prog, domain_bonus in _PROGRAM_DOMAIN_BONUS.items():
            amount = sum(domain_bonus.get(d, 0.0) for d in course.domains)
            bonus += (program == prog) * amount

        # 68 points ~ a 2.25 grade; ability and affinity move it up or down.
        points = 68.0 + 11.0 * traits["general"] + 7.5 * affinity + bonus
        points = np.clip(points + rng.normal(0, 4.0, n_students), 5.0, 100.0)

        grades = _round_to_grade_step(points).astype(float)
        grades[program != course.program] = np.nan
        data[course.column] = grades

        mine = program == course.program
        masked = np.where(mine, points, np.nan)
        all_points.append(masked)

    # GWA blends the student's own courses with unmodelled GE subjects.
    core_mean_points = np.nanmean(np.vstack(all_points), axis=0)
    gwa_points = np.clip(
        0.75 * core_mean_points + 0.25 * (68.0 + 11.0 * traits["general"])
        + rng.normal(0, 2.5, n_students),
        5.0, 100.0,
    )
    data["gwa"] = _round_to_grade_step(gwa_points)

    # ------------------------------------------------------------- skills ---
    # Each student has a self-report bias: some consistently tick more boxes.
    # This is a real source of noise in self-reported instruments.
    report_bias = rng.normal(0.0, 0.45, n_students)

    for col, (loading, base_rate) in _SKILL_TRAIT_LOADING.items():
        response = _trait_response(traits, loading, rng, n_students, noise=0.70) + report_bias
        data[col] = _logistic_draw(rng, response, traits, base_rate, n_students, slope=0.90)

    # ----------------------------------------------------- certifications ---
    # Q7 first: engagement (people orientation + ability) drives whether the
    # student has any credential at all. Everything in Q8/Q9 is conditional.
    engagement = 0.55 * traits["people"] + 0.45 * traits["general"]
    has_cert = _logistic_draw(rng, engagement, traits, _HAS_CERT_BASE_RATE, n_students, slope=0.80)
    data["has_certification"] = has_cert

    for domain in cfg.CERT_DOMAINS:
        response = _trait_response(traits, _CERT_TRAIT_LOADING[domain], rng, n_students, noise=0.5)
        drawn = _logistic_draw(rng, response, traits, _CERT_BASE_RATE[domain], n_students)
        data[f"cert_{domain}"] = drawn * has_cert

    for cert_type, rate in _CERT_TYPE_RATE.items():
        drawn = (rng.random(n_students) < rate).astype(int)
        data[f"cert_type_{cert_type}"] = drawn * has_cert

    frame = pd.DataFrame(data)

    # A student who said "yes" must have named at least one kind of credential,
    # or the record contradicts itself the way a real cleaning log would flag.
    type_cols = list(cfg.CERT_TYPE_KEYS)
    empty = (frame["has_certification"] == 1) & (frame[type_cols].sum(axis=1) == 0)
    frame.loc[empty, "cert_type_technical_seminar"] = 1

    return frame


def _zscore(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=np.nanmean(values) if np.isfinite(np.nanmean(values)) else 0.0)
    std = values.std()
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def _assign_work_categories(
    engineered: pd.DataFrame,
    rng: np.random.Generator,
    calibration_rounds: int = 120,
) -> np.ndarray:
    """Sample a work category per student from the random-utility model."""
    categories = list(cfg.WORK_CATEGORIES)
    n = len(engineered)

    # Pre-compute the deterministic part of each category's utility.
    systematic = np.zeros((n, len(categories)))
    for j, category in enumerate(categories):
        total = np.zeros(n)
        for column, weight in CATEGORY_UTILITY[category].items():
            if column not in engineered.columns:
                raise KeyError(f"Utility driver '{column}' is not an engineered column.")
            total += weight * _zscore(engineered[column])
        systematic[:, j] = UTILITY_STRENGTH * total

    # Fixed noise draw so calibration adjusts intercepts, not luck.
    gumbel = rng.gumbel(0.0, 1.0, size=(n, len(categories)))
    targets = np.array([TARGET_DISTRIBUTION[c] for c in categories])
    intercepts = np.log(targets)

    labels_idx = None
    for _ in range(calibration_rounds):
        labels_idx = np.argmax(systematic + intercepts + gumbel, axis=1)
        observed = np.bincount(labels_idx, minlength=len(categories)) / n
        # Nudge intercepts toward the target mix (damped multiplicative update).
        observed = np.clip(observed, 1e-4, None)
        adjustment = 0.35 * np.log(targets / observed)
        if np.max(np.abs(adjustment)) < 1e-3:
            break
        intercepts = intercepts + adjustment

    return np.array(categories)[labels_idx]


def _inject_missing_values(
    df: pd.DataFrame, rng: np.random.Generator, missing_rate: float
) -> pd.DataFrame:
    """Blank out a few SELF-REPORTED cells, the way a real Google Form would.

    Registrar-sourced columns (grades, GWA, program) are left complete on
    purpose -- those come from official records, not from the student.
    """
    if missing_rate <= 0:
        return df

    out = df.copy()
    optional = list(cfg.ALL_SKILL_KEYS) + list(cfg.CERT_FEATURE_KEYS) + list(cfg.CERT_TYPE_KEYS)
    for col in optional:
        mask = rng.random(len(out)) < missing_rate
        if mask.any():
            out[col] = out[col].astype(float)
            out.loc[mask, col] = np.nan
    return out


def generate_dataset(
    n_students: int = 400,
    seed: int = cfg.RANDOM_STATE,
    missing_rate: float = 0.015,
) -> pd.DataFrame:
    """Build a complete, labelled synthetic cohort.

    Parameters
    ----------
    n_students   : cohort size. PSU's real fourth-year BSCS+BSIT population is
                   smaller than this; 400 is used so that 5-fold CV on a 70%
                   training split still leaves usable folds for the minority
                   Managerial class. Set this to your real N once you have it.
    seed         : reproducibility (the thesis fixes random_state = 42).
    missing_rate : fraction of self-reported cells left blank, so the data
                   cleaning step in the pipeline is exercised rather than
                   decorative.
    """
    rng = np.random.default_rng(seed)

    raw = _generate_raw_profiles(rng, n_students)
    engineered = feat.build_features(raw)
    raw[cfg.TARGET_COLUMN] = _assign_work_categories(engineered, rng)

    raw = _inject_missing_values(raw, rng, missing_rate)

    # Emit columns in the documented schema order.
    return raw[list(feat.RAW_COLUMNS)]


def save_dataset(df: pd.DataFrame, path=None):
    from pathlib import Path

    path = Path(path) if path else cfg.DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def describe_dataset(df: pd.DataFrame) -> str:
    """Human-readable summary used by the CLI and by Chapter V profiling."""
    # Blanks in the other program's prospectus are by design, not missing data.
    structural = sum(
        int((df["program"] != course.program).sum()) for course in cfg.ALL_COURSES
    )
    lines = [
        f"Students            : {len(df)}",
        f"Columns             : {len(df.columns)}",
        "Programs            : " + ", ".join(
            f"{k}={v}" for k, v in df['program'].value_counts().items()
        ),
        f"Course grade slots  : {len(cfg.BSCS_COURSES)} BSCS / {len(cfg.BSIT_COURSES)} BSIT, "
        f"folded into {len(cfg.DOMAIN_KEYS)} shared academic domains",
        f"GWA (1.00 best)     : mean {df['gwa'].mean():.3f}, "
        f"best {df['gwa'].min():.2f}, lowest {df['gwa'].max():.2f}",
        f"Credential holders  : {int(df['has_certification'].sum())} "
        f"({df['has_certification'].mean():.1%} answered 'Yes' to Q7)",
        f"Missing cells       : {int(df.isna().sum().sum()) - structural} "
        f"(excluding {structural} blanks for the other program's prospectus)",
        "",
        "Work category distribution (the ML target):",
    ]
    counts = df[cfg.TARGET_COLUMN].value_counts()
    for category in cfg.WORK_CATEGORIES:
        count = int(counts.get(category, 0))
        share = count / len(df) * 100 if len(df) else 0.0
        bar = "#" * int(round(share / 2))
        lines.append(f"  {category:<12} {count:>4}  {share:5.1f}%  {bar}")

    majority = counts.idxmax()
    lines += [
        "",
        f"ZeroR baseline would always answer '{majority}' "
        f"-> null accuracy floor = {counts.max() / len(df):.2%}",
    ]
    return "\n".join(lines)
