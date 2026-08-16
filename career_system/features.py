"""
Feature engineering for the PSU Career Category & Degree Alignment System.

This module turns the RAW questionnaire/registrar columns into the numeric
feature matrix the classifiers consume. Three of the theories in Chapter III
are implemented here as real, inspectable computations rather than prose:

  * Holland's Theory of Vocational Choice (RIASEC)
        -> `add_riasec_features`, driven by config.RIASEC_MAP
  * Expectancy-Value Theory,  Motivation(S) = E x V
        -> `add_evt_features`
  * Competency "signals" that feed career matching and degree alignment
        -> `add_signal_features`

RAW SCHEMA (what a CSV must contain -- see `validate_raw_frame`)
    student_id            str
    program               'BSCS' | 'BSIT'
    gwa                   float, 1.00-5.00 (1.00 is highest)
    grade_<course_key>    float, 1.00-5.00, one per config.CORE_COURSES
    cert_<domain>         0/1, one per config.CERT_DOMAINS
    seminar_units         int, >= 0
    tech_<skill>          int 1-5   (self-reported Likert)
    hard_<skill>          int 1-5
    soft_<skill>          int 1-5
    work_category         str, one of config.WORK_CATEGORIES   (target; optional
                          when scoring a brand-new student)

FEATURE SCHEMA (what the models see)
    Every `grade_*` and `gwa` column is converted to 0-100 GRADE POINTS so that
    HIGHER ALWAYS MEANS BETTER. This matters: on the Philippine 1.00-5.00 scale
    a raw grade of 1.00 is excellent, which would otherwise make every model and
    every SHAP plot read backwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------- #
# Raw schema helpers
# --------------------------------------------------------------------------- #
GRADE_COLUMNS = tuple(f"grade_{c.key}" for c in cfg.CORE_COURSES)

RAW_FEATURE_COLUMNS = (
    ("program", "gwa")
    + GRADE_COLUMNS
    + cfg.CERT_FEATURE_KEYS
    + ("seminar_units",)
    + cfg.ALL_SKILL_KEYS
)

RAW_COLUMNS = ("student_id",) + RAW_FEATURE_COLUMNS + (cfg.TARGET_COLUMN,)


def validate_raw_frame(df: pd.DataFrame, require_target: bool = True) -> None:
    """Raise a helpful error if a user-supplied CSV is missing anything.

    Called before every training run so that swapping in the real Google Forms
    export fails loudly and early instead of silently producing garbage.
    """
    required = list(RAW_FEATURE_COLUMNS)
    if require_target:
        required.append(cfg.TARGET_COLUMN)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} required column(s): {missing}\n"
            f"Expected schema is documented in career_system/features.py. "
            f"Run `python main.py generate-data` to produce a correctly shaped "
            f"template you can compare against."
        )

    bad_program = set(df["program"].unique()) - set(cfg.PROGRAMS)
    if bad_program:
        raise ValueError(f"Unknown program value(s): {sorted(bad_program)}. Expected {cfg.PROGRAMS}.")

    if require_target:
        bad_target = set(df[cfg.TARGET_COLUMN].unique()) - set(cfg.WORK_CATEGORIES)
        if bad_target:
            raise ValueError(
                f"Unknown work_category value(s): {sorted(bad_target)}. "
                f"Expected {cfg.WORK_CATEGORIES}."
            )

    for col in ("gwa",) + GRADE_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = series[(series < cfg.GRADE_BEST) | (series > cfg.GRADE_WORST)]
        if len(out_of_range):
            raise ValueError(
                f"Column '{col}' has {len(out_of_range)} value(s) outside the "
                f"{cfg.GRADE_BEST:.2f}-{cfg.GRADE_WORST:.2f} grading scale "
                f"(e.g. {out_of_range.iloc[0]}). If your registrar exports "
                f"percentage grades, convert them first."
            )

    for col in cfg.ALL_SKILL_KEYS:
        series = pd.to_numeric(df[col], errors="coerce")
        bad = series[(series < cfg.LIKERT_MIN) | (series > cfg.LIKERT_MAX)]
        if len(bad):
            raise ValueError(
                f"Column '{col}' has {len(bad)} value(s) outside the "
                f"{cfg.LIKERT_MIN}-{cfg.LIKERT_MAX} Likert range."
            )


# --------------------------------------------------------------------------- #
# Small scaling utilities
# --------------------------------------------------------------------------- #
def likert_to_percent(series) -> np.ndarray:
    """Map a 1-5 Likert response onto 0-100 so it is comparable to grade points."""
    values = np.asarray(series, dtype=float)
    span = cfg.LIKERT_MAX - cfg.LIKERT_MIN
    return np.clip((values - cfg.LIKERT_MIN) / span * 100.0, 0.0, 100.0)


# --------------------------------------------------------------------------- #
# Step 1 - convert grades so that higher = better
# --------------------------------------------------------------------------- #
def add_grade_points(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 1.00-5.00 grades with 0-100 points and add summary statistics."""
    out = df.copy()

    out["gwa_points"] = cfg.grade_to_points(out["gwa"].astype(float))
    for col in GRADE_COLUMNS:
        out[col] = cfg.grade_to_points(out[col].astype(float))

    grades = out[list(GRADE_COLUMNS)]
    out["core_grade_mean"] = grades.mean(axis=1)
    # Low std = consistent performer; high std = spiky, specialised profile.
    out["core_grade_std"] = grades.std(axis=1, ddof=0)

    return out


# --------------------------------------------------------------------------- #
# Step 2 - certification and skill aggregates
# --------------------------------------------------------------------------- #
def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["cert_count"] = out[list(cfg.CERT_FEATURE_KEYS)].sum(axis=1).astype(float)

    for group_name, keys in (
        ("tech_skill_mean", cfg.TECH_SKILL_KEYS),
        ("hard_skill_mean", cfg.HARD_SKILL_KEYS),
        ("soft_skill_mean", cfg.SOFT_SKILL_KEYS),
    ):
        out[group_name] = out[list(keys)].mean(axis=1).astype(float)

    return out


# --------------------------------------------------------------------------- #
# Step 3 - competency signals (0-100)
# --------------------------------------------------------------------------- #
# Within a signal, the three evidence groups are combined with these weights.
# Academic grades carry the most weight because they are registrar-verified;
# self-reported skills carry less; certificates are a smaller booster signal.
_SIGNAL_GROUP_WEIGHTS = {"courses": 0.45, "skills": 0.40, "certs": 0.15}


def compute_signal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 0-100 competency signal score for every student.

    `df` must already have gone through `add_grade_points`.
    Returns a frame of columns named 'sig_<signal key>'.
    """
    result = {}

    for signal in cfg.SIGNALS:
        parts: dict[str, pd.Series] = {}

        if signal.courses:
            cols = [f"grade_{k}" for k in signal.courses if f"grade_{k}" in df.columns]
            if cols:
                parts["courses"] = df[cols].mean(axis=1)

        skill_frames = []
        for prefix, names in (
            ("tech_", signal.tech_skills),
            ("hard_", signal.hard_skills),
            ("soft_", signal.soft_skills),
        ):
            for name in names:
                col = f"{prefix}{name}"
                if col in df.columns:
                    skill_frames.append(pd.Series(likert_to_percent(df[col]), index=df.index))
        if skill_frames:
            parts["skills"] = pd.concat(skill_frames, axis=1).mean(axis=1)

        if signal.certs:
            cols = [f"cert_{c}" for c in signal.certs if f"cert_{c}" in df.columns]
            if cols:
                parts["certs"] = df[cols].mean(axis=1) * 100.0

        # Weighted mean over whichever evidence groups exist, computed PER ROW so
        # that a blank questionnaire cell only removes that group for that
        # student. Doing this with a plain weighted sum would be wrong: one NaN
        # term turns the whole score into NaN, which would silently corrupt the
        # alignment percentages of every student who skipped a single question.
        group_frame = pd.DataFrame(parts, index=df.index)
        weights = pd.Series({g: _SIGNAL_GROUP_WEIGHTS[g] for g in group_frame.columns})

        weighted_total = group_frame.mul(weights, axis=1).sum(axis=1, skipna=True)
        available_weight = group_frame.notna().mul(weights, axis=1).sum(axis=1)

        score = weighted_total.div(available_weight.where(available_weight > 0))
        result[f"sig_{signal.key}"] = score.clip(0.0, 100.0)

    return pd.DataFrame(result, index=df.index)


def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, compute_signal_matrix(df)], axis=1)


# --------------------------------------------------------------------------- #
# Step 4 - Holland RIASEC archetypes
# --------------------------------------------------------------------------- #
def add_riasec_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate competency signals into the six Holland vocational archetypes.

    Chapter III: "This theory guides Feature engineering, which will allow the
    researchers to qualitatively group prospective courses into industry-specific
    markers." That grouping is config.RIASEC_MAP.
    """
    out = df.copy()
    for archetype, signal_keys in cfg.RIASEC_MAP.items():
        cols = [f"sig_{k}" for k in signal_keys if f"sig_{k}" in out.columns]
        out[f"riasec_{archetype}"] = out[cols].mean(axis=1)
    return out


def riasec_code(row: pd.Series) -> str:
    """Return the three-letter Holland summary code (e.g. 'IRC') for a student."""
    scores = {a: row.get(f"riasec_{a}", 0.0) for a in cfg.RIASEC_MAP}
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]
    return "".join(a[0].upper() for a in top3)


# --------------------------------------------------------------------------- #
# Step 5 - Expectancy-Value Theory:  Motivation(S) = E x V
# --------------------------------------------------------------------------- #
def add_evt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implement the EVT formula stated in Chapter III of the thesis.

        E (Expectancy) = academic performance   -> core grades + GWA
        V (Value)      = skills + credentials   -> self-reported skills,
                                                   certifications, seminar units
        Motivation(S)  = E x V   (rescaled back onto 0-100 for readability)
    """
    out = df.copy()

    expectancy = 0.5 * out["gwa_points"] + 0.5 * out["core_grade_mean"]

    skill_component = likert_to_percent(
        out[list(cfg.ALL_SKILL_KEYS)].mean(axis=1)
    )
    cert_component = (out["cert_count"] / max(len(cfg.CERT_DOMAINS), 1)) * 100.0
    # Seminar units are open-ended; 12 units is treated as a "fully credentialed"
    # ceiling so the component stays on the same 0-100 footing as the others.
    seminar_component = np.clip(out["seminar_units"] / 12.0, 0.0, 1.0) * 100.0

    value = 0.50 * skill_component + 0.30 * cert_component + 0.20 * seminar_component

    out["evt_expectancy"] = expectancy
    out["evt_value"] = value
    out["evt_motivation"] = (expectancy * value) / 100.0

    return out


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the whole raw -> engineered transformation in the documented order."""
    out = add_grade_points(df)
    out = add_aggregate_features(out)
    out = add_signal_features(out)
    out = add_riasec_features(out)
    out = add_evt_features(out)
    out["program_code"] = (out["program"] == "BSIT").astype(int)  # BSCS=0, BSIT=1
    return out


# --------------------------------------------------------------------------- #
# Feature-set selection
# --------------------------------------------------------------------------- #
# `core`       = only what the student actually reported (no derived variables).
# `engineered` = only the theory-derived variables (signals, RIASEC, EVT).
# `full`       = both. This is the default and what the thesis describes.
#
# Having all three lets you run a small ablation study and state, with evidence,
# whether the RIASEC/EVT feature engineering actually earned its place.
_CORE_FEATURES = (
    ("program_code", "gwa_points")
    + GRADE_COLUMNS
    + cfg.CERT_FEATURE_KEYS
    + ("seminar_units",)
    + cfg.ALL_SKILL_KEYS
)

_ENGINEERED_FEATURES = (
    ("core_grade_mean", "core_grade_std", "cert_count",
     "tech_skill_mean", "hard_skill_mean", "soft_skill_mean")
    + cfg.SIGNAL_FEATURE_KEYS
    + cfg.RIASEC_FEATURE_KEYS
    + ("evt_expectancy", "evt_value", "evt_motivation")
)

FEATURE_SETS = {
    "core": _CORE_FEATURES,
    "engineered": ("program_code",) + _ENGINEERED_FEATURES,
    "full": _CORE_FEATURES + _ENGINEERED_FEATURES,
}


def get_feature_columns(feature_set: str = "full") -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. Choose from {sorted(FEATURE_SETS)}."
        )
    return list(FEATURE_SETS[feature_set])


def make_xy(
    df: pd.DataFrame,
    feature_set: str = "full",
    require_target: bool = True,
):
    """Convenience wrapper: raw frame -> (X, y, feature_names).

    Returns y=None when the frame has no target column (scoring a new student).
    """
    validate_raw_frame(df, require_target=require_target)
    engineered = build_features(df)
    columns = get_feature_columns(feature_set)

    X = engineered[columns].astype(float)
    y = engineered[cfg.TARGET_COLUMN] if cfg.TARGET_COLUMN in engineered.columns else None
    return X, y, columns
