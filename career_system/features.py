"""
Feature engineering for the PSU Explainable Career Category Recommendation System.

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
    student_id                  str
    program                     'BSCS' | 'BSIT'
    gwa                         float, 1.00-5.00 (1.00 is highest)
    grade_bscs_<course>         float, 1.00-5.00, one per config.BSCS_COURSES
    grade_bsit_<course>         float, 1.00-5.00, one per config.BSIT_COURSES
    has_certification           0/1          (Q7)
    cert_type_<type>            0/1          (Q9)
    cert_<domain>               0/1          (Q8, read through config.CERT_KEYWORDS)
    tech_<skill>                0/1          (Q10 checkbox)
    hard_<skill>                0/1          (Q11 checkbox)
    soft_<skill>                0/1          (Q12 checkbox)
    work_category               str, one of config.WORK_CATEGORIES  (target;
                                optional when scoring a brand-new student)

A BSCS student has blanks in every `grade_bsit_*` column and vice versa. That is
expected: the two programs sat different prospectuses.

FEATURE SCHEMA (what the models see)
    The per-course grade columns are NOT fed to the model. Half of them are
    structurally blank for any given student, so a tree would learn to read the
    degree program off the pattern of missing values instead of off the grades.
    Instead each course is folded into one of the ten shared ACADEMIC_DOMAINS
    and the model sees `acad_<domain>`: the student's mean grade in that domain,
    expressed in 0-100 GRADE POINTS so that HIGHER ALWAYS MEANS BETTER. This
    matters: on the Philippine 1.00-5.00 scale a raw grade of 1.00 is excellent,
    which would otherwise make every model and every SHAP plot read backwards.

    Skills are binary because the questionnaire asked for checkboxes
    ("Check all the skills you have actually used"), not ratings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------- #
# Raw schema helpers
# --------------------------------------------------------------------------- #
GRADE_COLUMNS = cfg.GRADE_COLUMNS

RAW_FEATURE_COLUMNS = (
    ("program", "gwa")
    + GRADE_COLUMNS
    + ("has_certification",)
    + cfg.CERT_TYPE_KEYS
    + cfg.CERT_FEATURE_KEYS
    + cfg.ALL_SKILL_KEYS
)

RAW_COLUMNS = ("student_id",) + RAW_FEATURE_COLUMNS + (cfg.TARGET_COLUMN,)

BINARY_COLUMNS = (
    ("has_certification",) + cfg.CERT_TYPE_KEYS + cfg.CERT_FEATURE_KEYS + cfg.ALL_SKILL_KEYS
)


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
            f"Dataset is missing {len(missing)} required column(s): {missing[:12]}"
            f"{' ...' if len(missing) > 12 else ''}\n"
            f"Expected schema is documented in career_system/features.py. Run "
            f"`python main.py generate-data` to produce a correctly shaped "
            f"template you can compare against, or rebuild the real export with "
            f"`python scripts/build_model_dataset.py`."
        )

    bad_program = set(df["program"].unique()) - set(cfg.PROGRAMS)
    if bad_program:
        raise ValueError(f"Unknown program value(s): {sorted(bad_program)}. Expected {cfg.PROGRAMS}.")

    if require_target:
        bad_target = set(df[cfg.TARGET_COLUMN].dropna().unique()) - set(cfg.WORK_CATEGORIES)
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

    for col in BINARY_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        bad = series[~series.isna() & ~series.isin([0, 1])]
        if len(bad):
            raise ValueError(
                f"Column '{col}' has {len(bad)} value(s) that are neither 0 nor 1 "
                f"(e.g. {bad.iloc[0]}). Part IV of the questionnaire is a set of "
                f"checkboxes, so every skill and certification column is binary. "
                f"If you are importing an older Likert-coded file, recode it: "
                f"4-5 -> 1, 1-3 -> 0, and say so in Chapter IV."
            )

    # A student should only have grades from their own prospectus. This catches a
    # mis-wired column map long before it reaches a confusion matrix.
    for program in cfg.PROGRAMS:
        rows = df["program"] == program
        if not rows.any():
            continue
        foreign = [c.column for c in cfg.ALL_COURSES if c.program != program]
        filled = df.loc[rows, foreign].notna().sum().sum()
        if filled:
            raise ValueError(
                f"{int(filled)} grade value(s) belonging to the other program's "
                f"prospectus are filled in for {program} students. Each student "
                f"answers only their own course list; check the column mapping in "
                f"scripts/build_model_dataset.py."
            )


# --------------------------------------------------------------------------- #
# Small scaling utilities
# --------------------------------------------------------------------------- #
def checkbox_to_percent(series) -> np.ndarray:
    """Map a 0/1 checkbox onto 0-100 so it is comparable to grade points.

    A ticked box is evidence the student has used the skill; it is not a claim
    about how good they are at it. Treating "ticked" as 100 keeps the yes/no
    information exact and invents no gradations the form never collected.
    """
    values = np.asarray(series, dtype=float)
    return np.clip(values, 0.0, 1.0) * 100.0


# --------------------------------------------------------------------------- #
# Step 1 - fold the two prospectuses into shared academic domains
# --------------------------------------------------------------------------- #
def add_grade_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add `gwa_points` and one `acad_<domain>` score per academic domain.

    The raw `grade_*` columns are left on the frame untouched (profiling reports
    them on the original 1.00-5.00 transcript scale) but they are not model
    features -- see the module docstring.
    """
    out = df.copy()
    out["gwa_points"] = cfg.grade_to_points(pd.to_numeric(out["gwa"], errors="coerce"))

    # Points version of every course grade, computed once.
    points = pd.DataFrame(index=out.index)
    for column in GRADE_COLUMNS:
        points[column] = cfg.grade_to_points(pd.to_numeric(out[column], errors="coerce"))
        # grade_to_points() clips NaN-free; restore the blanks it would hide.
        points.loc[pd.to_numeric(out[column], errors="coerce").isna(), column] = np.nan

    for domain in cfg.DOMAIN_KEYS:
        # Each student only has one program's courses, so taking the row mean
        # over BOTH prospectuses' columns for this domain is exactly the mean of
        # whichever set is filled in.
        columns = [c.column for c in cfg.ALL_COURSES if domain in c.domains]
        out[f"acad_{domain}"] = points[columns].mean(axis=1, skipna=True)

    domains = out[list(cfg.ACAD_FEATURE_KEYS)]
    out["core_grade_mean"] = domains.mean(axis=1, skipna=True)
    # Low std = consistent performer; high std = spiky, specialised profile.
    out["core_grade_std"] = domains.std(axis=1, ddof=0, skipna=True)

    return out


# --------------------------------------------------------------------------- #
# Step 2 - certification and skill aggregates
# --------------------------------------------------------------------------- #
def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["cert_count"] = out[list(cfg.CERT_FEATURE_KEYS)].sum(axis=1).astype(float)
    out["cert_type_count"] = out[list(cfg.CERT_TYPE_KEYS)].sum(axis=1).astype(float)

    for name, keys in (
        ("tech_skill_count", cfg.TECH_SKILL_KEYS),
        ("hard_skill_count", cfg.HARD_SKILL_KEYS),
        ("soft_skill_count", cfg.SOFT_SKILL_KEYS),
    ):
        out[name] = out[list(keys)].sum(axis=1).astype(float)

    return out


# --------------------------------------------------------------------------- #
# Step 3 - competency signals (0-100)
# --------------------------------------------------------------------------- #
# Within a signal, the three evidence groups are combined with these weights.
# Academic grades carry the most weight because they are registrar-verified;
# self-reported checkboxes carry less; certificates are a smaller booster.
_SIGNAL_GROUP_WEIGHTS = {"courses": 0.45, "skills": 0.40, "certs": 0.15}


def compute_signal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 0-100 competency signal score for every student.

    `df` must already have gone through `add_grade_points`.
    Returns a frame of columns named 'sig_<signal key>'.
    """
    result = {}

    for signal in cfg.SIGNALS:
        parts: dict[str, pd.Series] = {}

        if signal.domains:
            cols = [f"acad_{d}" for d in signal.domains if f"acad_{d}" in df.columns]
            if cols:
                parts["courses"] = df[cols].mean(axis=1, skipna=True)

        skill_frames = []
        for prefix, names in (
            ("tech_", signal.tech_skills),
            ("hard_", signal.hard_skills),
            ("soft_", signal.soft_skills),
        ):
            for name in names:
                col = f"{prefix}{name}"
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors="coerce")
                    scaled = pd.Series(checkbox_to_percent(values), index=df.index)
                    skill_frames.append(scaled.where(values.notna()))
        if skill_frames:
            parts["skills"] = pd.concat(skill_frames, axis=1).mean(axis=1, skipna=True)

        if signal.certs:
            cols = [f"cert_{c}" for c in signal.certs if f"cert_{c}" in df.columns]
            if cols:
                parts["certs"] = df[cols].mean(axis=1, skipna=True) * 100.0

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
    """Aggregate competency signals into the six Holland vocational archetypes."""
    out = df.copy()
    for archetype, signal_keys in cfg.RIASEC_MAP.items():
        cols = [f"sig_{k}" for k in signal_keys if f"sig_{k}" in out.columns]
        out[f"riasec_{archetype}"] = out[cols].mean(axis=1, skipna=True)
    return out


def riasec_code(row: pd.Series) -> str:
    """Return the three-letter Holland summary code (e.g. 'IRC') for a student."""
    scores = {a: row.get(f"riasec_{a}", 0.0) for a in cfg.RIASEC_MAP}
    scores = {a: (0.0 if v is None or pd.isna(v) else float(v)) for a, v in scores.items()}
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]
    return "".join(a[0].upper() for a in top3)


# --------------------------------------------------------------------------- #
# Step 5 - Expectancy-Value Theory:  Motivation(S) = E x V
# --------------------------------------------------------------------------- #
def add_evt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implement the EVT formula stated in Chapter III of the thesis.

        E (Expectancy) = academic performance   -> domain grades + GWA
        V (Value)      = skills + credentials   -> ticked skills, recognised
                                                   certification domains and
                                                   credential types
        Motivation(S)  = E x V   (rescaled back onto 0-100 for readability)
    """
    out = df.copy()

    expectancy = 0.5 * out["gwa_points"] + 0.5 * out["core_grade_mean"]

    skill_component = (
        out[list(cfg.ALL_SKILL_KEYS)].mean(axis=1, skipna=True) * 100.0
    )
    cert_component = (out["cert_count"] / max(len(cfg.CERT_DOMAINS), 1)) * 100.0
    # Credential breadth replaces the old open-ended "seminar units" count,
    # which the approved questionnaire never asks for. Q9 offers five kinds of
    # credential, so holding all five is the ceiling.
    type_component = (out["cert_type_count"] / max(len(cfg.CERTIFICATION_TYPES), 1)) * 100.0

    value = 0.50 * skill_component + 0.30 * cert_component + 0.20 * type_component

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
# `core`       = only what the student actually reported, on a scale both
#                programs share (domain grades, certificates, ticked skills).
# `engineered` = only the theory-derived variables (signals, RIASEC, EVT).
# `full`       = both. This is the default and what the thesis describes.
#
# Having all three lets you run a small ablation study and state, with evidence,
# whether the RIASEC/EVT feature engineering actually earned its place.
_CORE_FEATURES = (
    ("program_code", "gwa_points")
    + cfg.ACAD_FEATURE_KEYS
    + ("has_certification",)
    + cfg.CERT_TYPE_KEYS
    + cfg.CERT_FEATURE_KEYS
    + cfg.ALL_SKILL_KEYS
)

_ENGINEERED_FEATURES = (
    ("core_grade_mean", "core_grade_std", "cert_count", "cert_type_count",
     "tech_skill_count", "hard_skill_count", "soft_skill_count")
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
