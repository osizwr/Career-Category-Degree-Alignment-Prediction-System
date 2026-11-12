"""
The Chapter IV labelling rubric, implemented.

    "Each researcher assigns a career category label using the following
     criteria/rubric, grounded in the Expectancy-Value framework (Eccles &
     Wigfield, 2002) adopted in this study, wherein academic performance
     represents the 'Expectancy' component and professional readiness
     represents the 'Value' component:

        Core course performance ...... 30%   grades in courses directly aligned
                                             to a given career category
        GWA .......................... 20%   overall cumulative performance
        Certifications / seminars .... 25%
        Self-reported skills ......... 25%

     A student's profile is scored against each candidate career category, and
     the category with the highest composite score is assigned as that
     student's preliminary label."

WHY THIS FILE EXISTS, AND WHAT IT IS NOT
----------------------------------------
Chapter IV says two labelers score every profile against this rubric and an
adviser adjudicates disagreements. That human step is what makes the labels
defensible. This module is the rubric arithmetic only -- it reproduces what a
labeler computes, so a complete pipeline can be run and demonstrated before (or
alongside) the manual coding.

It is NOT a substitute for the labelers, and the numbers it produces must never
be reported as model accuracy without the following caveat, stated plainly:

    The rubric scores a student on exactly the variables the classifier uses as
    predictors. A label produced by it is a deterministic function of the
    features, so a classifier trained on it is measuring whether it can
    re-learn this formula, not whether it can predict a real career. Expect
    high accuracy and report it as a PIPELINE CHECK.

To report real performance you need labels from an observed outcome: the two
researchers' independent manual coding, an OJT placement record, or a graduate
tracer study. When you have those, put them in the `work_category` column and
ignore this module entirely.

ONE PROPERTY OF THE RUBRIC WORTH KNOWING BEFORE THE DEFENSE
-----------------------------------------------------------
Only two of the four criteria can change which category wins.

  * GWA (20%) is a single number, identical for every candidate category, so it
    raises or lowers all four composites by the same amount and never decides
    between them. It sets the level, not the ranking.
  * Certifications (25%) decide nothing for a student who holds none -- which,
    in the batch collected so far, is most of them.

So in practice the label is driven by core course performance (30%) and ticked
skills (25%), re-weighted to 55/45 between them. That is not a bug in this
implementation; it follows from the rubric as written in Chapter IV. If the
panel expects GWA to carry a fifth of the decision, the rubric needs a
category-relative GWA term -- or the chapter needs a sentence saying GWA
expresses overall readiness rather than discriminating between categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import features as feat


def _weighted_percent(frame: pd.DataFrame, weights: dict, scale: float = 1.0) -> pd.Series:
    """Row-wise weighted mean over 0-100 columns, renormalised for blanks.

    A student who skipped a question loses that term rather than the whole
    score, which is the same convention `features.compute_signal_matrix` uses.
    """
    columns = [c for c in weights if c in frame.columns]
    if not columns:
        return pd.Series(np.nan, index=frame.index)

    values = frame[columns].apply(pd.to_numeric, errors="coerce") * scale
    w = pd.Series({c: float(weights[c]) for c in columns})

    total = values.mul(w, axis=1).sum(axis=1, skipna=True)
    available = values.notna().mul(w, axis=1).sum(axis=1)
    return total.div(available.where(available > 0))


def _category_components(engineered: pd.DataFrame, profile) -> pd.DataFrame:
    """The four rubric components for one work category, as 0-100 columns."""
    return pd.DataFrame({
        "core_courses": _weighted_percent(
            engineered, {f"acad_{d}": w for d, w in profile.domains.items()}
        ),
        "gwa": pd.to_numeric(engineered["gwa_points"], errors="coerce"),
        "certifications": _weighted_percent(
            engineered, {f"cert_{d}": w for d, w in profile.certs.items()}, scale=100.0
        ),
        "skills": _weighted_percent(engineered, profile.skills, scale=100.0),
    })


def _composite(components: pd.DataFrame) -> pd.Series:
    """Combine the four components with the 30/20/25/25 weights."""
    w = pd.Series(cfg.RUBRIC_WEIGHTS)
    total = components.mul(w, axis=1).sum(axis=1, skipna=True)
    available = components.notna().mul(w, axis=1).sum(axis=1)
    return total.div(available.where(available > 0))


def _engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either a raw frame or one that already has the domain scores."""
    return df if "acad_programming" in df.columns else feat.add_grade_points(df)


def rubric_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Composite rubric score (0-100) for every student x work category."""
    engineered = _engineered(df)
    scores = {
        category: _composite(_category_components(engineered, profile))
        for category, profile in cfg.WORK_CATEGORY_PROFILES.items()
    }
    return pd.DataFrame(scores, index=engineered.index)[list(cfg.WORK_CATEGORIES)]


def rubric_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per-student, per-category table of the four rubric components.

    This is the worksheet a labeler would fill in, and what you show a panelist
    who asks "how exactly did this student end up labelled Supervisory?".
    """
    engineered = _engineered(df)
    ids = engineered.get("student_id", pd.Series(engineered.index, index=engineered.index))

    blocks = []
    for category, profile in cfg.WORK_CATEGORY_PROFILES.items():
        components = _category_components(engineered, profile)
        blocks.append(pd.DataFrame({
            "student_id": ids,
            "work_category": category,
            f"core_courses ({cfg.RUBRIC_WEIGHTS['core_courses']:.0%})":
                components["core_courses"].round(2),
            f"gwa ({cfg.RUBRIC_WEIGHTS['gwa']:.0%})": components["gwa"].round(2),
            f"certifications ({cfg.RUBRIC_WEIGHTS['certifications']:.0%})":
                components["certifications"].round(2),
            f"skills ({cfg.RUBRIC_WEIGHTS['skills']:.0%})": components["skills"].round(2),
            # Computed from this block's own components, never joined back from a
            # separate table: the two would be in different row orders.
            "composite": _composite(components).round(2),
        }))

    table = pd.concat(blocks, ignore_index=True)
    return table.sort_values(["student_id", "composite"], ascending=[True, False])


def assign_work_categories(df: pd.DataFrame) -> pd.Series:
    """The preliminary label: the highest-scoring category per student."""
    return rubric_scores(df).idxmax(axis=1)


def label_margin(df: pd.DataFrame) -> pd.Series:
    """Gap between the winning and runner-up category, in rubric points.

    Chapter IV sends disagreements to a consensus discussion. A small margin is
    where two labelers are most likely to disagree, so these are the profiles
    worth putting in front of the adviser first.
    """
    scores = rubric_scores(df)
    ordered = np.sort(scores.to_numpy(dtype=float), axis=1)
    return pd.Series(ordered[:, -1] - ordered[:, -2], index=scores.index).round(2)
