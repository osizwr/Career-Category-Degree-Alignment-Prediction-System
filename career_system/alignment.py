"""
Degree alignment, career-track matching and Palawan employer mapping.

This module answers SOP 3 / Objective 3:

    "compute the percentage alignment relative to the computing disciplines and
     map these profiles to specific local enterprises, business process
     outsourcing hubs, and government agencies in Palawan"

Three computations live here:

  1. DEGREE ALIGNMENT %   - how closely a student's competency profile matches
                            the BSCS and the BSIT discipline profiles
                            (config.DISCIPLINE_PROFILES, derived from CMO 25).
  2. CAREER-TRACK MATCH % - the "92.8%" figure in the appendix mockups. It
                            blends the student's competency fit for a role with
                            the classifier's probability for that role's work
                            category, so the machine learning result actually
                            steers the ranking instead of decorating it.
  3. SKILL-GAP ANALYSIS   - which competency signals fall short for the
                            recommended track. This is what turns a prediction
                            into the "suggested actions" chips.

Every score is a transparent weighted mean of 0-100 competency signals, so any
number the system shows a student can be traced back to specific grades,
certificates and skills. That traceability is the whole point of the XAI framing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from . import features as feat


# --------------------------------------------------------------------------- #
# Weighted signal scoring
# --------------------------------------------------------------------------- #
def _weighted_signal_score(signal_row: pd.Series, weights: dict) -> float:
    """Weighted mean of 0-100 competency signals -> a single 0-100 score.

    Signals that are missing for this student are dropped and the remaining
    weights are renormalised, so an incomplete questionnaire produces a score
    based on less evidence rather than no score at all.
    """
    normalised = cfg.normalise_weights(weights)

    total, available = 0.0, 0.0
    for signal_key, weight in normalised.items():
        value = signal_row.get(f"sig_{signal_key}", np.nan)
        if value is None or pd.isna(value):
            continue
        total += weight * float(value)
        available += weight

    if available <= 0:
        return float("nan")
    return float(np.clip(total / available, 0.0, 100.0))


# --------------------------------------------------------------------------- #
# 1. Degree alignment
# --------------------------------------------------------------------------- #
@dataclass
class DegreeAlignment:
    bscs_percent: float
    bsit_percent: float
    enrolled_program: str

    @property
    def own_program_percent(self) -> float:
        return self.bscs_percent if self.enrolled_program == "BSCS" else self.bsit_percent

    @property
    def stronger_program(self) -> str:
        return "BSCS" if self.bscs_percent >= self.bsit_percent else "BSIT"

    @property
    def is_aligned_with_enrolment(self) -> bool:
        return self.stronger_program == self.enrolled_program

    def verdict(self) -> str:
        """One sentence a counsellor can read out loud."""
        own = self.own_program_percent
        label = cfg.PROGRAM_LABELS[self.enrolled_program]

        if own >= 75:
            band = "strongly aligned"
        elif own >= 60:
            band = "aligned"
        elif own >= 45:
            band = "moderately aligned"
        else:
            band = "weakly aligned"

        text = (f"This profile is {band} with {label} "
                f"({own:.1f}% discipline alignment).")

        if not self.is_aligned_with_enrolment:
            other = self.stronger_program
            gap = abs(self.bscs_percent - self.bsit_percent)
            if gap >= 5:
                text += (f" Note that the competency profile leans "
                         f"{gap:.1f} points closer to {cfg.PROGRAM_LABELS[other]}; "
                         f"consider {other}-flavoured electives, certifications and OJT placements.")
        return text


def compute_degree_alignment(signal_row: pd.Series, program: str) -> DegreeAlignment:
    """Percentage alignment of one student's profile to BSCS and to BSIT."""
    return DegreeAlignment(
        bscs_percent=_weighted_signal_score(signal_row, cfg.DISCIPLINE_PROFILES["BSCS"]),
        bsit_percent=_weighted_signal_score(signal_row, cfg.DISCIPLINE_PROFILES["BSIT"]),
        enrolled_program=program,
    )


def compute_alignment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Degree alignment for a whole cohort (used for the institutional report)."""
    engineered = feat.build_features(df)
    rows = []
    for _, row in engineered.iterrows():
        alignment = compute_degree_alignment(row, row["program"])
        rows.append({
            "student_id": row.get("student_id", ""),
            "program": row["program"],
            "bscs_alignment_pct": round(alignment.bscs_percent, 2),
            "bsit_alignment_pct": round(alignment.bsit_percent, 2),
            "own_program_alignment_pct": round(alignment.own_program_percent, 2),
            "stronger_program": alignment.stronger_program,
            "aligned_with_enrolment": alignment.is_aligned_with_enrolment,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Career-track matching
# --------------------------------------------------------------------------- #
# How much of the match score comes from measured competency fit vs. from the
# classifier's confidence in the work category. 75/25 keeps the score dominated
# by evidence the student can see and act on, while still letting the model
# reorder tracks.
COMPETENCY_WEIGHT = 0.75
MODEL_WEIGHT = 0.25


@dataclass
class TrackMatch:
    track: cfg.CareerTrack
    match_percent: float
    competency_fit: float
    category_probability: float
    contributing_signals: pd.Series   # signal key -> weighted contribution

    @property
    def name(self) -> str:
        return self.track.name

    @property
    def work_category(self) -> str:
        return self.track.work_category

    def top_signals(self, n: int = 4) -> list:
        top = self.contributing_signals.sort_values(ascending=False).head(n)
        return [(cfg.SIGNAL_BY_KEY[k].label, float(v)) for k, v in top.items()]


def score_career_tracks(
    signal_row: pd.Series,
    category_probabilities: dict[str, float] | None = None,
) -> list[TrackMatch]:
    """Rank every configured career track for one student.

    `category_probabilities` maps a work category to the classifier's predicted
    probability. Pass None to score on competency fit alone (useful for
    sanity-checking the rule layer without the model in the loop).
    """
    matches = []

    for track in cfg.CAREER_TRACKS:
        weights = cfg.normalise_weights(track.weights)

        contributions = {}
        total, available = 0.0, 0.0
        for signal_key, weight in weights.items():
            value = signal_row.get(f"sig_{signal_key}", np.nan)
            if value is None or pd.isna(value):
                continue
            contributions[signal_key] = weight * float(value)
            total += weight * float(value)
            available += weight

        # Same per-row renormalisation as the degree alignment: a missing signal
        # must not drag a track's match score to zero (or to NaN).
        fit = float(np.clip(total / available, 0.0, 100.0)) if available > 0 else 0.0

        if category_probabilities is None:
            match = fit
            probability = float("nan")
        else:
            probability = float(category_probabilities.get(track.work_category, 0.0))
            match = COMPETENCY_WEIGHT * fit + MODEL_WEIGHT * (probability * 100.0)

        matches.append(TrackMatch(
            track=track,
            match_percent=round(float(np.clip(match, 0.0, 100.0)), 1),
            competency_fit=round(fit, 1),
            category_probability=probability,
            contributing_signals=pd.Series(contributions),
        ))

    matches.sort(key=lambda m: m.match_percent, reverse=True)
    return matches


# --------------------------------------------------------------------------- #
# 3. Skill-gap analysis  ->  suggested actions
# --------------------------------------------------------------------------- #
@dataclass
class SkillGap:
    signal_key: str
    label: str
    current: float
    weight: float
    shortfall: float     # how far below the target band, weighted by importance


def analyse_skill_gaps(signal_row: pd.Series, track: cfg.CareerTrack,
                       target: float = 75.0, min_weight: float = 0.08) -> list[SkillGap]:
    """Find the competencies holding a student back from a target track.

    Only signals that genuinely matter for the track (weight >= min_weight) are
    considered, so the advice stays short and relevant instead of listing every
    weak area a student has.
    """
    weights = cfg.normalise_weights(track.weights)
    gaps = []

    for signal_key, weight in weights.items():
        if weight < min_weight:
            continue
        current = float(signal_row.get(f"sig_{signal_key}", 0.0))
        if current < target:
            gaps.append(SkillGap(
                signal_key=signal_key,
                label=cfg.SIGNAL_BY_KEY[signal_key].label,
                current=round(current, 1),
                weight=round(weight, 3),
                shortfall=round((target - current) * weight, 2),
            ))

    gaps.sort(key=lambda g: g.shortfall, reverse=True)
    return gaps


def suggested_actions(track: cfg.CareerTrack, gaps: list[SkillGap], limit: int = 5) -> list[str]:
    """The action chips shown under the recommendation (see Figure 4.2)."""
    actions = list(track.actions)[:limit]

    # If a heavily weighted competency is badly short, lead with a concrete
    # certification suggestion instead of the generic action list.
    if gaps and gaps[0].shortfall > 6.0 and track.certifications:
        primary = f"Pursue: {track.certifications[0]}"
        actions = [primary] + [a for a in actions if a != primary][: limit - 1]

    return actions


# --------------------------------------------------------------------------- #
# 4. Palawan employer mapping
# --------------------------------------------------------------------------- #
def employers_for_track(track: cfg.CareerTrack, limit_per_sector: int = 2) -> list[dict]:
    """Local internship / employment destinations for a career track."""
    destinations = []
    for sector_key in track.sectors:
        for employer in cfg.PALAWAN_EMPLOYERS.get(sector_key, [])[:limit_per_sector]:
            destinations.append({
                "sector_key": sector_key,
                "sector": cfg.SECTOR_LABELS.get(sector_key, sector_key),
                "name": employer.name,
                "locality": employer.locality,
                "note": employer.note,
            })
    return destinations


def sector_summary(track: cfg.CareerTrack) -> str:
    labels = [cfg.SECTOR_LABELS.get(s, s) for s in track.sectors]
    return ", ".join(labels)


# --------------------------------------------------------------------------- #
# Cohort-level view (for PSU-CS, not for the individual student)
# --------------------------------------------------------------------------- #
def cohort_signal_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean competency signal per program -- the curriculum-review table."""
    engineered = feat.build_features(df)
    columns = list(cfg.SIGNAL_FEATURE_KEYS)

    grouped = engineered.groupby("program")[columns].mean().T
    grouped.index = [cfg.SIGNAL_BY_KEY[k[len("sig_"):]].label for k in grouped.index]
    grouped["Cohort mean"] = engineered[columns].mean().values
    grouped.index.name = "Competency signal"
    return grouped.round(2)


def cohort_track_demand(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """How often each career track lands in a student's top-N recommendations.

    Lets OSAS see, before deployment season, roughly how many students will be
    chasing which kind of placement.
    """
    engineered = feat.build_features(df)
    tally: dict[str, int] = {}

    for _, row in engineered.iterrows():
        for match in score_career_tracks(row, category_probabilities=None)[:top_n]:
            tally[match.name] = tally.get(match.name, 0) + 1

    frame = pd.DataFrame(
        sorted(tally.items(), key=lambda kv: kv[1], reverse=True),
        columns=["Career track", f"Times in top {top_n}"],
    )
    frame["Share of cohort"] = (frame[f"Times in top {top_n}"] / len(engineered)).map(lambda v: f"{v:.1%}")
    frame["Work category"] = frame["Career track"].map(
        {t.name: t.work_category for t in cfg.CAREER_TRACKS}
    )
    frame["Primary Palawan sectors"] = frame["Career track"].map(
        {t.name: sector_summary(t) for t in cfg.CAREER_TRACKS}
    )
    return frame
