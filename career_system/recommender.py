"""
The decision-support layer: one student in, one full recommendation out.

This is the console equivalent of the appendix mockups (Figures 4.2-4.4). For a
single fourth-year student it produces:

    * the predicted broad WORK CATEGORY and the model's confidence
    * the % DEGREE ALIGNMENT to BSCS and to BSIT
    * the Holland RIASEC summary code
    * the ranked TOP CAREER TRACKS with match scores
    * the SHAP "why was this recommended?" breakdown
    * SKILL GAPS and SUGGESTED ACTIONS
    * PALAWAN internship / employment destinations

Nothing here invents information. Every number traces back to a grade, a
certificate or a self-reported skill, which is the property that makes the
system defensible as an explainable decision-support tool rather than a
black box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import alignment as align
from . import config as cfg
from . import explain as xai
from . import features as feat
from . import models as mdl


# --------------------------------------------------------------------------- #
# Report container
# --------------------------------------------------------------------------- #
@dataclass
class StudentReport:
    student_id: str
    program: str
    predicted_category: str
    confidence: float
    category_probabilities: dict
    degree_alignment: align.DegreeAlignment
    riasec_code: str
    riasec_scores: dict
    track_matches: list
    skill_gaps: list
    actions: list
    employers: list
    explanation: xai.LocalExplanation | None = None
    signal_scores: dict = field(default_factory=dict)
    academic_summary: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ text
    def render(self, width: int = 78) -> str:
        """Format the report the way the dashboard would present it."""
        line = "=" * width
        thin = "-" * width
        top = self.track_matches[0]

        out = [
            line,
            "  CAREER RECOMMENDATION REPORT".ljust(width),
            f"  Student {self.student_id}  |  {cfg.PROGRAM_LABELS[self.program]}  |  4th Year",
            line,
            "",
            "  PREDICTED WORK CATEGORY",
            thin,
            f"    {self.predicted_category.upper()}      "
            f"(model confidence {self.confidence:.1%})",
            f"    {cfg.WORK_CATEGORY_DESCRIPTIONS[self.predicted_category]}",
            "",
            "    Probability across all categories:",
        ]
        for category, probability in sorted(
            self.category_probabilities.items(), key=lambda kv: kv[1], reverse=True
        ):
            bar = "#" * int(round(probability * 34))
            out.append(f"      {category:<12} {probability:6.1%}  {bar}")

        # ------------------------------------------------------ alignment ---
        alignment = self.degree_alignment
        out += [
            "",
            "  DEGREE ALIGNMENT  (SOP 3)",
            thin,
            f"    BS Computer Science      {alignment.bscs_percent:5.1f}%  "
            f"{self._bar(alignment.bscs_percent)}",
            f"    BS Information Tech.     {alignment.bsit_percent:5.1f}%  "
            f"{self._bar(alignment.bsit_percent)}",
            "",
            f"    {self._wrap(alignment.verdict(), width - 6, indent=4)}",
            "",
            f"  HOLLAND RIASEC CODE: {self.riasec_code}",
            thin,
        ]
        for archetype, score in sorted(self.riasec_scores.items(), key=lambda kv: kv[1], reverse=True):
            out.append(f"    {archetype.title():<15} {score:5.1f}  {self._bar(score, 26)}")

        # --------------------------------------------------------- tracks ---
        out += [
            "",
            "  TOP CAREER RECOMMENDATIONS",
            thin,
        ]
        for rank, match in enumerate(self.track_matches, start=1):
            out.append(f"    {rank}. {match.name:<36} {match.match_percent:5.1f}%  "
                       f"{self._bar(match.match_percent, 20)}")
            out.append(f"       {match.track.description}")
            out.append(f"       Work category: {match.work_category}   |   "
                       f"Competency fit: {match.competency_fit:.1f}%")
            drivers = ", ".join(f"{label}" for label, _ in match.top_signals(3))
            out.append(f"       Strongest drivers: {drivers}")
            out.append("")

        # ---------------------------------------------------- explanation ---
        out += ["  WHY WAS THIS RECOMMENDED?  (SHAP explanation)", thin]
        if self.explanation is not None:
            out.append(f"    Base rate for '{self.predicted_category}' "
                       f"= {self.explanation.base_value:+.3f} log-odds; the features below "
                       f"moved it.")
            out.append("")
            for _, row in self.explanation.top_positive(5).iterrows():
                out.append(f"      {row['pretty_name']:<48} {row['shap_value']:+.3f}  "
                           f"{'#' * int(min(abs(row['shap_value']) * 40, 24))}")
            negatives = self.explanation.top_negative(3)
            if len(negatives):
                out.append("")
                out.append("    Pushing AWAY from this category:")
                for _, row in negatives.iterrows():
                    out.append(f"      {row['pretty_name']:<48} {row['shap_value']:+.3f}")
        else:
            out.append("    (SHAP unavailable - install the `shap` package for this section.)")

        # ------------------------------------------------------- guidance ---
        out += ["", f"  SKILL GAPS FOR '{top.name}'", thin]
        if self.skill_gaps:
            for gap in self.skill_gaps[:4]:
                out.append(f"    {gap.label:<44} {gap.current:5.1f}/100  "
                           f"(weight {gap.weight:.2f} in this role)")
        else:
            out.append("    No significant gaps - this profile already meets the target band.")

        out += ["", "  SUGGESTED ACTIONS", thin,
                "    " + "  |  ".join(self.actions)]

        # ------------------------------------------------------ employers ---
        out += ["", "  PALAWAN INTERNSHIP / EMPLOYMENT DESTINATIONS", thin]
        current_sector = None
        for employer in self.employers:
            if employer["sector"] != current_sector:
                current_sector = employer["sector"]
                out.append(f"    [{current_sector}]")
            out.append(f"      - {employer['name']}  ({employer['locality']})")
            if employer["note"]:
                out.append(f"        {employer['note']}")

        out += [
            "",
            line,
            "  Generated by an XGBoost model with SHAP explanations. This is guidance,",
            "  not a placement decision, and does not guarantee hiring outcomes.",
            line,
        ]
        return "\n".join(out)

    @staticmethod
    def _bar(percent: float, width: int = 30) -> str:
        if percent is None or not np.isfinite(percent):
            return "?" * width  # no evidence at all for this score
        filled = int(round(np.clip(percent, 0, 100) / 100 * width))
        return "#" * filled + "." * (width - filled)

    @staticmethod
    def _wrap(text: str, width: int, indent: int = 4) -> str:
        import textwrap
        pad = " " * indent
        return textwrap.fill(text, width=width, initial_indent="", subsequent_indent=pad)

    # ------------------------------------------------------------------ data
    def to_dict(self) -> dict:
        """JSON-serialisable form -- this is what a web API would return."""
        return {
            "student_id": self.student_id,
            "program": self.program,
            "predicted_work_category": self.predicted_category,
            "confidence": round(self.confidence, 4),
            "category_probabilities": {k: round(v, 4) for k, v in self.category_probabilities.items()},
            "degree_alignment": {
                "bscs_percent": round(self.degree_alignment.bscs_percent, 2),
                "bsit_percent": round(self.degree_alignment.bsit_percent, 2),
                "own_program_percent": round(self.degree_alignment.own_program_percent, 2),
                "stronger_program": self.degree_alignment.stronger_program,
                "aligned_with_enrolment": self.degree_alignment.is_aligned_with_enrolment,
                "verdict": self.degree_alignment.verdict(),
            },
            "riasec": {"code": self.riasec_code,
                       "scores": {k: round(v, 2) for k, v in self.riasec_scores.items()}},
            "top_careers": [
                {
                    "rank": i,
                    "name": m.name,
                    "work_category": m.work_category,
                    "match_percent": m.match_percent,
                    "competency_fit": m.competency_fit,
                    "description": m.track.description,
                    "top_signals": [{"signal": s, "contribution": round(v, 2)} for s, v in m.top_signals()],
                }
                for i, m in enumerate(self.track_matches, start=1)
            ],
            "shap_explanation": (
                [
                    {"feature": r["pretty_name"],
                     "contribution": round(float(r["shap_value"]), 4),
                     "value": round(float(r["raw_value"]), 3)}
                    for _, r in self.explanation.contributions.head(8).iterrows()
                ] if self.explanation is not None else []
            ),
            "skill_gaps": [
                {"competency": g.label, "current": g.current, "role_weight": g.weight}
                for g in self.skill_gaps
            ],
            "suggested_actions": self.actions,
            "palawan_destinations": self.employers,
            "signal_scores": {k: round(v, 2) for k, v in self.signal_scores.items()},
            "academic_summary": self.academic_summary,
        }


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
class CareerRecommender:
    """Wraps a trained classifier and produces full student reports."""

    def __init__(self, model, label_encoder, feature_names, feature_set: str = "full"):
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names = list(feature_names)
        self.feature_set = feature_set

    # ------------------------------------------------------------ builders
    @classmethod
    def from_benchmark(cls, result) -> "CareerRecommender":
        """Use the boosting model that a benchmark run just fitted."""
        name = mdl.primary_model_name(result.fitted_models)
        return cls(
            model=result.fitted_models[name],
            label_encoder=result.label_encoder,
            feature_names=result.feature_names,
            feature_set=result.split_summary.get("feature_set", "full"),
        )

    @classmethod
    def load(cls, model_dir=None, tag: str = "") -> "CareerRecommender":
        """Reload a previously saved model from outputs/models/."""
        model_dir = model_dir or cfg.MODEL_DIR
        suffix = f"_{tag}" if tag else ""

        candidates = sorted(model_dir.glob(f"xgboost{suffix}.joblib")) or \
                     sorted(model_dir.glob(f"histgradientboosting{suffix}.joblib"))
        if not candidates:
            raise FileNotFoundError(
                f"No trained boosting model found in {model_dir}. "
                f"Run `python main.py train` first."
            )

        bundle = joblib.load(model_dir / f"label_encoder{suffix}.joblib")
        return cls(
            model=joblib.load(candidates[0]),
            label_encoder=bundle["label_encoder"],
            feature_names=bundle["feature_names"],
        )

    @classmethod
    def fit_new(cls, df: pd.DataFrame, feature_set: str = "full",
                random_state: int = cfg.RANDOM_STATE) -> "CareerRecommender":
        """Train a fresh model on the entire dataset (no hold-out).

        Use this only for producing recommendations after the benchmark has
        already established how well the approach performs. Fitting on all the
        data here is intentional: once the methodology is validated there is no
        reason to throw away 30% of the cohort when serving students.
        """
        from sklearn.preprocessing import LabelEncoder

        X, y_raw, feature_names = feat.make_xy(df, feature_set=feature_set)
        encoder = LabelEncoder().fit(cfg.WORK_CATEGORIES)
        y = encoder.transform(y_raw)

        models = mdl.build_models(len(encoder.classes_), random_state=random_state)
        model = models[mdl.primary_model_name(models)]
        model.fit(X, y)

        return cls(model, encoder, feature_names, feature_set)

    # ---------------------------------------------------------- prediction
    def _prepare(self, student) -> pd.DataFrame:
        """Accept a dict, a Series or a one-row DataFrame; return a raw frame."""
        if isinstance(student, dict):
            frame = pd.DataFrame([student])
        elif isinstance(student, pd.Series):
            frame = student.to_frame().T
        elif isinstance(student, pd.DataFrame):
            frame = student.head(1).copy()
        else:
            raise TypeError(f"Unsupported student type: {type(student)!r}")
        return frame.reset_index(drop=True)

    def recommend(self, student, top_n: int = 3, explain: bool = True) -> StudentReport:
        raw = self._prepare(student)
        student_id = str(raw.get("student_id", pd.Series(["student"])).iloc[0])
        program = str(raw["program"].iloc[0])

        # Engineered view (signals, RIASEC, EVT) + the model's feature matrix.
        engineered = feat.build_features(raw)
        signal_row = engineered.iloc[0]
        X = engineered[self.feature_names].astype(float)

        probabilities = self.model.predict_proba(X)[0]
        classes = list(self.label_encoder.classes_)
        category_probabilities = {c: float(p) for c, p in zip(classes, probabilities)}
        predicted_index = int(np.argmax(probabilities))
        predicted_category = classes[predicted_index]

        # Rank the specific career tracks, model-informed.
        matches = align.score_career_tracks(signal_row, category_probabilities)[:top_n]
        top = matches[0]

        gaps = align.analyse_skill_gaps(signal_row, top.track)
        actions = align.suggested_actions(top.track, gaps)
        employers = align.employers_for_track(top.track)

        explanation = None
        if explain and xai.SHAP_AVAILABLE:
            try:
                explanation = xai.explain_student(self.model, X, classes, student_id=student_id)
            except Exception as exc:  # pragma: no cover - explanation is optional
                print(f"    [warn] SHAP explanation unavailable for {student_id}: {exc}")

        return StudentReport(
            student_id=student_id,
            program=program,
            predicted_category=predicted_category,
            confidence=float(probabilities[predicted_index]),
            category_probabilities=category_probabilities,
            degree_alignment=align.compute_degree_alignment(signal_row, program),
            riasec_code=feat.riasec_code(signal_row),
            riasec_scores={a: float(signal_row[f"riasec_{a}"]) for a in cfg.RIASEC_MAP},
            track_matches=matches,
            skill_gaps=gaps,
            actions=actions,
            employers=employers,
            explanation=explanation,
            signal_scores={
                cfg.SIGNAL_BY_KEY[k].label: float(signal_row[f"sig_{k}"]) for k in cfg.SIGNAL_KEYS
            },
            academic_summary={
                "gwa": float(raw["gwa"].iloc[0]),
                "gwa_points": round(float(signal_row["gwa_points"]), 1),
                "core_grade_mean_points": round(float(signal_row["core_grade_mean"]), 1),
                "certifications": int(signal_row["cert_count"]),
                "seminar_units": (
                    None if pd.isna(raw["seminar_units"].iloc[0])
                    else int(raw["seminar_units"].iloc[0])
                ),
                "evt_motivation": round(float(signal_row["evt_motivation"]), 1),
            },
        )

    # ------------------------------------------------------------- batching
    def recommend_batch(self, df: pd.DataFrame, top_n: int = 3,
                        explain: bool = False) -> pd.DataFrame:
        """Summary row per student -- what OSAS would export for the whole cohort."""
        rows = []
        for _, student in df.iterrows():
            report = self.recommend(student, top_n=top_n, explain=explain)
            row = {
                "student_id": report.student_id,
                "program": report.program,
                "predicted_work_category": report.predicted_category,
                "confidence": round(report.confidence, 4),
                "bscs_alignment_pct": round(report.degree_alignment.bscs_percent, 2),
                "bsit_alignment_pct": round(report.degree_alignment.bsit_percent, 2),
                "own_program_alignment_pct": round(report.degree_alignment.own_program_percent, 2),
                "aligned_with_enrolment": report.degree_alignment.is_aligned_with_enrolment,
                "riasec_code": report.riasec_code,
            }
            for rank, match in enumerate(report.track_matches, start=1):
                row[f"career_{rank}"] = match.name
                row[f"career_{rank}_match_pct"] = match.match_percent
            row["primary_sectors"] = align.sector_summary(report.track_matches[0].track)
            rows.append(row)

        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_report(report: StudentReport, outdir=None) -> dict:
    outdir = outdir or cfg.REPORT_DIR
    slug = report.student_id.replace("-", "_").lower()

    text_path = outdir / f"report_{slug}.txt"
    text_path.write_text(report.render())

    json_path = outdir / f"report_{slug}.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))

    return {"text": text_path, "json": json_path}
