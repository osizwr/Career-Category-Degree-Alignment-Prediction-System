"""
SHAP explainability layer (Objective 4 / SOP 4).

Answers the "why was this recommended?" panel in Figures 4.2-4.4 by attributing
each prediction back to the individual grades, certifications and skills that
produced it.

Two levels of explanation:

  GLOBAL  -- which features drive work-category classification across the whole
             cohort. This is what PSU-CS uses to review the curriculum.
  LOCAL   -- which of THIS student's features pushed them toward THIS category.
             This is what the student sees.

Both are also rendered as complete sentences (narrate_student, narrate_global).
SHAP itself outputs numbers -- one signed contribution per feature per class --
so the narration is the last mile: it reads each contribution together with the
value that produced it and states, in prose, what the model did and by how much.
The wording comes from config.py, so a clause names the questionnaire item the
student actually answered; nothing is interpreted by hand, and changing one
grade changes the sentences.

A note on units: for a multi-class model SHAP values are contributions to the
raw margin (log-odds) of a class, not to a probability. A value of +0.28 means
"this feature pushed the log-odds of the predicted category up by 0.28". They
sum exactly to (margin - base value), which is the additivity property the
thesis cites from Lundberg & Lee (2017).

If the `shap` package is unavailable the module degrades to permutation
importance so the pipeline still runs; the CLI says so loudly when it happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Getting data into the shape the final estimator sees
# --------------------------------------------------------------------------- #
def transform_for_model(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every preprocessing step of a pipeline, skipping the samplers.

    SMOTE only acts during `fit`; at prediction time the data flows straight
    through it. Reproducing that here means SHAP sees exactly the matrix the
    booster was asked to score.
    """
    data = X
    for name, step in pipeline.steps[:-1]:
        if hasattr(step, "fit_resample") and not hasattr(step, "transform"):
            continue  # a sampler: inactive outside fit()
        data = step.transform(data)

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=X.columns, index=X.index)
    return data


def final_estimator(pipeline):
    return pipeline.steps[-1][1]


# --------------------------------------------------------------------------- #
# SHAP value computation
# --------------------------------------------------------------------------- #
@dataclass
class ShapBundle:
    """SHAP output normalised into one predictable shape.

    values : ndarray (n_samples, n_features, n_classes)
    """

    values: np.ndarray
    base_values: np.ndarray
    data: pd.DataFrame
    feature_names: list
    class_names: list
    method: str = "shap"

    def for_class(self, class_index: int) -> np.ndarray:
        return self.values[:, :, class_index]

    def mean_abs_importance(self) -> pd.DataFrame:
        """Mean |SHAP| per feature, per class, plus an overall column."""
        per_class = np.abs(self.values).mean(axis=0)          # (features, classes)
        frame = pd.DataFrame(per_class, index=self.feature_names, columns=self.class_names)
        frame["Overall"] = frame.mean(axis=1)
        frame["Feature"] = [cfg.pretty_feature_name(f) for f in self.feature_names]
        return frame.sort_values("Overall", ascending=False)


def compute_shap(pipeline, X: pd.DataFrame, class_names: list,
                 max_samples: int = 400, random_state: int = cfg.RANDOM_STATE) -> ShapBundle:
    """Run TreeExplainer over (a sample of) X and normalise the output shape."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("The `shap` package is not installed.")

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    transformed = transform_for_model(pipeline, X)
    model = final_estimator(pipeline)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)

    # SHAP returns (n, f) for binary/regression and (n, f, c) for multi-class.
    if values.ndim == 2:
        values = values[:, :, None]
    if base.ndim == 1:
        base = base[:, None]

    return ShapBundle(
        values=values,
        base_values=base,
        data=transformed,
        feature_names=list(X.columns),
        class_names=list(class_names),
    )


def fallback_importance(pipeline, X: pd.DataFrame, y, class_names: list) -> ShapBundle:
    """Permutation importance stand-in when `shap` cannot be imported."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X, y, n_repeats=8, random_state=cfg.RANDOM_STATE, n_jobs=-1
    )
    # Broadcast the single importance vector across classes so the downstream
    # plotting code does not need to special-case it.
    values = np.tile(
        result.importances_mean[None, :, None], (len(X), 1, len(class_names))
    )
    return ShapBundle(
        values=values,
        base_values=np.zeros((len(X), len(class_names))),
        data=transform_for_model(pipeline, X),
        feature_names=list(X.columns),
        class_names=list(class_names),
        method="permutation_importance",
    )


# --------------------------------------------------------------------------- #
# Global plots
# --------------------------------------------------------------------------- #
def plot_global_importance(bundle: ShapBundle, top_n: int = 20, outdir=None):
    """Horizontal bar chart of the most influential features overall."""
    outdir = outdir or cfg.FIGURE_DIR
    importance = bundle.mean_abs_importance().head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(importance) + 2.0))
    bottom = np.zeros(len(importance))
    palette = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6"]

    for i, category in enumerate(bundle.class_names):
        widths = importance[category].values
        ax.barh(importance["Feature"], widths, left=bottom,
                label=category, color=palette[i % len(palette)], edgecolor="white", height=0.72)
        bottom += widths

    ax.set_xlabel("Mean |SHAP value|  (average contribution to the class margin)")
    ax.set_title(f"Global Feature Importance - Top {top_n}\n"
                 "Which academic records, certifications and skills drive work-category prediction")
    ax.legend(title="Work category", frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = outdir / "shap_global_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_beeswarm_per_class(bundle: ShapBundle, outdir=None, top_n: int = 14) -> list:
    """One SHAP beeswarm per work category (the classic summary plot)."""
    if not SHAP_AVAILABLE or bundle.method != "shap":
        return []

    outdir = outdir or cfg.FIGURE_DIR
    pretty = [cfg.pretty_feature_name(f) for f in bundle.feature_names]
    paths = []

    for index, category in enumerate(bundle.class_names):
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            bundle.for_class(index),
            bundle.data,
            feature_names=pretty,
            max_display=top_n,
            show=False,
            plot_size=None,
        )
        plt.title(f"SHAP Summary - '{category}' work category\n"
                  "Red = high feature value, blue = low; right = pushes toward this category",
                  fontsize=11)
        plt.tight_layout()
        path = outdir / f"shap_beeswarm_{category.lower()}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path)

    return paths


def plot_signal_importance(bundle: ShapBundle, outdir=None):
    """Roll feature importance up to the competency-signal level.

    Individual course grades are hard to act on; "Networking & Communications is
    the biggest driver of Technical placement" is something the College of
    Sciences can actually use for curriculum review.
    """
    outdir = outdir or cfg.FIGURE_DIR
    importance = bundle.mean_abs_importance()

    grouped: dict[str, float] = {}
    for feature, row in importance.iterrows():
        label = _feature_group(feature)
        grouped[label] = grouped.get(label, 0.0) + float(row["Overall"])

    series = pd.Series(grouped).sort_values()

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(series) + 1.8))
    bars = ax.barh(series.index, series.values, color="#2563eb", edgecolor="white", height=0.7)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Summed mean |SHAP value|")
    ax.set_title("Feature Importance Grouped by Evidence Type\n"
                 "Aggregated view for curriculum and program review")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = outdir / "shap_grouped_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _feature_group(feature: str) -> str:
    if feature.startswith("acad_") or feature.startswith("grade_"):
        return "Core course grades (by domain)"
    if feature.startswith("tech_"):
        return "Technical skills (self-reported)"
    if feature.startswith("hard_"):
        return "Hard skills (self-reported)"
    if feature.startswith("soft_"):
        return "Soft skills (self-reported)"
    if feature.startswith("cert_type_"):
        return "Credential types (seminar / workshop / exam)"
    if feature.startswith("cert_") or feature == "has_certification":
        return "Professional certifications"
    if feature.startswith("sig_"):
        return "Competency signals (engineered)"
    if feature.startswith("riasec_"):
        return "RIASEC archetypes (engineered)"
    if feature.startswith("evt_"):
        return "Expectancy-Value score (engineered)"
    if feature in ("gwa_points", "core_grade_mean", "core_grade_std"):
        return "Academic averages"
    if feature.endswith("_count"):
        return "Profile breadth counts (engineered)"
    return "Other"


# --------------------------------------------------------------------------- #
# Local (per-student) explanation
# --------------------------------------------------------------------------- #
@dataclass
class LocalExplanation:
    student_id: str
    predicted_category: str
    confidence: float
    contributions: pd.DataFrame   # feature, pretty_name, shap_value, raw_value
    base_value: float

    # Everything below is what turns the numbers into sentences: the full
    # (features x classes) SHAP matrix, the class order it is indexed by, and
    # the feature order its rows follow (which is NOT the order `contributions`
    # ends up in, because that frame is re-sorted by |SHAP|).
    class_names: tuple = ()
    probabilities: tuple = ()
    feature_order: tuple = ()
    class_values: np.ndarray | None = None

    def top_positive(self, n: int = 5) -> pd.DataFrame:
        positive = self.contributions[self.contributions["shap_value"] > 0]
        return positive.head(n)

    def top_negative(self, n: int = 3) -> pd.DataFrame:
        negative = self.contributions[self.contributions["shap_value"] < 0]
        return negative.sort_values("shap_value").head(n)

    # ------------------------------------------------------------ additivity
    def total_contribution(self) -> float:
        return float(self.contributions["shap_value"].sum())

    def margin(self) -> float:
        """base value + every contribution = the raw margin the model produced.

        This is the Lundberg & Lee (2017) additivity property, and the sentence
        generator quotes it so a panelist can check the arithmetic.
        """
        return float(self.base_value + self.total_contribution())

    # ------------------------------------------------------ class comparison
    def ranked_categories(self) -> list:
        return sorted(zip(self.class_names, self.probabilities),
                      key=lambda kv: kv[1], reverse=True)

    def runner_up(self):
        """(category, probability) of the category that came second, or None."""
        ranked = self.ranked_categories()
        return ranked[1] if len(ranked) > 1 else None

    def contrast_with_runner_up(self):
        """The feature that most favours the prediction over the runner-up.

        A straight subtraction of two SHAP columns: how much more this
        student's answer was worth to the winning category than to the one
        that came closest. Returns (pretty_name, delta, runner_up_name).
        """
        runner = self.runner_up()
        if runner is None or self.class_values is None or not len(self.feature_order):
            return None

        names = list(self.class_names)
        if self.predicted_category not in names or runner[0] not in names:
            return None

        values = np.asarray(self.class_values)
        delta = values[:, names.index(self.predicted_category)] \
            - values[:, names.index(runner[0])]
        index = int(np.argmax(delta))
        if not np.isfinite(delta[index]) or delta[index] <= 0:
            return None
        return (cfg.pretty_feature_name(self.feature_order[index]),
                float(delta[index]), runner[0])

    # --------------------------------------------------------------- prose
    def narrative(self, style: str = "human", **kwargs) -> list:
        """The explanation as complete sentences. See narrate_student()."""
        return narrate_student(self, style=style, **kwargs)


def explain_student(pipeline, X_row: pd.DataFrame, class_names: list,
                    student_id: str = "student") -> LocalExplanation:
    """Explain one student's predicted work category, Figure 4.2 style."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("The `shap` package is required for per-student explanations.")

    transformed = transform_for_model(pipeline, X_row)
    model = final_estimator(pipeline)

    probabilities = pipeline.predict_proba(X_row)[0]
    class_index = int(np.argmax(probabilities))

    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 2:
        values = values[:, :, None]
    if base.ndim == 1:
        base = base[:, None]

    shap_row = values[0, :, class_index]
    base_value = float(base[0, class_index])

    contributions = pd.DataFrame({
        "feature": X_row.columns,
        "pretty_name": [cfg.pretty_feature_name(c) for c in X_row.columns],
        "shap_value": shap_row,
        "raw_value": transformed.iloc[0].values,
    }).sort_values("shap_value", ascending=False, key=abs).reset_index(drop=True)
    contributions = contributions.reindex(
        contributions["shap_value"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return LocalExplanation(
        student_id=student_id,
        predicted_category=class_names[class_index],
        confidence=float(probabilities[class_index]),
        contributions=contributions,
        base_value=base_value,
        class_names=tuple(class_names),
        probabilities=tuple(float(p) for p in probabilities),
        feature_order=tuple(X_row.columns),
        class_values=values[0],          # (n_features, n_classes)
    )


def plot_local_explanation(explanation: LocalExplanation, top_n: int = 8, outdir=None):
    """The horizontal +/- bar chart shown in the appendix mockups."""
    outdir = outdir or cfg.FIGURE_DIR
    top = explanation.contributions.head(top_n).iloc[::-1]

    colours = ["#10b981" if v > 0 else "#ef4444" for v in top["shap_value"]]

    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(top) + 2.0))
    bars = ax.barh(top["pretty_name"], top["shap_value"], color=colours,
                   edgecolor="white", height=0.68)
    ax.bar_label(bars, fmt="%+.3f", padding=4, fontsize=9)
    ax.axvline(0, color="#334155", lw=1)

    ax.set_xlabel("SHAP contribution to the predicted category's log-odds")
    ax.set_title(
        f"Why was '{explanation.predicted_category}' recommended for "
        f"{explanation.student_id}?\n"
        f"Model confidence: {explanation.confidence:.1%}   "
        f"(green = pushed toward, red = pushed away)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    margin = max(abs(top["shap_value"]).max(), 0.01) * 0.35
    ax.set_xlim(top["shap_value"].min() - margin, top["shap_value"].max() + margin)
    fig.tight_layout()

    path = outdir / f"shap_local_{explanation.student_id.replace('-', '_').lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_importance_table(bundle: ShapBundle, outdir=None):
    outdir = outdir or cfg.TABLE_DIR
    path = outdir / "shap_feature_importance.csv"
    table = bundle.mean_abs_importance()
    table.index.name = "raw_feature"
    table.to_csv(path)
    return path


# --------------------------------------------------------------------------- #
# From SHAP values to sentences
# --------------------------------------------------------------------------- #
# SHAP gives numbers: one signed contribution per feature per class, summing
# exactly to the model's margin. The functions below do the one step SHAP
# leaves to the application -- putting those numbers into prose -- and they do
# it from the SHAP output and config.py alone.
#
# There is no hand-written interpretation in here. Every clause is read off a
# contribution and the value that produced it, so changing one grade changes
# the sentences; nothing is asserted that the model did not do.

_GROUP_PHRASES = {
    "Core course grades (by domain)": "core course grades",
    "Technical skills (self-reported)": "self-reported technical skills",
    "Hard skills (self-reported)": "self-reported hard skills",
    "Soft skills (self-reported)": "self-reported soft skills",
    "Credential types (seminar / workshop / exam)": "the kind of credential reported",
    "Professional certifications": "professional certifications",
    "Competency signals (engineered)": "the engineered competency signals",
    "RIASEC archetypes (engineered)": "the Holland RIASEC scores",
    "Expectancy-Value score (engineered)": "the Expectancy-Value motivation score",
    "Academic averages": "GWA and overall grade averages",
    "Profile breadth counts (engineered)": "how many skills and credentials were reported",
    "Other": "other features",
}


def _group_phrase(feature: str) -> str:
    return _GROUP_PHRASES.get(_feature_group(feature), "other features")


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


def join_clauses(items, conjunction: str = "and") -> str:
    """'a', 'a and b', 'a, b and c' -- so the sentences read like sentences."""
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _points_clause(points: float) -> str:
    grade = float(cfg.points_to_grade(points))
    return f"{points:.1f} grade points, about {grade:.2f} on the 1.00-5.00 scale"


def describe_value(feature: str, value) -> str:
    """One clause saying what this student's value for `feature` actually is.

    SHAP supplies the contribution; this supplies the noun phrase, read off
    config.py so the wording matches the questionnaire the student answered.
    A checkbox becomes "you ticked it", a domain mean becomes the grade it
    corresponds to on the PSU scale, and so on.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "this answer could not be read"
    if not np.isfinite(value):
        return "you left this blank, so the cohort median stood in for it"

    if feature in cfg.SKILL_BY_COLUMN:
        return "you ticked it" if value >= 0.5 else "you left it unticked"
    if feature in cfg.COURSE_BY_COLUMN:
        return f"you reported a grade of {value:.2f}"
    if feature.startswith("acad_"):
        return f"you averaged {_points_clause(value)}"
    if feature == "core_grade_mean":
        return f"across every domain you averaged {_points_clause(value)}"
    if feature == "gwa_points":
        return f"your GWA works out to {_points_clause(value)}"
    if feature == "core_grade_std":
        return (f"your domain averages spread {value:.1f} points around that mean"
                if value > 0.05 else "your domain averages barely vary")
    if feature == "has_certification":
        return ("you answered Yes to holding a certification, seminar or training"
                if value >= 0.5 else
                "you answered No to holding any certification, seminar or training")
    if feature.startswith("cert_type_"):
        return "you reported one" if value >= 0.5 else "you reported none of this kind"
    if feature == "cert_count":
        n = int(round(value))
        return f"{n} subject {_plural(n, 'domain')} recognised from the credentials you named"
    if feature == "cert_type_count":
        n = int(round(value))
        return f"{n} credential {_plural(n, 'type')} reported"
    if feature.startswith("cert_"):
        return "you named one" if value >= 0.5 else "you named none in this subject"
    if feature == "program_code":
        return "you are enrolled in BSIT" if value >= 0.5 else "you are enrolled in BSCS"
    if feature.endswith("_skill_count"):
        total = len(cfg.SKILL_GROUPS.get(feature.split("_")[0], ()))
        return f"you ticked {int(round(value))} of the {total}"
    if feature.startswith(("sig_", "riasec_", "evt_")):
        return f"you score {value:.1f} out of 100"
    return f"your value is {value:.2f}"


def describe_value_human(feature: str, value) -> str:
    """A natural, counselor-style description of what the student reported for `feature`."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unrecorded response"
    if not np.isfinite(value):
        return "no answer provided (cohort average applied)"

    if feature in cfg.SKILL_BY_COLUMN:
        return "practical hands-on experience verified" if value >= 0.5 else "not reported as an active skill"
    if feature in cfg.COURSE_BY_COLUMN:
        return f"course grade of {value:.2f}"
    if feature.startswith("acad_"):
        return f"academic average of {_points_clause(value)}"
    if feature == "core_grade_mean":
        return f"overall coursework average of {_points_clause(value)}"
    if feature == "gwa_points":
        return f"cumulative GWA equivalent to {_points_clause(value)}"
    if feature == "core_grade_std":
        return (f"coursework performance varied by {value:.1f} points across domains"
                if value > 0.05 else "very uniform grades across all subjects")
    if feature == "has_certification":
        return ("holds external industry certifications or formal training"
                if value >= 0.5 else
                "no external certifications reported")
    if feature.startswith("cert_type_"):
        return "credential on record" if value >= 0.5 else "none reported"
    if feature == "cert_count":
        n = int(round(value))
        return f"{n} certified subject {_plural(n, 'domain')}"
    if feature == "cert_type_count":
        n = int(round(value))
        return f"{n} credential {_plural(n, 'type')} completed"
    if feature.startswith("cert_"):
        return "certified in this specialization" if value >= 0.5 else "not certified in this subject"
    if feature == "program_code":
        return "enrolled in BS Information Technology" if value >= 0.5 else "enrolled in BS Computer Science"
    if feature.endswith("_skill_count"):
        total = len(cfg.SKILL_GROUPS.get(feature.split("_")[0], ()))
        return f"{int(round(value))} of {total} skills applied"
    if feature.startswith("riasec_"):
        return f"interest score of {value:.1f}/100"
    if feature.startswith("sig_"):
        return f"competency score of {value:.1f}/100"
    if feature.startswith("evt_"):
        return f"motivation score of {value:.1f}/100"
    return f"score of {value:.2f}"


_POSITIVE_LEADS = (
    "The strongest single reason is",
    "The second-largest contribution is",
    "Third comes",
    "Fourth comes",
    "Fifth comes",
)


def narrate_student(explanation: LocalExplanation, top_n: int = 5,
                    negative_n: int = 3, style: str = "human") -> list:
    """Explain one prediction in complete sentences, generated from SHAP.

    Parameters
    ----------
    explanation : LocalExplanation
        The per-student SHAP explanation object.
    top_n : int
        Number of positive features to narrate in detail.
    negative_n : int
        Number of opposing features to cite.
    style : str
        'human' (default) for an intuitive, counselor-style narrative;
        'technical' for the log-odds Lundberg & Lee (2017) additivity breakdown.
    """
    if style == "technical":
        return _narrate_student_technical(explanation, top_n=top_n, negative_n=negative_n)
    return _narrate_student_human(explanation, top_n=top_n, negative_n=negative_n)


def _narrate_student_human(explanation: LocalExplanation, top_n: int = 5,
                           negative_n: int = 3) -> list:
    """Intuitive, advisor-grade narrative explaining the career recommendation."""
    category = explanation.predicted_category.upper()
    contributions = explanation.contributions
    sentences: list[str] = []

    if contributions.empty:
        return [f"The system recommends {category} with {explanation.confidence:.1%} "
                f"confidence based on your general profile."]

    positive = contributions[contributions["shap_value"] > 0] \
        .sort_values("shap_value", ascending=False)
    negative = contributions[contributions["shap_value"] < 0] \
        .sort_values("shap_value")
    positive_total = float(positive["shap_value"].sum())
    negative_total = float(negative["shap_value"].sum())

    # ---- 1. Advisor Overview ---------------------------------------------- #
    opening = (
        f"Based on your profile, the system recommends {category} as your primary "
        f"career match with {explanation.confidence:.1%} confidence"
    )
    runner_up = explanation.runner_up()
    if runner_up:
        opening += f", notably ahead of {runner_up[0].upper()} ({runner_up[1]:.1%})."
    else:
        opening += "."

    opening += (
        f" For {explanation.student_id}, this recommendation is substantiated by "
        f"{len(positive)} distinct profile strengths that heavily outweigh "
        f"{len(negative)} counter-indicators."
    )
    sentences.append(opening)

    # ---- 2. Primary Positive Drivers -------------------------------------- #
    denominator = positive_total if positive_total > 1e-9 else float("nan")
    human_leads = (
        "The primary driver is",
        "Second is",
        "Third is",
        "Fourth is",
        "Fifth is",
    )
    for rank, (_, row) in enumerate(positive.head(top_n).iterrows()):
        lead = human_leads[rank] if rank < len(human_leads) else f"Rank {rank + 1} factor is"
        share = float(row["shap_value"]) / denominator
        tail = f" -- accounting for {share:.0%} of all positive evidence" if np.isfinite(share) else ""
        desc = describe_value_human(row["feature"], row["raw_value"])
        sentences.append(
            f"{lead} {row['pretty_name']} ({desc}){tail}."
        )

    remainder = positive.iloc[top_n:]
    if len(remainder):
        sentences.append(
            f"Beyond these primary drivers, an additional {len(remainder)} supporting indicators "
            f"consistently reinforce this trajectory across your coursework, interests, and competencies."
        )

    # ---- 3. Opposing / Counterbalancing Factors --------------------------- #
    if len(negative):
        clauses = [
            f"{row['pretty_name']} ({describe_value_human(row['feature'], row['raw_value'])})"
            for _, row in negative.head(negative_n).iterrows()
        ]
        sentences.append(
            f"Pulling slightly in the other direction, factors such as {join_clauses(clauses)} "
            f"leaned toward alternative paths. However, these counter-influences were far too small "
            f"to offset the overwhelming evidence in favor of {category}."
        )

    # ---- 4. Overall Evidence Distribution --------------------------------- #
    grouped: dict[str, float] = {}
    for _, row in contributions.iterrows():
        phrase = _group_phrase(row["feature"])
        grouped[phrase] = grouped.get(phrase, 0.0) + abs(float(row["shap_value"]))
    total_absolute = sum(grouped.values())
    if total_absolute > 1e-9:
        ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:3]
        clauses = [f"{value / total_absolute:.0%} from {phrase}" for phrase, value in ranked]
        sentences.append(
            f"Looking at your overall profile, the recommendation is shaped most strongly by "
            f"{join_clauses(clauses)}."
        )

    # ---- 5. Distinction from Runner-Up ------------------------------------ #
    contrast = explanation.contrast_with_runner_up()
    if contrast is not None:
        pretty, delta, other = contrast
        sentences.append(
            f"Between {category} and your runner-up category ({other.upper()}), the most decisive "
            f"differentiator is {pretty}: your background here provided the key advantage tipping the "
            f"balance toward {category}."
        )

    # ---- 6. Transparency & Fidelity Assurance ----------------------------- #
    sentences.append(
        f"This explanation is derived mathematically using SHAP (Shapley Additive Explanations) across "
        f"all {len(contributions)} features evaluated by the model, ensuring transparent and personalized "
        f"guidance rooted strictly in your responses."
    )
    return sentences


def _narrate_student_technical(explanation: LocalExplanation, top_n: int = 5,
                               negative_n: int = 3) -> list:
    """The original mathematical Lundberg & Lee (2017) log-odds additivity breakdown."""
    category = explanation.predicted_category.upper()
    contributions = explanation.contributions
    sentences: list[str] = []

    if contributions.empty:
        return [f"The model chose {category} with {explanation.confidence:.1%} "
                f"confidence, but SHAP returned no feature contributions to explain it."]

    positive = contributions[contributions["shap_value"] > 0] \
        .sort_values("shap_value", ascending=False)
    negative = contributions[contributions["shap_value"] < 0] \
        .sort_values("shap_value")
    positive_total = float(positive["shap_value"].sum())
    negative_total = float(negative["shap_value"].sum())

    # ---- 1. how the margin was assembled ---------------------------------- #
    opening = (
        f"SHAP starts from the cohort base rate for {category} "
        f"({explanation.base_value:+.3f} log-odds) and then adds one contribution "
        f"for every feature: for {explanation.student_id} the {len(positive)} "
        f"supporting {_plural(len(positive), 'feature')} are worth "
        f"{positive_total:+.3f} and the {len(negative)} opposing "
        f"{_plural(len(negative), 'one')} {negative_total:+.3f}, a net "
        f"{explanation.total_contribution():+.3f} that leaves {category} first at "
        f"{explanation.confidence:.1%} confidence"
    )
    runner_up = explanation.runner_up()
    opening += (f", ahead of {runner_up[0].upper()} at {runner_up[1]:.1%}."
                if runner_up else ".")
    sentences.append(opening)

    # ---- 2. the ranked reasons in favour ---------------------------------- #
    denominator = positive_total if positive_total > 1e-9 else float("nan")
    for rank, (_, row) in enumerate(positive.head(top_n).iterrows()):
        lead = _POSITIVE_LEADS[rank] if rank < len(_POSITIVE_LEADS) else "Then comes"
        share = float(row["shap_value"]) / denominator
        tail = (f" -- {share:.0%} of all the evidence in its favour"
                if np.isfinite(share) else "")
        sentences.append(
            f"{lead} {row['pretty_name']}: "
            f"{describe_value(row['feature'], row['raw_value'])}, which raised the "
            f"log-odds of {category} by {float(row['shap_value']):.3f}{tail}."
        )

    remainder = positive.iloc[top_n:]
    if len(remainder):
        sentences.append(
            f"The other {len(remainder)} supporting "
            f"{_plural(len(remainder), 'feature')} add "
            f"{float(remainder['shap_value'].sum()):+.3f} between them, so none of "
            f"them changes the outcome on its own."
        )

    # ---- 3. what worked against it ---------------------------------------- #
    if len(negative):
        clauses = [
            f"{row['pretty_name']} ({describe_value(row['feature'], row['raw_value'])}) "
            f"cost {abs(float(row['shap_value'])):.3f}"
            for _, row in negative.head(negative_n).iterrows()
        ]
        verdict = ("not enough to displace"
                   if positive_total + negative_total > 0 else "set against")
        sentences.append(
            f"Pulling the other way, {join_clauses(clauses)} log-odds; the "
            f"{len(negative)} opposing {_plural(len(negative), 'feature')} remove "
            f"{abs(negative_total):.3f} in total, {verdict} the "
            f"{positive_total:.3f} in favour."
        )

    # ---- 4. which kind of evidence dominated ------------------------------ #
    grouped: dict[str, float] = {}
    for _, row in contributions.iterrows():
        phrase = _group_phrase(row["feature"])
        grouped[phrase] = grouped.get(phrase, 0.0) + abs(float(row["shap_value"]))
    total_absolute = sum(grouped.values())
    if total_absolute > 1e-9:
        ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:3]
        # Only the first clause carries the verb, so the list reads as one
        # sentence instead of three stacked copies of the same phrase.
        clauses = [f"{value / total_absolute:.0%} of the movement comes from {phrase}"
                   if position == 0 else f"{value / total_absolute:.0%} from {phrase}"
                   for position, (phrase, value) in enumerate(ranked)]
        sentences.append(f"Grouped by kind of evidence, {join_clauses(clauses)}.")

    # ---- 5. what separated it from the runner-up -------------------------- #
    contrast = explanation.contrast_with_runner_up()
    if contrast is not None:
        pretty, delta, other = contrast
        sentences.append(
            f"What most separates {category} from {other.upper()} is {pretty}: the "
            f"same answer is worth {delta:.3f} more log-odds to {category} than to "
            f"{other.upper()}, which is where the gap between the two categories "
            f"mainly comes from."
        )

    # ---- 6. the additivity check ------------------------------------------ #
    sentences.append(
        f"Every figure above is a SHAP value in log-odds of {category}: all "
        f"{len(contributions)} of them sum to {explanation.total_contribution():+.3f}, "
        f"which added to the base rate reproduces the model's raw margin of "
        f"{explanation.margin():+.3f} exactly, so the explanation accounts for the "
        f"whole prediction and nothing has been added by hand."
    )
    return sentences


def narrate_global(bundle: ShapBundle, top_n: int = 5) -> list:
    """Explain the cohort-level SHAP output in complete sentences.

    This is the paragraph the curriculum-review section of Chapter IV needs:
    which evidence the model leans on across all students, per work category,
    with the direction of each effect taken from the SHAP values themselves.
    """
    importance = bundle.mean_abs_importance()
    sentences: list[str] = []

    # The same sentences have to stay true when `shap` is missing and the
    # pipeline falls back to permutation importance, so the metric is named
    # from the bundle rather than assumed.
    is_shap = bundle.method == "shap"
    metric = "mean |SHAP|" if is_shap else "importance"

    if not is_shap:
        sentences.append(
            f"The `shap` package was unavailable, so these sentences describe "
            f"{bundle.method.replace('_', ' ')} instead: it ranks features by how "
            f"much accuracy drops when each is shuffled, which gives magnitude but "
            f"neither a sign nor a per-student breakdown."
        )

    sentences.append(
        f"{'SHAP' if is_shap else 'The fallback'} scored {len(bundle.data)} students "
        f"across {len(bundle.feature_names)} features and {len(bundle.class_names)} "
        f"work categories; the figures below are "
        + ("mean absolute SHAP values, that is, the average size of a feature's "
           "contribution to a category's log-odds."
           if is_shap else
           "permutation-importance scores, that is, the average drop in accuracy "
           "when one feature is shuffled.")
    )

    leaders = importance.head(top_n)
    sentences.append(
        "Across the whole cohort the strongest predictor is "
        f"{leaders.iloc[0]['Feature']} ({metric} {leaders.iloc[0]['Overall']:.3f}), "
        "followed by "
        + join_clauses([f"{row['Feature']} ({row['Overall']:.3f})"
                        for _, row in leaders.iloc[1:].iterrows()])
        + "."
    )

    # Direction per category, straight from the sign of the value/SHAP relation.
    order = list(bundle.feature_names)
    for index, category in enumerate(bundle.class_names):
        column = importance[category].sort_values(ascending=False)
        feature = column.index[0]
        values = bundle.for_class(index)[:, order.index(feature)]
        data = np.asarray(bundle.data[feature], dtype=float)

        if not is_shap:
            direction = ("permutation importance cannot say which way it pushes -- "
                         "install `shap` for that")
        elif np.std(data) < 1e-12 or np.std(values) < 1e-12:
            direction = "its effect is flat in this sample"
        else:
            correlation = float(np.corrcoef(data, values)[0, 1])
            if correlation > 0.2:
                direction = "higher values push students toward it"
            elif correlation < -0.2:
                direction = "lower values push students toward it"
            else:
                direction = "its effect is not monotonic -- the beeswarm plot shows why"
        sentences.append(
            f"For the {category.upper()} category the biggest driver is "
            f"{cfg.pretty_feature_name(feature)} ({metric} {column.iloc[0]:.3f}), and "
            f"{direction}."
        )

    grouped: dict[str, float] = {}
    for feature, row in importance.iterrows():
        phrase = _group_phrase(feature)
        grouped[phrase] = grouped.get(phrase, 0.0) + float(row["Overall"])
    total = sum(grouped.values())
    if total > 1e-9:
        ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:4]
        clauses = [f"{phrase} carry {value / total:.0%} of the total influence"
                   if position == 0 else f"{phrase} {value / total:.0%}"
                   for position, (phrase, value) in enumerate(ranked)]
        sentences.append(
            f"Grouped by kind of evidence, {join_clauses(clauses)} -- the split "
            f"the College of Sciences would review the curriculum against."
        )
    return sentences


def format_sentences(sentences, width: int = 78, indent: int = 4,
                     bullet: str = "") -> str:
    """Wrap generated sentences for a fixed-width console or text report."""
    import textwrap

    pad = " " * indent
    blocks = []
    for sentence in sentences:
        first = pad + bullet
        blocks.append(textwrap.fill(
            sentence,
            width=width,
            initial_indent=first,
            subsequent_indent=pad + " " * len(bullet),
        ))
    return "\n\n".join(blocks)


def save_global_narrative(bundle: ShapBundle, outdir=None):
    """Write the cohort-level explanation out as prose, ready to quote."""
    outdir = outdir or cfg.REPORT_DIR
    path = outdir / "shap_global_explanation.txt"
    header = [
        "SHAP GLOBAL EXPLANATION",
        "=" * 70,
        "Generated from the SHAP values computed by career_system.explain.",
        "Mean |SHAP| is the average contribution a feature makes to a work",
        "category's log-odds; it carries no sign, so direction is reported",
        "separately from the relationship between each value and its SHAP value.",
        "",
    ]
    body = format_sentences(narrate_global(bundle), width=70, indent=0)
    path.write_text("\n".join(header) + "\n" + body + "\n")
    return path
