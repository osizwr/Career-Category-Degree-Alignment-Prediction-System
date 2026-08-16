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
    if feature.startswith("grade_"):
        return "Core course grades"
    if feature.startswith("tech_"):
        return "Technical skills (self-reported)"
    if feature.startswith("hard_"):
        return "Hard skills (self-reported)"
    if feature.startswith("soft_"):
        return "Soft skills (self-reported)"
    if feature.startswith("cert_"):
        return "Professional certifications"
    if feature.startswith("sig_"):
        return "Competency signals (engineered)"
    if feature.startswith("riasec_"):
        return "RIASEC archetypes (engineered)"
    if feature.startswith("evt_"):
        return "Expectancy-Value score (engineered)"
    if feature in ("gwa_points", "core_grade_mean", "core_grade_std"):
        return "Academic averages"
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

    def top_positive(self, n: int = 5) -> pd.DataFrame:
        positive = self.contributions[self.contributions["shap_value"] > 0]
        return positive.head(n)

    def top_negative(self, n: int = 3) -> pd.DataFrame:
        negative = self.contributions[self.contributions["shap_value"] < 0]
        return negative.sort_values("shap_value").head(n)


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
