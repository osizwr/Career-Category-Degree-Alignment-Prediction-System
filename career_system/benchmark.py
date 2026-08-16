"""
Side-by-side benchmark of XGBoost vs Logistic Regression vs ZeroR.

Implements the evaluation plan in Chapter IV of the thesis:

  * 70-30 stratified train/test split
  * Stratified 5-Fold cross-validation on the TRAINING portion only
  * Accuracy, Precision, Recall and F1 (macro and weighted)
  * A confusion matrix per model
  * Percentage improvement of each model over the ZeroR null accuracy floor
  * A McNemar test, so "the model is significantly better" is a measured claim
    rather than an assertion

Everything it produces is written to outputs/ as CSV, TXT and PNG so the tables
and figures can be dropped straight into Chapter V.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config as cfg
from . import features as feat
from . import models as mdl


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkResult:
    """Everything one benchmark run produced, ready to be saved or printed."""

    holdout_metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    confusion_matrices: dict
    reports: dict
    fitted_models: dict = field(repr=False, default_factory=dict)
    label_encoder: LabelEncoder = field(repr=False, default=None)
    feature_names: list = field(repr=False, default_factory=list)
    predictions: dict = field(repr=False, default_factory=dict)
    significance: dict = field(default_factory=dict)
    split_summary: dict = field(default_factory=dict)

    @property
    def class_names(self):
        return list(self.label_encoder.classes_)


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _score_row(name: str, y_true, y_pred) -> dict:
    """All four thesis metrics for one model, macro and weighted."""
    with warnings.catch_warnings():
        # A model may legitimately never predict a minority class; zero_division
        # already handles it and the warning only adds noise to the console.
        warnings.simplefilter("ignore")
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }


SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
}


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on the two models' disagreements.

    Counts only the cases where exactly one model is right. Under the null
    hypothesis ("the two models are equally accurate") those wins split 50/50,
    so an exact binomial test on the discordant pairs gives the p-value.
    """
    a_right = np.asarray(pred_a) == np.asarray(y_true)
    b_right = np.asarray(pred_b) == np.asarray(y_true)

    only_a = int(np.sum(a_right & ~b_right))
    only_b = int(np.sum(~a_right & b_right))
    discordant = only_a + only_b

    if discordant == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0,
                "significant": False,
                "note": "The two models made identical predictions on every test case."}

    p_value = binomtest(only_a, discordant, 0.5).pvalue
    return {
        "only_a": only_a,
        "only_b": only_b,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "note": (f"{only_a} cases only model A got right vs {only_b} only model B "
                 f"got right (n discordant = {discordant})."),
    }


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def run_benchmark(
    df: pd.DataFrame,
    feature_set: str = "full",
    test_size: float = cfg.TEST_SIZE,
    cv_folds: int = cfg.CV_FOLDS,
    random_state: int = cfg.RANDOM_STATE,
    use_smote: bool = True,
    verbose: bool = True,
) -> BenchmarkResult:
    """Train and evaluate all three classifiers on one dataset."""
    X, y_raw, feature_names = feat.make_xy(df, feature_set=feature_set, require_target=True)

    # XGBoost needs integer targets; encoding once keeps all three models
    # working on identical inputs so the comparison stays fair.
    encoder = LabelEncoder().fit(cfg.WORK_CATEGORIES)
    y = encoder.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"  Feature set        : {feature_set}  ({len(feature_names)} features)")
        print(f"  Training / Testing : {len(X_train)} / {len(X_test)}  "
              f"({100 * (1 - test_size):.0f}-{100 * test_size:.0f} stratified split)")
        print(f"  SMOTE              : {'enabled (inside CV pipeline)' if use_smote else 'disabled'}")
        print(f"  Cross-validation   : Stratified {cv_folds}-Fold on the training set\n")

    models = mdl.build_models(
        n_classes=len(encoder.classes_), random_state=random_state, use_smote=use_smote
    )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    holdout_rows, cv_rows = [], []
    confusion_matrices, reports, fitted, predictions = {}, {}, {}, {}

    for name, pipeline in models.items():
        if verbose:
            print(f"  -> {name}")

        # ---- Stratified K-Fold cross-validation (training set only) --------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate(
                pipeline, X_train, y_train,
                cv=skf, scoring=SCORING, n_jobs=1, error_score="raise",
            )
        cv_rows.append({
            "Model": name,
            "CV Accuracy (mean)": cv["test_accuracy"].mean(),
            "CV Accuracy (std)": cv["test_accuracy"].std(),
            "CV Precision macro": cv["test_precision_macro"].mean(),
            "CV Recall macro": cv["test_recall_macro"].mean(),
            "CV F1 macro (mean)": cv["test_f1_macro"].mean(),
            "CV F1 macro (std)": cv["test_f1_macro"].std(),
            "CV F1 weighted": cv["test_f1_weighted"].mean(),
            "Fit time (s)": cv["fit_time"].mean(),
            "_fold_accuracies": list(cv["test_accuracy"]),
            "_fold_f1": list(cv["test_f1_macro"]),
        })

        # ---- Final fit on the full training set, evaluated on the hold-out --
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        holdout_rows.append(_score_row(name, y_test, y_pred))
        confusion_matrices[name] = confusion_matrix(y_test, y_pred, labels=range(len(encoder.classes_)))
        reports[name] = classification_report(
            y_test, y_pred,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
            digits=4,
        )
        fitted[name] = pipeline
        predictions[name] = y_pred

    holdout = pd.DataFrame(holdout_rows).set_index("Model")
    cv_table = pd.DataFrame(cv_rows).set_index("Model")

    # ---- Improvement over the ZeroR null accuracy floor ---------------------
    zeror_accuracy = holdout.loc["ZeroR", "Accuracy"]
    holdout["Gain over ZeroR (pp)"] = (holdout["Accuracy"] - zeror_accuracy) * 100
    holdout["Relative Improvement"] = (holdout["Accuracy"] / zeror_accuracy) - 1.0

    # ---- Statistical significance ------------------------------------------
    boosting = mdl.primary_model_name(models)
    significance = {
        f"{boosting} vs ZeroR": mcnemar_test(y_test, predictions[boosting], predictions["ZeroR"]),
        f"{boosting} vs Logistic Regression": mcnemar_test(
            y_test, predictions[boosting], predictions["Logistic Regression"]
        ),
        "Logistic Regression vs ZeroR": mcnemar_test(
            y_test, predictions["Logistic Regression"], predictions["ZeroR"]
        ),
    }

    split_summary = {
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": test_size,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "feature_set": feature_set,
        "n_features": len(feature_names),
        "smote": use_smote,
        "zeror_floor": float(zeror_accuracy),
        "train_class_counts": {
            encoder.classes_[i]: int(c) for i, c in enumerate(np.bincount(y_train, minlength=len(encoder.classes_)))
        },
        "test_class_counts": {
            encoder.classes_[i]: int(c) for i, c in enumerate(np.bincount(y_test, minlength=len(encoder.classes_)))
        },
    }

    return BenchmarkResult(
        holdout_metrics=holdout,
        cv_metrics=cv_table,
        confusion_matrices=confusion_matrices,
        reports=reports,
        fitted_models=fitted,
        label_encoder=encoder,
        feature_names=feature_names,
        predictions={**predictions, "_y_test": y_test},
        significance=significance,
        split_summary=split_summary,
    )


# --------------------------------------------------------------------------- #
# Console output
# --------------------------------------------------------------------------- #
def print_results(result: BenchmarkResult) -> None:
    metric_cols = [
        "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
        "F1 (weighted)", "Gain over ZeroR (pp)",
    ]

    print("\n" + "=" * 78)
    print("HOLD-OUT TEST RESULTS  (30% never seen during training)")
    print("=" * 78)
    display = result.holdout_metrics[metric_cols].copy()
    for col in metric_cols[:-1]:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    display["Gain over ZeroR (pp)"] = result.holdout_metrics["Gain over ZeroR (pp)"].map(
        lambda v: f"{v:+.2f}"
    )
    print(display.to_string())

    print("\n" + "=" * 78)
    print(f"STRATIFIED {result.split_summary['cv_folds']}-FOLD CROSS-VALIDATION  (training set only)")
    print("=" * 78)
    cv_display = result.cv_metrics[[
        "CV Accuracy (mean)", "CV Accuracy (std)", "CV F1 macro (mean)", "CV F1 macro (std)"
    ]].copy()
    print(cv_display.map(lambda v: f"{v:.4f}").to_string())

    print("\n" + "=" * 78)
    print("STATISTICAL SIGNIFICANCE  (exact McNemar test on the hold-out set)")
    print("=" * 78)
    for comparison, stats in result.significance.items():
        verdict = "SIGNIFICANT" if stats["significant"] else "not significant"
        print(f"  {comparison}")
        print(f"      p = {stats['p_value']:.5f}  ->  {verdict} at alpha = 0.05")
        print(f"      {stats['note']}")

    print("\n" + "=" * 78)
    print("PER-CLASS CLASSIFICATION REPORTS")
    print("=" * 78)
    for name, report in result.reports.items():
        print(f"\n--- {name} ---")
        print(report)

    print("=" * 78)
    print("CONFUSION MATRICES  (rows = actual, columns = predicted)")
    print("=" * 78)
    for name, matrix in result.confusion_matrices.items():
        frame = pd.DataFrame(matrix, index=result.class_names, columns=result.class_names)
        print(f"\n--- {name} ---")
        print(frame.to_string())

    _print_verdict(result)


def _print_verdict(result: BenchmarkResult) -> None:
    """Plain-language summary the researchers can paraphrase in Chapter V."""
    boosting = mdl.primary_model_name(result.fitted_models)
    acc = result.holdout_metrics["Accuracy"]
    floor = result.split_summary["zeror_floor"]

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(f"  ZeroR always answers the majority class and scores {floor:.2%}. "
          f"Any model at or below\n  this number has learned nothing.")
    print(f"  {boosting:<22} {acc[boosting]:.2%}  "
          f"({(acc[boosting] - floor) * 100:+.2f} pp over the floor)")
    print(f"  {'Logistic Regression':<22} {acc['Logistic Regression']:.2%}  "
          f"({(acc['Logistic Regression'] - floor) * 100:+.2f} pp over the floor)")

    gap = acc[boosting] - acc["Logistic Regression"]
    sig = result.significance[f"{boosting} vs Logistic Regression"]
    if gap > 0 and sig["significant"]:
        print(f"\n  {boosting} beats the linear baseline by {gap * 100:.2f} pp and the "
              f"difference is\n  statistically significant (p = {sig['p_value']:.4f}). This is the "
              f"evidence that the\n  relationships in the student profiles are non-linear.")
    elif gap > 0:
        print(f"\n  {boosting} leads by {gap * 100:.2f} pp, but McNemar returns "
              f"p = {sig['p_value']:.4f}, so on this\n  test-set size the gap is NOT statistically "
              f"significant. Report it as such -- a larger\n  cohort is needed before claiming "
              f"boosting is definitively superior.")
    else:
        print(f"\n  The linear baseline matched or beat {boosting} here "
              f"({gap * 100:+.2f} pp). Worth reporting\n  honestly: it suggests the signal in this "
              f"dataset is mostly linear.")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
_PALETTE = {"ZeroR": "#94a3b8", "Logistic Regression": "#f59e0b", "_boost": "#2563eb"}


def _model_colour(name: str) -> str:
    return _PALETTE.get(name, _PALETTE["_boost"])


def plot_confusion_matrices(result: BenchmarkResult, outdir=None) -> list:
    outdir = outdir or cfg.FIGURE_DIR
    paths = []
    classes = result.class_names

    for name, matrix in result.confusion_matrices.items():
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        # Row-normalised shading so the minority classes stay readable.
        row_sums = matrix.sum(axis=1, keepdims=True)
        shaded = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

        im = ax.imshow(shaded, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted work category")
        ax.set_ylabel("Actual work category")
        ax.set_title(f"Confusion Matrix - {name}\n(hold-out test set, n={matrix.sum()})")

        for i in range(len(classes)):
            for j in range(len(classes)):
                pct = shaded[i, j] * 100
                ax.text(j, i, f"{matrix[i, j]}\n{pct:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if shaded[i, j] > 0.55 else "#1e293b")

        fig.colorbar(im, ax=ax, label="Share of actual class")
        fig.tight_layout()
        path = outdir / f"confusion_matrix_{name.split('(')[0].strip().lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_model_comparison(result: BenchmarkResult, outdir=None):
    outdir = outdir or cfg.FIGURE_DIR
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    data = result.holdout_metrics[metrics]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    names = list(data.index)
    x = np.arange(len(metrics))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, data.loc[name].values, width,
                      label=name, color=_model_colour(name), edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    floor = result.split_summary["zeror_floor"]
    ax.axhline(floor, ls="--", lw=1.2, color="#ef4444", zorder=0)
    ax.text(len(metrics) - 0.45, floor + 0.012,
            f"ZeroR null accuracy floor = {floor:.1%}",
            color="#ef4444", fontsize=8, ha="right")

    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Side-by-Side Benchmark on the Hold-Out Test Set\n"
                 "Work Category Classification, BSCS & BSIT fourth-year students")
    ax.legend(frameon=False, ncols=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = outdir / "model_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cv_distribution(result: BenchmarkResult, outdir=None):
    """Box plot of the per-fold accuracies -- shows stability, not just the mean."""
    outdir = outdir or cfg.FIGURE_DIR
    names = list(result.cv_metrics.index)
    folds = [result.cv_metrics.loc[n, "_fold_accuracies"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bp = ax.boxplot(folds, tick_labels=names, patch_artist=True, widths=0.55)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(_model_colour(name))
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#0f172a")

    for i, values in enumerate(folds, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.035, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=18,
                   color="#0f172a", alpha=0.65, zorder=3)

    ax.set_ylabel("Fold accuracy")
    ax.set_title(f"Stratified {result.split_summary['cv_folds']}-Fold Cross-Validation Stability\n"
                 "(each dot is one fold of the training set)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = outdir / "cv_fold_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame, outdir=None):
    """Bar chart of the work-category imbalance that justifies SMOTE."""
    outdir = outdir or cfg.FIGURE_DIR
    counts = df[cfg.TARGET_COLUMN].value_counts().reindex(cfg.WORK_CATEGORIES).fillna(0)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#2563eb", "#f59e0b", "#10b981", "#8b5cf6"], edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3)

    mean_count = counts.mean()
    ax.axhline(mean_count, ls="--", lw=1.1, color="#64748b")
    ax.text(len(counts) - 0.5, mean_count * 1.03,
            "perfectly balanced would be here", ha="right", fontsize=8, color="#64748b")

    ax.set_ylabel("Number of students")
    ax.set_title("Work Category Distribution\n"
                 "Class imbalance is what SMOTE is applied to correct (training folds only)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = outdir / "class_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #
def tune_boosting(
    df: pd.DataFrame,
    feature_set: str = "full",
    n_iter: int = 40,
    cv_folds: int = cfg.CV_FOLDS,
    random_state: int = cfg.RANDOM_STATE,
) -> dict:
    """Randomised search over the booster, scored by macro F1.

    Macro F1 rather than accuracy on purpose: with a 54% majority class,
    accuracy rewards a model that quietly ignores the Managerial students.
    Macro F1 forces every category to be predicted competently.

    Searching only on the TRAINING split keeps the hold-out honest -- tuning
    against the test set is a subtle but very common way to inflate results.
    """
    from sklearn.model_selection import RandomizedSearchCV

    X, y_raw, _ = feat.make_xy(df, feature_set=feature_set)
    encoder = LabelEncoder().fit(cfg.WORK_CATEGORIES)
    y = encoder.transform(y_raw)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=random_state, stratify=y
    )

    models = mdl.build_models(len(encoder.classes_), random_state=random_state)
    pipeline = models[mdl.primary_model_name(models)]

    if mdl.XGBOOST_AVAILABLE:
        grid = {
            "clf__n_estimators": [150, 250, 400, 600],
            "clf__max_depth": [2, 3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.12, 0.2],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 2, 4, 6],
            "clf__reg_lambda": [0.5, 1.0, 1.5, 3.0, 6.0],
            "clf__gamma": [0.0, 0.05, 0.2],
        }
    else:
        grid = {
            "clf__max_iter": [150, 250, 400],
            "clf__max_depth": [2, 3, 4, 6],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "clf__l2_regularization": [0.5, 1.0, 3.0],
        }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            pipeline, grid, n_iter=n_iter, scoring="f1_macro",
            cv=skf, random_state=random_state, n_jobs=-1, refit=False,
        )
        search.fit(X_train, y_train)

        default_score = cross_validate(
            pipeline, X_train, y_train, cv=skf, scoring="f1_macro"
        )["test_score"].mean()

    results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    path = cfg.TABLE_DIR / "hyperparameter_search.csv"
    results.to_csv(path, index=False)

    return {
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "best_score": float(search.best_score_),
        "default_score": float(default_score),
        "results": results,
        "path": path,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_results(result: BenchmarkResult, tag: str = "") -> dict:
    """Write every table, report and model artefact to outputs/."""
    suffix = f"_{tag}" if tag else ""
    written = {}

    holdout_path = cfg.TABLE_DIR / f"holdout_metrics{suffix}.csv"
    result.holdout_metrics.round(6).to_csv(holdout_path)
    written["holdout_metrics"] = holdout_path

    cv_path = cfg.TABLE_DIR / f"cv_metrics{suffix}.csv"
    result.cv_metrics.drop(columns=["_fold_accuracies", "_fold_f1"]).round(6).to_csv(cv_path)
    written["cv_metrics"] = cv_path

    for name, matrix in result.confusion_matrices.items():
        slug = name.split("(")[0].strip().lower().replace(" ", "_")
        path = cfg.TABLE_DIR / f"confusion_matrix_{slug}{suffix}.csv"
        pd.DataFrame(matrix, index=result.class_names, columns=result.class_names).to_csv(path)
        written[f"confusion_{slug}"] = path

    report_path = cfg.TABLE_DIR / f"classification_reports{suffix}.txt"
    with open(report_path, "w") as fh:
        for name, report in result.reports.items():
            fh.write(f"=== {name} ===\n{report}\n\n")
    written["classification_reports"] = report_path

    meta_path = cfg.TABLE_DIR / f"benchmark_metadata{suffix}.json"
    with open(meta_path, "w") as fh:
        json.dump(
            {"split": result.split_summary,
             "significance": result.significance,
             "classes": result.class_names,
             "features": result.feature_names},
            fh, indent=2,
        )
    written["metadata"] = meta_path

    for name, model in result.fitted_models.items():
        slug = name.split("(")[0].strip().lower().replace(" ", "_")
        path = cfg.MODEL_DIR / f"{slug}{suffix}.joblib"
        joblib.dump(model, path)
        written[f"model_{slug}"] = path

    bundle_path = cfg.MODEL_DIR / f"label_encoder{suffix}.joblib"
    joblib.dump(
        {"label_encoder": result.label_encoder, "feature_names": result.feature_names},
        bundle_path,
    )
    written["label_encoder"] = bundle_path

    return written
