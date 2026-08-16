"""
Classifiers and pipelines for the work-category benchmark.

Three models, exactly as specified in the thesis:

    ZeroR                -> the "null accuracy floor" (Termmedi et al., 2023)
    Logistic Regression  -> parametric, linear baseline (softmax / multinomial)
    XGBoost              -> the primary regularised tree-boosting model
                            (Chen & Guestrin, 2016)

ONE METHODOLOGICAL POINT WORTH DEFENDING
----------------------------------------
Scaling and SMOTE are placed INSIDE an imbalanced-learn Pipeline, not applied to
the dataset up front. This matters more than it sounds:

  * If you scale before splitting, the test set's mean and variance leak into
    the training set and every reported metric is optimistic.
  * If you SMOTE before splitting, synthetic points generated from test-set
    neighbours end up in training. Accuracy can jump 5-15 points for free, and
    the result is not reproducible on new students.

Putting both steps in the pipeline means `cross_validate` refits them on each
training fold only, so the 30% hold-out and every validation fold stay clean.

ZeroR deliberately gets NO SMOTE. Its whole purpose is to report what you would
score by always answering the genuine majority class; balancing its input first
would destroy the very floor it is meant to measure.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline

from . import config as cfg

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful degradation
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGBOOST_AVAILABLE = False


# --------------------------------------------------------------------------- #
# ZeroR
# --------------------------------------------------------------------------- #
class ZeroRClassifier(BaseEstimator, ClassifierMixin):
    """Zero Rule baseline: ignore every predictor, always answer the majority class.

    Implements the accuracy definition given in Chapter III:

        Accuracy_ZeroR = (1/n) * SUM_i  I( y_i == C_maj )

    Kept as an explicit class rather than sklearn's DummyClassifier so the code
    matches the formula in the paper line for line, and so `predict_proba`
    returns the empirical class frequencies (useful when the report shows a
    "confidence" next to a ZeroR prediction).
    """

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_frequencies_ = counts / counts.sum()
        self.majority_class_ = self.classes_[np.argmax(counts)]
        self.majority_share_ = float(np.max(counts) / counts.sum())
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "majority_class_")
        return np.full(np.asarray(X).shape[0], self.majority_class_, dtype=self.classes_.dtype)

    def predict_proba(self, X):
        check_is_fitted(self, "majority_class_")
        n = np.asarray(X).shape[0]
        return np.tile(self.class_frequencies_, (n, 1))

    def __repr__(self):
        return "ZeroRClassifier()"


# --------------------------------------------------------------------------- #
# SMOTE that will not crash on a tiny minority class
# --------------------------------------------------------------------------- #
class AdaptiveSMOTE(BaseEstimator):
    """SMOTE wrapper that shrinks k_neighbors to fit the smallest class.

    Plain SMOTE raises when a class has fewer than k_neighbors+1 members. Inside
    5-fold CV the Managerial class can drop to ~20 students, and on a real PSU
    cohort it may be smaller still. This picks the largest workable k and skips
    resampling entirely (with the data passed through untouched) if even k=1 is
    impossible, so a small real dataset degrades gracefully instead of aborting
    the whole benchmark.
    """

    def __init__(self, random_state=cfg.RANDOM_STATE, max_k=5):
        self.random_state = random_state
        self.max_k = max_k

    def fit_resample(self, X, y):
        y_arr = np.asarray(y)
        _, counts = np.unique(y_arr, return_counts=True)
        smallest = int(counts.min())

        k = min(self.max_k, smallest - 1)
        if k < 1:
            # Only one member in some class: nothing to interpolate between.
            self.k_neighbors_used_ = 0
            return X, y

        self.k_neighbors_used_ = k
        sampler = SMOTE(random_state=self.random_state, k_neighbors=k)
        return sampler.fit_resample(X, y)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def _boosting_estimator(n_classes: int, random_state: int):
    """Return XGBoost, or a scikit-learn gradient booster if xgboost is absent."""
    if XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.90,
            colsample_bytree=0.80,
            min_child_weight=2,
            # The regularisation terms are the Omega penalty in the thesis's
            # objective function; they are what stop the model from memorising
            # a single College of Sciences cohort.
            reg_lambda=1.5,
            reg_alpha=0.10,
            gamma=0.05,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=4, learning_rate=0.08,
        l2_regularization=1.5, random_state=random_state,
    )


def build_models(n_classes: int = len(cfg.WORK_CATEGORIES),
                 random_state: int = cfg.RANDOM_STATE,
                 use_smote: bool = True) -> dict:
    """Build the three competing pipelines, keyed by display name.

    Each value is a fitted-on-demand sklearn/imblearn Pipeline, so the caller can
    treat all three identically in cross-validation.
    """
    median_impute = SimpleImputer(strategy="median")

    # --- ZeroR: no scaling, no SMOTE, no features used at all ---------------
    zero_r = SkPipeline([
        ("impute", median_impute),
        ("clf", ZeroRClassifier()),
    ])

    # --- Logistic Regression: needs imputation + scaling ---------------------
    logreg_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_smote:
        logreg_steps.append(("smote", AdaptiveSMOTE(random_state=random_state)))
    logreg_steps.append((
        "clf",
        LogisticRegression(
            solver="lbfgs",          # softmax / multinomial by default
            C=1.0,
            max_iter=3000,
            random_state=random_state,
            n_jobs=-1,
        ),
    ))

    # --- XGBoost: trees do not need scaling, but SMOTE needs complete data ---
    xgb_steps = [("impute", SimpleImputer(strategy="median"))]
    if use_smote:
        xgb_steps.append(("smote", AdaptiveSMOTE(random_state=random_state)))
    xgb_steps.append(("clf", _boosting_estimator(n_classes, random_state)))

    boosting_name = "XGBoost" if XGBOOST_AVAILABLE else "HistGradientBoosting (XGBoost fallback)"

    return {
        "ZeroR": zero_r,
        "Logistic Regression": ImbPipeline(logreg_steps) if use_smote else SkPipeline(logreg_steps),
        boosting_name: ImbPipeline(xgb_steps) if use_smote else SkPipeline(xgb_steps),
    }


def primary_model_name(models: dict) -> str:
    """Name of the boosting model inside a `build_models` dictionary."""
    for name in models:
        if "XGBoost" in name or "Boosting" in name:
            return name
    raise KeyError("No boosting model found in the model dictionary.")
