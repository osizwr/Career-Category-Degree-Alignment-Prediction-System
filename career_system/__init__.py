"""
PSU Career Category & Degree Alignment System
=============================================

Reference implementation for:

    "Predictive Modeling of Career Category and Degree Alignment for CS and IT
     Students Using Ensemble and Baseline Classifiers"
    Cajilo, M. J. M., Donghil, Z. B., Mahinay, J. V., & Sibayan, G. P. U.
    College of Sciences, Palawan State University.

Module map
----------
config          domain knowledge you edit: prospectus, skills, certifications,
                career tracks, Palawan employers, BSCS/BSIT discipline profiles
features        raw questionnaire columns -> engineered matrix (RIASEC, EVT,
                competency signals)
data_generator  synthetic cohort used until the real AY 2026-2027 data exists
models          ZeroR / Logistic Regression / XGBoost + leak-free SMOTE pipelines
benchmark       70-30 split, Stratified 5-Fold CV, Accuracy/Precision/Recall/F1
explain         SHAP global and per-student explanations
alignment       % degree alignment, career-track matching, employer mapping
recommender     ties everything into a single per-student report
cli             command line entry points
"""

__version__ = "1.0.0"

__all__ = [
    "config",
    "features",
    "data_generator",
    "models",
    "benchmark",
    "explain",
    "alignment",
    "recommender",
]
