"""
PSU Explainable Career Category Recommendation System
=====================================================

Reference implementation for:

    "An Explainable Career Category Recommendation System for Computer Science
     and Information Technology Graduating Students using Ensemble and Baseline
     Classifiers"
    Cajilo, M. J. M., Donghil, Z. B., Mahinay, J. V., & Sibayan, G. P. U.
    College of Sciences, Palawan State University.

Module map
----------
config          domain knowledge you edit: the two prospectuses, the skill
                checklists, certifications, career tracks, Palawan employers,
                BSCS/BSIT discipline profiles, the Chapter IV labelling rubric
features        raw questionnaire columns -> engineered matrix (academic
                domains, competency signals, RIASEC, EVT)
labelling       the Chapter IV scoring rubric: 30% core courses, 20% GWA,
                25% certifications, 25% skills
data_generator  synthetic cohort used until the real AY 2026-2027 data exists
models          ZeroR / Logistic Regression / XGBoost + leak-free SMOTE pipelines
benchmark       70-30 split, Stratified 5-Fold CV, Accuracy/Precision/Recall/F1
explain         SHAP global and per-student explanations
alignment       % degree alignment, career-track matching, employer mapping
recommender     ties everything into a single per-student report
interactive     the questionnaire a student answers to test their own profile
cli             command line entry points
"""

__version__ = "2.0.0"

__all__ = [
    "config",
    "features",
    "labelling",
    "data_generator",
    "models",
    "benchmark",
    "explain",
    "alignment",
    "recommender",
    "interactive",
]
