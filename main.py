#!/usr/bin/env python3
"""
PSU Explainable Career Category Recommendation System - entry point.

    An Explainable Career Category Recommendation System for Computer Science
    and Information Technology Graduating Students using Ensemble and Baseline
    Classifiers

    Cajilo, M. J. M. | Donghil, Z. B. | Mahinay, J. V. | Sibayan, G. P. U.
    College of Sciences, Palawan State University

Quick start
-----------
    python main.py                    run the whole pipeline, then offer to
                                      predict your own career category
    python main.py predict            answer the questionnaire yourself
    python main.py --help             list every command

Individual steps
----------------
    python main.py generate-data      build the synthetic cohort
    python main.py profile            descriptive statistics        (SOP 1)
    python main.py train              XGBoost vs LogReg vs ZeroR    (SOP 2)
    python main.py explain            SHAP feature importance       (SOP 4)
    python main.py recommend          one student's full report     (SOP 3)
    python main.py predict            interactive: your own profile
    python main.py batch              cohort-wide export
    python main.py ablation           test the feature engineering
    python main.py tune               hyperparameter search

Running on the real cohort
--------------------------
    python scripts/clean_real_data.py       Google Forms export -> clean CSV
    python scripts/build_model_dataset.py   clean CSV -> model schema
    python main.py --data data/real-cohort-model.csv
"""

import sys

from career_system.cli import main

if __name__ == "__main__":
    sys.exit(main())
