#!/usr/bin/env python3
"""
PSU Career Category & Degree Alignment System - entry point.

    Predictive Modeling of Career Category and Degree Alignment for CS and IT
    Students Using Ensemble and Baseline Classifiers

    Cajilo, M. J. M. | Donghil, Z. B. | Mahinay, J. V. | Sibayan, G. P. U.
    College of Sciences, Palawan State University

Quick start
-----------
    python main.py                    run the whole pipeline
    python main.py --help             list every command

Individual steps
----------------
    python main.py generate-data      build the synthetic cohort
    python main.py profile            descriptive statistics        (SOP 1)
    python main.py train              XGBoost vs LogReg vs ZeroR    (SOP 2)
    python main.py explain            SHAP feature importance       (SOP 4)
    python main.py recommend          one student's full report     (SOP 3)
    python main.py batch              cohort-wide export
    python main.py ablation           test the feature engineering
    python main.py tune               hyperparameter search
"""

import sys

from career_system.cli import main

if __name__ == "__main__":
    sys.exit(main())
