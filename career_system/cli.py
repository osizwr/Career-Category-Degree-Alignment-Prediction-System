"""
Command line interface.

    python main.py generate-data      build the synthetic cohort CSV
    python main.py profile            descriptive statistics (SOP 1)
    python main.py train              XGBoost vs LogReg vs ZeroR (SOP 2)
    python main.py explain            SHAP global feature importance (SOP 4)
    python main.py recommend          one student's full report (SOP 3)
    python main.py predict            answer the questionnaire yourself
    python main.py batch              cohort-wide recommendation export
    python main.py ablation           does the feature engineering actually help?
    python main.py tune               randomised hyperparameter search
    python main.py run-all            everything, in order

Run `python main.py <command> --help` for the options of any single command.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

from . import alignment as align
from . import benchmark as bm
from . import config as cfg
from . import data_generator as dg
from . import explain as xai
from . import features as feat
from . import interactive as ui
from . import labelling
from . import models as mdl
from . import profiling
from . import recommender as rec


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def banner(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}")
    print(f" {title}")
    print(f"{char * 78}")


def step(message: str) -> None:
    print(f"\n>> {message}")


def load_dataset(path=None, required: bool = True) -> pd.DataFrame:
    path = Path(path) if path else cfg.DATASET_PATH
    if not path.exists():
        if not required:
            return None
        print(f"\n[!] No dataset at {path}")
        print("    Run `python main.py generate-data` first, or pass --data /path/to/your.csv")
        sys.exit(1)

    df = pd.read_csv(path)
    feat.validate_raw_frame(df, require_target=True)
    return df


def shown_path(path) -> str:
    """Project-relative when it is inside the project, absolute when it is not."""
    try:
        return str(Path(path).relative_to(cfg.PROJECT_ROOT))
    except ValueError:
        return str(path)


def report_written(paths: dict) -> None:
    print("\n  Files written:")
    for label, path in paths.items():
        if label.startswith("_"):
            continue
        print(f"    {shown_path(path)}")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_generate_data(args) -> pd.DataFrame:
    banner("STEP 1 - GENERATE SYNTHETIC COHORT")
    print(textwrap.dedent("""
        The real dataset is collected from active fourth-year BSCS/BSIT students
        in the first semester of AY 2026-2027. Until then this generator produces
        a cohort with the same schema so the whole pipeline can be built, tested
        and demonstrated. See career_system/data_generator.py for exactly how the
        work-category labels are produced -- you will be asked about this.
    """).strip())

    df = dg.generate_dataset(
        n_students=args.n, seed=args.seed, missing_rate=args.missing_rate
    )
    path = dg.save_dataset(df, args.out)

    print()
    print(dg.describe_dataset(df))
    print(f"\n  Saved to {shown_path(path)}")

    bm.plot_class_distribution(df)
    print(f"  Figure   outputs/figures/class_distribution.png")
    return df


def cmd_profile(args) -> None:
    banner("STEP 2 - DESCRIPTIVE PROFILE OF THE COHORT  (SOP 1)")
    df = load_dataset(args.data)
    written = profiling.run_profiling(df, verbose=True)
    report_written(written)


def cmd_train(args) -> bm.BenchmarkResult:
    banner("STEP 3 - BENCHMARK: XGBoost vs LOGISTIC REGRESSION vs ZeroR  (SOP 2)")
    df = load_dataset(args.data)

    if not mdl.XGBOOST_AVAILABLE:
        print("\n[!] xgboost is not installed; falling back to scikit-learn's")
        print("    HistGradientBoostingClassifier. Install xgboost to match the thesis exactly:")
        print("        pip install xgboost\n")

    result = bm.run_benchmark(
        df,
        feature_set=args.feature_set,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.seed,
        use_smote=not args.no_smote,
        verbose=True,
    )

    bm.print_results(result)

    step("Writing tables, figures and fitted models")
    written = bm.save_results(result, tag=args.tag)
    written["fig_comparison"] = bm.plot_model_comparison(result)
    written["fig_cv"] = bm.plot_cv_distribution(result)
    for path in bm.plot_confusion_matrices(result):
        written[f"fig_{path.stem}"] = path
    report_written(written)

    return result


def cmd_explain(args) -> None:
    banner("STEP 4 - SHAP EXPLAINABILITY  (SOP 4)")
    df = load_dataset(args.data)

    if not xai.SHAP_AVAILABLE:
        print("\n[!] The `shap` package is not installed, so this step will use")
        print("    permutation importance instead. To match the thesis:")
        print("        pip install shap\n")

    step("Training the boosting model on the full cohort")
    recommender = rec.CareerRecommender.fit_new(df, feature_set=args.feature_set, random_state=args.seed)
    X, y, _ = feat.make_xy(df, feature_set=args.feature_set)
    classes = list(recommender.label_encoder.classes_)

    step("Computing SHAP values")
    if xai.SHAP_AVAILABLE:
        bundle = xai.compute_shap(recommender.model, X, classes, max_samples=args.max_samples)
    else:
        bundle = xai.fallback_importance(
            recommender.model, X, recommender.label_encoder.transform(y), classes
        )

    importance = bundle.mean_abs_importance()
    print("\n  Top 15 features driving work-category classification")
    print("  " + "-" * 74)
    top = importance.head(15)[["Feature", "Overall"] + classes]
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}", max_colwidth=46))

    # Read straight off the SHAP values: which features the model leans on,
    # in which direction, and how the influence splits across kinds of
    # evidence. This is the paragraph curriculum review actually quotes.
    print("\n  Interpretation for curriculum review (PSU-CS):\n")
    print(xai.format_sentences(xai.narrate_global(bundle), width=76, indent=4))

    step("Writing SHAP figures")
    written = {
        "importance_table": xai.save_importance_table(bundle),
        "narrative": xai.save_global_narrative(bundle),
        "fig_global": xai.plot_global_importance(bundle, top_n=args.top_n),
        "fig_grouped": xai.plot_signal_importance(bundle),
    }
    for path in xai.plot_beeswarm_per_class(bundle):
        written[f"fig_{path.stem}"] = path
    report_written(written)


def cmd_recommend(args) -> None:
    banner("STEP 5 - STUDENT CAREER RECOMMENDATION  (SOP 3)")
    df = load_dataset(args.data)

    step("Preparing the recommendation model")
    recommender = _load_recommender(args, df)

    if args.student_id:
        matches = df[df["student_id"] == args.student_id]
        if matches.empty:
            print(f"\n[!] No student with id '{args.student_id}'.")
            print(f"    Available ids look like: {', '.join(df['student_id'].head(3))} ...")
            sys.exit(1)
        selection = matches
    else:
        selection = df.sample(args.count, random_state=args.seed)

    for _, student in selection.iterrows():
        report = recommender.recommend(student, top_n=args.top_n)
        style = getattr(args, "explanation_style", "human")
        print("\n" + report.render(explanation_style=style))

        written = rec.save_report(report, explanation_style=style)
        if report.explanation is not None:
            written["figure"] = xai.plot_local_explanation(report.explanation)
        report_written(written)


def _load_recommender(args, df=None):
    """The fitted model, from disk if a previous run saved one."""
    try:
        recommender = rec.CareerRecommender.load(tag=args.tag)
        print("  Loaded the previously trained model from outputs/models/")
        return recommender
    except FileNotFoundError:
        pass

    if df is None:
        df = load_dataset(getattr(args, "data", None))
    print("  No saved model found - training a fresh one on the full cohort")
    return rec.CareerRecommender.fit_new(df, random_state=args.seed)


def cmd_predict(args, recommender=None) -> None:
    """Answer the questionnaire as yourself and get the model's verdict."""
    banner("TEST YOUR OWN PROFILE - INTERACTIVE CAREER CATEGORY PREDICTION")

    if args.answers:
        record = ui.load_answers(args.answers)
        print(f"\n  Replaying the answers in {args.answers}")
    else:
        try:
            record = ui.run_questionnaire(student_id=args.student_id or "YOU")
        except ui.Abort as exc:
            if str(exc) == "no more input":
                print("\n\n[!] Ran out of input before the questionnaire finished.")
                print("    `predict` reads answers from the terminal. To run it without")
                print("    one, save the answers once and replay them:")
                print("        python main.py predict --save-answers me.json")
                print("        python main.py predict --answers me.json")
                return
            print("\n  Cancelled. Nothing was saved.")
            return

    if args.save_answers:
        ui.save_answers(record, args.save_answers)
        print(f"\n  Answers saved to {args.save_answers}")

    ui.print_answer_summary(record)

    frame = ui.record_to_frame(record)

    if recommender is None:
        step("Preparing the recommendation model")
        recommender = _load_recommender(args)

    # The Chapter IV rubric, scored on these answers. Shown next to the model so
    # a panelist can see where the two agree and where they do not.
    rubric = labelling.rubric_scores(frame).iloc[0].sort_values(ascending=False)

    step("Scoring your profile")
    report = recommender.recommend(frame.iloc[0], top_n=args.top_n)
    style = getattr(args, "explanation_style", "human")
    print("\n" + report.render(explanation_style=style))

    print("\n  CHAPTER IV LABELLING RUBRIC, FOR COMPARISON")
    print("  " + "-" * 74)
    print("    Core courses 30%  |  GWA 20%  |  Certifications 25%  |  Skills 25%")
    for category, score in rubric.items():
        if pd.isna(score):
            continue
        bar = "#" * int(round(score / 100 * 34))
        print(f"      {category:<12} {score:5.1f}  {bar}")
    rubric_pick = rubric.idxmax()
    if rubric_pick == report.predicted_category:
        print(f"\n    The rubric and the model agree on '{rubric_pick}'.")
    else:
        print(f"\n    The rubric scores '{rubric_pick}' highest while the model "
              f"predicts\n    '{report.predicted_category}'. Disagreement means the "
              f"model has learned something\n    the flat weighted average does not "
              f"capture -- or that this profile sits\n    near a boundary. Chapter IV "
              f"sends exactly these cases to the adviser.")

    ui.print_next_steps(report)

    if args.save_report:
        written = rec.save_report(report, explanation_style=style)
        report_written(written)


def cmd_batch(args) -> None:
    banner("COHORT-WIDE RECOMMENDATION EXPORT")
    df = load_dataset(args.data)

    try:
        recommender = rec.CareerRecommender.load(tag=args.tag)
    except FileNotFoundError:
        recommender = rec.CareerRecommender.fit_new(df, random_state=args.seed)

    step(f"Scoring {len(df)} students")
    table = recommender.recommend_batch(df, top_n=args.top_n, explain=False)

    path = cfg.TABLE_DIR / "cohort_recommendations.csv"
    table.to_csv(path, index=False)

    print("\n  Predicted work-category distribution across the cohort:")
    counts = table["predicted_work_category"].value_counts()
    for category in cfg.WORK_CATEGORIES:
        count = int(counts.get(category, 0))
        print(f"    {category:<12} {count:>4}  ({count / len(table):5.1%})")

    aligned = table["aligned_with_enrolment"].sum()
    print(f"\n  Students whose competency profile best matches their enrolled program: "
          f"{aligned}/{len(table)} ({aligned / len(table):.1%})")

    print(f"\n  Mean confidence: {table['confidence'].mean():.1%}")
    print("    [!] Do not quote this as an accuracy figure. The saved model was fitted on")
    print("        70% of this cohort, so most of these students were part of its training")
    print("        data and it is unusually sure about them. The honest performance numbers")
    print("        are the hold-out results from `python main.py train`.")
    print("\n  Sample of the export:")
    print(table.head(8)[
        ["student_id", "program", "predicted_work_category", "career_1", "career_1_match_pct"]
    ].to_string(index=False))

    report_written({"cohort_recommendations": path})


def cmd_ablation(args) -> None:
    banner("ABLATION - DOES THE THEORY-DRIVEN FEATURE ENGINEERING EARN ITS PLACE?")
    print(textwrap.dedent("""
        Runs the same benchmark three times on three feature sets:

          core        only what the student actually reported (grades, certs, skills)
          engineered  only the derived variables (competency signals, RIASEC, EVT)
          full        both -- what the thesis proposes

        If `full` does not beat `core`, say so in Chapter V. An honest negative
        result about your own feature engineering is far stronger than a claim
        the numbers do not support.
    """).strip())

    df = load_dataset(args.data)
    rows = []

    for feature_set in ("core", "engineered", "full"):
        step(f"Feature set: {feature_set}")
        result = bm.run_benchmark(
            df, feature_set=feature_set, random_state=args.seed, verbose=False
        )
        boosting = mdl.primary_model_name(result.fitted_models)
        for model_name in result.holdout_metrics.index:
            rows.append({
                "Feature set": feature_set,
                "N features": result.split_summary["n_features"],
                "Model": model_name,
                "Accuracy": result.holdout_metrics.loc[model_name, "Accuracy"],
                "F1 (macro)": result.holdout_metrics.loc[model_name, "F1 (macro)"],
                "CV Accuracy": result.cv_metrics.loc[model_name, "CV Accuracy (mean)"],
            })
        print(f"   {boosting}: accuracy {result.holdout_metrics.loc[boosting, 'Accuracy']:.4f}, "
              f"F1 macro {result.holdout_metrics.loc[boosting, 'F1 (macro)']:.4f}")

    table = pd.DataFrame(rows)
    print("\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    path = cfg.TABLE_DIR / "ablation_feature_sets.csv"
    table.to_csv(path, index=False)
    report_written({"ablation": path})


def cmd_tune(args) -> None:
    banner("HYPERPARAMETER SEARCH FOR THE BOOSTING MODEL")
    df = load_dataset(args.data)
    result = bm.tune_boosting(
        df, feature_set=args.feature_set, n_iter=args.n_iter,
        cv_folds=args.cv_folds, random_state=args.seed,
    )

    print("\n  Best parameters found:")
    for key, value in result["best_params"].items():
        print(f"    {key:<24} {value}")
    print(f"\n  Best CV F1 (macro): {result['best_score']:.4f}")
    print(f"  Current defaults  : {result['default_score']:.4f}")

    if result["best_score"] > result["default_score"] + 0.005:
        print("\n  The search found a meaningfully better configuration. Copy the")
        print("  parameters above into `_boosting_estimator` in career_system/models.py.")
    else:
        print("\n  The search did NOT beat the current defaults by a meaningful margin,")
        print("  which usually means performance is limited by the data, not the model.")

    report_written({"tuning_results": result["path"]})


def cmd_run_all(args) -> None:
    banner("PSU CAREER CATEGORY & DEGREE ALIGNMENT SYSTEM - FULL PIPELINE", "#")
    print("  Predictive Modeling of Career Category and Degree Alignment for")
    print("  CS and IT Students Using Ensemble and Baseline Classifiers")
    print("  Palawan State University - College of Sciences")

    existing = load_dataset(args.data, required=False)
    if existing is None or args.regenerate:
        cmd_generate_data(args)
    else:
        source = Path(args.data) if args.data else cfg.DATASET_PATH
        print(f"\n  Using the existing dataset at {shown_path(source)} "
              f"({len(existing)} students).")
        print("  Pass --regenerate to build a fresh one.")

    cmd_profile(args)
    cmd_train(args)
    cmd_explain(args)
    cmd_recommend(args)
    cmd_batch(args)

    banner("PIPELINE COMPLETE", "#")
    print("  Everything is under outputs/:")
    print("    outputs/tables/    metrics, confusion matrices, descriptive profiles")
    print("    outputs/figures/   comparison charts, confusion matrices, SHAP plots")
    print("    outputs/models/    the fitted pipelines (joblib)")
    print("    outputs/reports/   per-student recommendation reports")
    print("\n  Next: `python main.py ablation` to test whether your feature")
    print("  engineering actually helps, and `python main.py tune` before you")
    print("  finalise the model on real data.")

    # The model is trained; now let whoever is watching put their own profile
    # through it. This is the demonstration a panelist asks for.
    if not args.no_interactive and ui.offer(
        "The model is trained. Answer the questionnaire yourself and see "
        "which work\n  category it places you in?"
    ):
        cmd_predict(args)
    elif not args.no_interactive:
        print("\n  To try it yourself later: python main.py predict")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="PSU Career Category & Degree Alignment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            examples
              python main.py                              run the whole pipeline
              python main.py generate-data --n 500
              python main.py train --feature-set core
              python main.py train --no-smote             show what SMOTE is worth
              python main.py recommend --student-id PSU-0007
              python main.py predict                      answer the form yourself
              python main.py ablation
        """),
    )
    sub = parser.add_subparsers(dest="command")

    def common(p):
        p.add_argument("--data", help="path to a dataset CSV (default: data/psu_computing_students.csv)")
        p.add_argument("--seed", type=int, default=cfg.RANDOM_STATE, help="random seed (default: 42)")
        p.add_argument("--tag", default="", help="suffix for saved artefacts, e.g. --tag v2")
        return p

    p = common(sub.add_parser("generate-data", help="build the synthetic cohort CSV"))
    p.add_argument("--n", type=int, default=400, help="number of students (default: 400)")
    p.add_argument("--missing-rate", type=float, default=0.015,
                   help="fraction of self-reported cells left blank (default: 0.015)")
    p.add_argument("--out", help="output CSV path")
    p.set_defaults(func=cmd_generate_data)

    p = common(sub.add_parser("profile", help="descriptive statistics of the cohort (SOP 1)"))
    p.set_defaults(func=cmd_profile)

    p = common(sub.add_parser("train", help="benchmark the three classifiers (SOP 2)"))
    p.add_argument("--feature-set", choices=sorted(feat.FEATURE_SETS), default="full")
    p.add_argument("--test-size", type=float, default=cfg.TEST_SIZE, help="default: 0.30")
    p.add_argument("--cv-folds", type=int, default=cfg.CV_FOLDS, help="default: 5")
    p.add_argument("--no-smote", action="store_true", help="disable SMOTE to measure its effect")
    p.set_defaults(func=cmd_train)

    p = common(sub.add_parser("explain", help="SHAP global feature importance (SOP 4)"))
    p.add_argument("--feature-set", choices=sorted(feat.FEATURE_SETS), default="full")
    p.add_argument("--top-n", type=int, default=20, help="features to plot (default: 20)")
    p.add_argument("--max-samples", type=int, default=400,
                   help="students to explain; lower is faster (default: 400)")
    p.set_defaults(func=cmd_explain)

    p = common(sub.add_parser("recommend", help="full report for one student (SOP 3)"))
    p.add_argument("--student-id", help="e.g. PSU-0007; omit to pick at random")
    p.add_argument("--count", type=int, default=1, help="how many random students (default: 1)")
    p.add_argument("--top-n", type=int, default=3, help="career tracks to show (default: 3)")
    p.add_argument("--explanation-style", choices=["human", "technical"], default="human",
                   help="style of explanation narrative: 'human' (advisor-friendly, default) or 'technical' (log-odds)")
    p.set_defaults(func=cmd_recommend)

    p = common(sub.add_parser(
        "predict", help="answer the questionnaire yourself and get a prediction"))
    p.add_argument("--student-id", default="YOU", help="label for the report (default: YOU)")
    p.add_argument("--top-n", type=int, default=3, help="career tracks to show (default: 3)")
    p.add_argument("--answers", help="replay a saved JSON answer file instead of asking")
    p.add_argument("--save-answers", help="write the answers to a JSON file")
    p.add_argument("--save-report", action="store_true",
                   help="also write the report to outputs/reports/")
    p.add_argument("--explanation-style", choices=["human", "technical"], default="human",
                   help="style of explanation narrative: 'human' (advisor-friendly, default) or 'technical' (log-odds)")
    p.set_defaults(func=cmd_predict)

    p = common(sub.add_parser("batch", help="recommendation export for the whole cohort"))
    p.add_argument("--top-n", type=int, default=3)
    p.set_defaults(func=cmd_batch)

    p = common(sub.add_parser("ablation", help="compare core / engineered / full feature sets"))
    p.set_defaults(func=cmd_ablation)

    p = common(sub.add_parser("tune", help="randomised hyperparameter search"))
    p.add_argument("--feature-set", choices=sorted(feat.FEATURE_SETS), default="full")
    p.add_argument("--n-iter", type=int, default=40, help="search iterations (default: 40)")
    p.add_argument("--cv-folds", type=int, default=cfg.CV_FOLDS)
    p.set_defaults(func=cmd_tune)

    p = common(sub.add_parser("run-all", help="run every step in order (default)"))
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--missing-rate", type=float, default=0.015)
    p.add_argument("--out", default=None)
    p.add_argument("--feature-set", choices=sorted(feat.FEATURE_SETS), default="full")
    p.add_argument("--test-size", type=float, default=cfg.TEST_SIZE)
    p.add_argument("--cv-folds", type=int, default=cfg.CV_FOLDS)
    p.add_argument("--no-smote", action="store_true")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument("--student-id", default=None)
    p.add_argument("--answers", default=None)
    p.add_argument("--save-answers", default=None)
    p.add_argument("--save-report", action="store_true")
    p.add_argument("--no-interactive", action="store_true",
                   help="skip the 'test your own profile' prompt at the end")
    p.add_argument("--regenerate", action="store_true", help="rebuild the dataset even if one exists")
    p.set_defaults(func=cmd_run_all)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    # No subcommand -> run the whole pipeline, which is what a first-time user
    # (or a panelist told to "just run it") will expect.
    if not argv or argv[0].startswith("-"):
        argv = ["run-all"] + argv

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n[error] {exc}")
        return 1

    return 0
