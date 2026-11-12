#!/usr/bin/env python3
"""
Adapt the cleaned survey export onto the schema career_system expects.

    .venv/bin/python scripts/build_model_dataset.py [--only-4th-year] [--out PATH]

Input :  data/real-data-clean.csv   (produced by scripts/clean_real_data.py)
Output:  data/real-cohort-model.csv (consumed by `python main.py --data ...`)

WHAT THIS SCRIPT USED TO DO, AND WHY IT NO LONGER DOES IT
---------------------------------------------------------
The previous version of this file carried four hand-written translation tables
because career_system was built against a guessed instrument: a single shared
course list, 1-5 Likert skills and an open-ended "seminar units" count. Six
model skills had no counterpart in the real form and were left permanently
blank; five more shared a single checkbox and were therefore perfectly
correlated by construction; two course slots were empty for every BSCS student
and one for every BSIT student.

career_system/config.py now mirrors the approved questionnaire item for item,
so none of that translation is needed. Course grades, skills and credential
types carry across one-to-one. The only interpretation left is reading Q8's
free-text credential names into subject domains, and that lives in
config.CERT_KEYWORDS where the interactive questionnaire reads it too.

READ THIS BEFORE QUOTING ANY ACCURACY FIGURE
--------------------------------------------
`work_category` was never asked in the survey. This script derives it with the
Chapter IV scoring rubric (career_system/labelling.py): core course performance
30%, GWA 20%, certifications 25%, self-reported skills 25%, highest composite
wins.

That makes the label a deterministic function of the features. A classifier
trained on it is measuring whether it can re-learn a formula this repository
already contains -- not whether it can predict real careers. Expect very high
accuracy and report it as a pipeline check, never as evidence about student
outcomes. Real labels have to come from an actual outcome: the two researchers'
independent manual coding described under "Qualified Labelers", an OJT
placement record, or a graduate tracer study.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from career_system import config as cfg  # noqa: E402
from career_system import features as feat  # noqa: E402
from career_system import labelling  # noqa: E402

DEFAULT_IN = ROOT / "data" / "real-data-clean.csv"
DEFAULT_OUT = ROOT / "data" / "real-cohort-model.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default=DEFAULT_IN, type=Path)
    ap.add_argument("--out", dest="dst", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--only-4th-year", action="store_true",
                    help="drop respondents whose flags say they are not 4th-year students")
    args = ap.parse_args()

    clean = pd.read_csv(args.src)
    print(f"Read {args.src.name}: {len(clean)} respondents")

    if args.only_4th_year:
        keep = clean["is_enrolled_4th_year"] == 1
        dropped = clean.loc[~keep, "student_id"].tolist()
        clean = clean[keep].reset_index(drop=True)
        print(f"  --only-4th-year: dropped {len(dropped)} ({', '.join(dropped) or 'none'})")

    raw = pd.DataFrame({
        "student_id": clean["student_id"],
        "program": clean["program"],
        "gwa": clean["gwa"],
    })

    # ---------------------------------------------------- 1. course grades --
    # The cleaned export already names these grade_<program>_<course>, which is
    # exactly what config.Course.column produces. A missing column means the
    # prospectus in config.py and the one in clean_real_data.py have drifted
    # apart, so say so rather than silently blanking the course.
    missing_courses = [c.column for c in cfg.ALL_COURSES if c.column not in clean.columns]
    if missing_courses:
        raise SystemExit(
            f"{args.src.name} has no column for {len(missing_courses)} prospectus "
            f"course(s): {missing_courses}\n"
            f"config.BSCS_COURSES / BSIT_COURSES and scripts/clean_real_data.py "
            f"must describe the same questionnaire."
        )
    for course in cfg.ALL_COURSES:
        column = pd.to_numeric(clean[course.column], errors="coerce")
        # A respondent only answers their own prospectus; blank the rest so the
        # validator's cross-program check has something true to confirm.
        raw[course.column] = column.where(clean["program"] == course.program)

    # ------------------------------------------------- 2. certifications ----
    raw["has_certification"] = pd.to_numeric(
        clean["has_certification"], errors="coerce"
    ).fillna(0).astype(int)

    for key in cfg.CERTIFICATION_TYPES:
        column = f"cert_type_{key}"
        if column in clean.columns:
            raw[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0).astype(int)
        else:
            # The form offers this option but no respondent picked it, so
            # clean_real_data.py never created the column.
            raw[column] = 0

    names = clean["certification_names"].fillna("").astype(str)
    for domain in cfg.CERT_DOMAINS:
        patterns = [p for p, d in cfg.CERT_KEYWORDS if d == domain]
        hit = pd.Series(False, index=clean.index)
        for pattern in patterns:
            hit |= names.str.contains(pattern, case=False, regex=True, na=False)
        raw[f"cert_{domain}"] = hit.astype(int)

    # -------------------------------------------------------- 3. skills ----
    # One-to-one: the cleaned column is skill_<group>_<key>, the model column is
    # <group>_<key>. A blank means the respondent skipped the whole question.
    for skill in cfg.ALL_SKILLS:
        source = f"skill_{skill.group}_{skill.key}"
        if source not in clean.columns:
            raise SystemExit(
                f"{args.src.name} has no column '{source}'. config.py and "
                f"clean_real_data.py disagree about Part IV of the questionnaire."
            )
        raw[f"{skill.group}_{skill.key}"] = pd.to_numeric(clean[source], errors="coerce")

    # Fails loudly here rather than inside main.py if a mapping went wrong.
    feat.validate_raw_frame(raw, require_target=False)

    # ---- derive the label (see the module docstring: this is circular) ------
    raw[cfg.TARGET_COLUMN] = labelling.assign_work_categories(raw).values
    margins = labelling.label_margin(raw)

    feat.validate_raw_frame(raw, require_target=True)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    raw[list(feat.RAW_COLUMNS)].to_csv(args.dst, index=False)

    breakdown_path = args.dst.with_name(args.dst.stem + "-rubric.csv")
    labelling.rubric_breakdown(raw).to_csv(breakdown_path, index=False)

    # ----------------------------- audit ------------------------------------ #
    print(f"Wrote {args.dst}: {len(raw)} rows x {len(feat.RAW_COLUMNS)} columns")
    print(f"Wrote {breakdown_path}: the per-student rubric worksheet\n")

    print("Course grades carried across one-to-one:")
    for program in cfg.PROGRAMS:
        rows = raw["program"] == program
        courses = cfg.COURSES_BY_PROGRAM[program]
        reported = sum(int(raw.loc[rows, c.column].notna().sum()) for c in courses)
        possible = int(rows.sum()) * len(courses)
        print(f"  {program}: {len(courses)} courses, {reported}/{possible} grades reported "
              f"({reported / max(possible, 1):.0%})")

    print("\nAcademic domain coverage (what the model actually reads):")
    engineered = feat.add_grade_points(raw)
    for domain, label in cfg.ACADEMIC_DOMAINS.items():
        series = engineered[f"acad_{domain}"]
        print(f"  {label:<42} {int(series.notna().sum()):>4}/{len(raw)} students, "
              f"mean {series.mean():5.1f} points")

    said_yes = int((raw["has_certification"] == 1).sum())
    flags = int(raw[[f"cert_{d}" for d in cfg.CERT_DOMAINS]].to_numpy().sum())
    print(f"\nCertifications: {said_yes} respondents answered 'Yes' to Q7; "
          f"{flags} subject-domain flag(s) recognised from Q8 free text")
    for domain in cfg.CERT_DOMAINS:
        n = int(raw[f"cert_{domain}"].sum())
        if n:
            print(f"  cert_{domain}: {n}")
    no_name = int(((raw["has_certification"] == 1) & (names == "")).sum())
    if no_name:
        print(f"  {no_name} named no credential, so no subject domain could be set")

    print("\nSkills ticked (Part IV):")
    for group, skills in cfg.SKILL_GROUPS.items():
        for skill in skills:
            series = raw[f"{group}_{skill.key}"]
            print(f"  {group}_{skill.key:<22} {int(series.sum()):>4} of "
                  f"{int(series.notna().sum())} who answered")

    print(f"\nDerived {cfg.TARGET_COLUMN} distribution (Chapter IV rubric):")
    counts = raw[cfg.TARGET_COLUMN].value_counts()
    for category in cfg.WORK_CATEGORIES:
        n = int(counts.get(category, 0))
        print(f"  {category:<12} {n:>4}  ({n / len(raw):5.1%})")

    close = int((margins < 2.0).sum())
    print(f"\n  {close} student(s) are within 2.0 rubric points of a different category.")
    print("  Chapter IV sends exactly these to the consensus discussion, and then")
    print("  to the adviser. They are listed in the rubric worksheet above.")

    smallest = counts.min()
    if len(counts) < 2:
        print("\n  [!] Only one class present. Classification is impossible: train/explain/ablation")
        print("      cannot run, and this is a property of the rule, not of your students.")
    elif smallest < 5:
        print(f"\n  [!] Smallest class has {smallest} member(s). A 70-30 stratified split with 5-fold CV")
        print("      needs several per class; expect SMOTE to skip it and CV folds to be unstable.")

    print("\n  [!] work_category above is COMPUTED FROM the features by the Chapter IV")
    print("      rubric, not observed. Accuracy from `main.py train` on this file")
    print("      measures rule recovery only. Replace it with the labelers' manual")
    print("      coding, OJT placements or tracer-study outcomes before Chapter V.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
