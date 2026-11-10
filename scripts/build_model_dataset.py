#!/usr/bin/env python3
"""
Adapt the cleaned survey export onto the schema career_system expects.

    .venv/bin/python scripts/build_model_dataset.py [--only-4th-year] [--out PATH]

Input :  data/real-data-clean.csv   (produced by scripts/clean_real_data.py)
Output:  data/real-cohort-model.csv (consumed by `python main.py --data ...`)

The survey instrument and the model schema do not line up, so every mapping
decision is in this file, in one of four tables, and the script prints exactly
what it did. Nothing is mapped silently.

    BSCS_COURSE_MAP / BSIT_COURSE_MAP   real course grades -> grade_<course_key>
    SKILL_MAP                           survey checkboxes  -> tech_/hard_/soft_
    CERT_KEYWORDS                       certificate text   -> cert_<domain>

READ THIS BEFORE QUOTING ANY ACCURACY FIGURE
--------------------------------------------
`work_category` was never asked in the survey. This script derives it with a
rule: a student's competency signals are scored against the 15 career tracks in
config.CAREER_TRACKS, and the work category of the best-matching track becomes
the label (the same scoring the recommender shows students, with the model taken
out of the loop).

That makes the label a deterministic function of the features. A classifier
trained on it is therefore measuring whether it can re-learn a formula this
repository already contains -- not whether it can predict real careers. Expect
very high accuracy and report it as a pipeline check, never as evidence about
student outcomes. Real labels have to come from an actual outcome: an intended
career-category question, an OJT placement record, or a graduate tracer study.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from career_system import alignment as align  # noqa: E402
from career_system import config as cfg  # noqa: E402
from career_system import features as feat  # noqa: E402

DEFAULT_IN = ROOT / "data" / "real-data-clean.csv"
DEFAULT_OUT = ROOT / "data" / "real-cohort-model.csv"

# --------------------------------------------------------------------------- #
# 1. Course grades:  config course key -> survey column(s), averaged
# --------------------------------------------------------------------------- #
# The two programs sat different prospectuses, so each gets its own map. A key
# absent from a map has no counterpart in that program's curriculum and is left
# blank for every student in it (the model pipelines median-impute).
BSCS_COURSE_MAP = {
    "intro_computing": ["grade_bscs_intro_computing"],
    # BSCS reported one programming-languages course rather than Programming 1/2.
    "programming_1": ["grade_bscs_programming_languages"],
    "discrete_structures": [
        "grade_bscs_discrete_structures_1",
        "grade_bscs_discrete_structures_2",
        "grade_bscs_automata_formal_languages",
    ],
    "data_structures": [
        "grade_bscs_data_structures_algorithms",
        "grade_bscs_algorithms_complexity",
    ],
    "oop": ["grade_bscs_oop"],
    "information_management": ["grade_bscs_information_management"],
    "computer_architecture": ["grade_bscs_architecture_organization"],
    "operating_systems": ["grade_bscs_operating_systems"],
    "networks": ["grade_bscs_networks_communications"],
    "information_security": ["grade_bscs_info_assurance_security"],
    "hci": ["grade_bscs_hci"],
    "software_engineering": ["grade_bscs_software_engineering_1"],
    "app_development": ["grade_bscs_app_dev_emerging_tech"],
    "professional_practice": ["grade_bscs_social_issues_prof_practice"],
    # no counterpart: programming_2, web_systems
}

BSIT_COURSE_MAP = {
    "intro_computing": ["grade_bsit_intro_computing"],
    "programming_1": ["grade_bsit_programming_1"],
    "programming_2": ["grade_bsit_programming_2"],
    "discrete_structures": ["grade_bsit_discrete_mathematics"],
    "data_structures": ["grade_bsit_data_structures_algorithms"],
    # Closest BSIT equivalent to OOP: the integrative-programming course.
    "oop": ["grade_bsit_integrative_programming_1"],
    "information_management": [
        "grade_bsit_information_management_1",
        "grade_bsit_advanced_database_systems",
    ],
    "computer_architecture": ["grade_bsit_embedded_systems"],
    "networks": [
        "grade_bsit_network_technologies_1",
        "grade_bsit_network_technologies_2",
    ],
    "information_security": [
        "grade_bsit_info_assurance_security_1",
        "grade_bsit_info_assurance_security_2",
    ],
    "web_systems": ["grade_bsit_web_systems_1"],
    # HCI in BSIT is spread across the interface / visual-media courses.
    "hci": [
        "grade_bsit_intro_hci",
        "grade_bsit_graphics_visual_computing",
        "grade_bsit_multimedia_systems",
    ],
    "software_engineering": ["grade_bsit_systems_integration_architecture_1"],
    "app_development": ["grade_bsit_app_dev_emerging_tech_1"],
    "professional_practice": ["grade_bsit_social_professional_issues"],
    # no counterpart: operating_systems
    # deliberately unused (student-chosen or no shared equivalent):
    #   grade_bsit_it_elective_1/2/3, grade_bsit_gis
}

# --------------------------------------------------------------------------- #
# 2. Skills:  model skill column -> survey checkbox column
# --------------------------------------------------------------------------- #
# The questionnaire asked which skills a student HAS (checkboxes); the model
# expects a 1-5 Likert rating of each. There is no way to recover intensity
# that was never collected, so a ticked box becomes 5 and an unticked box 1 --
# the Likert extremes, which features.likert_to_percent maps to 100 and 0. This
# preserves the yes/no information exactly and invents no gradations.
# A model skill with no survey counterpart is left blank (median-imputed) and
# listed in the output as never-measured.
SKILL_MAP = {
    "tech_programming": "skill_tech_programming",
    "tech_web_development": "skill_tech_web_mobile_dev",
    # One checkbox covered "a website or Android app", so it feeds both.
    "tech_mobile_development": "skill_tech_web_mobile_dev",
    "tech_database_management": "skill_tech_database_sql",
    "tech_networking": "skill_tech_networking",
    # "Setting up a LAN, configuring routers or switches" is the closest thing
    # the survey has to systems administration.
    "tech_systems_administration": "skill_tech_networking",
    "tech_technical_troubleshooting": "skill_tech_it_support",
    # The survey listed data analysis under hard skills; the model calls it technical.
    "tech_data_analytics": "skill_hard_data_analysis",
    "hard_office_productivity": "skill_hard_office_productivity",
    # Encoding was not asked separately; the Excel/Word/Workspace box is the proxy.
    "hard_data_encoding": "skill_hard_office_productivity",
    "hard_project_management": "skill_hard_project_management",
    "hard_documentation": "skill_hard_research_writing",
    "hard_technical_writing": "skill_hard_research_writing",
    "soft_communication": "skill_soft_communication",
    "soft_teamwork": "skill_soft_teamwork",
    "soft_leadership": "skill_soft_leadership",
    "soft_problem_solving": "skill_soft_problem_solving",
    # "Finding and fixing an error in your program" is the analytical item asked.
    "soft_critical_thinking": "skill_soft_problem_solving",
    "soft_adaptability": "skill_soft_adaptability",
    # never measured: tech_cybersecurity, tech_cloud_computing,
    #                 hard_systems_analysis, hard_quality_assurance_testing,
    #                 soft_time_management, soft_customer_service
}

# Survey columns that intentionally feed no model column.
UNUSED_SURVEY_SKILLS = ["skill_hard_graphic_ui_design"]

# --------------------------------------------------------------------------- #
# 3. Certifications:  free-text certificate name -> cert_<domain> flag
# --------------------------------------------------------------------------- #
CERT_KEYWORDS = [
    (r"cyber|security", "security"),
    (r"blockchain|web3|\bsui\b|bull or bear|build the future", "programming"),
    (r"database|\bsql\b|oracle|mysql", "database"),
    (r"azure|\baws\b|cloud", "web_cloud"),
    (r"\bai\b|data analytic|data science|power ?bi|machine learning", "data_analytics"),
    (r"network|ccna|network\+", "networking"),
    (r"office|excel|workspace|microsoft word", "office_productivity"),
    (r"leadership|officership|project management|governance", "leadership_management"),
]


def mean_of(row: pd.Series, columns: list[str]) -> float | None:
    values = [row[c] for c in columns if c in row.index and pd.notna(row[c])]
    return round(sum(values) / len(values), 4) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    rows = []
    for _, src in clean.iterrows():
        course_map = BSCS_COURSE_MAP if src["program"] == "BSCS" else BSIT_COURSE_MAP
        rec = {
            "student_id": src["student_id"],
            "program": src["program"],
            "gwa": src["gwa"],
        }
        for course in cfg.CORE_COURSES:
            rec[f"grade_{course.key}"] = mean_of(src, course_map.get(course.key, []))

        names = "" if pd.isna(src["certification_names"]) else str(src["certification_names"])
        domains = {d for pattern, d in CERT_KEYWORDS if re.search(pattern, names, re.IGNORECASE)}
        for domain in cfg.CERT_DOMAINS:
            rec[f"cert_{domain}"] = int(domain in domains)

        # Never asked in the questionnaire. Constant 0 means the seminar term of
        # the EVT value score contributes nothing for every student.
        rec["seminar_units"] = 0

        for model_col in cfg.ALL_SKILL_KEYS:
            survey_col = SKILL_MAP.get(model_col)
            if survey_col is None or pd.isna(src[survey_col]):
                rec[model_col] = None
            else:
                rec[model_col] = 5 if int(src[survey_col]) == 1 else 1

        rows.append(rec)

    raw = pd.DataFrame(rows)

    # Fails loudly here rather than inside main.py if a mapping went wrong.
    feat.validate_raw_frame(raw, require_target=False)

    # ---- derive the label (see the module docstring: this is circular) ------
    engineered = feat.build_features(raw)
    labels, top_tracks = [], []
    for _, row in engineered.iterrows():
        best = align.score_career_tracks(row, category_probabilities=None)[0]
        labels.append(best.work_category)
        top_tracks.append(best.track.name)
    raw[cfg.TARGET_COLUMN] = labels

    feat.validate_raw_frame(raw, require_target=True)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.dst, index=False)

    # ----------------------------- audit ------------------------------------ #
    print(f"Wrote {args.dst}: {len(raw)} rows x {len(raw.columns)} columns\n")

    print("Course slots with no counterpart in the prospectus (left blank, median-imputed):")
    for program, cmap in (("BSCS", BSCS_COURSE_MAP), ("BSIT", BSIT_COURSE_MAP)):
        gaps = [c.key for c in cfg.CORE_COURSES if c.key not in cmap]
        print(f"  {program}: {', '.join(gaps) or 'none'}")
    used = set(sum(BSCS_COURSE_MAP.values(), []) + sum(BSIT_COURSE_MAP.values(), []))
    unused = [c for c in clean.columns if c.startswith("grade_") and c not in used]
    print(f"  survey course columns not used: {', '.join(unused) or 'none'}")

    never = [k for k in cfg.ALL_SKILL_KEYS if k not in SKILL_MAP]
    print(f"\nSkills the questionnaire never measured ({len(never)} of {len(cfg.ALL_SKILL_KEYS)}), left blank:")
    print(f"  {', '.join(never)}")
    print(f"  survey skill columns not used: {', '.join(UNUSED_SURVEY_SKILLS)}")
    shared = [m for m in SKILL_MAP if list(SKILL_MAP.values()).count(SKILL_MAP[m]) > 1]
    print(f"  model columns sharing a checkbox (perfectly correlated by construction): {', '.join(shared)}")

    cert_cols = [f"cert_{d}" for d in cfg.CERT_DOMAINS]
    print(f"\nCertification domains recognised from free text: {int(raw[cert_cols].to_numpy().sum())} flags set")
    for domain in cfg.CERT_DOMAINS:
        n = int(raw[f"cert_{domain}"].sum())
        if n:
            print(f"  cert_{domain}: {n}")
    said_yes = int((clean["has_certification"] == 1).sum())
    no_name = int(((clean["has_certification"] == 1) & clean["certification_names"].isna()).sum())
    print(f"  {said_yes} respondents reported a credential; {no_name} named none, so no domain could be set")

    print(f"\nDerived {cfg.TARGET_COLUMN} distribution:")
    counts = raw[cfg.TARGET_COLUMN].value_counts()
    for category in cfg.WORK_CATEGORIES:
        n = int(counts.get(category, 0))
        print(f"  {category:<12} {n:>4}  ({n / len(raw):5.1%})")
    smallest = counts.min()
    if len(counts) < 2:
        print("\n  [!] Only one class present. Classification is impossible: train/explain/ablation")
        print("      cannot run, and this is a property of the rule, not of your students.")
    elif smallest < 5:
        print(f"\n  [!] Smallest class has {smallest} member(s). A 70-30 stratified split with 5-fold CV")
        print("      needs several per class; expect SMOTE to skip it and CV folds to be unstable.")

    print("\nBest-matching career track (what the rule used):")
    for name, n in pd.Series(top_tracks).value_counts().items():
        print(f"  {name:<42} {n:>4}")

    print("\n  [!] work_category above is COMPUTED FROM the features, not observed.")
    print("      Accuracy from `main.py train` on this file measures rule recovery only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
