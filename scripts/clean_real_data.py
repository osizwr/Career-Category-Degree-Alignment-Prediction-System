#!/usr/bin/env python3
"""
Clean the Google Forms export (data/real-data.xlsx) into one tidy CSV.

    .venv/bin/python scripts/clean_real_data.py [input.xlsx] [output.csv]

One row per respondent. No respondent is silently dropped: rows that may need
excluding (not a 4th-year student, identical grades to another respondent,
off-scale grades, ambiguous answers) are kept and listed in the `flags` column
so the exclusion decision stays visible in the methodology. The only rows
removed are those without informed consent (none in the current export).

Cleaning rules
--------------
* Consent questions 1-2: rows without "Yes" to both are excluded; the columns
  are then dropped because they are constant.
* student_id: R001.. in sheet order; source_row points back to the .xlsx row.
* program: "Bachelor of Science in ... (BSCS)" -> "BSCS" / "BSIT".
* academic_status: spelling variants unified ("Third-year student" and
  "3rd year" -> "3rd year", "5th Year" -> "5th year", ...).
* is_enrolled_4th_year: screening question 3, Yes -> 1, No -> 0.
* Course grades: BSCS and BSIT answered different course lists, so each block
  keeps its own columns (grade_bscs_*, grade_bsit_*); the other program's block
  is blank by design. Text typos are repaired ("1 .75" -> 1.75,
  "2..25" -> 2.25). Values are otherwise kept as entered, including 4.00
  (conditional), 5.00 (failed) and off-scale values such as 1.6.
* gwa: recomputed as the unweighted mean of the respondent's non-blank course
  grades. The sheet's formula (=SUM(...)/17 or /24) skipped text grades and
  divided blanks in as zero, which understated GWA for four respondents.
  Note this is the mean of the surveyed courses, not a registrar GWA.
* Certifications: has_certification is question 7 (Yes -> 1, No -> 0, the
  ambiguous "Yes, No" -> blank). The credential-type answers were a
  multi-select split across three columns in option order, so they do not
  line up with the certificate names; they become one 0/1 column per type.
  Certificate names are joined with "; ".
* Skills (questions 10-12): multi-select answers split across five columns
  each. Every listed option becomes a 0/1 column. Typed-in "Other" answers are
  kept verbatim in skill_*_other, and also set the matching option's column
  when they plainly describe it (WRITE_IN_MAP). A respondent who skipped the
  whole question gets blanks, not zeros.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.utils import column_index_from_string

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "real-data.xlsx"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "real-data-clean.csv"
SHEET = "Form Responses 1"

# PSU grading scale (1.00 best, 3.00 pass, 4.00 conditional, 5.00 fail).
GRADE_SCALE = {1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0}

# (sheet column, header text in the sheet, output column). Headers are checked
# so a re-exported sheet with moved columns fails loudly instead of mislabelling.
BSCS_COURSES = [
    ("H", "Introduction to Computing", "grade_bscs_intro_computing"),
    ("I", "Data Structures and Algorithms", "grade_bscs_data_structures_algorithms"),
    ("J", "Information Management", "grade_bscs_information_management"),
    ("K", "Applications Development and Emerging Technologies", "grade_bscs_app_dev_emerging_tech"),
    ("L", "Discrete Structures 1", "grade_bscs_discrete_structures_1"),
    ("M", "Discrete Structures 2", "grade_bscs_discrete_structures_2"),
    ("N", "Object-oriented Programming", "grade_bscs_oop"),
    ("O", "Algorithms and Complexity", "grade_bscs_algorithms_complexity"),
    ("P", "Automata Theory and Formal Languages", "grade_bscs_automata_formal_languages"),
    ("Q", "Architecture and Organization", "grade_bscs_architecture_organization"),
    ("R", "Information Assurance and Security", "grade_bscs_info_assurance_security"),
    ("S", "Human Computer Interaction", "grade_bscs_hci"),
    ("T", "Networks and Communications", "grade_bscs_networks_communications"),
    ("U", "Operating Systems", "grade_bscs_operating_systems"),
    ("V", "Programming Languages", "grade_bscs_programming_languages"),
    ("W", "Software Engineering 1", "grade_bscs_software_engineering_1"),
    ("X", "Social Issues and Professional Practice", "grade_bscs_social_issues_prof_practice"),
]
BSIT_COURSES = [
    # Google Forms appended " 2" to this header because the BSCS block already
    # had the same question title; the course is Introduction to Computing.
    ("AV", "Introduction to Computing 2", "grade_bsit_intro_computing"),
    ("AW", "Computer Programming 1", "grade_bsit_programming_1"),
    ("AX", "Discrete Mathematics", "grade_bsit_discrete_mathematics"),
    ("AY", "Introduction to Human Computer Interaction", "grade_bsit_intro_hci"),
    ("AZ", "Computer Programming 2", "grade_bsit_programming_2"),
    ("BA", "Graphics and Visual Computing", "grade_bsit_graphics_visual_computing"),
    ("BB", "Data Structures and Algorithms", "grade_bsit_data_structures_algorithms"),
    ("BC", "IT Elective 1", "grade_bsit_it_elective_1"),
    ("BD", "IT Elective 2", "grade_bsit_it_elective_2"),
    ("BE", "Information Management 1", "grade_bsit_information_management_1"),
    ("BF", "Network Technologies 1", "grade_bsit_network_technologies_1"),
    ("BG", "Integrative Programming and Technologies 1", "grade_bsit_integrative_programming_1"),
    ("BH", "Systems Integration and Architecture 1", "grade_bsit_systems_integration_architecture_1"),
    ("BI", "Advanced Database Systems", "grade_bsit_advanced_database_systems"),
    ("BJ", "Network Technologies 2", "grade_bsit_network_technologies_2"),
    ("BK", "Information Assurance and Security 1", "grade_bsit_info_assurance_security_1"),
    ("BL", "Web Systems and Technologies 1", "grade_bsit_web_systems_1"),
    ("BM", "Multimedia Systems", "grade_bsit_multimedia_systems"),
    ("BN", "IT Elective 3", "grade_bsit_it_elective_3"),
    ("BO", "Social and Professional Issues", "grade_bsit_social_professional_issues"),
    ("BP", "Application Development and Emerging Technologies 1", "grade_bsit_app_dev_emerging_tech_1"),
    ("BQ", "Geographic Information System", "grade_bsit_gis"),
    ("BR", "Embedded System", "grade_bsit_embedded_systems"),
    ("BS", "Information Assurance and Security 2", "grade_bsit_info_assurance_security_2"),
]

# Multi-select skill questions: (first col, last col, [(option regex, output column)]).
# Regexes tolerate the form's own typos ("Buidling", "Settinng") and plural slips.
SKILL_GROUPS = {
    "tech": ("AF", "AJ", [
        (r"writing programs in python, java, c\+\+, javascript and other programming languages?", "skill_tech_programming"),
        (r"bui\w*ing a website or android app", "skill_tech_web_mobile_dev"),
        (r"designing tables and writing sql queries", "skill_tech_database_sql"),
        (r"set\w* up a lan, configuring routers or switches", "skill_tech_networking"),
        (r"helping users with computer problems, fixing hardware or software", "skill_tech_it_support"),
    ]),
    "hard": ("AK", "AO", [
        (r"analyzing data with excel or python, making charts", "skill_hard_data_analysis"),
        (r"making poster or app screen in canva or figma", "skill_hard_graphic_ui_design"),
        (r"advanced excel formulas, word, google workspace", "skill_hard_office_productivity"),
        (r"planning task using agile/scrum or trello", "skill_hard_project_management"),
        (r"writing research papers or technical documents", "skill_hard_research_writing"),
    ]),
    "soft": ("AP", "AT", [
        (r"presenting your project, writing a clear email or report", "skill_soft_communication"),
        (r"sharing tasks fairly in a group project", "skill_soft_teamwork"),
        (r"leading a group or organizing an activity", "skill_soft_leadership"),
        (r"finding and fixing an error in your program", "skill_soft_problem_solving"),
        (r"learning a new tool because the project needed it", "skill_soft_adaptability"),
    ]),
}

# Typed-in "Other" answers that plainly describe one of the listed options.
# Anything not listed here stays only in the skill_*_other text column.
WRITE_IN_MAP = {
    "programming in c, python, c#, c++": "skill_tech_programming",
    "db management and optimizing db": "skill_tech_database_sql",
    "web and mobile development": "skill_tech_web_mobile_dev",
    "networking and cybersecurity cryptography": "skill_tech_networking",
    "designing good user interface and user experience": "skill_hard_graphic_ui_design",
    "finding and fixing easy to miss problems": "skill_soft_problem_solving",
    "reporting and presentation": "skill_soft_communication",
}

# Q9 of the questionnaire. Every option the form offers gets a column, even
# when nobody in this batch picked it -- otherwise the next batch silently
# loses that answer. config.CERTIFICATION_TYPES holds the same five.
CERT_TYPES = [
    (r"certification examination", "cert_type_certification_exam"),
    (r"technical seminar", "cert_type_technical_seminar"),
    (r"workshop", "cert_type_workshop"),
    (r"training program", "cert_type_training_program"),
    (r"webinar", "cert_type_webinar"),
]

STATUS_MAP = {
    "fourth-year student": "4th year",
    "third-year student": "3rd year",
    "3rd year": "3rd year",
    "5th year": "5th year",
    "6th year": "6th year",
    "irregular": "Irregular",
}
PROGRAM_MAP = {
    "Bachelor of Science in Computer Science (BSCS)": "BSCS",
    "Bachelor of Science in Information Technology (BSIT)": "BSIT",
}


def squash(text) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return " ".join(str(text).split())


def col(letter: str) -> int:
    return column_index_from_string(letter) - 1


def parse_grade(value, where: str, log: list) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    repaired = re.sub(r"\.+", ".", re.sub(r"\s+", "", value))
    try:
        grade = float(repaired)
    except ValueError:
        raise ValueError(f"{where}: cannot read grade {value!r}") from None
    log.append(f"{where}: grade text {value!r} -> {grade}")
    return grade


def parse_skills(cells, options, group: str, where: str, log: list) -> dict:
    out = {name: 0 for _, name in options}
    other_col = f"skill_{group}_other"
    texts = [squash(c) for c in cells if c is not None and squash(c)]
    if not texts:
        return {**{k: pd.NA for k in out}, other_col: pd.NA}
    write_ins = []
    for text in texts:
        rest = text
        for pattern, name in options:
            if re.search(pattern, rest, re.IGNORECASE):
                out[name] = 1
                rest = re.sub(pattern, "", rest, flags=re.IGNORECASE)
        rest = rest.strip(" ,")
        if rest:
            write_ins.append(rest)
            mapped = WRITE_IN_MAP.get(rest.lower())
            if mapped:
                out[mapped] = 1
                log.append(f"{where}: write-in {rest!r} -> {mapped}")
    out[other_col] = "; ".join(write_ins) if write_ins else pd.NA
    return out


def main() -> int:
    ws_val = openpyxl.load_workbook(SRC, data_only=True)[SHEET]
    header = [squash(c.value) if c.value is not None else "" for c in ws_val[1]]

    for letter, expected, _ in BSCS_COURSES + BSIT_COURSES:
        if header[col(letter)] != expected:
            raise SystemExit(f"Column {letter}: expected {expected!r}, found {header[col(letter)]!r}")

    log: list[str] = []
    records = []
    excluded_no_consent = []
    for row_no, row in enumerate(ws_val.iter_rows(min_row=2, values_only=True), start=2):
        if all(v is None for v in row):
            continue
        if squash(row[col("B")]) != "Yes" or squash(row[col("C")]) != "Yes":
            excluded_no_consent.append(row_no)
            continue

        sid = f"R{len(records) + 1:03d}"
        where = f"{sid} (sheet row {row_no})"
        flags = []

        program = PROGRAM_MAP[squash(row[col("E")])]
        status_raw = squash(row[col("F")]).lower()
        if status_raw not in STATUS_MAP:
            raise SystemExit(f"{where}: unknown academic status {row[col('F')]!r}; add it to STATUS_MAP")
        enrolled = {"Yes": 1, "No": 0}[squash(row[col("D")])]
        if not enrolled:
            flags.append("not_4th_year")

        rec = {
            "student_id": sid,
            "source_row": row_no,
            "timestamp": row[col("A")].replace(microsecond=0),
            "program": program,
            "academic_status": STATUS_MAP[status_raw],
            "is_enrolled_4th_year": enrolled,
        }

        own, other = (BSCS_COURSES, BSIT_COURSES) if program == "BSCS" else (BSIT_COURSES, BSCS_COURSES)
        if any(row[col(letter)] is not None for letter, _, _ in other):
            raise SystemExit(f"{where}: {program} respondent has grades in the other program's block")
        grades = {}
        for letter, _, name in own:
            g = parse_grade(row[col(letter)], f"{where} {name}", log)
            grades[name] = g
            if g is None:
                flags.append(f"missing_grade:{name}")
            elif g not in GRADE_SCALE:
                flags.append(f"off_scale_grade:{name}={g:g}")
        present = [g for g in grades.values() if g is not None]
        rec["gwa"] = round(sum(present) / len(present), 4)
        sheet_gwa = row[col("G") if program == "BSCS" else col("AU")]
        if sheet_gwa is None or round(sheet_gwa, 4) != rec["gwa"]:
            log.append(
                f"{where}: gwa {sheet_gwa} in sheet -> {rec['gwa']} recomputed "
                f"(mean of {len(present)} grades)"
            )
        for _, _, name in BSCS_COURSES + BSIT_COURSES:
            rec[name] = grades.get(name)

        cert_answer = squash(row[col("Y")])
        rec["has_certification"] = {"Yes": 1, "No": 0}.get(cert_answer, pd.NA)
        if rec["has_certification"] is pd.NA:
            flags.append(f"ambiguous_cert_answer:{cert_answer}")
        names = [squash(row[col(c)]).strip(" ,") for c in ("Z", "AA", "AB") if row[col(c)] is not None]
        rec["certification_names"] = "; ".join(n for n in names if n) or pd.NA
        type_text = " | ".join(squash(row[col(c)]) for c in ("AC", "AD", "AE") if row[col(c)] is not None)
        for pattern, name in CERT_TYPES:
            rec[name] = int(bool(re.search(pattern, type_text, re.IGNORECASE)))
        if cert_answer == "No" and (names or type_text):
            flags.append("cert_details_but_answered_no")

        for group, (first, last, options) in SKILL_GROUPS.items():
            cells = row[col(first): col(last) + 1]
            parsed = parse_skills(cells, options, group, where, log)
            if parsed[options[0][1]] is pd.NA:
                flags.append(f"skipped_{group}_skills")
            elif not any(parsed[name] for _, name in options):
                # Answered, but only with free text that matches no listed
                # option: the 0s mean "no listed option", not "no skills".
                flags.append(f"only_write_in_{group}_skills")
            rec.update(parsed)

        rec["flags"] = flags
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Identical grade vectors are very unlikely by chance: flag both sides.
    grade_cols = [n for _, _, n in BSCS_COURSES + BSIT_COURSES]
    key = df[["program"] + grade_cols].apply(lambda r: "|".join(map(str, r)), axis=1)
    for _, ids in df.groupby(key)["student_id"]:
        if len(ids) > 1:
            for idx in ids.index:
                others = ",".join(i for i in ids if i != df.at[idx, "student_id"])
                df.at[idx, "flags"] = df.at[idx, "flags"] + [f"same_grades_as:{others}"]

    df["flags"] = df["flags"].map(lambda f: "; ".join(f) if f else pd.NA)
    int_cols = [c for c in df.columns if c.startswith(("skill_", "cert_type_")) and not c.endswith("_other")]
    int_cols += ["has_certification", "is_enrolled_4th_year"]
    df[int_cols] = df[int_cols].astype("Int64")

    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DST, index=False)

    print(f"Read {SRC.name}: {len(records) + len(excluded_no_consent)} responses")
    print(f"Excluded for missing consent: {len(excluded_no_consent)} {excluded_no_consent or ''}")
    print(f"Wrote {DST}: {len(df)} rows x {len(df.columns)} columns")
    print(f"\nRepairs ({len(log)}):")
    for line in log:
        print("  " + line)
    flagged = df[df["flags"].notna()]
    print(f"\nFlagged rows ({len(flagged)}):")
    for _, r in flagged.iterrows():
        print(
            f"  {r['student_id']} (sheet row {r['source_row']}, {r['program']}, "
            f"{r['academic_status']}): {r['flags']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
