"""
Central configuration for the PSU Explainable Career Category Recommendation System.

Everything the researchers are expected to EDIT before the final defense lives
in this one file:

  * BSCS_COURSES / BSIT_COURSES ... the two prospectuses, exactly as printed in
                                    the Student Profile and Skills Questionnaire
  * ACADEMIC_DOMAINS ............ the 10 shared competency domains both
                                  prospectuses are folded into, so a BSCS and a
                                  BSIT profile become comparable to the model
  * CERTIFICATION_TYPES ......... Q9 of the questionnaire
  * CERTIFICATION_DOMAINS ....... keyword groups used to read Q8's free text
  * TECHNICAL/HARD/SOFT_SKILLS .. the Part IV checkbox inventories (Q10-Q12)
  * WORK_CATEGORIES ............. the four broad work categories (the ML target)
  * WORK_CATEGORY_PROFILES ...... the labelling rubric of Chapter IV
  * CAREER_TRACKS ............... specific job roles + their competency profiles
  * PALAWAN_EMPLOYERS ........... local internship / employment destinations
  * DISCIPLINE_PROFILES ......... BSCS vs BSIT weights used for % degree alignment

Reference: Cajilo, Donghil, Mahinay & Sibayan (2026), "An Explainable Career
Category Recommendation System for Computer Science and Information Technology
Graduating Students using Ensemble and Baseline Classifiers", College of
Sciences, Palawan State University.

WHAT CHANGED WHEN THE QUESTIONNAIRE ARRIVED
-------------------------------------------
The earlier version of this file guessed at the instrument: a single CMO-style
core-course list shared by both programs, 1-5 Likert skill ratings, and an
open-ended "seminar units" count. The approved questionnaire does none of those
things. It asks each program for its own prospectus, it asks for skills as
CHECKBOXES (you either ticked it or you did not), and it asks for certifications
as a yes/no plus a free-text name and a type. This file now mirrors the
instrument item for item; nothing is asked of a student that the form does not
collect, and nothing the form collects is thrown away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

DATASET_PATH = DATA_DIR / "psu_computing_students.csv"

for _d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Experiment settings  (these mirror Chapter IV of the thesis)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.30        # "a 70% training set and a 30% held-out test set"
CV_FOLDS = 5            # "Stratified K-Fold Cross-Validation (k=5)"


# --------------------------------------------------------------------------- #
# Grading scale
# --------------------------------------------------------------------------- #
# Philippine state-university scale: 1.00 is the HIGHEST mark, 3.00 is the
# passing mark, 5.00 is a failure. Because "lower is better" is the opposite of
# what most ML models assume, every grade is converted to 0-100 "grade points"
# (higher = better) before it reaches a model. See grade_to_points().
GRADE_BEST = 1.00
GRADE_PASS = 3.00
GRADE_WORST = 5.00
LOWER_GRADE_IS_BETTER = True


def grade_to_points(grade):
    """Convert a 1.00-5.00 grade into 0-100 points where higher = better.

    1.00 -> 100.0   (excellent)
    1.75 ->  81.25
    3.00 ->  50.0   (passing)
    5.00 ->   0.0   (failed)
    """
    grade = np.asarray(grade, dtype=float)
    points = (GRADE_WORST - grade) / (GRADE_WORST - GRADE_BEST) * 100.0
    return np.clip(points, 0.0, 100.0)


def points_to_grade(points):
    """Inverse of grade_to_points(): 0-100 points back to a 1.00-5.00 grade."""
    points = np.asarray(points, dtype=float)
    return GRADE_WORST - (points / 100.0) * (GRADE_WORST - GRADE_BEST)


# --------------------------------------------------------------------------- #
# Degree programs
# --------------------------------------------------------------------------- #
PROGRAMS = ("BSCS", "BSIT")

PROGRAM_LABELS = {
    "BSCS": "BS Computer Science",
    "BSIT": "BS Information Technology",
}


# --------------------------------------------------------------------------- #
# Shared academic competency domains
# --------------------------------------------------------------------------- #
# BSCS reports 17 courses and BSIT reports 24 DIFFERENT courses, so a raw
# per-course feature matrix would be half-empty for every student and the model
# could identify the program from the pattern of blanks alone. Instead every
# course is tagged with one or more of the domains below, and the model sees the
# student's MEAN GRADE POINTS PER DOMAIN. Each domain is covered by both
# prospectuses, so `acad_networking` means the same thing for a BSCS and a BSIT
# student even though it is computed from different courses.
#
# This is also what the thesis's labelling rubric means by "grades in courses
# directly aligned to a given career category (e.g. networking courses ->
# Cybersecurity track)".
ACADEMIC_DOMAINS: dict[str, str] = {
    "foundations":            "Computing Foundations",
    "programming":            "Programming & Software Construction",
    "theory_analytics":       "Algorithmic Thinking & Analytics",
    "data_management":        "Data & Database Management",
    "networking":             "Networking & Communications",
    "security":               "Information Assurance & Security",
    "web_design":             "Web, Interface & Multimedia",
    "systems_engineering":    "Systems Analysis & Engineering",
    "infrastructure_support": "Infrastructure & Platform Technologies",
    "professional_practice":  "Social & Professional Practice",
}

DOMAIN_KEYS = tuple(ACADEMIC_DOMAINS)
ACAD_FEATURE_KEYS = tuple(f"acad_{d}" for d in DOMAIN_KEYS)


# --------------------------------------------------------------------------- #
# Prospectus-based core courses, per program  (Q6 of the questionnaire)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Course:
    """One prospectus course, exactly as it appears on the questionnaire."""

    key: str            # snake_case; the CSV column is grade_<program>_<key>
    title: str          # the course title as printed on the form
    program: str        # 'BSCS' or 'BSIT'
    domains: tuple      # which ACADEMIC_DOMAINS this course feeds

    @property
    def column(self) -> str:
        return f"grade_{self.program.lower()}_{self.key}"


# -- BSCS prospectus: 17 courses, in questionnaire order --------------------- #
BSCS_COURSES: tuple[Course, ...] = (
    Course("intro_computing", "Introduction to Computing", "BSCS", ("foundations",)),
    Course("data_structures_algorithms", "Data Structures and Algorithms", "BSCS",
           ("programming", "theory_analytics")),
    Course("information_management", "Information Management", "BSCS", ("data_management",)),
    Course("app_dev_emerging_tech", "Applications Development and Emerging Technologies", "BSCS",
           ("programming", "systems_engineering")),
    Course("discrete_structures_1", "Discrete Structures 1", "BSCS", ("theory_analytics",)),
    Course("discrete_structures_2", "Discrete Structures 2", "BSCS", ("theory_analytics",)),
    Course("oop", "Object-oriented Programming", "BSCS", ("programming",)),
    Course("algorithms_complexity", "Algorithms and Complexity", "BSCS", ("theory_analytics",)),
    Course("automata_formal_languages", "Automata Theory and Formal Languages", "BSCS",
           ("theory_analytics",)),
    Course("architecture_organization", "Architecture and Organization", "BSCS",
           ("infrastructure_support",)),
    Course("info_assurance_security", "Information Assurance and Security", "BSCS", ("security",)),
    Course("hci", "Human Computer Interaction", "BSCS", ("web_design",)),
    Course("networks_communications", "Networks and Communications", "BSCS", ("networking",)),
    Course("operating_systems", "Operating Systems", "BSCS", ("infrastructure_support",)),
    Course("programming_languages", "Programming Languages", "BSCS", ("programming",)),
    Course("software_engineering_1", "Software Engineering 1", "BSCS", ("systems_engineering",)),
    Course("social_issues_prof_practice", "Social Issues and Professional Practice", "BSCS",
           ("professional_practice",)),
)

# -- BSIT prospectus: 24 courses, in questionnaire order --------------------- #
# The three IT electives are student-chosen, so they carry no domain: they are
# still collected and reported, but they do not feed a comparable domain score.
BSIT_COURSES: tuple[Course, ...] = (
    Course("intro_computing", "Introduction to Computing", "BSIT", ("foundations",)),
    Course("programming_1", "Computer Programming 1", "BSIT", ("programming",)),
    Course("discrete_mathematics", "Discrete Mathematics", "BSIT", ("theory_analytics",)),
    Course("intro_hci", "Introduction to Human Computer Interaction", "BSIT", ("web_design",)),
    Course("programming_2", "Computer Programming 2", "BSIT", ("programming",)),
    Course("graphics_visual_computing", "Graphics and Visual Computing", "BSIT", ("web_design",)),
    Course("data_structures_algorithms", "Data Structures and Algorithms", "BSIT",
           ("programming", "theory_analytics")),
    Course("it_elective_1", "IT Elective 1", "BSIT", ()),
    Course("it_elective_2", "IT Elective 2", "BSIT", ()),
    Course("information_management_1", "Information Management 1", "BSIT", ("data_management",)),
    Course("network_technologies_1", "Network Technologies 1", "BSIT", ("networking",)),
    Course("integrative_programming_1", "Integrative Programming and Technologies 1", "BSIT",
           ("programming", "systems_engineering")),
    Course("systems_integration_architecture_1", "Systems Integration and Architecture 1", "BSIT",
           ("systems_engineering",)),
    Course("advanced_database_systems", "Advanced Database Systems", "BSIT", ("data_management",)),
    Course("network_technologies_2", "Network Technologies 2", "BSIT", ("networking",)),
    Course("info_assurance_security_1", "Information Assurance and Security 1", "BSIT", ("security",)),
    Course("web_systems_1", "Web Systems and Technologies 1", "BSIT", ("web_design",)),
    Course("multimedia_systems", "Multimedia Systems", "BSIT", ("web_design",)),
    Course("it_elective_3", "IT Elective 3", "BSIT", ()),
    Course("social_professional_issues", "Social and Professional Issues", "BSIT",
           ("professional_practice",)),
    Course("app_dev_emerging_tech_1", "Application Development and Emerging Technologies 1", "BSIT",
           ("programming", "systems_engineering")),
    Course("gis", "Geographic Information System", "BSIT", ("data_management",)),
    Course("embedded_systems", "Embedded System", "BSIT", ("infrastructure_support",)),
    Course("info_assurance_security_2", "Information Assurance and Security 2", "BSIT", ("security",)),
)

COURSES_BY_PROGRAM: dict[str, tuple[Course, ...]] = {
    "BSCS": BSCS_COURSES,
    "BSIT": BSIT_COURSES,
}

ALL_COURSES: tuple[Course, ...] = BSCS_COURSES + BSIT_COURSES
GRADE_COLUMNS: tuple[str, ...] = tuple(c.column for c in ALL_COURSES)
COURSE_BY_COLUMN: dict[str, Course] = {c.column: c for c in ALL_COURSES}


def domain_courses(program: str, domain: str) -> tuple[Course, ...]:
    """Every course of one program that feeds one academic domain."""
    return tuple(c for c in COURSES_BY_PROGRAM[program] if domain in c.domains)


def _check_domain_coverage() -> None:
    """Both prospectuses must cover every domain, or the feature is not comparable."""
    for program in PROGRAMS:
        missing = [d for d in DOMAIN_KEYS if not domain_courses(program, d)]
        if missing:
            raise ValueError(
                f"{program} has no course feeding academic domain(s) {missing}. "
                f"Every domain must be covered by both prospectuses, otherwise "
                f"`acad_{missing[0]}` is structurally blank for all {program} "
                f"students and the model can read the program off the blanks. "
                f"Either tag another {program} course with that domain or remove "
                f"the domain from ACADEMIC_DOMAINS."
            )


_check_domain_coverage()


# --------------------------------------------------------------------------- #
# Certifications, seminars and training  (Part III, Q7-Q9)
# --------------------------------------------------------------------------- #
# Q9 asks what KIND of credential it was. Each becomes a 0/1 feature.
CERTIFICATION_TYPES: dict[str, str] = {
    "certification_exam": "Certification examination",
    "technical_seminar": "Technical seminar",
    "workshop": "Workshop",
    "training_program": "Training program",
    "webinar": "Webinar",
}

CERT_TYPE_KEYS = tuple(f"cert_type_{t}" for t in CERTIFICATION_TYPES)

# Q8 asks for the credential's NAME as free text. These keyword groups turn that
# text into subject-domain flags, which is what lets a certificate contribute to
# the right competency signal instead of being a single undifferentiated count.
# Add patterns here as new credential names appear in the responses.
CERTIFICATION_DOMAINS: dict[str, list[str]] = {
    "networking": ["Cisco CCNA", "CompTIA Network+", "Cisco NetAcad Networking Essentials"],
    "security": ["CompTIA Security+", "Cisco CyberOps Associate", "Cybersecurity Seminar/Workshop"],
    "programming": ["Oracle Java Certification", "Python Programming Certificate", "C++/C# Certification"],
    "data_analytics": ["Google Data Analytics", "Microsoft Power BI", "AI / Data Science Seminar"],
    "web_cloud": ["AWS Cloud Practitioner", "Microsoft Azure Fundamentals", "Web Development Bootcamp"],
    "database": ["Oracle Database SQL", "MySQL Certification", "Database Administration Seminar"],
    "office_productivity": ["Microsoft Office Specialist", "Google Workspace Certificate"],
    "leadership_management": ["Leadership Training / LDT", "Project Management Seminar", "Student Org Officership"],
}

CERT_DOMAINS = tuple(CERTIFICATION_DOMAINS)
CERT_FEATURE_KEYS = tuple(f"cert_{d}" for d in CERT_DOMAINS)

# Free-text name -> subject domain. Used by scripts/build_model_dataset.py and by
# the interactive questionnaire, so both read a credential the same way.
CERT_KEYWORDS: tuple[tuple[str, str], ...] = (
    (r"cyber|security|hacking|forensic", "security"),
    (r"blockchain|web3|\bsui\b|bull or bear|build the future", "programming"),
    (r"database|\bsql\b|oracle|mysql|postgres", "database"),
    (r"azure|\baws\b|cloud|devops", "web_cloud"),
    (r"\bai\b|artificial intelligence|data analytic|data science|power ?bi|machine learning", "data_analytics"),
    (r"network|ccna|comptia|routing|switching", "networking"),
    (r"office|excel|workspace|microsoft word|spreadsheet", "office_productivity"),
    (r"leadership|officership|project management|governance|scrum|agile", "leadership_management"),
    (r"python|java|c\+\+|programming|software dev|web dev|android|flutter", "programming"),
)


# --------------------------------------------------------------------------- #
# Self-reported skills  (Part IV, Q10-Q12)  --  CHECKBOXES, not ratings
# --------------------------------------------------------------------------- #
# The instrument says "Check all the skills you have actually used", so every
# skill is BINARY: 1 = ticked, 0 = not ticked, blank = the student skipped the
# whole question. Do not reintroduce a 1-5 scale here; intensity was never
# collected and inventing it would put numbers in Chapter V that no student
# ever reported.
SKILL_PRESENT, SKILL_ABSENT = 1, 0


@dataclass(frozen=True)
class Skill:
    key: str        # column is <group>_<key>, e.g. tech_programming
    group: str      # 'tech' | 'hard' | 'soft'
    label: str      # short name for tables and SHAP plots
    prompt: str     # the checkbox text, verbatim from the questionnaire


TECHNICAL_SKILLS: tuple[Skill, ...] = (
    Skill("programming", "tech", "Programming",
          "Writing programs in Python, Java, C++, JavaScript and other programming languages"),
    Skill("web_mobile_dev", "tech", "Web / Mobile Development",
          "Building a website or Android app"),
    Skill("database_sql", "tech", "Database & SQL",
          "Designing tables and writing SQL queries"),
    Skill("networking", "tech", "Networking",
          "Setting up a LAN, configuring routers or switches"),
    Skill("it_support", "tech", "IT Support & Troubleshooting",
          "Helping users with computer problems, fixing hardware or software"),
)

HARD_SKILLS: tuple[Skill, ...] = (
    Skill("data_analysis", "hard", "Data Analysis",
          "Analyzing data with Excel or Python, making charts"),
    Skill("graphic_ui_design", "hard", "Graphic / UI-UX Design",
          "Making a poster or app screen in Canva or Figma"),
    Skill("office_productivity", "hard", "Office Productivity",
          "Advanced Excel formulas, Word, Google Workspace"),
    Skill("project_management", "hard", "Project Management",
          "Planning tasks using Agile/Scrum or Trello"),
    Skill("research_writing", "hard", "Research & Technical Writing",
          "Writing research papers or technical documents"),
)

SOFT_SKILLS: tuple[Skill, ...] = (
    Skill("communication", "soft", "Communication",
          "Presenting your project, writing a clear email or report"),
    Skill("teamwork", "soft", "Teamwork",
          "Sharing tasks fairly in a group project"),
    Skill("leadership", "soft", "Leadership",
          "Leading a group or organizing an activity"),
    Skill("problem_solving", "soft", "Problem Solving",
          "Finding and fixing an error in your program"),
    Skill("adaptability", "soft", "Adaptability",
          "Learning a new tool because the project needed it"),
)

ALL_SKILLS: tuple[Skill, ...] = TECHNICAL_SKILLS + HARD_SKILLS + SOFT_SKILLS

SKILL_GROUPS: dict[str, tuple[Skill, ...]] = {
    "tech": TECHNICAL_SKILLS,
    "hard": HARD_SKILLS,
    "soft": SOFT_SKILLS,
}

SKILL_GROUP_LABELS = {
    "tech": "Technical skill",
    "hard": "Hard skill",
    "soft": "Soft skill",
}

TECH_SKILL_KEYS = tuple(f"tech_{s.key}" for s in TECHNICAL_SKILLS)
HARD_SKILL_KEYS = tuple(f"hard_{s.key}" for s in HARD_SKILLS)
SOFT_SKILL_KEYS = tuple(f"soft_{s.key}" for s in SOFT_SKILLS)
ALL_SKILL_KEYS = TECH_SKILL_KEYS + HARD_SKILL_KEYS + SOFT_SKILL_KEYS

SKILL_BY_COLUMN: dict[str, Skill] = {f"{s.group}_{s.key}": s for s in ALL_SKILLS}


# --------------------------------------------------------------------------- #
# Competency signals
# --------------------------------------------------------------------------- #
# A "signal" is a 0-100 composite score built from the domain grade points, the
# ticked skills and the recognised certificates. Signals are the shared
# vocabulary used by four parts of the system, which is what keeps the whole
# thing internally consistent:
#
#   1. RIASEC feature engineering  (Holland's Theory of Vocational Choice)
#   2. the work-category labelling rubric of Chapter IV
#   3. career-track match scoring  (the "Match Score" in the mockups)
#   4. BSCS / BSIT degree alignment %
@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    domains: tuple = ()          # keys from ACADEMIC_DOMAINS
    tech_skills: tuple = ()      # keys from TECHNICAL_SKILLS
    hard_skills: tuple = ()      # keys from HARD_SKILLS
    soft_skills: tuple = ()      # keys from SOFT_SKILLS
    certs: tuple = ()            # keys from CERT_DOMAINS


SIGNALS: tuple[Signal, ...] = (
    Signal(
        "programming", "Programming & Software Construction",
        domains=("programming",),
        tech_skills=("programming", "web_mobile_dev"),
        certs=("programming",),
    ),
    Signal(
        "theory_analytics", "Algorithmic Thinking & Analytics",
        domains=("theory_analytics",),
        hard_skills=("data_analysis",),
        soft_skills=("problem_solving",),
        certs=("data_analytics",),
    ),
    Signal(
        "data_management", "Data & Database Management",
        domains=("data_management",),
        tech_skills=("database_sql",),
        hard_skills=("data_analysis",),
        certs=("database",),
    ),
    Signal(
        "networking", "Networking & Communications",
        domains=("networking",),
        tech_skills=("networking",),
        certs=("networking",),
    ),
    Signal(
        "security", "Information Assurance & Security",
        domains=("security", "networking"),
        tech_skills=("networking",),
        certs=("security",),
    ),
    Signal(
        "web_design", "Web, Interface & Multimedia",
        domains=("web_design",),
        tech_skills=("web_mobile_dev",),
        hard_skills=("graphic_ui_design",),
        certs=("web_cloud",),
    ),
    Signal(
        "systems_engineering", "Systems Analysis & Engineering",
        domains=("systems_engineering", "foundations"),
        hard_skills=("project_management", "research_writing"),
    ),
    Signal(
        "infrastructure_support", "Infrastructure & Technical Support",
        domains=("infrastructure_support",),
        tech_skills=("it_support", "networking"),
    ),
    Signal(
        "documentation_clerical", "Documentation & Office Operations",
        domains=("foundations",),
        hard_skills=("office_productivity", "research_writing"),
        certs=("office_productivity",),
    ),
    Signal(
        "leadership_management", "Leadership & Project Management",
        domains=("professional_practice",),
        hard_skills=("project_management",),
        soft_skills=("leadership",),
        certs=("leadership_management",),
    ),
    Signal(
        "communication_service", "Communication & Client Service",
        domains=("professional_practice",),
        hard_skills=("research_writing",),
        soft_skills=("communication", "teamwork", "adaptability"),
    ),
)

SIGNAL_KEYS = tuple(s.key for s in SIGNALS)
SIGNAL_BY_KEY = {s.key: s for s in SIGNALS}
SIGNAL_FEATURE_KEYS = tuple(f"sig_{k}" for k in SIGNAL_KEYS)


# --------------------------------------------------------------------------- #
# Holland's RIASEC mapping (Chapter III: Theory of Vocational Choice)
# --------------------------------------------------------------------------- #
# Each RIASEC archetype is the mean of the competency signals that express it.
RIASEC_MAP: dict[str, tuple[str, ...]] = {
    "realistic":     ("networking", "infrastructure_support"),
    "investigative": ("theory_analytics", "programming", "security"),
    "artistic":      ("web_design",),
    "social":        ("communication_service",),
    "enterprising":  ("leadership_management",),
    "conventional":  ("documentation_clerical", "data_management"),
}

RIASEC_FEATURE_KEYS = tuple(f"riasec_{k}" for k in RIASEC_MAP)


# --------------------------------------------------------------------------- #
# Broad work categories  ->  the machine-learning TARGET variable
# --------------------------------------------------------------------------- #
# Chapter I, Career Category Classification: "the process of assigning students
# to broad employment categories such as technical, clerical, supervisory, or
# managerial roles."
WORK_CATEGORIES: tuple[str, ...] = ("Technical", "Clerical", "Supervisory", "Managerial")

WORK_CATEGORY_DESCRIPTIONS = {
    "Technical": "Hands-on engineering, development, infrastructure and security roles.",
    "Clerical":  "Records, encoding, documentation and front-line support roles.",
    "Supervisory": "Team lead and coordination roles overseeing technical staff.",
    "Managerial": "Planning, resourcing and unit-level IT management roles.",
}

TARGET_COLUMN = "work_category"


# --------------------------------------------------------------------------- #
# The labelling rubric  (Chapter IV, "Scoring Criteria")
# --------------------------------------------------------------------------- #
# "A student's profile is scored against each candidate career category, and the
#  category with the highest composite score is assigned as that student's
#  preliminary label."
#
#   Core course performance ...... 30%
#   GWA .......................... 20%
#   Certifications / seminars .... 25%
#   Self-reported skills ......... 25%
#
# READ THIS BEFORE QUOTING ANY ACCURACY FIGURE --------------------------------
# The rubric scores a student on the SAME variables the classifier uses as
# predictors. Any label produced by it is therefore a deterministic function of
# the features, and a model trained on it is measuring whether it can re-learn
# this formula -- not whether it can predict real careers. Use
# `labelling.assign_work_categories` only to bootstrap a pipeline, and state the
# circularity plainly in Chapter V. Defensible labels have to come from an
# observed outcome: the researchers' independent manual coding described in
# "Qualified Labelers", an OJT placement record, or a graduate tracer study.
# -----------------------------------------------------------------------------
RUBRIC_WEIGHTS = {
    "core_courses": 0.30,
    "gwa": 0.20,
    "certifications": 0.25,
    "skills": 0.25,
}


@dataclass(frozen=True)
class WorkCategoryProfile:
    """What evidence counts toward one work category in the labelling rubric."""

    domains: dict        # academic domain key -> weight
    certs: dict          # certification domain key -> weight
    skills: dict         # skill column -> weight


WORK_CATEGORY_PROFILES: dict[str, WorkCategoryProfile] = {
    "Technical": WorkCategoryProfile(
        domains={"programming": 0.24, "theory_analytics": 0.14, "networking": 0.14,
                 "security": 0.12, "data_management": 0.12, "infrastructure_support": 0.12,
                 "web_design": 0.08, "systems_engineering": 0.04},
        certs={"programming": 0.22, "networking": 0.20, "security": 0.18,
               "database": 0.16, "web_cloud": 0.14, "data_analytics": 0.10},
        skills={"tech_programming": 0.24, "tech_web_mobile_dev": 0.18,
                "tech_database_sql": 0.18, "tech_networking": 0.16,
                "tech_it_support": 0.12, "soft_problem_solving": 0.12},
    ),
    "Clerical": WorkCategoryProfile(
        domains={"foundations": 0.26, "data_management": 0.24, "web_design": 0.16,
                 "professional_practice": 0.16, "infrastructure_support": 0.10,
                 "systems_engineering": 0.08},
        certs={"office_productivity": 0.46, "database": 0.26, "data_analytics": 0.16,
               "networking": 0.12},
        skills={"hard_office_productivity": 0.30, "hard_research_writing": 0.20,
                "tech_it_support": 0.18, "soft_communication": 0.18,
                "soft_teamwork": 0.14},
    ),
    "Supervisory": WorkCategoryProfile(
        domains={"professional_practice": 0.30, "systems_engineering": 0.22,
                 "infrastructure_support": 0.16, "networking": 0.14,
                 "data_management": 0.10, "foundations": 0.08},
        certs={"leadership_management": 0.44, "networking": 0.20, "security": 0.18,
               "office_productivity": 0.18},
        skills={"hard_project_management": 0.26, "soft_leadership": 0.24,
                "soft_teamwork": 0.18, "soft_communication": 0.18,
                "soft_adaptability": 0.14},
    ),
    "Managerial": WorkCategoryProfile(
        domains={"professional_practice": 0.32, "systems_engineering": 0.26,
                 "data_management": 0.16, "foundations": 0.14, "security": 0.12},
        certs={"leadership_management": 0.46, "office_productivity": 0.22,
               "data_analytics": 0.18, "security": 0.14},
        skills={"soft_leadership": 0.26, "hard_project_management": 0.22,
                "hard_research_writing": 0.18, "soft_communication": 0.18,
                "soft_adaptability": 0.16},
    ),
}


# --------------------------------------------------------------------------- #
# Specific career tracks (the ranked "Top Career Recommendations" panel)
# --------------------------------------------------------------------------- #
# Chapter IV: "each career recommendation is presented at two levels: the broad
# career work category predicted by the XGBoost model, and an illustrative job
# role within that category." The work category is the ML target; these tracks
# are the illustrative roles.
@dataclass(frozen=True)
class CareerTrack:
    key: str
    name: str
    work_category: str
    description: str
    weights: dict          # signal key -> weight (normalised automatically)
    sectors: tuple         # keys into PALAWAN_EMPLOYERS
    certifications: tuple  # recommended certifications to pursue
    actions: tuple         # "suggested actions" chips


CAREER_TRACKS: tuple[CareerTrack, ...] = (
    # ---------------------------------------------------------- Technical ---
    CareerTrack(
        "software_developer", "Software Developer", "Technical",
        "Design, develop and maintain software applications and systems.",
        {"programming": 0.40, "systems_engineering": 0.20, "theory_analytics": 0.20,
         "web_design": 0.12, "data_management": 0.08},
        ("private_tech", "it_bpm", "government"),
        ("Oracle Java Certification", "Python Programming Certificate"),
        ("Build Projects", "Create Portfolio", "Join Internships", "Learn Full-Stack", "Earn Certifications"),
    ),
    CareerTrack(
        "data_analyst", "Data Analyst", "Technical",
        "Collect, analyze and interpret data to help organizations make informed decisions.",
        {"theory_analytics": 0.35, "data_management": 0.32, "programming": 0.15,
         "systems_engineering": 0.10, "documentation_clerical": 0.08},
        ("private_tech", "government", "finance", "academe_utilities"),
        ("Google Data Analytics", "Microsoft Power BI"),
        ("Learn Power BI", "Learn Tableau", "Master SQL", "Build Data Projects", "Take Analytics Course"),
    ),
    CareerTrack(
        "network_administrator", "Network Administrator", "Technical",
        "Manage and maintain network infrastructure, security and system operations.",
        {"networking": 0.38, "infrastructure_support": 0.30, "security": 0.20,
         "systems_engineering": 0.07, "documentation_clerical": 0.05},
        ("academe_utilities", "government", "tourism_hospitality", "it_bpm"),
        ("Cisco CCNA", "CompTIA Network+"),
        ("Take CCNA", "Study Security+", "Learn Linux Administration", "Practice Network Configuration",
         "Gain Industry Experience"),
    ),
    CareerTrack(
        "cybersecurity_analyst", "Cybersecurity Analyst", "Technical",
        "Protect institutional systems and data against threats and intrusions.",
        {"security": 0.42, "networking": 0.26, "infrastructure_support": 0.16,
         "theory_analytics": 0.10, "documentation_clerical": 0.06},
        ("government", "finance", "academe_utilities", "it_bpm"),
        ("CompTIA Security+", "Cisco CyberOps Associate"),
        ("Study Security+", "Practice on TryHackMe", "Learn Incident Response", "Join CTF Competitions"),
    ),
    CareerTrack(
        "web_developer", "Web Developer", "Technical",
        "Build and maintain responsive websites and web-based information systems.",
        {"web_design": 0.42, "programming": 0.30, "data_management": 0.14,
         "systems_engineering": 0.08, "communication_service": 0.06},
        ("private_tech", "tourism_hospitality", "government"),
        ("Web Development Bootcamp", "AWS Cloud Practitioner"),
        ("Build a Portfolio Site", "Learn React/Laravel", "Practice Responsive Design", "Freelance Projects"),
    ),
    CareerTrack(
        "database_administrator", "Database Administrator", "Technical",
        "Design, secure and tune institutional databases and reporting pipelines.",
        {"data_management": 0.44, "security": 0.16, "infrastructure_support": 0.16,
         "theory_analytics": 0.14, "documentation_clerical": 0.10},
        ("academe_utilities", "government", "finance"),
        ("Oracle Database SQL", "MySQL Certification"),
        ("Master SQL", "Learn Database Tuning", "Study Backup & Recovery", "Take Oracle Certification"),
    ),
    CareerTrack(
        "qa_tester", "QA / Software Tester", "Technical",
        "Validate software quality through structured test design and defect reporting.",
        {"systems_engineering": 0.34, "programming": 0.22, "documentation_clerical": 0.22,
         "theory_analytics": 0.12, "communication_service": 0.10},
        ("private_tech", "it_bpm"),
        ("ISTQB Foundation Seminar", "Automation Testing Workshop"),
        ("Write Test Cases", "Learn Selenium", "Study ISTQB", "Join QA Internship"),
    ),
    CareerTrack(
        "technical_support", "Technical Support Specialist", "Technical",
        "Resolve hardware, software and connectivity issues for end users.",
        {"infrastructure_support": 0.36, "communication_service": 0.24, "networking": 0.20,
         "documentation_clerical": 0.12, "security": 0.08},
        ("it_bpm", "tourism_hospitality", "academe_utilities"),
        ("CompTIA A+", "CompTIA Network+"),
        ("Practice Troubleshooting", "Learn Ticketing Systems", "Improve Communication", "Apply for BPO Internship"),
    ),
    CareerTrack(
        "multimedia_developer", "Multimedia / UI-UX Developer", "Technical",
        "Design interfaces and interactive media for web, mobile and information systems.",
        {"web_design": 0.46, "programming": 0.20, "communication_service": 0.14,
         "systems_engineering": 0.12, "documentation_clerical": 0.08},
        ("private_tech", "tourism_hospitality", "it_bpm"),
        ("UI/UX Design Workshop", "Adobe / Figma Certificate"),
        ("Build a Design Portfolio", "Learn Figma Prototyping", "Study Usability Testing", "Join Design Sprints"),
    ),
    # ----------------------------------------------------------- Clerical ---
    CareerTrack(
        "data_encoder", "IT Clerk / Data Encoder", "Clerical",
        "Encode, validate and maintain institutional records and datasets.",
        {"documentation_clerical": 0.50, "data_management": 0.22, "communication_service": 0.14,
         "systems_engineering": 0.08, "programming": 0.06},
        ("government", "academe_utilities", "finance", "it_bpm"),
        ("Microsoft Office Specialist", "Google Workspace Certificate"),
        ("Improve Typing Speed", "Master Excel", "Learn Records Management", "Practice Data Validation"),
    ),
    CareerTrack(
        "records_officer", "Records / Documentation Officer", "Clerical",
        "Prepare technical documentation, manuals and compliance records.",
        {"documentation_clerical": 0.46, "communication_service": 0.20, "systems_engineering": 0.16,
         "leadership_management": 0.10, "data_management": 0.08},
        ("government", "academe_utilities"),
        ("Microsoft Office Specialist", "Technical Writing Seminar"),
        ("Practice Technical Writing", "Learn Documentation Tools", "Study ISO Documentation"),
    ),
    CareerTrack(
        "helpdesk_associate", "Helpdesk / Customer Service Associate", "Clerical",
        "Handle front-line client interactions for IT-BPM and service desks.",
        {"communication_service": 0.46, "documentation_clerical": 0.22, "infrastructure_support": 0.18,
         "networking": 0.08, "leadership_management": 0.06},
        ("it_bpm", "tourism_hospitality", "finance"),
        ("Customer Service Seminar", "Microsoft Office Specialist"),
        ("Practice Voice & Accent", "Learn CRM Tools", "Improve Active Listening", "Apply to BPO Hubs"),
    ),
    # -------------------------------------------------------- Supervisory ---
    CareerTrack(
        "it_team_lead", "IT Team Lead / Systems Supervisor", "Supervisory",
        "Oversee a small technical team, assign work and review deliverables.",
        {"leadership_management": 0.34, "infrastructure_support": 0.20, "systems_engineering": 0.18,
         "communication_service": 0.16, "networking": 0.12},
        ("it_bpm", "private_tech", "academe_utilities", "government"),
        ("Leadership Training / LDT", "ITIL Foundation Seminar"),
        ("Lead a Class Project", "Join Org Officership", "Study ITIL", "Mentor Junior Students"),
    ),
    CareerTrack(
        "project_coordinator", "Technical Project Coordinator", "Supervisory",
        "Coordinate schedules, resources and reporting across technical projects.",
        {"leadership_management": 0.34, "communication_service": 0.24, "systems_engineering": 0.20,
         "documentation_clerical": 0.14, "data_management": 0.08},
        ("private_tech", "government", "it_bpm"),
        ("Project Management Seminar", "Scrum Fundamentals"),
        ("Learn Agile/Scrum", "Practice Gantt Planning", "Lead Capstone Coordination", "Study PM Tools"),
    ),
    # --------------------------------------------------------- Managerial ---
    CareerTrack(
        "it_project_manager", "IT Project Manager", "Managerial",
        "Plan, budget and deliver institutional IT projects end-to-end.",
        {"leadership_management": 0.40, "systems_engineering": 0.22, "communication_service": 0.18,
         "documentation_clerical": 0.10, "data_management": 0.10},
        ("private_tech", "government", "it_bpm", "finance"),
        ("Project Management Seminar", "PMP/CAPM Review"),
        ("Study PMBOK", "Lead a Capstone Team", "Learn Risk Management", "Build Stakeholder Skills"),
    ),
    CareerTrack(
        "mis_manager", "MIS / IT Operations Manager", "Managerial",
        "Direct the information systems unit, policies and service delivery.",
        {"leadership_management": 0.36, "systems_engineering": 0.20, "data_management": 0.16,
         "infrastructure_support": 0.16, "communication_service": 0.12},
        ("government", "academe_utilities", "finance", "tourism_hospitality"),
        ("Leadership Training / LDT", "IT Governance Seminar"),
        ("Study IT Governance", "Learn Budget Planning", "Practice Policy Writing", "Pursue Graduate Studies"),
    ),
)

TRACK_BY_KEY = {t.key: t for t in CAREER_TRACKS}
TRACKS_BY_CATEGORY: dict[str, list[CareerTrack]] = {c: [] for c in WORK_CATEGORIES}
for _t in CAREER_TRACKS:
    TRACKS_BY_CATEGORY[_t.work_category].append(_t)


# --------------------------------------------------------------------------- #
# Palawan / Puerto Princesa employer mapping
# --------------------------------------------------------------------------- #
# IMPORTANT FOR THE RESEARCHERS -------------------------------------------- #
# These entries are drawn from the establishments named in Chapter II of the
# thesis plus the sector categories described by the DICT Digital Cities
# programme. They are PLACEHOLDERS: verify each organisation's current
# internship capacity with OSAS / the OJT coordinator before publishing the
# system, and add or remove entries here rather than in the source code.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Employer:
    name: str
    sector_label: str
    locality: str
    note: str = ""


PALAWAN_EMPLOYERS: dict[str, list[Employer]] = {
    "it_bpm": [
        Employer("SupportNinja Puerto Princesa", "IT-BPM", "Puerto Princesa City",
                 "Technical support, back-office and service-desk operations."),
        Employer("Local BPO / call-centre hubs", "IT-BPM", "Puerto Princesa City",
                 "Entry-level technical and customer support pipelines."),
    ],
    "government": [
        Employer("City Government of Puerto Princesa - MIS Office", "Government", "Puerto Princesa City",
                 "Smart City / e-governance systems, e-permits and tax systems."),
        Employer("Provincial Government of Palawan - ICT Unit", "Government", "Palawan",
                 "Provincial information systems and records digitisation."),
        Employer("DICT MIMAROPA Field Office", "Government", "MIMAROPA Region",
                 "Digital Cities programme and national ICT initiatives."),
    ],
    "academe_utilities": [
        Employer("Palawan State University - ICT / MIS Office", "Academe", "Puerto Princesa City",
                 "Institutional servers, student information systems and databases."),
        Employer("Palawan Electric Cooperative (PALECO)", "Public Utility", "Palawan",
                 "Utility billing systems, network and database administration."),
        Employer("Puerto Princesa City Water District", "Public Utility", "Puerto Princesa City",
                 "Records systems, network infrastructure and support."),
    ],
    "tourism_hospitality": [
        Employer("Hotels and resorts (tourism IT)", "Tourism & Hospitality", "Palawan",
                 "Property management systems, guest networks and IT security."),
        Employer("Tour and travel operators", "Tourism & Hospitality", "Puerto Princesa City",
                 "Booking platforms and front-office systems."),
    ],
    "private_tech": [
        Employer("Local software houses and tech startups", "Private Technology", "Puerto Princesa City",
                 "Software solutions for tourism, retail and logistics."),
        Employer("Regional IT solutions providers", "Private Technology", "Palawan",
                 "Systems integration, web development and deployment."),
    ],
    "finance": [
        Employer("Rural banks and cooperatives", "Finance", "Palawan",
                 "Core banking support, data encoding and information security."),
        Employer("Lending and microfinance institutions", "Finance", "Puerto Princesa City",
                 "Records management and reporting systems."),
    ],
}

SECTOR_LABELS = {
    "it_bpm": "IT-BPM / Business Process Outsourcing",
    "government": "Government & Public Administration",
    "academe_utilities": "Academe & Public Utilities",
    "tourism_hospitality": "Tourism & Hospitality",
    "private_tech": "Private Technology Enterprises",
    "finance": "Banking & Finance",
}


# --------------------------------------------------------------------------- #
# BSCS vs BSIT discipline profiles  ->  "% degree alignment"
# --------------------------------------------------------------------------- #
# Weights follow CMO No. 25 s.2015 and the two PSU prospectuses: BSCS emphasises
# algorithmic problem solving and software design; BSIT emphasises application,
# integration and management of computing technologies (networking, security,
# infrastructure, web and multimedia).
DISCIPLINE_PROFILES: dict[str, dict] = {
    "BSCS": {
        "programming": 0.26,
        "theory_analytics": 0.24,
        "systems_engineering": 0.16,
        "data_management": 0.10,
        "web_design": 0.08,
        "security": 0.06,
        "networking": 0.05,
        "infrastructure_support": 0.05,
    },
    "BSIT": {
        "networking": 0.22,
        "infrastructure_support": 0.20,
        "security": 0.16,
        "data_management": 0.14,
        "web_design": 0.12,
        "systems_engineering": 0.10,
        "programming": 0.06,
    },
}


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def pretty_feature_name(column: str) -> str:
    """Turn a raw feature column into a label a panelist can read.

    'grade_bsit_network_technologies_1' -> 'Network Technologies 1 (BSIT Grade)'
    'acad_networking'                   -> 'Networking & Communications (Grades)'
    'tech_networking'                   -> 'Networking (Technical skill)'
    'cert_networking'                   -> 'Networking Certification'
    'sig_security'                      -> 'Information Assurance & Security (Signal)'
    """
    if column in COURSE_BY_COLUMN:
        course = COURSE_BY_COLUMN[column]
        return f"{course.title} ({course.program} Grade)"
    if column.startswith("acad_"):
        key = column[len("acad_"):]
        return f"{ACADEMIC_DOMAINS.get(key, key)} (Grades)"
    if column in SKILL_BY_COLUMN:
        skill = SKILL_BY_COLUMN[column]
        return f"{skill.label} ({SKILL_GROUP_LABELS[skill.group]})"
    if column.startswith("cert_type_"):
        key = column[len("cert_type_"):]
        return f"{CERTIFICATION_TYPES.get(key, key)} (Credential type)"
    if column.startswith("cert_"):
        return f"{column[len('cert_'):].replace('_', ' ').title()} Certification"
    if column.startswith("sig_"):
        sig = SIGNAL_BY_KEY.get(column[len("sig_"):])
        return f"{sig.label} (Signal)" if sig else column
    if column.startswith("riasec_"):
        return f"RIASEC: {column[len('riasec_'):].title()}"

    special = {
        "gwa": "Cumulative GWA",
        "gwa_points": "Cumulative GWA (Points)",
        "program": "Degree Program",
        "program_code": "Degree Program (BSCS / BSIT)",
        "has_certification": "Holds Any Credential",
        "cert_count": "Number of Certification Domains",
        "cert_type_count": "Number of Credential Types",
        "evt_motivation": "Expectancy-Value Motivation Score",
        "evt_expectancy": "EVT: Expectancy (Academic)",
        "evt_value": "EVT: Value (Skills & Credentials)",
        "core_grade_mean": "Mean Domain Grade (Points)",
        "core_grade_std": "Consistency of Domain Grades",
        "tech_skill_count": "Technical Skills Ticked",
        "hard_skill_count": "Hard Skills Ticked",
        "soft_skill_count": "Soft Skills Ticked",
    }
    return special.get(column, column.replace("_", " ").title())


def normalise_weights(weights: dict) -> dict:
    """Rescale a weight dictionary so the values sum to 1.0."""
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Weight dictionary must have a positive total.")
    return {k: v / total for k, v in weights.items()}
