"""
Central configuration for the PSU Career Category & Degree Alignment System.

Everything the researchers are expected to EDIT before the final defense lives
in this one file:

  * CORE_COURSES ............ the prospectus-based core courses used as features
  * CERTIFICATION_DOMAINS ... the professional / seminar certification groupings
  * TECHNICAL/HARD/SOFT_SKILLS  the self-reported skill inventories
  * WORK_CATEGORIES ......... the four broad work categories (the ML target)
  * CAREER_TRACKS ........... specific job roles + their competency profiles
  * PALAWAN_EMPLOYERS ....... local internship / employment destinations
  * DISCIPLINE_PROFILES ..... BSCS vs BSIT weights used for % degree alignment

Reference: Cajilo, Donghil, Mahinay & Sibayan (2026),
"Predictive Modeling of Career Category and Degree Alignment for CS and IT
Students Using Ensemble and Baseline Classifiers", Palawan State University.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
TEST_SIZE = 0.30        # thesis: "70-30 ratio: 70% training, 30% testing"
CV_FOLDS = 5            # thesis: "Stratified K-Fold Cross-Validation (with k=5)"


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
# Prospectus-based core courses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Course:
    """One prospectus core course used as a predictor variable."""

    key: str            # snake_case -> becomes the CSV / feature column name
    code: str           # course code as printed on the prospectus
    title: str          # human-readable title (used in SHAP plots)
    year_level: int     # 1..3 (thesis scope: first to third academic year)
    domains: tuple      # competency signals this course feeds


# NOTE FOR THE RESEARCHERS -------------------------------------------------- #
# These are the CMO No. 25 s.2015 common core courses shared by BSCS and BSIT.
# Replace the codes/titles with the exact PSU College of Sciences prospectus
# before data collection; the rest of the system adapts automatically.
# --------------------------------------------------------------------------- #
CORE_COURSES: tuple[Course, ...] = (
    Course("intro_computing", "CC 101", "Introduction to Computing", 1,
           ("programming", "systems_engineering")),
    Course("programming_1", "CC 102", "Computer Programming 1", 1,
           ("programming",)),
    Course("programming_2", "CC 103", "Computer Programming 2", 1,
           ("programming",)),
    Course("discrete_structures", "CC 104", "Discrete Structures / Quantitative Methods", 1,
           ("theory_analytics",)),
    Course("data_structures", "CC 105", "Data Structures and Algorithms", 2,
           ("programming", "theory_analytics")),
    Course("oop", "CC 106", "Object-Oriented Programming", 2,
           ("programming", "systems_engineering")),
    Course("information_management", "CC 107", "Information Management (Database Systems)", 2,
           ("data_management",)),
    Course("computer_architecture", "CC 108", "Computer Architecture and Organization", 2,
           ("networking", "infrastructure_support")),
    Course("operating_systems", "CC 109", "Operating Systems / Platform Technologies", 2,
           ("infrastructure_support", "networking")),
    Course("web_systems", "CC 110", "Web Systems and Technologies", 2,
           ("web_design", "programming")),
    Course("networks", "CC 111", "Networks and Communication", 3,
           ("networking", "infrastructure_support")),
    Course("information_security", "CC 112", "Information Assurance and Security", 3,
           ("security",)),
    Course("hci", "CC 113", "Human Computer Interaction", 3,
           ("web_design",)),
    Course("software_engineering", "CC 114", "Software Engineering / Systems Analysis and Design", 3,
           ("systems_engineering",)),
    Course("app_development", "CC 115", "Application Development and Emerging Technologies", 3,
           ("programming", "systems_engineering")),
    Course("professional_practice", "CC 116", "Social Issues and Professional Practice", 3,
           ("leadership_management", "communication_service")),
)

COURSE_KEYS = tuple(c.key for c in CORE_COURSES)
COURSE_BY_KEY = {c.key: c for c in CORE_COURSES}


# --------------------------------------------------------------------------- #
# Professional / seminar certifications
# --------------------------------------------------------------------------- #
# Each domain becomes a 0/1 feature ("has the student earned any certificate in
# this domain?"). The example certificate names are shown to students in the
# questionnaire and in the "suggested actions" of the recommendation report.
CERTIFICATION_DOMAINS: dict[str, list[str]] = {
    "networking": ["Cisco CCNA", "CompTIA Network+", "Cisco NetAcad Networking Essentials"],
    "security": ["CompTIA Security+", "Cisco CyberOps Associate", "Cybersecurity Seminar/Workshop"],
    "programming": ["Oracle Java Certification", "Python Programming Certificate", "C++/C# Certification"],
    "data_analytics": ["Google Data Analytics", "Microsoft Power BI", "Data Science Seminar"],
    "web_cloud": ["AWS Cloud Practitioner", "Microsoft Azure Fundamentals", "Web Development Bootcamp"],
    "database": ["Oracle Database SQL", "MySQL Certification", "Database Administration Seminar"],
    "office_productivity": ["Microsoft Office Specialist", "Google Workspace Certificate"],
    "leadership_management": ["Leadership Training / LDT", "Project Management Seminar", "Student Org Officership"],
}

CERT_DOMAINS = tuple(CERTIFICATION_DOMAINS.keys())
CERT_FEATURE_KEYS = tuple(f"cert_{d}" for d in CERT_DOMAINS)


# --------------------------------------------------------------------------- #
# Self-reported skill inventories (Likert 1-5)
# --------------------------------------------------------------------------- #
LIKERT_MIN, LIKERT_MAX = 1, 5

TECHNICAL_SKILLS: tuple[str, ...] = (
    "programming",
    "web_development",
    "database_management",
    "networking",
    "cybersecurity",
    "data_analytics",
    "systems_administration",
    "mobile_development",
    "cloud_computing",
    "technical_troubleshooting",
)

HARD_SKILLS: tuple[str, ...] = (
    "documentation",
    "technical_writing",
    "data_encoding",
    "office_productivity",
    "systems_analysis",
    "project_management",
    "quality_assurance_testing",
)

SOFT_SKILLS: tuple[str, ...] = (
    "communication",
    "teamwork",
    "leadership",
    "problem_solving",
    "adaptability",
    "time_management",
    "critical_thinking",
    "customer_service",
)

TECH_SKILL_KEYS = tuple(f"tech_{s}" for s in TECHNICAL_SKILLS)
HARD_SKILL_KEYS = tuple(f"hard_{s}" for s in HARD_SKILLS)
SOFT_SKILL_KEYS = tuple(f"soft_{s}" for s in SOFT_SKILLS)
ALL_SKILL_KEYS = TECH_SKILL_KEYS + HARD_SKILL_KEYS + SOFT_SKILL_KEYS


# --------------------------------------------------------------------------- #
# Competency signals
# --------------------------------------------------------------------------- #
# A "signal" is a 0-100 composite score built from the raw grades, skills and
# certifications. Signals are the shared vocabulary used by three parts of the
# system, which is what keeps the whole thing internally consistent:
#
#   1. RIASEC feature engineering  (Holland's Theory of Vocational Choice)
#   2. career-track match scoring  (the "92.8%" in Figure 4.2)
#   3. BSCS / BSIT degree alignment %
#
# Each signal declares which courses / skills / certificates contribute to it.
@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    courses: tuple = ()          # course keys
    tech_skills: tuple = ()      # names from TECHNICAL_SKILLS
    hard_skills: tuple = ()      # names from HARD_SKILLS
    soft_skills: tuple = ()      # names from SOFT_SKILLS
    certs: tuple = ()            # names from CERT_DOMAINS


SIGNALS: tuple[Signal, ...] = (
    Signal(
        "programming", "Programming & Software Construction",
        courses=("programming_1", "programming_2", "oop", "data_structures", "app_development"),
        tech_skills=("programming", "mobile_development"),
        certs=("programming",),
    ),
    Signal(
        "theory_analytics", "Algorithmic Thinking & Analytics",
        courses=("discrete_structures", "data_structures"),
        tech_skills=("data_analytics",),
        soft_skills=("problem_solving", "critical_thinking"),
        certs=("data_analytics",),
    ),
    Signal(
        "data_management", "Data & Database Management",
        courses=("information_management",),
        tech_skills=("database_management", "data_analytics"),
        certs=("database",),
    ),
    Signal(
        "networking", "Networking & Communications",
        courses=("networks", "computer_architecture"),
        tech_skills=("networking", "systems_administration"),
        certs=("networking",),
    ),
    Signal(
        "security", "Information Assurance & Security",
        courses=("information_security", "networks"),
        tech_skills=("cybersecurity",),
        certs=("security",),
    ),
    Signal(
        "web_design", "Web & Interface Development",
        courses=("web_systems", "hci"),
        tech_skills=("web_development", "cloud_computing"),
        certs=("web_cloud",),
    ),
    Signal(
        "systems_engineering", "Systems Analysis & Engineering",
        courses=("software_engineering", "intro_computing", "app_development"),
        hard_skills=("systems_analysis", "quality_assurance_testing"),
    ),
    Signal(
        "infrastructure_support", "Infrastructure & Technical Support",
        courses=("operating_systems", "computer_architecture", "networks"),
        tech_skills=("technical_troubleshooting", "systems_administration"),
    ),
    Signal(
        "documentation_clerical", "Documentation & Office Operations",
        hard_skills=("documentation", "technical_writing", "data_encoding", "office_productivity"),
        certs=("office_productivity",),
    ),
    Signal(
        "leadership_management", "Leadership & Project Management",
        courses=("professional_practice",),
        hard_skills=("project_management",),
        soft_skills=("leadership", "time_management"),
        certs=("leadership_management",),
    ),
    Signal(
        "communication_service", "Communication & Client Service",
        soft_skills=("communication", "teamwork", "customer_service", "adaptability"),
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
WORK_CATEGORIES: tuple[str, ...] = ("Technical", "Clerical", "Supervisory", "Managerial")

WORK_CATEGORY_DESCRIPTIONS = {
    "Technical": "Hands-on engineering, development, infrastructure and security roles.",
    "Clerical":  "Records, encoding, documentation and front-line support roles.",
    "Supervisory": "Team lead and coordination roles overseeing technical staff.",
    "Managerial": "Planning, resourcing and unit-level IT management roles.",
}

TARGET_COLUMN = "work_category"


# --------------------------------------------------------------------------- #
# Specific career tracks (the ranked "Top Career Recommendations" panel)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CareerTrack:
    key: str
    name: str
    work_category: str
    description: str
    weights: dict          # signal key -> weight (normalised automatically)
    sectors: tuple         # keys into PALAWAN_EMPLOYERS
    certifications: tuple  # recommended certifications to pursue
    actions: tuple         # "suggested actions" chips (see Figure 4.2)


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
# BSCS vs BSIT discipline profiles  ->  "% degree alignment" (SOP #3)
# --------------------------------------------------------------------------- #
# Weights follow CMO No. 25 s.2015: BSCS emphasises algorithmic problem solving
# and software design; BSIT emphasises application, integration and management
# of computing technologies (networking, security, infrastructure).
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

    'grade_networks'         -> 'Networks and Communication (Grade)'
    'tech_cybersecurity'     -> 'Cybersecurity (Technical Skill)'
    'cert_networking'        -> 'Networking Certification'
    'sig_security'           -> 'Information Assurance & Security (Signal)'
    """
    if column.startswith("grade_"):
        key = column[len("grade_"):]
        course = COURSE_BY_KEY.get(key)
        return f"{course.title} (Grade)" if course else column
    if column.startswith("tech_"):
        return f"{column[len('tech_'):].replace('_', ' ').title()} (Technical Skill)"
    if column.startswith("hard_"):
        return f"{column[len('hard_'):].replace('_', ' ').title()} (Hard Skill)"
    if column.startswith("soft_"):
        return f"{column[len('soft_'):].replace('_', ' ').title()} (Soft Skill)"
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
        "cert_count": "Number of Certifications",
        "seminar_units": "Seminar Units Earned",
        "evt_motivation": "Expectancy-Value Motivation Score",
        "evt_expectancy": "EVT: Expectancy (Academic)",
        "evt_value": "EVT: Value (Skills & Credentials)",
        "core_grade_mean": "Mean Core Course Grade (Points)",
        "core_grade_std": "Consistency of Core Grades",
        "tech_skill_mean": "Mean Technical Skill",
        "hard_skill_mean": "Mean Hard Skill",
        "soft_skill_mean": "Mean Soft Skill",
    }
    return special.get(column, column.replace("_", " ").title())


def normalise_weights(weights: dict) -> dict:
    """Rescale a weight dictionary so the values sum to 1.0."""
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Weight dictionary must have a positive total.")
    return {k: v / total for k, v in weights.items()}
