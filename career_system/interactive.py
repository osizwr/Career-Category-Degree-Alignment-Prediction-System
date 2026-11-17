"""
The interactive questionnaire: a student answers, the trained model replies.

This is the console stand-in for the web form described in Chapter IV. It walks
one person through the approved Student Profile and Skills Questionnaire --
degree program first, then GWA, then that program's prospectus, then Part III
(certifications) and Part IV (the three skill checklists) -- assembles exactly
the raw row the pipeline expects, and hands it to the fitted XGBoost model.

What comes back is the full StudentReport: the predicted broad work category
with the model's confidence, the illustrative job roles inside it, the BSCS/BSIT
degree alignment, the Holland code, the SHAP explanation, the skill gaps and the
Palawan destinations.

    python main.py predict                 answer the questionnaire now
    python main.py predict --answers a.json  replay a saved set of answers
    python main.py predict --save-answers a.json

Nothing typed here is written to the training data. The answers produce one
prediction and, optionally, one JSON file the student can keep.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap

import pandas as pd

from . import config as cfg
from . import features as feat


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
WIDTH = 78


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _heading(title: str) -> None:
    print(f"\n{_rule('=')}")
    print(f"  {title}")
    print(_rule("="))


def _section(title: str, blurb: str = "") -> None:
    print(f"\n  {title}")
    print("  " + _rule()[2:])
    if blurb:
        print(textwrap.fill(blurb, width=WIDTH - 4,
                            initial_indent="    ", subsequent_indent="    "))


class Abort(Exception):
    """The student pressed Ctrl-D or typed 'quit'."""


def _ask(prompt: str) -> str:
    try:
        answer = input(prompt)
    except EOFError as exc:
        raise Abort("no more input") from exc
    if answer.strip().lower() in {"quit", "exit"}:
        raise Abort("cancelled")
    return answer.strip()


# --------------------------------------------------------------------------- #
# Primitive prompts
# --------------------------------------------------------------------------- #
def ask_choice(question: str, options: list[tuple[str, str]]) -> str:
    """Single choice from a numbered list. Returns the chosen key."""
    print(f"\n  {question}")
    for index, (_, label) in enumerate(options, start=1):
        print(f"    [{index}] {label}")

    while True:
        answer = _ask(f"  Enter 1-{len(options)}: ")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print(f"    Please type a number from 1 to {len(options)}.")


def ask_yes_no(question: str, default: bool | None = None) -> bool:
    suffix = {True: " [Y/n]: ", False: " [y/N]: ", None: " [y/n]: "}[default]
    while True:
        answer = _ask(f"  {question}{suffix}").lower()
        if not answer and default is not None:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("    Please answer y or n.")


def ask_grade(label: str, allow_blank: bool = True) -> float | None:
    """One grade on the 1.00-5.00 Philippine scale (1.00 is highest)."""
    hint = "  (Enter to skip)" if allow_blank else ""
    while True:
        answer = _ask(f"    {label:<52}{hint} ")
        if not answer:
            if allow_blank:
                return None
            print("      This one is required.")
            continue
        try:
            value = float(answer)
        except ValueError:
            print("      Type a number like 1.75, or press Enter to skip.")
            continue
        if not (cfg.GRADE_BEST <= value <= cfg.GRADE_WORST):
            print(f"      Grades run from {cfg.GRADE_BEST:.2f} (highest) to "
                  f"{cfg.GRADE_WORST:.2f} (failed).")
            continue
        return value


def ask_checkboxes(question: str, items: list[str]) -> list[int]:
    """Multi-select. Returns the 0-based indices the student ticked."""
    print(f"\n  {question}")
    for index, label in enumerate(items, start=1):
        print(f"    [{index}] {label}")
    print("    Type the numbers you want, separated by commas (e.g. 1,3,5).")
    print("    Press Enter or type 'none' if none of them apply.")

    while True:
        answer = _ask("  Your answer: ").lower()
        if not answer or answer == "none":
            return []
        if answer in {"all", "*"}:
            return list(range(len(items)))

        picks, bad = [], []
        for token in re.split(r"[,\s]+", answer):
            if not token:
                continue
            if token.isdigit() and 1 <= int(token) <= len(items):
                picks.append(int(token) - 1)
            else:
                bad.append(token)
        if bad:
            print(f"    Not a valid choice: {', '.join(bad)}. "
                  f"Use numbers from 1 to {len(items)}.")
            continue
        return sorted(set(picks))


# --------------------------------------------------------------------------- #
# Certificate free text -> subject domains
# --------------------------------------------------------------------------- #
def recognise_cert_domains(names: str) -> set[str]:
    """Read Q8's free text the same way scripts/build_model_dataset.py does."""
    return {
        domain for pattern, domain in cfg.CERT_KEYWORDS
        if re.search(pattern, names or "", re.IGNORECASE)
    }


# --------------------------------------------------------------------------- #
# The questionnaire
# --------------------------------------------------------------------------- #
def run_questionnaire(student_id: str = "YOU") -> dict:
    """Walk one person through the form and return a raw answer record."""
    _heading("STUDENT PROFILE AND SKILLS QUESTIONNAIRE")
    print(textwrap.dedent(f"""
          Answer as yourself. Nothing you type is saved to the training data or
          sent anywhere -- it produces one prediction on this machine.

          Type 'quit' at any prompt to stop.
    """).rstrip())

    record: dict = {"student_id": student_id}

    # ---------------------------------------------------- Part II: profile ---
    _section("PART II - STUDENT PROFILE")
    program = ask_choice(
        "What is your degree program?",
        [(p, f"{cfg.PROGRAM_LABELS[p]} ({p})") for p in cfg.PROGRAMS],
    )
    record["program"] = program
    print(f"\n    -> Using the {program} prospectus "
          f"({len(cfg.COURSES_BY_PROGRAM[program])} courses).")

    print()
    record["gwa"] = ask_grade("Cumulative General Weighted Average (GWA)",
                              allow_blank=False)

    # ------------------------------------------------------------- grades ---
    courses = cfg.COURSES_BY_PROGRAM[program]
    for course in cfg.ALL_COURSES:
        record[course.column] = None

    _section(
        "CORE COURSE GRADES",
        f"The model reads your grades through {len(cfg.DOMAIN_KEYS)} shared "
        f"competency domains, so a blank course only weakens the domain it "
        f"belongs to -- it does not invalidate your profile."
    )
    enter_grades = ask_yes_no(
        f"Enter your {len(courses)} {program} course grades now?", default=True
    )

    if enter_grades:
        print(f"\n    1.00 is the highest mark, 3.00 passes, 5.00 is a failure.")
        print("    Press Enter to skip any course you have not taken yet.\n")
        for course in courses:
            record[course.column] = ask_grade(course.title)
        entered = sum(1 for c in courses if record[c.column] is not None)
        print(f"\n    -> {entered} of {len(courses)} course grades recorded.")
    else:
        print("\n    -> Skipped. Your academic profile will rest on your GWA alone,")
        print("       which is 20% of the rubric rather than the full 50%.")

    # -------------------------------------------- Part III: certifications ---
    _section(
        "CERTIFICATIONS, SEMINARS AND TRAINING",
        "Professional certifications, technical seminars, workshops, webinars or "
        "training activities relevant to Computer Science, Information Technology "
        "or your intended career."
    )
    for key in cfg.CERT_TYPE_KEYS:
        record[key] = 0
    for domain in cfg.CERT_DOMAINS:
        record[f"cert_{domain}"] = 0

    has_cert = ask_yes_no("Have you completed any of these?", default=False)
    record["has_certification"] = int(has_cert)

    if has_cert:
        print("\n  Name the certifications, seminars, workshops, webinars or")
        print("     trainings you completed, separated by commas.")
        names = _ask("  Your answer: ")
        domains = recognise_cert_domains(names)
        for domain in domains:
            record[f"cert_{domain}"] = 1

        if domains:
            readable = ", ".join(sorted(d.replace("_", " ").title() for d in domains))
            print(f"\n    -> Recognised subject area(s): {readable}")
        elif names:
            print("\n    -> No subject area recognised from that text. It still counts")
            print("       as a credential, but it cannot be matched to a competency.")
            print("       Add a keyword for it in config.CERT_KEYWORDS if it should.")

        type_keys = list(cfg.CERTIFICATION_TYPES)
        picked = ask_checkboxes(
            "What type of credential or activity? (tick all that apply)",
            [cfg.CERTIFICATION_TYPES[k] for k in type_keys],
        )
        for index in picked:
            record[f"cert_type_{type_keys[index]}"] = 1
        if not picked:
            # Q7 said yes, so at least one kind of activity happened; recording
            # none would contradict the answer above.
            record["cert_type_technical_seminar"] = 1
            print("    -> Nothing ticked; recorded as a technical seminar.")

    # ------------------------------------------------------ Part IV: skills ---
    _section(
        "SKILLS",
        "Check all the skills you have ACTUALLY USED. There is no rating here: "
        "the questionnaire asks what you have done, not how good you are at it."
    )

    group_questions = {
        "tech": ("Technical skills", "specialised computing skills"),
        "hard": ("Hard skills", "measurable skills that are not specific to computing"),
        "soft": ("Soft skills", "personal and interpersonal skills"),
    }
    for group, skills in cfg.SKILL_GROUPS.items():
        number, blurb = group_questions[group]
        picked = ask_checkboxes(f"{number} ({blurb})", [s.prompt for s in skills])
        for index, skill in enumerate(skills):
            record[f"{group}_{skill.key}"] = int(index in picked)

    ticked = sum(int(record[k]) for k in cfg.ALL_SKILL_KEYS)
    print(f"\n    -> {ticked} of {len(cfg.ALL_SKILL_KEYS)} skills ticked.")

    return record


# --------------------------------------------------------------------------- #
# Record -> frame
# --------------------------------------------------------------------------- #
def record_to_frame(record: dict) -> pd.DataFrame:
    """One answer record -> the one-row raw frame the pipeline validates."""
    row = {"student_id": record.get("student_id", "YOU")}
    for column in feat.RAW_FEATURE_COLUMNS:
        row[column] = record.get(column)

    missing = [c for c in ("program", "gwa") if row.get(c) in (None, "")]
    if missing:
        raise ValueError(f"The answer record is missing required field(s): {missing}")

    frame = pd.DataFrame([row])
    feat.validate_raw_frame(frame, require_target=False)
    return frame


def load_answers(path) -> dict:
    with open(path) as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError(f"{path} should contain a single JSON object of answers.")
    return record


def save_answers(record: dict, path) -> None:
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)


# --------------------------------------------------------------------------- #
# What the student sees afterwards
# --------------------------------------------------------------------------- #
def print_answer_summary(record: dict) -> None:
    """Echo the answers back so the student can see what the model was given."""
    program = record["program"]
    courses = cfg.COURSES_BY_PROGRAM[program]
    entered = [c for c in courses if record.get(c.column) is not None]

    _section("YOUR ANSWERS, AS THE MODEL RECEIVED THEM")
    print(f"    Program ................. {cfg.PROGRAM_LABELS[program]} ({program})")
    print(f"    Cumulative GWA .......... {float(record['gwa']):.2f}")
    print(f"    Course grades entered ... {len(entered)} of {len(courses)}")
    if entered:
        best = min(entered, key=lambda c: record[c.column])
        worst = max(entered, key=lambda c: record[c.column])
        print(f"      strongest ............. {best.title} ({record[best.column]:.2f})")
        print(f"      weakest ............... {worst.title} ({record[worst.column]:.2f})")

    held = [d.replace("_", " ").title() for d in cfg.CERT_DOMAINS
            if record.get(f"cert_{d}")]
    print(f"    Credentials ............. "
          f"{'yes' if record.get('has_certification') else 'none reported'}"
          f"{' - ' + ', '.join(held) if held else ''}")

    for group, skills in cfg.SKILL_GROUPS.items():
        chosen = [s.label for s in skills if record.get(f"{group}_{s.key}")]
        label = cfg.SKILL_GROUP_LABELS[group] + "s"
        print(f"    {label:<23} {', '.join(chosen) if chosen else '(none ticked)'}")


def print_next_steps(report) -> None:
    top = report.track_matches[0]
    _section("WHAT TO DO WITH THIS")
    print(textwrap.fill(
        f"The model places you in the {report.predicted_category.upper()} work "
        f"category with {report.confidence:.1%} confidence, and the closest "
        f"illustrative role inside it is {top.name} at a {top.match_percent:.1f}% "
        f"match. The SHAP section above says, in order, which grades, "
        f"credentials and skills moved that prediction and by how much.",
        width=WIDTH - 4, initial_indent="    ", subsequent_indent="    "))
    print()
    print(textwrap.fill(
        "This is guidance for a career conversation, not a placement decision. "
        "It is computed from what you reported today: tick one more skill or "
        "finish one more certification and it will move. Nothing here guarantees "
        "or forecloses any hiring outcome.",
        width=WIDTH - 4, initial_indent="    ", subsequent_indent="    "))


def offer(prompt: str = "Would you like to test your own profile now?") -> bool:
    """Ask at the end of the pipeline, but only when someone is actually there."""
    if not sys.stdin.isatty():
        return False
    try:
        return ask_yes_no(prompt, default=False)
    except Abort:
        return False
