"""
Descriptive profiling of the cohort (SOP 1 / Objective 1).

    "What are the academic and professional characteristics of the fourth-year
     computing students based on their cumulative undergraduate input data...?"

Produces the descriptive tables and figures that belong at the start of
Chapter V, before any model results are discussed:

  * academic profile   - GWA and per-course grade distributions, by program
  * certification profile - how many students hold which credentials
  * skill profile      - mean self-reported technical / hard / soft skills
  * competency signals - the engineered 0-100 scores, by program
  * RIASEC profile     - distribution of Holland codes across the cohort
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import alignment as align
from . import config as cfg
from . import features as feat


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def academic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-course grade statistics on the original 1.00-5.00 scale.

    Reported in raw grades (not points) because that is what appears on a
    transcript and what a panelist will recognise. Remember: LOWER IS BETTER.

    BSCS and BSIT answered different course lists, so each course is reported
    only for the program that sat it. The `acad_*` domain scores below are what
    make the two programs comparable.
    """
    rows = []
    for course in cfg.ALL_COURSES:
        series = pd.to_numeric(
            df.loc[df["program"] == course.program, course.column], errors="coerce"
        ).dropna()
        if series.empty:
            rows.append({
                "Program": course.program,
                "Course": course.title,
                "Domain": ", ".join(cfg.ACADEMIC_DOMAINS[d] for d in course.domains) or "(elective)",
                "N reported": 0,
                "Mean grade": np.nan,
                "Std dev": np.nan,
                "Best": np.nan,
                "Lowest": np.nan,
            })
            continue
        rows.append({
            "Program": course.program,
            "Course": course.title,
            "Domain": ", ".join(cfg.ACADEMIC_DOMAINS[d] for d in course.domains) or "(elective)",
            "N reported": int(series.size),
            "Mean grade": round(series.mean(), 3),
            "Std dev": round(series.std(), 3),
            "Best": round(series.min(), 2),
            "Lowest": round(series.max(), 2),
        })

    frame = pd.DataFrame(rows)

    gwa = pd.to_numeric(df["gwa"], errors="coerce")
    summary = {
        "Program": "BOTH",
        "Course": "CUMULATIVE GWA",
        "Domain": "--",
        "N reported": int(gwa.notna().sum()),
        "Mean grade": round(gwa.mean(), 3),
        "Std dev": round(gwa.std(), 3),
        "Best": round(gwa.min(), 2),
        "Lowest": round(gwa.max(), 2),
    }
    return pd.concat([frame, pd.DataFrame([summary])], ignore_index=True)


def domain_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean grade POINTS per shared academic domain, by program.

    This is the table to read alongside the per-course one: it is the only view
    in which a BSCS and a BSIT profile are measured on the same ruler, and it is
    what the classifier actually sees.
    """
    engineered = feat.add_grade_points(df)
    rows = []
    for domain, label in cfg.ACADEMIC_DOMAINS.items():
        column = f"acad_{domain}"
        series = engineered[column]
        row = {
            "Academic domain": label,
            "Cohort mean (points)": round(series.mean(), 2),
            "Std dev": round(series.std(), 2),
        }
        for program in cfg.PROGRAMS:
            subset = engineered.loc[engineered["program"] == program, column]
            row[f"{program} mean"] = round(subset.mean(), 2) if len(subset) else np.nan
            row[f"{program} courses"] = len(cfg.domain_courses(program, domain))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Cohort mean (points)", ascending=False)


def certification_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Part III of the questionnaire: who holds what, and of what kind."""
    rows = []
    answered = pd.to_numeric(df["has_certification"], errors="coerce")
    holders = int(answered.sum())
    rows.append({
        "Item": "Q7: holds any credential",
        "Students": holders,
        "Share of cohort": f"{holders / max(answered.notna().sum(), 1):.1%}",
        "Detail": "seminars, workshops, webinars, training or certification exams",
    })

    for key, label in cfg.CERTIFICATION_TYPES.items():
        series = pd.to_numeric(df[f"cert_type_{key}"], errors="coerce")
        count = int(series.sum())
        rows.append({
            "Item": f"Q9 type: {label}",
            "Students": count,
            "Share of cohort": f"{count / max(series.notna().sum(), 1):.1%}",
            "Detail": "",
        })

    for domain in cfg.CERT_DOMAINS:
        series = pd.to_numeric(df[f"cert_{domain}"], errors="coerce")
        count = int(series.sum())
        rows.append({
            "Item": f"Q8 subject: {domain.replace('_', ' ').title()}",
            "Students": count,
            "Share of cohort": f"{count / max(series.notna().sum(), 1):.1%}",
            "Detail": ", ".join(cfg.CERTIFICATION_DOMAINS[domain][:2]),
        })

    engineered = feat.add_aggregate_features(df)
    total = engineered["cert_count"]
    rows.append({
        "Item": "-- subject domains per student --",
        "Students": f"mean {total.mean():.2f}",
        "Share of cohort": f"max {int(total.max())}",
        "Detail": f"{int((total == 0).sum())} students match no recognised subject domain",
    })
    return pd.DataFrame(rows)


def skill_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the cohort that ticked each Part IV checkbox, overall and by program.

    Objective 1 asks for "the percentage distribution of technical skills within
    the consolidated graduating computing cohort", which is exactly this table.
    """
    rows = []
    for group, skills in cfg.SKILL_GROUPS.items():
        for skill in skills:
            column = f"{group}_{skill.key}"
            series = pd.to_numeric(df[column], errors="coerce")
            answered = int(series.notna().sum())
            ticked = int(series.sum())
            row = {
                "Group": cfg.SKILL_GROUP_LABELS[group],
                "Skill": skill.label,
                "Questionnaire item": skill.prompt,
                "Students who ticked": ticked,
                "% of cohort": round(100.0 * ticked / max(answered, 1), 1),
            }
            for program in cfg.PROGRAMS:
                subset = pd.to_numeric(
                    df.loc[df["program"] == program, column], errors="coerce"
                )
                row[f"{program} %"] = (
                    round(100.0 * subset.sum() / max(int(subset.notna().sum()), 1), 1)
                    if len(subset) else np.nan
                )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["Group", "% of cohort"], ascending=[True, False])


def riasec_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the three-letter Holland summary codes."""
    engineered = feat.build_features(df)
    codes = engineered.apply(feat.riasec_code, axis=1)

    counts = codes.value_counts().head(12).reset_index()
    counts.columns = ["Holland code", "Students"]
    counts["Share"] = (counts["Students"] / len(engineered)).map(lambda v: f"{v:.1%}")

    first_letters = codes.str[0].value_counts()
    expanded = {"R": "Realistic", "I": "Investigative", "A": "Artistic",
                "S": "Social", "E": "Enterprising", "C": "Conventional"}
    counts["_"] = ""
    dominant = pd.DataFrame({
        "Holland code": [f"DOMINANT: {expanded.get(k, k)}" for k in first_letters.index],
        "Students": first_letters.values,
        "Share": [f"{v / len(engineered):.1%}" for v in first_letters.values],
        "_": "",
    })
    return pd.concat([counts, dominant], ignore_index=True).drop(columns="_")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_skill_means(df: pd.DataFrame, outdir=None):
    """Percentage of the cohort that ticked each Part IV skill checkbox."""
    outdir = outdir or cfg.FIGURE_DIR
    profile = skill_profile(df).sort_values("% of cohort")

    colours = {"Technical skill": "#2563eb", "Hard skill": "#f59e0b", "Soft skill": "#10b981"}
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(profile) + 2.0))
    bars = ax.barh(profile["Skill"], profile["% of cohort"],
                   color=[colours[g] for g in profile["Group"]], edgecolor="white", height=0.72)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

    ax.axvline(50.0, ls="--", lw=1, color="#64748b")
    ax.text(50.8, -0.6, "half the cohort", fontsize=8, color="#64748b")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Share of respondents who ticked the box (%)")
    ax.set_title("Self-Reported Skill Profile of the Cohort\n"
                 "Part IV checkboxes: technical, hard and soft skills students have actually used")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles, colours.keys(), title="Skill group", frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = outdir / "profile_skills.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_signal_radar(df: pd.DataFrame, outdir=None):
    """Radar chart comparing the mean competency signals of BSCS vs BSIT."""
    outdir = outdir or cfg.FIGURE_DIR
    profile = align.cohort_signal_profile(df)

    labels = list(profile.index)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8), subplot_kw={"projection": "polar"})
    for program, colour in (("BSCS", "#2563eb"), ("BSIT", "#f59e0b")):
        if program not in profile.columns:
            continue
        values = profile[program].tolist()
        values += values[:1]
        ax.plot(angles, values, lw=2, color=colour, label=cfg.PROGRAM_LABELS[program])
        ax.fill(angles, values, color=colour, alpha=0.13)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace(" & ", "\n& ") for l in labels], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="#64748b")
    ax.set_title("Mean Competency Signals by Degree Program\n"
                 "0-100 composite of core grades, certifications and self-reported skills",
                 pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.10), frameon=False)
    fig.tight_layout()

    path = outdir / "profile_signal_radar.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gwa_distribution(df: pd.DataFrame, outdir=None):
    outdir = outdir or cfg.FIGURE_DIR

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for program, colour in (("BSCS", "#2563eb"), ("BSIT", "#f59e0b")):
        subset = df.loc[df["program"] == program, "gwa"].astype(float)
        ax1.hist(subset, bins=np.arange(1.0, 3.3, 0.125), alpha=0.62,
                 label=f"{program} (n={len(subset)})", color=colour, edgecolor="white")
    ax1.set_xlabel("Cumulative GWA  (1.00 = highest)")
    ax1.set_ylabel("Number of students")
    ax1.set_title("GWA Distribution by Program")
    ax1.legend(frameon=False)
    ax1.invert_xaxis()  # best students appear on the right
    ax1.spines[["top", "right"]].set_visible(False)

    engineered = feat.build_features(df)
    for category, colour in zip(cfg.WORK_CATEGORIES,
                                ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6"]):
        subset = engineered.loc[engineered[cfg.TARGET_COLUMN] == category, "gwa_points"]
        if len(subset):
            ax2.scatter(subset, np.random.default_rng(1).normal(
                cfg.WORK_CATEGORIES.index(category), 0.09, len(subset)),
                alpha=0.55, s=22, color=colour, label=category)
    ax2.set_yticks(range(len(cfg.WORK_CATEGORIES)), cfg.WORK_CATEGORIES)
    ax2.set_xlabel("GWA in points (higher = better)")
    ax2.set_title("Academic Standing Across Work Categories")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = outdir / "profile_gwa.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_profiling(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Build every descriptive table and figure; return the paths written."""
    tables = {
        "profile_academic": academic_profile(df),
        "profile_academic_domains": domain_profile(df),
        "profile_certifications": certification_profile(df),
        "profile_skills": skill_profile(df),
        "profile_riasec": riasec_profile(df),
        "profile_competency_signals": align.cohort_signal_profile(df),
        "profile_degree_alignment": align.compute_alignment_table(df),
        "profile_track_demand": align.cohort_track_demand(df),
    }

    written = {}
    for name, table in tables.items():
        path = cfg.TABLE_DIR / f"{name}.csv"
        table.to_csv(path, index=(name == "profile_competency_signals"))
        written[name] = path

    written["figure_skills"] = plot_skill_means(df)
    written["figure_radar"] = plot_signal_radar(df)
    written["figure_gwa"] = plot_gwa_distribution(df)

    if verbose:
        print("\n  Academic profile by shared domain (0-100 points, higher is better)")
        print("  " + "-" * 74)
        print(tables["profile_academic_domains"][
            ["Academic domain", "Cohort mean (points)", "BSCS mean", "BSIT mean"]
        ].to_string(index=False, max_colwidth=42))
        print("    Per-course grades on the 1.00-5.00 transcript scale are in")
        print("    outputs/tables/profile_academic.csv, listed per prospectus.")

        print("\n  Certifications, seminars and training (Part III)")
        print("  " + "-" * 74)
        print(tables["profile_certifications"][
            ["Item", "Students", "Share of cohort"]
        ].head(14).to_string(index=False))

        print("\n  Most commonly ticked skills (Part IV)")
        print("  " + "-" * 74)
        skills = tables["profile_skills"].sort_values("% of cohort", ascending=False).head(10)
        print(skills[["Group", "Skill", "% of cohort", "BSCS %", "BSIT %"]].to_string(index=False))

        alignment = tables["profile_degree_alignment"]
        matched = alignment["aligned_with_enrolment"].sum()
        print("\n  Degree alignment summary")
        print("  " + "-" * 74)
        print(f"    Mean BSCS alignment ....... {alignment['bscs_alignment_pct'].mean():.1f}%")
        print(f"    Mean BSIT alignment ....... {alignment['bsit_alignment_pct'].mean():.1f}%")
        print(f"    Aligned with enrolment .... {matched}/{len(alignment)} "
              f"({matched / len(alignment):.1%}) students score highest on their own program")

        print("\n  Projected career-track demand (top-3 appearances)")
        print("  " + "-" * 74)
        print(tables["profile_track_demand"].head(8)[
            ["Career track", "Work category", "Share of cohort"]
        ].to_string(index=False))

    return written
