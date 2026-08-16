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
    """
    rows = []
    for course in cfg.CORE_COURSES:
        column = f"grade_{course.key}"
        series = df[column].astype(float)
        row = {
            "Course code": course.code,
            "Course": course.title,
            "Year": course.year_level,
            "Mean grade": round(series.mean(), 3),
            "Std dev": round(series.std(), 3),
            "Best": round(series.min(), 2),
            "Lowest": round(series.max(), 2),
        }
        for program in cfg.PROGRAMS:
            subset = df.loc[df["program"] == program, column].astype(float)
            row[f"{program} mean"] = round(subset.mean(), 3) if len(subset) else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)

    gwa = df["gwa"].astype(float)
    summary = {
        "Course code": "--",
        "Course": "CUMULATIVE GWA",
        "Year": "1-3",
        "Mean grade": round(gwa.mean(), 3),
        "Std dev": round(gwa.std(), 3),
        "Best": round(gwa.min(), 2),
        "Lowest": round(gwa.max(), 2),
    }
    for program in cfg.PROGRAMS:
        subset = df.loc[df["program"] == program, "gwa"].astype(float)
        summary[f"{program} mean"] = round(subset.mean(), 3) if len(subset) else np.nan

    return pd.concat([frame, pd.DataFrame([summary])], ignore_index=True)


def certification_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in cfg.CERT_DOMAINS:
        column = f"cert_{domain}"
        series = df[column].astype(float)
        holders = int(series.sum())
        rows.append({
            "Certification domain": domain.replace("_", " ").title(),
            "Students holding": holders,
            "Share of cohort": f"{holders / series.notna().sum():.1%}",
            "Example credentials": ", ".join(cfg.CERTIFICATION_DOMAINS[domain][:2]),
        })

    frame = pd.DataFrame(rows).sort_values("Students holding", ascending=False)

    engineered = feat.add_aggregate_features(df)
    total = engineered["cert_count"]
    footer = pd.DataFrame([{
        "Certification domain": "-- certificates per student --",
        "Students holding": f"mean {total.mean():.2f}",
        "Share of cohort": f"max {int(total.max())}",
        "Example credentials": f"{int((total == 0).sum())} students hold none",
    }])
    return pd.concat([frame, footer], ignore_index=True)


def skill_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean self-reported Likert score for every skill, overall and by program."""
    rows = []
    groups = (
        ("Technical", cfg.TECH_SKILL_KEYS),
        ("Hard", cfg.HARD_SKILL_KEYS),
        ("Soft", cfg.SOFT_SKILL_KEYS),
    )
    for group_label, keys in groups:
        for column in keys:
            series = df[column].astype(float)
            row = {
                "Group": group_label,
                "Skill": cfg.pretty_feature_name(column).rsplit(" (", 1)[0],
                "Mean (1-5)": round(series.mean(), 2),
                "Std dev": round(series.std(), 2),
                "% rating 4 or 5": f"{(series >= 4).sum() / series.notna().sum():.1%}",
            }
            for program in cfg.PROGRAMS:
                subset = df.loc[df["program"] == program, column].astype(float)
                row[f"{program} mean"] = round(subset.mean(), 2) if len(subset) else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["Group", "Mean (1-5)"], ascending=[True, False])


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
    outdir = outdir or cfg.FIGURE_DIR
    profile = skill_profile(df).sort_values("Mean (1-5)")

    colours = {"Technical": "#2563eb", "Hard": "#f59e0b", "Soft": "#10b981"}
    fig, ax = plt.subplots(figsize=(9, 0.30 * len(profile) + 2.0))
    bars = ax.barh(profile["Skill"], profile["Mean (1-5)"],
                   color=[colours[g] for g in profile["Group"]], edgecolor="white", height=0.72)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.axvline(3.0, ls="--", lw=1, color="#64748b")
    ax.text(3.02, -0.6, "neutral (3.0)", fontsize=8, color="#64748b")
    ax.set_xlim(0, 5.4)
    ax.set_xlabel("Mean self-reported proficiency (1 = very low, 5 = very high)")
    ax.set_title("Self-Reported Skill Profile of the Cohort\n"
                 "Technical, hard and soft skills as reported by fourth-year BSCS/BSIT students")

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
        print("\n  Academic profile (raw 1.00-5.00 grades, lower is better)")
        print("  " + "-" * 74)
        academic = tables["profile_academic"][["Course", "Mean grade", "BSCS mean", "BSIT mean"]]
        print(academic.to_string(index=False, max_colwidth=44))

        print("\n  Certification profile")
        print("  " + "-" * 74)
        print(tables["profile_certifications"][
            ["Certification domain", "Students holding", "Share of cohort"]
        ].to_string(index=False))

        print("\n  Top self-reported skills")
        print("  " + "-" * 74)
        skills = tables["profile_skills"].sort_values("Mean (1-5)", ascending=False).head(10)
        print(skills[["Group", "Skill", "Mean (1-5)", "% rating 4 or 5"]].to_string(index=False))

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
