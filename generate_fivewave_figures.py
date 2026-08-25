#!/usr/bin/env python3
"""Generate Scientific Reports figures for the audited five-wave analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import generate_scirep_figures as g

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullFormatter


FIVE = ROOT
PACKAGE = ROOT / "publication_outputs"
OUT = PACKAGE / "figures"
SOURCE = PACKAGE / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)
g.OUT = OUT


PANEL_BLUE = "#163A63"
TEXT_BLUE = "#243B53"
MUTED_TEXT = "#6B7C93"
PANEL_EDGE = "#C8D5E3"
COOL_EDGE = "#4F86A6"
COOL_FILL = "#F3F8FB"
WARM_EDGE = "#B96850"
WARM_FILL = "#FBF1EC"
DEATH_EDGE = "#7D8DA1"
DEATH_FILL = "#F1F4F7"
WORSENING = "#BE6148"
IMPROVEMENT = "#2A8F83"
MORTALITY = "#7B879B"


def final_hr(domain: str) -> pd.DataFrame:
    return pd.read_csv(FIVE / "models" / domain / "fivewave_final_full_household_robust_hr.csv")


def _panel(ax, edge=PANEL_EDGE):
    """Add a restrained rounded panel on a plain white background."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    patch = FancyBboxPatch(
        (0.6, 0.8), 98.8, 98.2,
        boxstyle="round,pad=0.30,rounding_size=2.4",
        facecolor="#FFFFFF", edgecolor=edge, linewidth=0.65, zorder=0,
    )
    ax.add_patch(patch)


def _panel_badge(ax, label, x=5.3, y=90.7):
    ax.text(x, y + 0.2, label, ha="center", va="center", color="white",
            fontsize=17.0, fontweight="bold", zorder=4,
            bbox={"boxstyle": "circle,pad=0.30", "facecolor": PANEL_BLUE, "edgecolor": "none"})


def _soft_box(ax, xy, width, height, title, detail, edge="#1670C5", face="#F7FBFF",
              title_fs=10.8, detail_fs=10.0, title_offset=3.2, detail_offset=-3.4,
              detail_weight="normal"):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.18,rounding_size=1.05",
        facecolor=face, edgecolor=edge, linewidth=0.95, zorder=3,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + width / 2, xy[1] + height / 2
    ax.text(cx, cy + title_offset, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=PANEL_BLUE, linespacing=1.05, zorder=4)
    ax.text(cx, cy + detail_offset, detail, ha="center", va="center", fontsize=detail_fs,
            fontweight=detail_weight, color=TEXT_BLUE, linespacing=1.12, zorder=4)
    return box


def _legend_arrow(ax, x, y, color, label):
    arrow = FancyArrowPatch(
        (x, y), (x + 5.4, y), arrowstyle="-|>", mutation_scale=10.5,
        linewidth=1.10, color=color, shrinkA=0, shrinkB=0, zorder=4,
    )
    ax.add_patch(arrow)
    ax.text(x + 6.4, y, label, va="center", fontsize=8.2, color=TEXT_BLUE)


def _flow_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18,
        linewidth=1.65, color=PANEL_BLUE, shrinkA=0, shrinkB=0, zorder=4,
    )
    ax.add_patch(arrow)


def draw_state_model(ax, states, exposure_label, title, label):
    """Draw one of the two state diagrams with identical visual grammar."""
    _panel(ax)
    _panel_badge(ax, label)
    ax.text(11.5, 90.7, title, va="center", fontsize=16.3, fontweight="bold", color=PANEL_BLUE)
    ax.text(12.0, 81.2, exposure_label, color="#236E80", fontsize=8.9, fontweight="bold")
    ax.text(12.0, 74.9, "Age updated at interval start; other covariates fixed at entry",
            color=TEXT_BLUE, fontsize=8.2)
    ax.text(12.0, 70.2, "Five waves; four adjacent intervals", color=MUTED_TEXT, fontsize=7.7)

    xs = [11.0, 41.0, 71.0]
    fills = [COOL_FILL, COOL_FILL, WARM_FILL]
    edges = [COOL_EDGE, COOL_EDGE, WARM_EDGE]
    for x, (code, desc), fill, edge in zip(xs, states, fills, edges):
        _soft_box(ax, (x, 43.0), 18.0, 21.8, code, desc, edge=edge, face=fill,
                  title_fs=11.3, detail_fs=8.2)
    _soft_box(ax, (41.5, 15.3), 17.0, 14.6, "D", "Death",
              edge=DEATH_EDGE, face=DEATH_FILL, title_fs=10.6, detail_fs=8.4)

    # Adjacent worsening and improvement.
    g.arrow(ax, (29.3, 57.1), (40.7, 57.1), WORSENING, lw=1.10)
    g.arrow(ax, (40.7, 50.6), (29.3, 50.6), IMPROVEMENT, lw=1.10)
    g.arrow(ax, (59.3, 57.1), (70.7, 57.1), WORSENING, lw=1.10)
    g.arrow(ax, (70.7, 50.6), (59.3, 50.6), IMPROVEMENT, lw=1.10)
    # Direct cross-level transitions.
    g.arrow(ax, (29.0, 62.2), (71.0, 62.2), WORSENING, "arc3,rad=-0.18", lw=1.10)
    g.arrow(ax, (71.0, 44.5), (29.0, 44.5), IMPROVEMENT, "arc3,rad=-0.18", lw=1.10)
    # Death is absorbing and reachable from every living state.
    for x in [20.0, 50.0, 80.0]:
        g.arrow(ax, (x, 42.8), (50.0, 30.1), MORTALITY, lw=1.10)

    legend = FancyBboxPatch(
        (11.0, 3.9), 78.0, 7.7,
        boxstyle="round,pad=0.18,rounding_size=1.5",
        facecolor="#FFFFFF", edgecolor=PANEL_EDGE, linewidth=0.65, zorder=2,
    )
    ax.add_patch(legend)
    _legend_arrow(ax, 14.0, 7.7, WORSENING, "Worsening")
    _legend_arrow(ax, 40.7, 7.7, IMPROVEMENT, "Improvement")
    _legend_arrow(ax, 68.2, 7.7, MORTALITY, "Mortality")


def figure1() -> None:
    pd.DataFrame([
        {"flow_stage": "Unique CHARLS IDs", "n": 26347, "excluded_reason": "Not age-eligible or age unavailable", "excluded_n": 2536},
        {"flow_stage": "Age ≥45 years", "n": 23811, "excluded_reason": "Missing eligible joint entry state", "excluded_n": 277},
        {"flow_stage": "Joint entry state and complete covariates", "n": 23369, "excluded_reason": "Incomplete entry covariates", "excluded_n": 165},
        {"flow_stage": "Primary analysis", "n": 21235, "excluded_reason": "No eligible adjacent-wave interval", "excluded_n": 2134},
        {"flow_stage": "Observed death endpoints", "n": 1973, "excluded_reason": "", "excluded_n": np.nan},
    ]).to_csv(SOURCE / "Fig1_cohort_flow_source.csv", index=False)
    pd.DataFrame([
        {"interval_start_wave": 2011, "interval_end_wave": 2013, "intervals": 14621},
        {"interval_start_wave": 2013, "interval_end_wave": 2015, "intervals": 14518},
        {"interval_start_wave": 2015, "interval_end_wave": 2018, "intervals": 16269},
        {"interval_start_wave": 2018, "interval_end_wave": 2020, "intervals": 16727},
    ]).to_csv(SOURCE / "Fig1_timeline_source.csv", index=False)

    fig = plt.figure(figsize=(10.8, 7.2), facecolor="white")
    top = fig.add_axes([0.012, 0.555, 0.976, 0.430])
    left = fig.add_axes([0.012, 0.018, 0.480, 0.512])
    right = fig.add_axes([0.508, 0.018, 0.480, 0.512])

    _panel(top)
    _panel_badge(top, "a", x=5.3, y=90.5)
    top.text(11.2, 90.5, "Cohort construction and five-wave survey timeline",
             va="center", fontsize=17.4, fontweight="bold", color=PANEL_BLUE)

    boxes = [
        (4.0, "Unique CHARLS IDs", "n = 26,347", 10.6, 9.6, 3.2, -3.4),
        (28.0, "Age ≥45 years", "n = 23,811", 10.6, 9.6, 3.2, -3.4),
        (52.0, "Joint entry state and\ncomplete covariates", "n = 23,369", 10.0, 9.4, 3.6, -4.1),
        (76.0, "Primary analysis", "21,235 participants\n62,135 adjacent-wave intervals\n1,973 deaths", 10.5, 8.0, 5.2, -3.0),
    ]
    for x, title, detail, title_fs, detail_fs, title_offset, detail_offset in boxes:
        _soft_box(top, (x, 49.0), 20.3, 25.3, title, detail,
                  edge=COOL_EDGE, face=COOL_FILL, title_fs=title_fs, detail_fs=detail_fs,
                  title_offset=title_offset, detail_offset=detail_offset)
    for start, end in ((24.7, 27.4), (48.7, 51.4), (72.7, 75.4)):
        _flow_arrow(top, (start, 61.7), (end, 61.7))

    top.plot([26.2, 26.2], [43.0, 49.0], color="#627DA8", lw=1.1, ls=(0, (1.2, 2.2)))
    top.plot([50.2, 50.2], [43.0, 49.0], color="#627DA8", lw=1.1, ls=(0, (1.2, 2.2)))
    top.plot([74.2, 74.2], [43.0, 49.0], color="#627DA8", lw=1.1, ls=(0, (1.2, 2.2)))
    # Centre each exclusion note directly below its dotted connector.
    top.text(26.2, 42.7, "Not age-eligible or age unavailable\nn = 2,536",
             ha="center", va="top", fontsize=7.7, color=TEXT_BLUE)
    top.text(50.2, 42.7, "Missing eligible joint entry state, n = 277\nIncomplete entry covariates, n = 165",
             ha="center", va="top", fontsize=7.2, color=TEXT_BLUE)
    top.text(74.2, 42.7, "No eligible adjacent-wave interval\nn = 2,134",
             ha="center", va="top", fontsize=7.7, color=TEXT_BLUE)

    positions = [14.0, 32.0, 50.0, 68.0, 86.0]
    years = [2011, 2013, 2015, 2018, 2020]
    counts = [14621, 14518, 16269, 16727]
    top.plot([14.0, 86.0], [16.3, 16.3], color=PANEL_BLUE, lw=2.25, solid_capstyle="round")
    for x, year in zip(positions, years):
        top.scatter(x, 16.3, s=118, color=IMPROVEMENT, edgecolor="white", linewidth=1.0, zorder=5)
        top.text(x, 8.5, str(year), ha="center", va="center", fontsize=11.7,
                 fontweight="bold", color=PANEL_BLUE)
    for left_x, right_x, count in zip(positions[:-1], positions[1:], counts):
        top.text((left_x + right_x) / 2, 21.1, f"{count:,} intervals",
                 ha="center", va="center", fontsize=8.2, color="#236E80", fontweight="bold")

    draw_state_model(left,
                     [("P0", "0 pain sites"), ("P1", "1 pain site"), ("P2", "≥2 pain\nsites")],
                     "Time-varying exposure: function state at interval start (F0/F1/F2)",
                     "Pain-state model", "b")
    draw_state_model(right,
                     [("F0", "No BADL or IADL\nlimitation"), ("F1", "IADL limitation\nonly"), ("F2", "≥1 BADL\nlimitation")],
                     "Time-varying exposure: pain state at interval start (P0/P1/P2)",
                     "Function-state model", "c")
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(OUT / "Fig1_study_design_fivewave.svg", format="svg",
                bbox_inches="tight", pad_inches=0.06, facecolor="white")
    g.save(fig, "Fig1_study_design_fivewave")


def forest_panel(ax, data, contrasts, order, title, primary):
    colors = [g.COL["blue"], g.COL["vermillion"]]; offsets = [-0.14, 0.14]
    y = np.arange(len(order))[::-1]
    for i, trans in enumerate(order):
        if trans == primary: ax.axhspan(y[i] - 0.43, y[i] + 0.43, color="#FFF4DF", zorder=0)
    for contrast, color, offset in zip(contrasts, colors, offsets):
        values = [data[(data.from_state == a) & (data.to_state == b) & (data.contrast == contrast)].iloc[0] for a, b in order]
        hr = np.array([v.hr for v in values]); lo = np.array([v.ci95_low for v in values]); hi = np.array([v.ci95_high for v in values])
        ax.errorbar(hr, y + offset, xerr=np.vstack((hr - lo, hi - hr)), fmt="o", color=color, ecolor=color,
                    ms=4.1, capsize=2.1, elinewidth=1.0, label=contrast, zorder=3)
    ax.axvline(1, color="#4B5563", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xlim(0.35, 4.8)
    ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y); ax.set_yticklabels([f"{a} → {b}" for a, b in order]); ax.set_xlabel("Adjusted hazard ratio (95% CI)")
    ax.set_title(title, loc="left", pad=8); ax.grid(axis="x", color="#E4E8ED", lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=7.0, handletextpad=0.4)


def aligned_panel_label(ax, label, x=-0.055, y=1.035):
    ax.text(x, y, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=g.COL["ink"])


def figure2() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), sharex=True)
    pain_order = [("P0", "P1"), ("P0", "P2"), ("P1", "P2"), ("P1", "P0"), ("P2", "P1"), ("P2", "P0")]
    function_order = [("F0", "F1"), ("F0", "F2"), ("F1", "F2"), ("F1", "F0"), ("F2", "F1"), ("F2", "F0")]
    aligned_panel_label(axes[0], "a"); forest_panel(axes[0], final_hr("pain"), ["F1 vs F0", "F2 vs F0"], pain_order,
                                               "Functional state and later pain transitions", ("P0", "P2"))
    aligned_panel_label(axes[1], "b"); forest_panel(axes[1], final_hr("function"), ["P1 vs P0", "P2 vs P0"], function_order,
                                               "Pain burden and later functional transitions", ("F0", "F2"))
    fig.subplots_adjust(wspace=0.36, left=0.105, right=0.985, top=0.88, bottom=0.15)
    g.save(fig, "Fig2_transition_HRs_fivewave")


def probability_panel(ax, domain, origin, exposures, states, colors, title, primary_dest):
    data = pd.read_csv(FIVE / "models" / domain / "fivewave_final_full_period_probabilities.csv")
    periods = ["2011-2013", "2013-2015", "2015-2018", "2018-2020"]
    ys = []; labels = []; keys = []
    for i, period in enumerate(periods):
        for j, exposure in enumerate(exposures):
            ys.append(i * 2.55 + j * 0.80); labels.append(f"{period}  |  {exposure}"); keys.append((period, exposure))
    for y, (period, exposure) in zip(ys, keys):
        left = 0.0
        for state, color in zip(states, colors):
            row = data[(data.period == period) & (data.exposure_state == exposure) &
                       (data.origin_state == origin) & (data.destination_state == state)].iloc[0]
            value = 100 * row.point_estimate
            ax.barh(y, value, left=left, height=0.58, color=color, edgecolor="white", linewidth=0.5)
            if state == primary_dest and value >= 4:
                text = ax.text(left + value / 2, y, f"{value:.1f}", ha="center", va="center", fontsize=6.2,
                               color="white", fontweight="bold")
                text.set_path_effects([patheffects.withStroke(linewidth=1.0, foreground="black", alpha=0.20)])
            left += value
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("Modelled two-year state probability (%)"); ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="x", color="#E5E7EB", lw=0.6); ax.set_axisbelow(True)


def figure3() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True)
    aligned_panel_label(axes[0], "a")
    probability_panel(axes[0], "pain", "P0", ["F0", "F2"], ["P0", "P1", "P2", "D"],
                      ["#CFE8E2", g.COL["sky"], g.COL["vermillion"], g.COL["death"]],
                      "Pain outcomes from P0, comparing F0 with F2", "P2")
    aligned_panel_label(axes[1], "b")
    probability_panel(axes[1], "function", "F0", ["P0", "P2"], ["F0", "F1", "F2", "D"],
                      ["#CFE8E2", g.COL["sky"], g.COL["vermillion"], g.COL["death"]],
                      "Functional outcomes from F0, comparing P0 with P2", "F2")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#CFE8E2", g.COL["sky"], g.COL["vermillion"], g.COL["death"]]]
    axes[0].legend(handles, ["No burden", "Intermediate", "Severe", "Death"], frameon=False, ncol=4,
                   loc="upper center", bbox_to_anchor=(0.60, 1.22), fontsize=7.0)
    fig.text(0.99, 0.012, "White labels show fitted severe-state probabilities.", ha="right", fontsize=6.8, color=g.COL["gray"])
    fig.subplots_adjust(hspace=0.46, left=0.205, right=0.985, top=0.90, bottom=0.10)
    g.save(fig, "Fig3_period_probabilities_fivewave")


def figure4() -> None:
    rows = []
    for domain in ("pain", "function"):
        audit = json.loads((FIVE / "models" / domain / "fivewave_primary_audit.json").read_text())
        values = [("78: family-shared", audit["shared_fit"]["negative_log_likelihood"]),
                  ("110: living-specific, death-shared", audit["primary_fit"]["negative_log_likelihood"]),
                  ("126: all transitions specific", audit["full_fit"]["negative_log_likelihood"])]
        reference = values[-1][1]
        for label, nll in values: rows.append({"domain": domain, "model": label, "delta_nll": nll - reference})
    source = pd.DataFrame(rows); source.to_csv(SOURCE / "Fig4_model_selection_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True)
    for ax, domain, label in zip(axes, ("pain", "function"), ("a", "b")):
        aligned_panel_label(ax, label, x=-0.06, y=1.04); data = source[source.domain == domain]
        y = np.arange(3)[::-1]; colors = [g.COL["gray"], g.COL["blue"], g.COL["vermillion"]]
        ax.barh(y, data.delta_nll, color=colors, height=0.58)
        for yi, value in zip(y, data.delta_nll): ax.text(value + max(data.delta_nll.max() * 0.025, 0.3), yi, f"{value:.1f}", va="center", fontsize=7.2)
        short_labels = ["Family-shared (78)", "Living-specific; death-shared (110)", "Transition-specific (126)"]
        ax.set_yticks(y); ax.set_yticklabels(short_labels)
        ax.set_title(f"{domain.capitalize()} model", loc="left", pad=8); ax.grid(axis="x", color="#E5E7EB", lw=0.6); ax.set_axisbelow(True)
        comparison = json.loads((FIVE / "models" / domain / "fivewave_final_full_audit.json").read_text())["comparisons"]
        p = comparison["primary_110_vs_full_126"]["p_value"]
        ax.text(0.98, 0.05, f"110 vs 126: P={p:.3f}", transform=ax.transAxes, ha="right", fontsize=7.1, color=g.COL["navy"])
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.supxlabel("Increase in negative log-likelihood relative to the transition-specific model", fontsize=8.5, y=0.04)
    fig.subplots_adjust(wspace=0.42, left=0.26, right=0.985, top=0.84, bottom=0.22)
    g.save(fig, "Fig4_model_structure_fivewave")


def supplementary_mortality() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True)
    for ax, domain, contrasts, title, label in [
        (axes[0], "pain", ["F1 vs F0", "F2 vs F0"], "Function and mortality in the pain model", "a"),
        (axes[1], "function", ["P1 vs P0", "P2 vs P0"], "Pain and mortality in the function model", "b")]:
        aligned_panel_label(ax, label); data = final_hr(domain); data = data[data.to_state == "D"]
        origins = ["P0", "P1", "P2"] if domain == "pain" else ["F0", "F1", "F2"]; y = np.arange(3)[::-1]
        for contrast, color, offset in zip(contrasts, [g.COL["blue"], g.COL["vermillion"]], [-0.13, 0.13]):
            values = [data[(data.from_state == origin) & (data.contrast == contrast)].iloc[0] for origin in origins]
            hr = np.array([v.hr for v in values]); lo = np.array([v.ci95_low for v in values]); hi = np.array([v.ci95_high for v in values])
            ax.errorbar(hr, y + offset, xerr=np.vstack((hr - lo, hi - hr)), fmt="o", color=color, ecolor=color,
                        capsize=2, ms=4, label=contrast)
        ax.axvline(1, color="#4B5563", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xlim(0.03, 40)
        ax.set_xticks([0.05, 0.1, 0.5, 1, 2, 10, 40]); ax.set_xticklabels(["0.05", "0.1", "0.5", "1", "2", "10", "40"]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(y); ax.set_yticklabels([f"{o} → D" for o in origins]); ax.set_title(title, loc="left")
        ax.set_xlabel("Adjusted hazard ratio (95% CI)"); ax.grid(axis="x", color="#E5E7EB", lw=0.6)
        ax.legend(frameon=False, fontsize=7.0, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    fig.subplots_adjust(wspace=0.32, left=0.13, right=0.99, top=0.88, bottom=0.28)
    g.save(fig, "FigS1_mortality_HRs_fivewave")


def supplementary_counts() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65))
    source_rows = []
    for ax, domain, states, title, label in [
        (axes[0], "pain", ["P0", "P1", "P2", "D"], "Observed pain-state endpoint changes", "a"),
        (axes[1], "function", ["F0", "F1", "F2", "D"], "Observed function-state endpoint changes", "b")]:
        aligned_panel_label(ax, label); data = pd.read_csv(FIVE / "models" / domain / "fivewave_primary_intervals.csv")
        data = data[data.to_state != "ALIVE_UNKNOWN"]
        origins = states[:-1]; matrix = np.zeros((3, 4))
        for i, origin in enumerate(origins):
            for j, destination in enumerate(states):
                matrix[i, j] = ((data.from_state == origin) & (data.to_state == destination)).sum()
                source_rows.append({"domain": domain, "from_state": origin, "to_state": destination, "count": int(matrix[i, j])})
        ax.imshow(np.log10(matrix + 1), cmap="Blues", vmin=0, vmax=np.log10(max(matrix.max(), 1)))
        for i in range(3):
            for j in range(4):
                value = int(matrix[i, j]); color = "white" if np.log10(value + 1) > 0.60 * np.log10(max(matrix.max(), 1)) else g.COL["ink"]
                ax.text(j, i, f"{value:,}" if value else "-", ha="center", va="center", fontsize=7.2, color=color, fontweight="bold" if value else "normal")
        ax.set_xticks(range(4)); ax.set_xticklabels(states); ax.set_yticks(range(3)); ax.set_yticklabels(origins)
        ax.set_xlabel("Endpoint state"); ax.set_ylabel("Interval-start state"); ax.set_title(title, loc="left")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2); ax.tick_params(which="minor", bottom=False, left=False)
    pd.DataFrame(source_rows).to_csv(SOURCE / "FigS2_transition_counts_source.csv", index=False)
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.99, top=0.88, bottom=0.16)
    g.save(fig, "FigS2_transition_counts_fivewave")


def supplementary_sensitivity() -> None:
    pain = pd.read_csv(FIVE / "models" / "pain" / "fivewave_final_sensitivity_summary.csv")
    function = pd.read_csv(FIVE / "models" / "function" / "fivewave_final_sensitivity_summary.csv")
    primary = pd.DataFrame([
        {"analysis": "primary five-wave", "domain": "pain", "hr": final_hr("pain").query("from_state=='P0' and to_state=='P2' and contrast=='F2 vs F0'").hr.iloc[0]},
        {"analysis": "primary five-wave", "domain": "function", "hr": final_hr("function").query("from_state=='F0' and to_state=='F2' and contrast=='P2 vs P0'").hr.iloc[0]},
    ])
    source = pd.concat([primary, pain[["analysis", "domain", "hr"]], function[["analysis", "domain", "hr"]]], ignore_index=True)
    source.to_csv(SOURCE / "FigS3_sensitivity_point_estimates_source.csv", index=False)
    labels = ["Five-wave primary", "Cross-sectional weighted", "Strict 11-item function", "2013 exit deaths only", "Excluding 2013 observations"]
    keys = ["primary five-wave", "cross-sectional-weighted", "strict-complete-11-item-function", "2013-exit-interview-deaths-only", "leave-2013-out"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    for ax, domain, title, label in zip(axes, ("pain", "function"),
                                        ("F2 versus F0 for P0 -> P2", "P2 versus P0 for F0 -> F2"), ("a", "b")):
        aligned_panel_label(ax, label); values = [source[(source.domain == domain) & (source.analysis == key)].hr.iloc[0] for key in keys]
        y = np.arange(len(values))[::-1]; ax.axhspan(y[0] - 0.42, y[0] + 0.42, color="#FFF4DF")
        ax.scatter(values, y, s=28, color=g.COL["navy"], zorder=3); ax.axvline(1, color="#4B5563", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(1.0, 2.6); ax.set_xlabel("Adjusted hazard ratio (point estimate)")
        ax.set_title(title, loc="left"); ax.grid(axis="x", color="#E5E7EB", lw=0.6)
        for value, yi in zip(values, y): ax.text(value + 0.035, yi, f"{value:.2f}", va="center", fontsize=7.0)
    fig.subplots_adjust(wspace=0.25, left=0.30, right=0.98, top=0.87, bottom=0.16)
    g.save(fig, "FigS3_sensitivity_point_estimates_fivewave")


def export_source_data() -> None:
    for domain in ("pain", "function"):
        final_hr(domain).to_csv(SOURCE / f"Fig2_{domain}_transition_HRs.csv", index=False)
        pd.read_csv(FIVE / "models" / domain / "fivewave_final_full_period_probabilities.csv").to_csv(
            SOURCE / f"Fig3_{domain}_period_probabilities.csv", index=False)
        pd.read_csv(FIVE / "models" / domain / "fivewave_final_sensitivity_summary.csv").to_csv(
            SOURCE / f"Sensitivity_{domain}_summary.csv", index=False)


def main() -> None:
    g.setup_style(); export_source_data(); figure1(); figure2(); figure3(); figure4()
    supplementary_mortality(); supplementary_counts(); supplementary_sensitivity()
    print(OUT)


if __name__ == "__main__":
    main()
