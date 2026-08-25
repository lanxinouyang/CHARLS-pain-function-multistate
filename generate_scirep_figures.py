#!/usr/bin/env python3
"""Generate publication figures for the Scientific Reports submission package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullFormatter


REV = ROOT / "models"
OUT = ROOT / "publication_outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "navy": "#183B56",
    "blue": "#3B82A0",
    "sky": "#7CB7D3",
    "teal": "#2A9D8F",
    "orange": "#D97706",
    "vermillion": "#C94C2C",
    "gray": "#697386",
    "lightgray": "#D9DEE7",
    "pale": "#F3F7FA",
    "death": "#9AA3AE",
    "ink": "#17212B",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": "#7A8696",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label):
    ax.text(
        -0.055,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COL["ink"],
        va="top",
    )


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(
        OUT / f"{stem}.tif",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def rounded_box(ax, xy, width, height, title, detail=None, face="#F7FAFC", edge=None, fs=8.0):
    edge = edge or COL["blue"]
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + width / 2, xy[1] + height / 2
    ax.text(cx, cy + (0.20 if detail else 0), title, ha="center", va="center", fontsize=fs, fontweight="bold")
    if detail:
        ax.text(cx, cy - 0.27, detail, ha="center", va="center", fontsize=fs - 0.35, color=COL["navy"])
    return box


def arrow(ax, start, end, color=None, connectionstyle="arc3", lw=1.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color or COL["gray"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def draw_state_model(ax, states, exposure_label, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.1)
    ax.axis("off")
    ax.set_title(title, loc="left", pad=5)
    xs = [0.35, 3.90, 7.45]
    for x, (code, desc), fill in zip(xs, states, ["#E8F4F1", "#EAF2F8", "#FCEFE9"]):
        rounded_box(ax, (x, 3.20), 2.2, 1.25, code, desc, face=fill, edge=COL["navy"], fs=7.5)
    rounded_box(ax, (4.0, 0.55), 2.0, 1.0, "D", "Death", face="#ECEFF2", edge=COL["death"], fs=7.8)

    # Adjacent and cross-level transitions are all allowed among living states.
    arrow(ax, (2.57, 3.98), (3.88, 3.98), COL["vermillion"])
    arrow(ax, (3.88, 3.55), (2.57, 3.55), COL["teal"])
    arrow(ax, (6.12, 3.98), (7.43, 3.98), COL["vermillion"])
    arrow(ax, (7.43, 3.55), (6.12, 3.55), COL["teal"])
    arrow(ax, (2.52, 4.38), (7.48, 4.38), COL["vermillion"], "arc3,rad=-0.18")
    arrow(ax, (7.48, 3.24), (2.52, 3.24), COL["teal"], "arc3,rad=-0.18")
    for x in [1.45, 5.0, 8.55]:
        arrow(ax, (x, 3.20), (5.0, 1.56), COL["death"], lw=0.8)

    ax.text(0.15, 5.32, exposure_label, color=COL["navy"], fontsize=7.7, fontweight="bold")
    ax.text(0.15, 4.88, "Covariates updated or fixed as prespecified", color=COL["gray"], fontsize=7.0)
    ax.plot([0.2, 0.68], [0.27, 0.27], color=COL["vermillion"], lw=2.0)
    ax.text(0.78, 0.27, "Worsening", va="center", fontsize=6.5)
    ax.plot([3.25, 3.73], [0.27, 0.27], color=COL["teal"], lw=2.0)
    ax.text(3.83, 0.27, "Improvement", va="center", fontsize=6.5)
    ax.plot([6.45, 6.93], [0.27, 0.27], color=COL["death"], lw=2.0)
    ax.text(7.03, 0.27, "Mortality", va="center", fontsize=6.5)


def figure1():
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], hspace=0.35, wspace=0.24)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    panel_label(ax, "a")
    ax.set_title("Cohort construction and survey timeline", loc="left", pad=5)

    boxes = [
        (0.20, "Unique CHARLS IDs", "n = 25,979"),
        (3.15, "Age >=45 years", "n = 23,578"),
        (6.10, "Joint entry state and\ncomplete covariates", "n = 23,055"),
        (9.05, "Primary analysis", "20,111 people\n46,994 intervals"),
    ]
    for x, title, detail in boxes:
        rounded_box(ax, (x, 3.02), 2.55, 1.30, title, detail, fs=7.35)
    for start in [(2.78, 3.67), (5.73, 3.67), (8.68, 3.67)]:
        arrow(ax, start, (start[0] + 0.33, start[1]))
    ax.text(1.47, 2.61, "Excluded: age <45\nn = 2,401", ha="center", va="top", fontsize=6.6, color=COL["gray"])
    ax.text(4.42, 2.61, "No joint entry state: 203\nIncomplete entry covariates: 320", ha="center", va="top", fontsize=6.25, color=COL["gray"])
    ax.text(7.38, 2.05, "No analyzable interval: 2,944 people\nMissing interval-start state: 795 intervals", ha="center", va="top", fontsize=6.25, color=COL["gray"])
    ax.text(10.32, 2.50, "Observed deaths: 1,712", ha="center", fontsize=6.8, color=COL["vermillion"], fontweight="bold")
    ax.plot([0.70, 11.30], [0.82, 0.82], color=COL["navy"], lw=1.8)
    timeline = [(1.0, "2011", "13,841"), (4.4, "2015", "16,321"), (7.7, "2018", "16,832"), (11.0, "2020", "")]
    for x, year, intervals in timeline:
        ax.scatter(x, 0.82, s=43, color=COL["teal"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x, 0.48, year, ha="center", va="top", fontsize=7.5, fontweight="bold")
        if intervals:
            ax.text(x + 1.65, 1.10, f"{intervals} intervals", ha="center", fontsize=6.4, color=COL["gray"])

    axb = fig.add_subplot(gs[1, 0])
    panel_label(axb, "b")
    draw_state_model(
        axb,
        [("P0", "0 pain sites"), ("P1", "1 pain site"), ("P2", ">=2 pain\nsites")],
        "Time-varying exposure: interval-start function (F0/F1/F2)",
        "Pain-state model",
    )
    axc = fig.add_subplot(gs[1, 1])
    panel_label(axc, "c")
    draw_state_model(
        axc,
        [("F0", "No BADL/IADL\nlimitation"), ("F1", "IADL only"), ("F2", ">=1 BADL\nlimitation")],
        "Time-varying exposure: interval-start pain (P0/P1/P2)",
        "Function-state model",
    )
    save(fig, "Fig1_study_design")


def load_active(domain):
    df = pd.read_csv(REV / domain / "age_updated_household_robust_hr.csv")
    return df[df.to_state != "D"].copy()


def forest_panel(ax, df, contrasts, order, title, primary):
    colors = [COL["blue"], COL["vermillion"]]
    offsets = [-0.14, 0.14]
    y = np.arange(len(order))[::-1]
    for idx, trans in enumerate(order):
        if trans == primary:
            ax.axhspan(y[idx] - 0.43, y[idx] + 0.43, color="#FFF4DF", zorder=0)
    for contrast, color, off in zip(contrasts, colors, offsets):
        vals = []
        for a, b in order:
            vals.append(df[(df.from_state == a) & (df.to_state == b) & (df.contrast == contrast)].iloc[0])
        hr = np.array([r.hr for r in vals])
        lo = np.array([r.cluster_ci95_low for r in vals])
        hi = np.array([r.cluster_ci95_high for r in vals])
        ax.errorbar(
            hr,
            y + off,
            xerr=np.vstack([hr - lo, hi - hr]),
            fmt="o",
            color=color,
            ecolor=color,
            ms=4.0,
            capsize=2.0,
            elinewidth=1.0,
            label=contrast,
            zorder=3,
        )
    ax.axvline(1, color="#4B5563", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.35, 4.8)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} -> {b}" for a, b in order])
    ax.set_xlabel("Adjusted hazard ratio (95% CI)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color="#E4E8ED", lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=7.0, handletextpad=0.4)


def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), sharex=True)
    pain_order = [("P0", "P1"), ("P0", "P2"), ("P1", "P2"), ("P1", "P0"), ("P2", "P1"), ("P2", "P0")]
    function_order = [("F0", "F1"), ("F0", "F2"), ("F1", "F2"), ("F1", "F0"), ("F2", "F1"), ("F2", "F0")]
    panel_label(axes[0], "a")
    forest_panel(axes[0], load_active("pain"), ["F1 vs F0", "F2 vs F0"], pain_order, "Function and pain transitions", ("P0", "P2"))
    panel_label(axes[1], "b")
    forest_panel(axes[1], load_active("function"), ["P1 vs P0", "P2 vs P0"], function_order, "Pain and function transitions", ("F0", "F2"))
    fig.subplots_adjust(wspace=0.32, left=0.12, right=0.99, top=0.90, bottom=0.14)
    save(fig, "Fig2_transition_HRs")


def probability_panel(ax, domain, origin, exposures, states, colors, title, primary_dest):
    d = pd.read_csv(REV / domain / "period_specific_probabilities.csv")
    periods = ["2011–2015", "2015–2018", "2018–2020"]
    ys, labels, rows = [], [], []
    for i, period in enumerate(periods):
        for j, exposure in enumerate(exposures):
            y = i * 2.75 + j * 0.86
            ys.append(y)
            labels.append(f"{period.replace('–', '-')}  |  {exposure}")
            rows.append((period, exposure))
    for y, (period, exposure) in zip(ys, rows):
        left = 0.0
        for state, color in zip(states, colors):
            r = d[(d.period == period) & (d.exposure_state == exposure) & (d.origin_state == origin) & (d.destination_state == state)].iloc[0]
            value = 100 * r.simulation_median
            ax.barh(y, value, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.5)
            if state == primary_dest and value >= 5:
                text = ax.text(left + value / 2, y, f"{value:.1f}", ha="center", va="center", fontsize=6.4, color="white", fontweight="bold")
                text.set_path_effects([patheffects.withStroke(linewidth=1.0, foreground="black", alpha=0.20)])
            left += value
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Two-year state probability (%)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)


def figure3():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.1), sharex=True)
    panel_label(axes[0], "a")
    probability_panel(
        axes[0], "pain", "P0", ["F0", "F2"], ["P0", "P1", "P2", "D"],
        ["#CFE8E2", COL["sky"], COL["vermillion"], COL["death"]],
        "Pain outcomes from P0 under F0 versus F2", "P2"
    )
    panel_label(axes[1], "b")
    probability_panel(
        axes[1], "function", "F0", ["P0", "P2"], ["F0", "F1", "F2", "D"],
        ["#CFE8E2", COL["sky"], COL["vermillion"], COL["death"]],
        "Function outcomes from F0 under P0 versus P2", "F2"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#CFE8E2", COL["sky"], COL["vermillion"], COL["death"]]]
    axes[0].legend(handles, ["No burden", "Intermediate", "Severe", "Death"], frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.58, 1.19), fontsize=7.0)
    fig.text(0.99, 0.012, "White labels show the severe-state probability.", ha="right", fontsize=6.8, color=COL["gray"])
    fig.subplots_adjust(hspace=0.42, left=0.20, right=0.99, top=0.91, bottom=0.11)
    save(fig, "Fig3_period_probabilities")


SENS = [
    ("Primary model", 2.95, 2.07, 4.19, 2.25, 1.99, 2.55),
    ("Arthritis excluded from count", 2.81, 2.29, 3.46, 2.31, 2.05, 2.59),
    ("CESD-10, smoking, alcohol", 2.41, 2.02, 2.88, 2.42, 2.15, 2.73),
    ("Physical activity", 2.49, 1.94, 3.19, 3.03, 2.57, 3.57),
    ("State history", 2.64, 1.92, 3.64, 2.06, 1.82, 2.34),
    ("Missingness IPW", 2.97, 2.33, 3.77, 2.27, 2.02, 2.55),
    ("Combined weights", 3.53, 2.60, 4.79, 2.36, 2.09, 2.66),
    ("Mixed exact death time", 2.92, 2.28, 3.74, 2.24, 2.00, 2.52),
    ("2011-2015 only", 2.42, 1.56, 3.76, 1.99, 1.58, 2.49),
    ("2018-2020 only", 3.07, 2.28, 4.13, 2.52, 2.14, 2.97),
]


def sensitivity_panel(ax, col, title):
    y = np.arange(len(SENS))[::-1]
    hr = np.array([x[col] for x in SENS])
    lo = np.array([x[col + 1] for x in SENS])
    hi = np.array([x[col + 2] for x in SENS])
    ax.axhspan(y[0] - 0.43, y[0] + 0.43, color="#FFF4DF", zorder=0)
    ax.errorbar(hr, y, xerr=np.vstack([hr - lo, hi - hr]), fmt="o", color=COL["navy"], ecolor=COL["blue"], capsize=2.0, ms=4.2, lw=1.0)
    ax.axvline(1, color="#4B5563", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1.25, 5.3)
    ax.set_xticks([1.5, 2, 3, 4, 5])
    ax.set_xticklabels(["1.5", "2", "3", "4", "5"])
    ax.set_yticks(y)
    ax.set_yticklabels([x[0] for x in SENS])
    ax.set_xlabel("Adjusted hazard ratio (95% CI)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)


def figure4():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.1), sharey=True)
    panel_label(axes[0], "a")
    sensitivity_panel(axes[0], 1, "F2 versus F0 for P0 -> P2")
    panel_label(axes[1], "b")
    sensitivity_panel(axes[1], 4, "P2 versus P0 for F0 -> F2")
    fig.subplots_adjust(wspace=0.20, left=0.30, right=0.99, top=0.90, bottom=0.13)
    save(fig, "Fig4_sensitivity")


def supplementary_mortality():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True)
    for ax, domain, contrasts, title, label in [
        (axes[0], "pain", ["F1 vs F0", "F2 vs F0"], "Function and mortality in the pain model", "a"),
        (axes[1], "function", ["P1 vs P0", "P2 vs P0"], "Pain and mortality in the function model", "b"),
    ]:
        panel_label(ax, label)
        df = pd.read_csv(REV / domain / "age_updated_household_robust_hr.csv")
        df = df[df.to_state == "D"]
        origins = ["P0", "P1", "P2"] if domain == "pain" else ["F0", "F1", "F2"]
        y = np.arange(3)[::-1]
        for contrast, color, off in zip(contrasts, [COL["blue"], COL["vermillion"]], [-0.13, 0.13]):
            vals = [df[(df.from_state == o) & (df.contrast == contrast)].iloc[0] for o in origins]
            hr = np.array([v.hr for v in vals]); lo = np.array([v.cluster_ci95_low for v in vals]); hi = np.array([v.cluster_ci95_high for v in vals])
            ax.errorbar(hr, y + off, xerr=np.vstack([hr - lo, hi - hr]), fmt="o", color=color, ecolor=color, capsize=2, ms=4, label=contrast)
        ax.axvline(1, color="#4B5563", ls="--", lw=0.8)
        ax.set_xscale("log"); ax.set_xlim(0.35, 4.8); ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(y); ax.set_yticklabels([f"{o} -> D" for o in origins]); ax.set_title(title, loc="left"); ax.set_xlabel("Adjusted hazard ratio (95% CI)")
        ax.grid(axis="x", color="#E5E7EB", lw=0.6); ax.legend(frameon=False, fontsize=7.0, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    fig.subplots_adjust(wspace=0.32, left=0.13, right=0.99, top=0.88, bottom=0.28)
    save(fig, "FigS1_mortality_HRs")


def supplementary_counts():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65))
    for ax, domain, states, title, label in [
        (axes[0], "pain", ["P0", "P1", "P2", "D"], "Observed pain-state endpoint changes", "a"),
        (axes[1], "function", ["F0", "F1", "F2", "D"], "Observed function-state endpoint changes", "b"),
    ]:
        panel_label(ax, label)
        c = pd.read_csv(REV / domain / "observed_transition_counts.csv")
        origins = states[:-1]
        mat = np.zeros((len(origins), len(states)))
        for i, origin in enumerate(origins):
            for j, dest in enumerate(states):
                row = c[(c.from_state == origin) & (c.to_state == dest)]
                mat[i, j] = int(row.observed_endpoint_transitions.iloc[0]) if len(row) else 0
        im = ax.imshow(np.log10(mat + 1), cmap="Blues", vmin=0, vmax=np.log10(max(mat.max(), 1)))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = int(mat[i, j])
                color = "white" if np.log10(val + 1) > 0.60 * np.log10(max(mat.max(), 1)) else COL["ink"]
                ax.text(j, i, f"{val:,}" if val else "—", ha="center", va="center", fontsize=7.2, color=color, fontweight="bold" if val else "normal")
        ax.set_xticks(range(len(states))); ax.set_xticklabels(states)
        ax.set_yticks(range(len(origins))); ax.set_yticklabels(origins)
        ax.set_xlabel("Endpoint state"); ax.set_ylabel("Interval-start state")
        ax.set_title(title, loc="left")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(states), 1), minor=True); ax.set_yticks(np.arange(-.5, len(origins), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2); ax.tick_params(which="minor", bottom=False, left=False)
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.99, top=0.88, bottom=0.16)
    save(fig, "FigS2_transition_counts")


def main():
    setup_style()
    figure1()
    figure2()
    figure3()
    figure4()
    supplementary_mortality()
    supplementary_counts()
    print(OUT)


if __name__ == "__main__":
    main()
