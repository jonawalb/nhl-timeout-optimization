"""Build publication-quality figures for the JQAS submission.

Conforms to JQAS guidelines:
- sans-serif (Helvetica/Arial), uniform style
- 300 dpi for halftone (color/grayscale)
- 1200 dpi for line drawings
- patterning (hatches) for bar charts rather than greyscale
- color allowed (free in JQAS)
- ~8pt lettering on axes
- self-explanatory legends and titles
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Load optimal-stopping module from sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimal_stopping import solve, SCORE_GRID, DT, T, implied_calltime_distribution

# JQAS-compliant style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
})

# Color palette: ColorBrewer-safe, journal-friendly
NAVY = "#1f3a93"
CRIMSON = "#a62630"
TEAL = "#117a65"
ORANGE = "#d48820"
GREY = "#525252"
LIGHT = "#cccccc"


def load_timeouts():
    return pd.read_parquet(DATA / "timeouts.parquet")


def load_plays():
    frames = [pd.read_parquet(p) for p in sorted(DATA.glob("plays_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


# ==== Figure 1: Timing distribution ====
def fig1_timing(t):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = np.arange(0, 70, 1)
    elapsed_min = t["total_elapsed_s"] / 60.0
    counts, _ = np.histogram(elapsed_min, bins=bins)
    ax.bar(bins[:-1], counts, width=1.0, color=NAVY, edgecolor="white", linewidth=0.4)
    for x, lbl in [(20, "P1/P2"), (40, "P2/P3"), (60, "End reg.")]:
        ax.axvline(x, color=GREY, linewidth=0.7, linestyle="--", alpha=0.7)
        ax.text(x, ax.get_ylim()[1] * 0.92, lbl, fontsize=8, color=GREY, ha="center")
    ax.set_xlabel("Elapsed game time at timeout call (minutes)")
    ax.set_ylabel("Number of timeouts")
    ax.set_xlim(0, 67)
    fig.savefig(FIG / "fig1_timing_distribution.pdf")
    fig.savefig(FIG / "fig1_timing_distribution.png", dpi=300)
    plt.close(fig)


# ==== Figure 2: Score-state distribution with hatching ====
def fig2_score_state(t):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sd = t["score_diff"].clip(-5, 5)
    counts = sd.value_counts().sort_index()
    diffs = counts.index.to_numpy()
    vals = counts.values
    # Patterning: trailing hatched, tied solid, leading dotted
    hatches = []
    colors = []
    for d in diffs:
        if d < 0:
            hatches.append("////")
            colors.append(CRIMSON)
        elif d == 0:
            hatches.append("")
            colors.append(NAVY)
        else:
            hatches.append("....")
            colors.append(TEAL)
    bars = ax.bar(diffs, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_xlabel("Calling-team score differential at moment of timeout (positive = leading)")
    ax.set_ylabel("Number of timeouts")
    ax.set_xticks(range(-5, 6))
    # Legend with patterns
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CRIMSON, hatch="////", edgecolor="black", label="Trailing"),
        Patch(facecolor=NAVY, edgecolor="black", label="Tied"),
        Patch(facecolor=TEAL, hatch="....", edgecolor="black", label="Leading"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False)
    fig.savefig(FIG / "fig2_score_state_distribution.pdf")
    fig.savefig(FIG / "fig2_score_state_distribution.png", dpi=300)
    plt.close(fig)


# ==== Figure 3: Joint timing × score-state heatmap ====
def fig3_joint(t):
    t2 = t.copy()
    t2["minute"] = (t2["total_elapsed_s"] // 60).clip(0, 64)
    t2["sd"] = t2["score_diff"].clip(-3, 3)
    pivot = t2.pivot_table(index="sd", columns="minute", values="game_id",
                          aggfunc="count", fill_value=0)
    pivot = pivot.reindex(index=range(-3, 4), fill_value=0)
    full_min = np.arange(0, 65)
    pivot = pivot.reindex(columns=full_min, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="YlOrRd",
                   extent=[full_min.min() - 0.5, full_min.max() + 0.5,
                           pivot.index.min() - 0.5, pivot.index.max() + 0.5])
    for x in [20, 40, 60]:
        ax.axvline(x, color=GREY, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("Elapsed game time at call (minutes)")
    ax.set_ylabel("Calling-team score differential")
    ax.set_yticks(range(-3, 4))
    cbar = fig.colorbar(im, ax=ax, label="Number of timeouts")
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(FIG / "fig3_timing_score_heatmap.pdf")
    fig.savefig(FIG / "fig3_timing_score_heatmap.png", dpi=300)
    plt.close(fig)


# ==== Figure 4: Strength state at call ====
def fig4_strength(t):
    t2 = t.copy()
    t2["caller_skater_diff"] = t2.apply(
        lambda r: r["home_skaters"] - r["away_skaters"]
        if r["calling_side"] == "home"
        else r["away_skaters"] - r["home_skaters"],
        axis=1,
    )
    counts = t2["caller_skater_diff"].value_counts().sort_index()
    counts = counts[(counts.index >= -2) & (counts.index <= 3)]
    labels = {
        -2: "5v3 SH",
        -1: "5v4 SH (or 6v5 EN-against)",
        0: "Even strength",
        1: "5v4 PP or 6v5 EN-for",
        2: "6v4 PP+EN or 5v3 PP",
        3: "6v3 PP+EN",
    }
    xs = list(counts.index)
    ys = list(counts.values)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(range(len(xs)), ys, color=NAVY, edgecolor="black", linewidth=0.5)
    # Hatch the >=+1 bars
    for i, x in enumerate(xs):
        if x >= 1:
            bars[i].set_hatch("xx")
            bars[i].set_facecolor(TEAL)
    ax.set_yticks(range(len(xs)))
    ax.set_yticklabels([labels.get(x, str(x)) for x in xs])
    ax.set_xlabel("Number of timeouts")
    ax.set_ylabel("")
    fig.savefig(FIG / "fig4_strength_state.pdf")
    fig.savefig(FIG / "fig4_strength_state.png", dpi=300)
    plt.close(fig)


# ==== Figure 5: Calling region (μ = 0.01) ====
def fig5_calling_region():
    sol = solve(mu=0.01)
    co = sol["call_optimal"].astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cmap = mpl.colors.ListedColormap([LIGHT, TEAL])
    ax.imshow(co, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
              extent=[0, T / 60, SCORE_GRID.min() - 0.5, SCORE_GRID.max() + 0.5])
    for x in [20, 40, 60]:
        ax.axvline(x, color=GREY, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("Elapsed game time (minutes)")
    ax.set_ylabel("Calling-team score differential")
    ax.set_yticks(range(-7, 8, 2))
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=LIGHT, label="Hold optimal"),
        Patch(facecolor=TEAL, label="Call optimal"),
    ], loc="upper left", frameon=False)
    fig.savefig(FIG / "fig5_calling_region_mu010.pdf")
    fig.savefig(FIG / "fig5_calling_region_mu010.png", dpi=300)
    plt.close(fig)


# ==== Figure 6: Calling region — multi-panel sensitivity ====
def fig6_sensitivity_panels():
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, sharey=True)
    cmap = mpl.colors.ListedColormap([LIGHT, TEAL])
    for ax, mu in zip(axes.flat, [0.005, 0.010, 0.020, 0.030]):
        sol = solve(mu=mu)
        co = sol["call_optimal"].astype(int)
        ax.imshow(co, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
                  extent=[0, T / 60, SCORE_GRID.min() - 0.5, SCORE_GRID.max() + 0.5])
        for x in [20, 40, 60]:
            ax.axvline(x, color=GREY, linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_title(f"μ = {mu}")
        ax.set_yticks(range(-6, 7, 3))
    for ax in axes[1, :]:
        ax.set_xlabel("Elapsed game time (minutes)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Score differential")
    fig.suptitle("Optimal calling region across boost-parameter values", y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_calling_region_sensitivity.pdf")
    fig.savefig(FIG / "fig6_calling_region_sensitivity.png", dpi=300)
    plt.close(fig)


# ==== Figure 7: Optimal vs empirical call distribution ====
def fig7_optimal_vs_empirical(t):
    sol = solve(mu=0.01)
    implied = implied_calltime_distribution(sol["call_optimal"], n_sim=10000)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.arange(0, 66, 1)
    if len(implied):
        ax.hist(implied["t_called_s"].values / 60.0, bins=bins, density=True,
                color=TEAL, alpha=0.65, edgecolor="black", linewidth=0.3,
                label=f"Optimal-stopping policy (μ = 0.01)")
    ax.hist(t["total_elapsed_s"].values / 60.0, bins=bins, density=True,
            color=CRIMSON, alpha=0.65, hatch="////", edgecolor="black",
            linewidth=0.3, label="Observed NHL coaches")
    for x in [20, 40, 60]:
        ax.axvline(x, color=GREY, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_xlabel("Elapsed game time at call (minutes)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 65)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(FIG / "fig7_optimal_vs_empirical.pdf")
    fig.savefig(FIG / "fig7_optimal_vs_empirical.png", dpi=300)
    plt.close(fig)


# ==== Figure 8: Usage rate by team ====
def fig8_team_heterogeneity(t):
    by_team = t.groupby("calling_team_abbr").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    bars = ax.barh(range(len(by_team)), by_team.values, color=NAVY,
                   edgecolor="black", linewidth=0.4)
    median = float(np.median(by_team.values))
    ax.axvline(median, color=CRIMSON, linewidth=1.0, linestyle="--",
               label=f"League median ({median:.0f})")
    ax.set_yticks(range(len(by_team)))
    ax.set_yticklabels(by_team.index, fontsize=8)
    ax.set_xlabel("Total timeouts called, 2022–2023 through 2024–2025 (3 seasons, 246 games each)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(FIG / "fig8_team_usage.pdf")
    fig.savefig(FIG / "fig8_team_usage.png", dpi=300)
    plt.close(fig)


# ==== Figure 9: Trends across seasons ====
def fig9_season_trend(t):
    by_season = t.groupby("season").size()
    seasons = sorted(by_season.index)
    labels = [f"{int(str(s)[:4])}–{str(s)[4:]}" for s in seasons]
    counts = [int(by_season[s]) for s in seasons]
    games_per_season = 1312
    rates = [c / (2 * games_per_season) for c in counts]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(range(len(seasons)), counts, color=NAVY, edgecolor="black",
                  linewidth=0.4, width=0.55)
    for i, v in enumerate(counts):
        ax.text(i, v + 3, f"{v}\n({rates[i]*100:.2f}%)", ha="center", fontsize=8)
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Regular season")
    ax.set_ylabel("Timeouts called league-wide")
    ax.set_ylim(0, max(counts) * 1.18)
    fig.savefig(FIG / "fig9_season_trend.pdf")
    fig.savefig(FIG / "fig9_season_trend.png", dpi=300)
    plt.close(fig)


# ==== Figure 10: WP-leverage map (the descriptive heatmap) ====
def fig10_wp_leverage():
    grid = pd.read_csv(DATA / "leverage_grid.csv")
    pv = grid.pivot(index="score_diff", columns="minute_of_game", values="leverage")
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    im = ax.imshow(pv.values, aspect="auto", origin="lower", cmap="viridis",
                   extent=[pv.columns.min(), pv.columns.max(),
                           pv.index.min() - 0.5, pv.index.max() + 0.5])
    for x in [20, 40, 60]:
        ax.axvline(x, color="white", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Elapsed game time (minutes)")
    ax.set_ylabel("Score differential")
    cbar = fig.colorbar(im, ax=ax, label="WP leverage  |∂π/∂d|")
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(FIG / "fig10_wp_leverage.pdf")
    fig.savefig(FIG / "fig10_wp_leverage.png", dpi=300)
    plt.close(fig)


def main():
    t = load_timeouts()
    fig1_timing(t)
    fig2_score_state(t)
    fig3_joint(t)
    fig4_strength(t)
    fig5_calling_region()
    fig6_sensitivity_panels()
    fig7_optimal_vs_empirical(t)
    fig8_team_heterogeneity(t)
    fig9_season_trend(t)
    fig10_wp_leverage()
    print(f"Wrote 10 figures to {FIG}")


if __name__ == "__main__":
    main()
