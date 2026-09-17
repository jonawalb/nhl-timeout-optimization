"""Descriptive + treatment-effect analysis for NHL timeout calls.

Outputs:
    - figures/timing_distribution.png
    - figures/score_diff_distribution.png
    - figures/wp_at_call.png
    - data/timeout_summary.csv
    - data/timeout_wp_effects.csv

The pre/post WP comparison is descriptive (timeouts are non-randomly assigned).
We frame the EU model in terms of *option value of holding the timeout* and
estimate a deterministic Bellman-style optimum for a calling-team trailing by k
goals at time t. See findings/findings.md for the writeup.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wp_model import (
    build_state_panel,
    fit_wp,
    label_home_winner,
    load_all_plays,
    predict_wp,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
FIND = ROOT / "findings"
FIND.mkdir(exist_ok=True)


def descriptive_timeout_summary(timeouts: pd.DataFrame) -> dict:
    """Top-line summary stats across the sample."""
    n_to = len(timeouts)
    n_games = timeouts["game_id"].nunique()
    by_period = timeouts["period"].value_counts().sort_index().to_dict()
    by_period_share = (
        timeouts["period"].value_counts(normalize=True).sort_index().to_dict()
    )
    median_elapsed = timeouts["total_elapsed_s"].median()
    pct_third = (timeouts["period"] == 3).mean()
    pct_last_5 = (
        (timeouts["period"] == 3) & (timeouts["time_remaining_in_period_s"] <= 300)
    ).mean()
    pct_trailing = (timeouts["score_diff"] < 0).mean()
    pct_tied = (timeouts["score_diff"] == 0).mean()
    pct_leading = (timeouts["score_diff"] > 0).mean()
    return {
        "n_timeouts": n_to,
        "n_games_with_pbp": n_games,
        "by_period": by_period,
        "by_period_share": by_period_share,
        "median_elapsed_s": float(median_elapsed),
        "median_elapsed_mmss": f"{int(median_elapsed//60):02d}:{int(median_elapsed%60):02d}",
        "pct_in_third_period": float(pct_third),
        "pct_in_last_5_min_of_3rd": float(pct_last_5),
        "pct_trailing": float(pct_trailing),
        "pct_tied": float(pct_tied),
        "pct_leading": float(pct_leading),
    }


def games_with_no_timeout(plays: pd.DataFrame, timeouts: pd.DataFrame) -> dict:
    """Share of (team-game) observations with zero timeouts called."""
    games = plays[["game_id", "home_team_id", "away_team_id"]].drop_duplicates()
    home_obs = games.rename(columns={"home_team_id": "team_id"})[["game_id", "team_id"]]
    home_obs["side"] = "home"
    away_obs = games.rename(columns={"away_team_id": "team_id"})[["game_id", "team_id"]]
    away_obs["side"] = "away"
    obs = pd.concat([home_obs, away_obs], ignore_index=True)
    called = timeouts[["game_id", "calling_team_id", "calling_side"]].rename(
        columns={"calling_team_id": "team_id", "calling_side": "side"}
    )
    called["called"] = 1
    merged = obs.merge(called, on=["game_id", "team_id", "side"], how="left")
    merged["called"] = merged["called"].fillna(0)
    return {
        "team_games": int(len(merged)),
        "team_games_with_timeout": int(merged["called"].sum()),
        "share_with_timeout": float(merged["called"].mean()),
    }


def fig_timing_distribution(timeouts: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    elapsed_min = timeouts["total_elapsed_s"] / 60.0
    ax.hist(elapsed_min, bins=60, color="#1f3a93", edgecolor="white")
    for x in [20, 40, 60]:
        ax.axvline(x, color="#444", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Game time when timeout called (minutes)")
    ax.set_ylabel("Number of timeouts")
    ax.set_title("When do NHL teams call their timeout?")
    ax.set_xlim(0, 65)
    fig.tight_layout()
    out = FIG / "timing_distribution.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_score_diff_distribution(timeouts: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sd = timeouts["score_diff"].clip(-5, 5)
    counts = sd.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#c0392b", edgecolor="white")
    ax.set_xlabel("Calling team score differential at time of timeout (+ = leading)")
    ax.set_ylabel("Number of timeouts")
    ax.set_title("Score state when timeouts are called")
    fig.tight_layout()
    out = FIG / "score_diff_distribution.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_wp_at_call(timeouts: pd.DataFrame, fit) -> Path:
    """For each timeout, compute calling-team WP at moment of call."""
    rows = []
    for _, t in timeouts.iterrows():
        score_diff_home = t["home_score_pre"] - t["away_score_pre"]
        skater_diff_home = t["home_skaters"] - t["away_skaters"]
        wp_home = predict_wp(fit, score_diff_home, 3600 - t["total_elapsed_s"], skater_diff_home)
        wp_caller = wp_home if t["calling_side"] == "home" else 1 - wp_home
        rows.append(
            {
                "game_id": t["game_id"],
                "total_elapsed_s": t["total_elapsed_s"],
                "score_diff": t["score_diff"],
                "wp_caller": wp_caller,
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["wp_caller"], bins=40, color="#16a085", edgecolor="white")
    ax.set_xlabel("Calling team's win probability at moment of timeout")
    ax.set_ylabel("Count")
    ax.set_title("How desperate are teams when they call timeout?")
    fig.tight_layout()
    out = FIG / "wp_at_call.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    df.to_csv(DATA / "wp_at_call.csv", index=False)
    return out


def value_of_timeout_grid(fit, delta_wp_call: float = 0.01) -> pd.DataFrame:
    """Bellman-style toy: for each (score_diff, time_remaining) state, compute
    immediate value of calling now vs. holding until any future state.

    Assumes calling timeout produces a deterministic +delta_wp_call boost to
    the calling team's win probability AT THE STATE WHERE IT IS CALLED.
    Since timeouts add small advantage (rest, tactics), this is the user-set
    parameter we'll vary.

    Optimal policy: call at state s if delta_wp_call > E[delta_wp_call gained
    in the future]. With a flat boost that doesn't depend on state, the
    *gain in win probability* depends on local concavity of WP. WP is most
    sensitive to small nudges where it sits near 0.5; near 0 or 1 the boost
    has near-zero marginal impact on win.

    We compute a simple state-grid score: |dWP/d(score_diff)| as a proxy for
    the leverage of a 'momentum nudge' at each state. Higher leverage =
    higher value of using the timeout there.
    """
    score_diffs = np.arange(-3, 4)
    time_remainings = np.arange(0, 3601, 60)  # every minute
    rows = []
    for sd in score_diffs:
        for trem in time_remainings:
            wp = predict_wp(fit, float(sd), float(trem))
            wp_up = predict_wp(fit, float(sd) + 0.5, float(trem))
            wp_down = predict_wp(fit, float(sd) - 0.5, float(trem))
            # Local "leverage" = sensitivity to small score-diff perturbation
            leverage = (wp_up - wp_down)
            rows.append(
                {
                    "score_diff": sd,
                    "time_remaining_s": trem,
                    "minute_of_game": (3600 - trem) / 60,
                    "wp": wp,
                    "leverage": leverage,
                }
            )
    return pd.DataFrame(rows)


def fig_leverage_heatmap(grid: pd.DataFrame) -> Path:
    pv = grid.pivot(index="score_diff", columns="minute_of_game", values="leverage")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pv.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    ax.set_xticks(np.linspace(0, len(pv.columns) - 1, 7))
    ax.set_xticklabels([f"{int(pv.columns[int(i)])}" for i in np.linspace(0, len(pv.columns) - 1, 7)])
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Score differential (calling team perspective)")
    ax.set_title("Win-probability leverage of a 'momentum nudge' across game states")
    fig.colorbar(im, ax=ax, label="ΔWP from ±0.5 score-diff perturbation")
    fig.tight_layout()
    out = FIG / "leverage_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def empirical_timeout_treatment(plays: pd.DataFrame, timeouts: pd.DataFrame, fit) -> pd.DataFrame:
    """For each timeout, compute calling-team WP just before the call, and
    again 5 minutes later (or at game end if earlier). The *change* is the
    raw 'treatment effect' — descriptive only; selection bias is heavy.

    Pair this with a comparison cohort: stoppages where no timeout was called
    in similar score-time states — provides a naive matched control.
    """
    rows = []
    plays_idx = plays.set_index(["game_id", "sort_order"]).sort_index()
    by_game = plays.groupby("game_id")
    for _, t in timeouts.iterrows():
        gid = t["game_id"]
        try:
            g_plays = by_game.get_group(gid).sort_values("sort_order")
        except KeyError:
            continue
        target_t = t["total_elapsed_s"] + 300  # 5 minutes later
        future = g_plays[g_plays["total_elapsed_s"] >= target_t]
        if len(future) == 0:
            future_row = g_plays.iloc[-1]
        else:
            future_row = future.iloc[0]
        score_diff_home_pre = t["home_score_pre"] - t["away_score_pre"]
        score_diff_home_post = future_row["home_score_pre"] - future_row["away_score_pre"]
        wp_home_pre = predict_wp(fit, score_diff_home_pre, 3600 - t["total_elapsed_s"])
        wp_home_post = predict_wp(
            fit,
            score_diff_home_post,
            max(0, 3600 - future_row["total_elapsed_s"]),
        )
        wp_caller_pre = wp_home_pre if t["calling_side"] == "home" else 1 - wp_home_pre
        wp_caller_post = wp_home_post if t["calling_side"] == "home" else 1 - wp_home_post
        rows.append(
            {
                "game_id": gid,
                "calling_side": t["calling_side"],
                "score_diff_pre": t["score_diff"],
                "elapsed_pre": t["total_elapsed_s"],
                "elapsed_post": future_row["total_elapsed_s"],
                "wp_caller_pre": wp_caller_pre,
                "wp_caller_post": wp_caller_post,
                "delta_wp": wp_caller_post - wp_caller_pre,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    timeouts = pd.read_parquet(DATA / "timeouts.parquet")
    plays = load_all_plays()

    summary = descriptive_timeout_summary(timeouts)
    coverage = games_with_no_timeout(plays, timeouts)
    summary.update({"coverage": coverage})

    pd.Series(
        {k: v for k, v in summary.items() if not isinstance(v, dict)}
    ).to_csv(DATA / "timeout_summary.csv", header=False)
    logger.info("descriptive summary: %s", summary)

    fig_timing_distribution(timeouts)
    fig_score_diff_distribution(timeouts)

    home_win = label_home_winner(plays)
    panel = build_state_panel(plays, step_s=60)
    fit = fit_wp(panel, home_win)
    fit.params.to_csv(DATA / "wp_model_params.csv")

    fig_wp_at_call(timeouts, fit)

    grid = value_of_timeout_grid(fit)
    grid.to_csv(DATA / "leverage_grid.csv", index=False)
    fig_leverage_heatmap(grid)

    treat = empirical_timeout_treatment(plays, timeouts, fit)
    treat.to_csv(DATA / "timeout_wp_effects.csv", index=False)
    logger.info(
        "mean ΔWP over 5 min after timeout (descriptive): %.4f (n=%d)",
        treat["delta_wp"].mean(),
        len(treat),
    )

    # Save a serializable summary dict
    import json
    summary_json = {**summary}
    summary_json["mean_delta_wp_5min"] = float(treat["delta_wp"].mean())
    summary_json["median_delta_wp_5min"] = float(treat["delta_wp"].median())
    summary_json["leverage_peak_state"] = (
        grid.loc[grid["leverage"].abs().idxmax()].to_dict()
    )
    with open(DATA / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    logger.info("wrote data/summary.json")


if __name__ == "__main__":
    main()
