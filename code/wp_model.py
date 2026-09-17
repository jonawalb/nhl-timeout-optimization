"""Build a simple in-game win-probability model from regular-season play-by-play.

We sample one row per (game, second) at coarse resolution (every 30 game-seconds)
and label each row with the eventual home-team result (W/L; ties resolved by
post-regulation outcome since 2005-06 — we use the final regulation+OT+SO winner).

Features:
    - score_diff (home - away)
    - total_elapsed_s (regulation seconds, capped at 3600)
    - skater_diff (home_skaters - away_skaters), capped at [-2, 2]
    - is_third_period
    - log time-remaining-in-regulation interactions

This is intentionally simple. Goal: a defensible WP curve sufficient to value
timeouts. We are NOT trying to beat published WP models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATA = Path(__file__).resolve().parents[1] / "data"


def load_all_plays() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("plays_*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("No plays_*.parquet found — run scrape_nhl.py first.")
    return pd.concat(frames, ignore_index=True)


def label_home_winner(plays: pd.DataFrame) -> pd.Series:
    """Determine the home-team winner per game using the last play's score and game-end events."""
    last = plays.sort_values(["game_id", "sort_order"]).groupby("game_id").tail(1)
    home_win = (last["home_score_post"] > last["away_score_post"]).astype(int)
    home_win.index = last["game_id"].values
    return home_win


def build_state_panel(plays: pd.DataFrame, step_s: int = 30) -> pd.DataFrame:
    """For each game, sample state at step_s intervals across regulation."""
    plays = plays.sort_values(["game_id", "sort_order"]).copy()
    plays["score_diff_pre"] = plays["home_score_pre"] - plays["away_score_pre"]
    plays["skater_diff"] = plays["home_skaters"] - plays["away_skaters"]
    keep_cols = [
        "game_id",
        "total_elapsed_s",
        "score_diff_pre",
        "skater_diff",
        "period",
    ]
    p = plays[keep_cols].dropna()
    p = p[p["period"].le(3)]
    # For each game, build a forward-fill at step_s grid up to last observed elapsed time
    grids = []
    for gid, g in p.groupby("game_id"):
        max_t = int(g["total_elapsed_s"].max())
        if max_t < 60:
            continue
        ts = np.arange(0, min(max_t, 3600) + 1, step_s)
        g_sorted = (
            g.sort_values("total_elapsed_s")
            .drop_duplicates(subset="total_elapsed_s", keep="last")
        )
        idx = pd.Index(ts, name="t")
        s = (
            g_sorted.set_index("total_elapsed_s")[["score_diff_pre", "skater_diff"]]
            .reindex(idx, method="ffill")
            .fillna(0)
        )
        s["game_id"] = gid
        s["t"] = s.index
        grids.append(s.reset_index(drop=True))
    panel = pd.concat(grids, ignore_index=True)
    panel["time_remaining_reg_s"] = (3600 - panel["t"]).clip(lower=0)
    panel["skater_diff"] = panel["skater_diff"].clip(-2, 2)
    panel["score_diff_pre"] = panel["score_diff_pre"].clip(-6, 6)
    return panel


def fit_wp(panel: pd.DataFrame, home_win: pd.Series) -> sm.GLM:
    df = panel.copy()
    df["home_win"] = df["game_id"].map(home_win).astype(float)
    df = df.dropna(subset=["home_win"])
    # Features: score_diff × sqrt(time-remaining-fraction) interactions are common
    # (Lock & Nettleton 2014 style). Keep simple: polynomials + interactions.
    df["trem_frac"] = df["time_remaining_reg_s"] / 3600.0
    df["sqrt_trem"] = np.sqrt(df["trem_frac"])
    df["sd_x_trem"] = df["score_diff_pre"] * df["trem_frac"]
    df["sd_x_sqrt_trem"] = df["score_diff_pre"] * df["sqrt_trem"]
    X = df[
        [
            "score_diff_pre",
            "trem_frac",
            "sqrt_trem",
            "sd_x_trem",
            "sd_x_sqrt_trem",
            "skater_diff",
        ]
    ]
    X = sm.add_constant(X)
    y = df["home_win"].values
    model = sm.GLM(y, X, family=sm.families.Binomial())
    fit = model.fit()
    logger.info("WP model fit on %d rows", len(df))
    return fit


def predict_wp(fit: sm.GLM, score_diff: float, time_remaining_s: float, skater_diff: float = 0.0) -> float:
    """Predict home-team win probability at a given state."""
    trem_frac = max(0.0, min(1.0, time_remaining_s / 3600.0))
    sqrt_trem = np.sqrt(trem_frac)
    row = pd.DataFrame(
        [
            {
                "const": 1.0,
                "score_diff_pre": score_diff,
                "trem_frac": trem_frac,
                "sqrt_trem": sqrt_trem,
                "sd_x_trem": score_diff * trem_frac,
                "sd_x_sqrt_trem": score_diff * sqrt_trem,
                "skater_diff": skater_diff,
            }
        ]
    )
    return float(fit.predict(row).iloc[0])


def main() -> None:
    plays = load_all_plays()
    home_win = label_home_winner(plays)
    panel = build_state_panel(plays)
    fit = fit_wp(panel, home_win)
    print(fit.summary())
    # Save a small artifact
    out = DATA / "wp_model_params.csv"
    fit.params.to_csv(out)
    logger.info("WP params written to %s", out)


if __name__ == "__main__":
    main()
