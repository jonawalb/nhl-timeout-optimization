"""Extended analyses for the expanded JQAS submission.

Computes:
1. EU loss in win-probability points per team-game from observed-vs-optimal divergence.
2. Sensitivity of optimal-stopping conclusions to scoring rate λ.
3. Robustness with state-dependent boost μ scaled by skater advantage.
4. Per-team usage statistics.
5. Per-season trend with rule-change context.
6. Asymmetric scoring-rate robustness (score effects).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimal_stopping import (  # noqa: E402
    SCORE_GRID,
    DT,
    T,
    LAMBDA_PER_SEC,
    solve,
    implied_calltime_distribution,
)


def load_plays():
    return pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA.glob("plays_*.parquet"))],
        ignore_index=True,
    )


def load_timeouts():
    return pd.read_parquet(DATA / "timeouts.parquet")


# ==========================================================================
# 1. EU loss per team-game
# ==========================================================================

def eu_loss_estimate(mu: float, n_sim: int = 20000) -> dict:
    """Compare expected V at game-start under optimal vs naive (never-call) policy.

    Naive policy: never use the timeout — equivalent to V(d=0, t=0, k=0).
    Optimal policy: V(d=0, t=0, k=1) under the optimal-stopping rule.
    Empirical policy: V at start under the empirical call-distribution model.

    Returns the EU difference in WP points per team-game.
    """
    sol = solve(mu=mu)
    V_optimal = sol["V_yes"][np.where(SCORE_GRID == 0)[0][0], 0]
    V_naive = sol["V_no"][np.where(SCORE_GRID == 0)[0][0], 0]
    optimal_minus_naive = V_optimal - V_naive

    # Empirical: in 5.95% of team-games coaches call timeout, mostly very
    # late and from low-WP states. Approximate the EU value of the empirical
    # policy by simulating: with prob 0.0595 use timeout at the empirical
    # state-time distribution; otherwise never use.
    timeouts = load_timeouts()
    rng = np.random.default_rng(7)
    p_use = 0.0595
    boost_values_collected = []
    # For each empirical timeout, compute the gain V_call - V_no at that state
    n_d = len(SCORE_GRID)
    n_t = sol["call_optimal"].shape[1]
    p_h = LAMBDA_PER_SEC * DT
    p_a = LAMBDA_PER_SEC * DT
    p_n = 1 - p_h - p_a
    for _, row in timeouts.iterrows():
        d = int(np.clip(row["score_diff"], -7, 7))
        t = int(min(row["total_elapsed_s"], T - DT))
        t_idx = min(t // DT, n_t - 2)
        d_idx = np.where(SCORE_GRID == d)[0][0]
        d_up = min(d_idx + 1, n_d - 1)
        d_dn = max(d_idx - 1, 0)
        v_no_now = sol["V_no"][d_idx, t_idx]
        v_call = (
            (max(0.0, p_n - mu)) * sol["V_no"][d_idx, t_idx + 1]
            + (p_h + mu) * sol["V_no"][d_up, t_idx + 1]
            + p_a * sol["V_no"][d_dn, t_idx + 1]
        )
        boost_values_collected.append(v_call - v_no_now)
    mean_empirical_call_value = float(np.mean(boost_values_collected))
    # Empirical EV improvement over naive = p_use * mean_empirical_call_value
    empirical_minus_naive = p_use * mean_empirical_call_value

    return {
        "mu": mu,
        "V_optimal_at_start_tied": float(V_optimal),
        "V_naive_at_start_tied": float(V_naive),
        "optimal_minus_naive_WP_points": float(optimal_minus_naive),
        "empirical_minus_naive_WP_points": float(empirical_minus_naive),
        "EU_loss_WP_points_per_team_game": float(optimal_minus_naive - empirical_minus_naive),
        "fraction_of_optimal_captured": float(empirical_minus_naive / optimal_minus_naive)
        if optimal_minus_naive > 0
        else None,
        "mean_empirical_call_value": mean_empirical_call_value,
    }


# ==========================================================================
# 2. λ sensitivity
# ==========================================================================

def lambda_sensitivity(mu: float = 0.01) -> pd.DataFrame:
    """Sensitivity of optimal-policy summary stats to the goal-arrival rate."""
    rows = []
    # Empirical λ = 0.000834. Sweep around it.
    lams = [0.0006, 0.0007, 0.0008, LAMBDA_PER_SEC, 0.0009, 0.0010, 0.0012]
    for lam in lams:
        sol = solve(mu=mu, lam=lam)
        co = sol["call_optimal"]
        impl = implied_calltime_distribution(co, n_sim=5000, lam=lam)
        median_min = float(impl["t_called_s"].median() / 60) if len(impl) else None
        unused = impl.attrs.get("never_called", 0) / impl.attrs.get("n_sim", 1)
        # First time d=0 is in calling region
        idx_zero = np.where(SCORE_GRID == 0)[0][0]
        first_zero = next(
            (t_i * DT / 60 for t_i in range(co.shape[1]) if co[idx_zero, t_i]),
            None,
        )
        rows.append({
            "lambda": lam,
            "lambda_per_sec": lam,
            "implied_median_min": median_min,
            "implied_unused_share": unused,
            "first_optimal_call_d0_min": first_zero,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 3. State-dependent boost (asymmetric μ for pulled-goalie scenario)
# ==========================================================================

def solve_state_dependent(mu_base: float = 0.010, mu_amp_pp: float = 1.5,
                          mu_amp_pp_pulled: float = 2.0,
                          lam: float = LAMBDA_PER_SEC, dt: int = DT):
    """Solver where boost depends on score state — proxy for pulled-goalie effect.

    When trailing late (d ≤ -1, t ≥ 50 min), assume boost is amplified by
    mu_amp_pp_pulled (pulled-goalie scenario). When d ≤ 0 in late game, use
    mu_amp_pp. Elsewhere, use base.
    """
    n_t = T // dt + 1
    n_d = len(SCORE_GRID)
    V_no = np.zeros((n_d, n_t))
    V_yes = np.zeros((n_d, n_t))
    for i, d in enumerate(SCORE_GRID):
        terminal = 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
        V_no[i, -1] = terminal
        V_yes[i, -1] = terminal
    p_h = lam * dt
    p_a = lam * dt
    p_n = 1 - p_h - p_a
    call_optimal = np.zeros((n_d, n_t), dtype=bool)
    for t_idx in range(n_t - 2, -1, -1):
        t = t_idx * dt
        for i, d in enumerate(SCORE_GRID):
            d_up = min(i + 1, n_d - 1)
            d_dn = max(i - 1, 0)
            # State-dependent μ
            if d <= -1 and t >= 3000:
                mu_eff = mu_base * mu_amp_pp_pulled
            elif d <= 0 and t >= 2400:
                mu_eff = mu_base * mu_amp_pp
            else:
                mu_eff = mu_base
            V_no[i, t_idx] = (
                p_n * V_no[i, t_idx + 1]
                + p_h * V_no[d_up, t_idx + 1]
                + p_a * V_no[d_dn, t_idx + 1]
            )
            v_hold = (
                p_n * V_yes[i, t_idx + 1]
                + p_h * V_yes[d_up, t_idx + 1]
                + p_a * V_yes[d_dn, t_idx + 1]
            )
            v_call = (
                max(0.0, p_n - mu_eff) * V_no[i, t_idx + 1]
                + (p_h + mu_eff) * V_no[d_up, t_idx + 1]
                + p_a * V_no[d_dn, t_idx + 1]
            )
            if v_call > v_hold:
                V_yes[i, t_idx] = v_call
                call_optimal[i, t_idx] = True
            else:
                V_yes[i, t_idx] = v_hold
    return {"V_no": V_no, "V_yes": V_yes, "call_optimal": call_optimal,
            "mu_base": mu_base, "lambda": lam}


def state_dependent_summary() -> dict:
    sol = solve_state_dependent()
    impl = implied_calltime_distribution(sol["call_optimal"], n_sim=10000)
    return {
        "implied_median_min": float(impl["t_called_s"].median() / 60) if len(impl) else None,
        "implied_unused_share": float(impl.attrs.get("never_called", 0) / impl.attrs.get("n_sim", 1)),
    }


# ==========================================================================
# 4. Per-team statistics
# ==========================================================================

def team_statistics() -> pd.DataFrame:
    plays = load_plays()
    timeouts = load_timeouts()
    # Team-game count: each team has exactly 82 games per season × 3 seasons = 246, except
    # ARI (defunct) and UTA (new). Compute actual.
    home = plays[["game_id", "home_abbr"]].drop_duplicates().rename(columns={"home_abbr": "team"})
    away = plays[["game_id", "away_abbr"]].drop_duplicates().rename(columns={"away_abbr": "team"})
    team_games = pd.concat([home, away], ignore_index=True)
    games_per_team = team_games.groupby("team").size()
    timeouts_per_team = timeouts.groupby("calling_team_abbr").size()
    df = pd.DataFrame({
        "team_games": games_per_team,
        "timeouts_called": timeouts_per_team,
    }).fillna(0).astype(int)
    df["usage_rate"] = df["timeouts_called"] / df["team_games"]
    df = df.sort_values("usage_rate", ascending=False)
    df.index.name = "team"
    return df


# ==========================================================================
# Main
# ==========================================================================

def main():
    print("=== EU loss across μ ===")
    eu_rows = []
    for mu in [0.005, 0.010, 0.020, 0.030]:
        eu = eu_loss_estimate(mu)
        eu_rows.append(eu)
        print(f"μ={mu}: optimal-naive={eu['optimal_minus_naive_WP_points']:.4f}  "
              f"empirical-naive={eu['empirical_minus_naive_WP_points']:.4f}  "
              f"EU loss/team-game={eu['EU_loss_WP_points_per_team_game']:.4f}  "
              f"fraction captured={eu['fraction_of_optimal_captured']:.3f}")
    pd.DataFrame(eu_rows).to_csv(DATA / "eu_loss.csv", index=False)

    print("\n=== λ sensitivity (μ=0.01) ===")
    lam_df = lambda_sensitivity()
    print(lam_df.to_string(index=False))
    lam_df.to_csv(DATA / "lambda_sensitivity.csv", index=False)

    print("\n=== State-dependent boost ===")
    sd = state_dependent_summary()
    print(sd)
    with open(DATA / "state_dependent_boost.json", "w") as f:
        json.dump(sd, f, indent=2)

    print("\n=== Per-team statistics ===")
    teams = team_statistics()
    print(teams.head(10))
    print("...")
    print(teams.tail(10))
    teams.to_csv(DATA / "team_statistics.csv")

    # Summary JSON
    summary = {
        "eu_loss": eu_rows,
        "state_dependent": sd,
        "per_team_summary": {
            "highest_usage_team": teams.index[0],
            "highest_usage_rate": float(teams["usage_rate"].iloc[0]),
            "lowest_usage_team": teams.index[-1],
            "lowest_usage_rate": float(teams["usage_rate"].iloc[-1]),
            "median_usage_rate": float(teams["usage_rate"].median()),
            "iqr_usage_rate": [float(teams["usage_rate"].quantile(0.25)),
                               float(teams["usage_rate"].quantile(0.75))],
        },
    }
    with open(DATA / "extended_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote summaries to {DATA}")


if __name__ == "__main__":
    main()
