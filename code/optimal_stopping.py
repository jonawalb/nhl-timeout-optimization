"""Numerical solution of the timeout real-options problem.

Models the timeout as a one-shot transient probability boost on the calling
team's chance of scoring in the immediately-following 30-second sequence (the
post-timeout shift / faceoff). After that sequence, the game continues with
the timeout no longer available.

State: d = score differential (calling team − opponent), t = elapsed seconds.
Time grid: Δt = 30 seconds. Score grid: d ∈ {-7, ..., 7}.

Goal-arrival model: each team Poisson(λ) with λ = 0.000834 / sec — the league
empirical regulation rate for both home and away (estimated from our 3-season
corpus, 6.01 reg goals/game).

Terminal value at t = T = 3600:
    V(d, T, k=0) = 1 if d > 0, 0.5 if d = 0 (OT/SO from tied is a coin flip,
    empirically close to truth), 0 if d < 0.

Definitions and recursion:
    p_h = λ Δt = baseline P(calling team scores in next Δt)
    p_a = λ Δt = baseline P(opponent scores in next Δt)
    Calling team's "boost": μ = added probability that calling team scores in
        the immediately-following 30-second window. Equivalently, the timeout
        shifts probability mass μ from "no-goal" to "calling team goal" in the
        post-call sequence. μ is the parameter we sweep.

    V(d, t, k=0) = baseline expected win prob without timeout in pocket
        = (1 - p_h - p_a) V(d, t+Δt, 0) + p_h V(d+1, t+Δt, 0) + p_a V(d-1, t+Δt, 0)

    Call action at (d, t, k=1):
        v_call(d, t) = (1 - p_h - p_a - μ) V(d, t+Δt, 0)
                     + (p_h + μ) V(d+1, t+Δt, 0)
                     + p_a V(d-1, t+Δt, 0)

    Hold action at (d, t, k=1):
        v_hold(d, t) = (1 - p_h - p_a) V(d, t+Δt, 1)
                     + p_h V(d+1, t+Δt, 1) + p_a V(d-1, t+Δt, 1)

    V(d, t, k=1) = max(v_call, v_hold)
    OptionValue(d, t) := V(d, t, k=1) − V(d, t, k=0) ≥ 0

This formulation makes the boost state-dependent through V(d±1) − V(d),
which is itself the local WP-leverage. Calling is most valuable where leverage
is highest. Holding is valuable when better leverage is expected later.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

# Empirical from our scrape: 23,644 regulation goals across 3,936 games
# = 6.007 goals/game ≈ 0.000834 / sec per team
LAMBDA_PER_SEC = 0.000834
DT = 30  # seconds per discretization step
T = 3600
SCORE_GRID = np.arange(-7, 8)  # d ∈ {-7,...,7}
ZERO = np.where(SCORE_GRID == 0)[0][0]


def solve(mu: float, lam: float = LAMBDA_PER_SEC, dt: int = DT) -> dict:
    """Backward-induction solver.

    mu = added P(calling team scores) in the 30-second post-timeout window.
    """
    n_t = T // dt + 1
    n_d = len(SCORE_GRID)

    V_no = np.zeros((n_d, n_t))
    V_yes = np.zeros((n_d, n_t))

    for i, d in enumerate(SCORE_GRID):
        terminal = 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
        V_no[i, -1] = terminal
        V_yes[i, -1] = terminal  # at T there's no time to call; option expires worthless

    p_h = lam * dt
    p_a = lam * dt
    p_n = 1 - p_h - p_a

    call_optimal = np.zeros((n_d, n_t), dtype=bool)
    for t_idx in range(n_t - 2, -1, -1):
        for i, d in enumerate(SCORE_GRID):
            d_up = min(i + 1, n_d - 1)
            d_dn = max(i - 1, 0)
            # No-timeout value via standard recursion
            V_no[i, t_idx] = (
                p_n * V_no[i, t_idx + 1]
                + p_h * V_no[d_up, t_idx + 1]
                + p_a * V_no[d_dn, t_idx + 1]
            )
            # Hold action: continuation with timeout still available
            v_hold = (
                p_n * V_yes[i, t_idx + 1]
                + p_h * V_yes[d_up, t_idx + 1]
                + p_a * V_yes[d_dn, t_idx + 1]
            )
            # Call action: shift probability mass μ from no-goal to calling-team goal
            # in the immediately following 30s. After that, no timeout in pocket.
            p_h_boost = p_h + mu
            p_n_boost = max(0.0, p_n - mu)
            v_call = (
                p_n_boost * V_no[i, t_idx + 1]
                + p_h_boost * V_no[d_up, t_idx + 1]
                + p_a * V_no[d_dn, t_idx + 1]
            )
            if v_call > v_hold:
                V_yes[i, t_idx] = v_call
                call_optimal[i, t_idx] = True
            else:
                V_yes[i, t_idx] = v_hold

    return {
        "V_no": V_no,
        "V_yes": V_yes,
        "OptionValue": V_yes - V_no,
        "call_optimal": call_optimal,
        "mu": mu,
        "lambda": lam,
        "dt": dt,
    }


def threshold_curve(call_optimal: np.ndarray) -> dict:
    """For each t, find the score-diff range where calling is optimal.

    Returns dict mapping t (in seconds) → (d_min, d_max) of the calling region.
    """
    out = {}
    n_t = call_optimal.shape[1]
    for t_idx in range(n_t):
        t = t_idx * DT
        opt = call_optimal[:, t_idx]
        if opt.any():
            ds = SCORE_GRID[opt]
            out[t] = (int(ds.min()), int(ds.max()))
        else:
            out[t] = None
    return out


def implied_calltime_distribution(call_optimal: np.ndarray, n_sim: int = 5000, lam: float = LAMBDA_PER_SEC) -> pd.DataFrame:
    """Monte Carlo: simulate game trajectories starting tied at t=0, holding the
    timeout. The first time the trajectory enters the calling region, record
    that time and state. This is the model-implied distribution of when an
    EU-optimizing coach would call timeout, conditional on being able to.
    """
    n_t = call_optimal.shape[1]
    rng = np.random.default_rng(42)
    p_h = lam * DT
    p_a = lam * DT
    rows = []
    never_called = 0
    for _ in range(n_sim):
        d = 0
        called = False
        for t_idx in range(n_t - 1):
            d_idx = np.where(SCORE_GRID == d)[0][0]
            if call_optimal[d_idx, t_idx]:
                rows.append({"t_called_s": t_idx * DT, "d_at_call": d, "called": True})
                called = True
                break
            r = rng.random()
            if r < p_h:
                d = min(d + 1, SCORE_GRID.max())
            elif r < p_h + p_a:
                d = max(d - 1, SCORE_GRID.min())
        if not called:
            # Game ended; would have called at T regardless under δ>0 if optional
            d_idx = np.where(SCORE_GRID == d)[0][0]
            if call_optimal[d_idx, -1]:
                rows.append({"t_called_s": T, "d_at_call": d, "called": True})
            else:
                never_called += 1
    df = pd.DataFrame(rows)
    df.attrs["never_called"] = never_called
    df.attrs["n_sim"] = n_sim
    return df


def empirical_calltime_distribution() -> pd.DataFrame:
    timeouts = pd.read_parquet(DATA / "timeouts.parquet")
    return pd.DataFrame({
        "t_called_s": timeouts["total_elapsed_s"].values,
        "d_at_call": timeouts["score_diff"].values,
        "period": timeouts["period"].values,
    })


def fig_optionvalue_heatmap(solution: dict, out_path: Path, mu: float) -> None:
    ov = solution["OptionValue"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(ov, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, T / 60, SCORE_GRID.min() - 0.5, SCORE_GRID.max() + 0.5])
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Score differential (calling team)")
    ax.set_title(f"Option value of holding the timeout (δ = {mu})")
    ax.axhline(0, color="white", linewidth=0.4, alpha=0.5)
    fig.colorbar(im, ax=ax, label="V(s,t,k=1) − V(s,t,k=0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_calling_region(solution: dict, out_path: Path, mu: float) -> None:
    co = solution["call_optimal"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(co, aspect="auto", origin="lower", cmap="RdYlGn",
              extent=[0, T / 60, SCORE_GRID.min() - 0.5, SCORE_GRID.max() + 0.5],
              vmin=0, vmax=1)
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Score differential (calling team)")
    ax.set_title(f"Calling region (green = call now is optimal). δ = {mu}")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_optimal_vs_empirical(implied_df: pd.DataFrame, empirical_df: pd.DataFrame, out_path: Path, mu: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins_min = np.arange(0, 66, 1)
    if len(implied_df):
        ax.hist(implied_df["t_called_s"].values / 60.0, bins=bins_min, alpha=0.55,
                color="#16a085", density=True,
                label=f"Optimal-stopping policy (μ={mu})")
    ax.hist(empirical_df["t_called_s"].values / 60.0, bins=bins_min, alpha=0.55,
            color="#c0392b", density=True, label="Observed NHL coaches")
    ax.axvline(20, color="grey", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.axvline(40, color="grey", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Density")
    ax.set_title(f"When should timeouts be called vs. when they are (μ = {mu})")
    ax.legend()
    ax.set_xlim(0, 65)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    mus = [0.005, 0.01, 0.02, 0.03]
    results = {}
    empirical = empirical_calltime_distribution()
    empirical_median_min = float(empirical["t_called_s"].median() / 60)
    empirical_in_p3 = float((empirical["t_called_s"] >= 2400).mean())
    empirical_unused_share = 1 - 468 / 7872

    summary_rows = []
    for mu in mus:
        sol = solve(mu=mu)
        co = sol["call_optimal"]
        threshold = threshold_curve(co)

        implied = implied_calltime_distribution(co, n_sim=5000)
        implied_median_min = float(implied["t_called_s"].median() / 60) if len(implied) else None
        implied_p3_share = float((implied["t_called_s"] >= 2400).mean()) if len(implied) else None
        implied_never = implied.attrs.get("never_called", 0)
        implied_unused_share = implied_never / implied.attrs.get("n_sim", 1)

        results[mu] = {
            "threshold_curve_sample": {
                str(t): threshold[t]
                for t in [0, 600, 1200, 1800, 2400, 3000, 3300, 3500, 3580]
                if t in threshold
            },
            "implied_median_call_minute": implied_median_min,
            "implied_share_in_third_period": implied_p3_share,
            "implied_unused_share": implied_unused_share,
            "first_time_calling_optimal_at_d_minus1_seconds": next(
                (t for t in sorted(threshold)
                 if threshold[t] is not None and threshold[t][0] <= -1 <= threshold[t][1]),
                None,
            ),
            "first_time_calling_optimal_at_d_zero_seconds": next(
                (t for t in sorted(threshold)
                 if threshold[t] is not None and threshold[t][0] <= 0 <= threshold[t][1]),
                None,
            ),
        }
        summary_rows.append({
            "mu": mu,
            "implied_median_call_min": implied_median_min,
            "implied_share_in_p3": implied_p3_share,
            "implied_unused_share": implied_unused_share,
            "first_optimal_call_at_d-1_min": (
                results[mu]["first_time_calling_optimal_at_d_minus1_seconds"] / 60
                if results[mu]["first_time_calling_optimal_at_d_minus1_seconds"] is not None else None
            ),
            "first_optimal_call_at_d0_min": (
                results[mu]["first_time_calling_optimal_at_d_zero_seconds"] / 60
                if results[mu]["first_time_calling_optimal_at_d_zero_seconds"] is not None else None
            ),
        })

        # figures
        fig_optionvalue_heatmap(sol, FIG / f"option_value_mu{int(mu*1000):03d}.png", mu)
        fig_calling_region(sol, FIG / f"calling_region_mu{int(mu*1000):03d}.png", mu)
        fig_optimal_vs_empirical(implied, empirical, FIG / f"optimal_vs_empirical_mu{int(mu*1000):03d}.png", mu)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(DATA / "optimal_stopping_summary.csv", index=False)
    summary_df["empirical_median_call_min"] = empirical_median_min
    summary_df["empirical_share_in_p3"] = empirical_in_p3
    summary_df["empirical_unused_share"] = empirical_unused_share

    print("\n=== EMPIRICAL ===")
    print(f"Median call minute: {empirical_median_min:.2f}")
    print(f"Share in P3: {empirical_in_p3:.3f}")
    print(f"Unused share: {empirical_unused_share:.3f}")
    print("\n=== MODEL-OPTIMAL ===")
    print(summary_df.to_string(index=False))

    with open(DATA / "optimal_stopping_results.json", "w") as f:
        json.dump({
            "empirical": {
                "median_call_minute": empirical_median_min,
                "share_in_p3": empirical_in_p3,
                "unused_share": empirical_unused_share,
                "n": int(len(empirical)),
            },
            "model": results,
        }, f, indent=2, default=str)

    logger.info("wrote %s and %s", DATA / "optimal_stopping_summary.csv", DATA / "optimal_stopping_results.json")


if __name__ == "__main__":
    main()
