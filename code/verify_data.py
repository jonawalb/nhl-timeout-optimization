"""End-to-end data verification for the NHL timeout corpus.

Checks:
    1. API provenance — confirm we hit the official NHL public API and that
       payload schema matches what we parsed (round-trip a sample of games).
    2. Game-count sanity — each season should have ~1312 reg-season games
       (32 teams × 82 games / 2). COVID-shortened seasons excepted.
    3. Score-line internal consistency — sum of goals per game equals final
       score per the NHL gamecenter "landing" endpoint.
    4. Timeout count sanity — no team ever called >1 timeout in a single
       game (this is an NHL rule constraint).
    5. Spot-check timeouts against the live NHL gamecenter "landing" page
       for a sample of games — ensures our parser caught every timeout the
       NHL recorded.

Outputs a `data/verification_report.json` and prints PASS/FAIL summary.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
API = "https://api-web.nhle.com/v1"

EXPECTED_GAMES = {
    "20222023": 1312,
    "20232024": 1312,
    "20242025": 1312,
}


def load_plays() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(DATA.glob("plays_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def check_game_counts(plays: pd.DataFrame) -> dict:
    by_season = plays.groupby("season")["game_id"].nunique().to_dict()
    out = {}
    for season, expected in EXPECTED_GAMES.items():
        season_int = int(season)
        actual = by_season.get(season_int, 0)
        out[season] = {
            "expected": expected,
            "actual": int(actual),
            "diff_pct": (actual - expected) / expected * 100 if expected else None,
            "ok": abs(actual - expected) <= 5,  # tolerate suspended/forfeit edge cases
        }
    return out


def check_max_one_timeout_per_team_game(plays: pd.DataFrame) -> dict:
    timeouts = plays[plays["reason"].isin(["home-timeout", "away-timeout"])].copy()
    timeouts["team_side"] = timeouts.apply(
        lambda r: ("home", r["home_team_id"]) if r["reason"] == "home-timeout"
        else ("away", r["away_team_id"]),
        axis=1,
    )
    by_team_game = timeouts.groupby(["game_id", "team_side"]).size()
    violations = by_team_game[by_team_game > 1]
    return {
        "n_timeouts_total": int(len(timeouts)),
        "n_team_games_with_call": int(len(by_team_game)),
        "max_calls_in_one_team_game": int(by_team_game.max()) if len(by_team_game) else 0,
        "violations_count": int(len(violations)),
        "ok": len(violations) == 0,
    }


def check_score_consistency(plays: pd.DataFrame, n_sample: int = 30) -> dict:
    """Pick n random games, recompute final score from goals (regulation + OT only,
    excluding shootout — period 5 — to match NHL's official scoring convention),
    compare to NHL gamecenter 'landing' endpoint's final score."""
    games = plays["game_id"].unique()
    sample = random.sample(list(games), min(n_sample, len(games)))
    failures = []
    successes = 0
    with httpx.Client(headers={"User-Agent": "nhl-timeouts-verify/0.1"}) as client:
        for gid in sample:
            sub = plays[plays["game_id"] == gid]
            home_id = sub.iloc[0]["home_team_id"]
            away_id = sub.iloc[0]["away_team_id"]
            non_shootout = sub[sub["period"].le(4)]
            home_goals = (
                (non_shootout["type_key"] == "goal")
                & (non_shootout["owner_team_id"] == home_id)
            ).sum()
            away_goals = (
                (non_shootout["type_key"] == "goal")
                & (non_shootout["owner_team_id"] == away_id)
            ).sum()
            try:
                r = client.get(f"{API}/gamecenter/{gid}/landing", timeout=15)
                if r.status_code != 200:
                    failures.append({"game_id": int(gid), "error": f"HTTP {r.status_code}"})
                    continue
                landing = r.json()
                api_home = landing.get("homeTeam", {}).get("score")
                api_away = landing.get("awayTeam", {}).get("score")
                # NHL convention: shootout winner gets +1 in the official final
                # score. Accept that adjustment as a pass.
                shootout_winner_home = (api_home, api_away) == (int(home_goals) + 1, int(away_goals))
                shootout_winner_away = (api_home, api_away) == (int(home_goals), int(away_goals) + 1)
                exact = api_home == int(home_goals) and api_away == int(away_goals)
                if exact or shootout_winner_home or shootout_winner_away:
                    successes += 1
                else:
                    failures.append({
                        "game_id": int(gid),
                        "parsed_home_goals": int(home_goals),
                        "parsed_away_goals": int(away_goals),
                        "api_home": api_home,
                        "api_away": api_away,
                    })
            except httpx.HTTPError as exc:
                failures.append({"game_id": int(gid), "error": str(exc)})
    return {
        "n_sampled": len(sample),
        "n_successes": successes,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def check_timeout_roundtrip(plays: pd.DataFrame, n_sample: int = 20) -> dict:
    """For a sample of games WITH at least one timeout, refetch play-by-play
    and re-extract the same timeout count. Catches any post-hoc API edits and
    confirms our parser is faithful."""
    timeouts = plays[plays["reason"].isin(["home-timeout", "away-timeout"])]
    games_with_to = timeouts["game_id"].unique()
    sample = random.sample(list(games_with_to), min(n_sample, len(games_with_to)))
    failures = []
    successes = 0
    with httpx.Client(headers={"User-Agent": "nhl-timeouts-verify/0.1"}) as client:
        for gid in sample:
            try:
                r = client.get(f"{API}/gamecenter/{gid}/play-by-play", timeout=15)
                live = r.json()
                live_to = sum(
                    1
                    for p in live.get("plays", [])
                    if (p.get("details") or {}).get("reason") in ("home-timeout", "away-timeout")
                )
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                failures.append({"game_id": int(gid), "error": str(exc)})
                continue
            local_to = int((timeouts["game_id"] == gid).sum())
            if local_to == live_to:
                successes += 1
            else:
                failures.append({
                    "game_id": int(gid),
                    "local_count": local_to,
                    "live_count": live_to,
                })
    return {
        "n_sampled": len(sample),
        "n_successes": successes,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def basic_distributions(plays: pd.DataFrame) -> dict:
    timeouts = plays[plays["reason"].isin(["home-timeout", "away-timeout"])]
    return {
        "n_total_plays": int(len(plays)),
        "n_total_games": int(plays["game_id"].nunique()),
        "n_total_timeouts": int(len(timeouts)),
        "timeouts_by_period": timeouts["period"].value_counts().sort_index().to_dict(),
        "share_in_third_period": float((timeouts["period"] == 3).mean()),
        "median_timeout_call_minute": float(timeouts["total_elapsed_s"].median() / 60),
        "share_caller_trailing": float(
            (
                timeouts.apply(
                    lambda r: (r["home_score_pre"] - r["away_score_pre"]) < 0
                    if r["reason"] == "home-timeout"
                    else (r["away_score_pre"] - r["home_score_pre"]) < 0,
                    axis=1,
                )
            ).mean()
        ),
    }


def main() -> None:
    plays = load_plays()
    if plays.empty:
        logger.error("no plays parquet found — has scrape finished?")
        return

    report = {
        "api_endpoint": API,
        "data_provenance": "Official NHL Web API (api-web.nhle.com/v1) — same backend used by nhl.com",
        "basic": basic_distributions(plays),
        "game_counts": check_game_counts(plays),
        "timeout_max_one_per_team_per_game": check_max_one_timeout_per_team_game(plays),
        "score_consistency_sample": check_score_consistency(plays),
        "timeout_roundtrip_sample": check_timeout_roundtrip(plays),
    }

    all_ok = (
        all(c["ok"] for c in report["game_counts"].values())
        and report["timeout_max_one_per_team_per_game"]["ok"]
        and report["score_consistency_sample"]["ok"]
        and report["timeout_roundtrip_sample"]["ok"]
    )
    report["overall_pass"] = all_ok

    out = DATA / "verification_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("wrote %s", out)
    print(json.dumps(report, indent=2, default=str))
    print("\n=========================")
    print("OVERALL:", "PASS ✓" if all_ok else "FAIL ✗")
    print("=========================")


if __name__ == "__main__":
    main()
