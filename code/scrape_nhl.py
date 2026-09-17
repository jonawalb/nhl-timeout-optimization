"""Scrape NHL play-by-play for regular-season games across seasons.

Extracts every play, plus a tagged frame of timeout events with context
(period, game-clock, score differential, strength state, home/away,
team identities). Output: data/plays_<season>.parquet and data/timeouts.parquet.

NHL public API: https://api-web.nhle.com/v1/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

API = "https://api-web.nhle.com/v1"
DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

# 32-team list (tri-codes). We'll iterate club-schedule-season for each.
TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
    # Legacy code for Arizona Coyotes (now Utah) — needed for older seasons
    "ARI",
]


def fetch_schedule_for_team(client: httpx.Client, team: str, season: str) -> list[dict]:
    """Return list of regular-season games for a team-season."""
    url = f"{API}/club-schedule-season/{team}/{season}"
    try:
        r = client.get(url, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("schedule fetch failed for %s %s: %s", team, season, exc)
        return []
    return [g for g in data.get("games", []) if g.get("gameType") == 2]


def fetch_play_by_play(client: httpx.Client, game_id: int) -> dict | None:
    url = f"{API}/gamecenter/{game_id}/play-by-play"
    for attempt in range(3):
        try:
            r = client.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except httpx.HTTPError as exc:
            logger.warning("pbp attempt %d failed for %s: %s", attempt + 1, game_id, exc)
        time.sleep(0.6 * (attempt + 1))
    return None


def parse_situation_code(code: str | None) -> tuple[int, int, int, int]:
    """situationCode is 4 digits: away_goalie | away_skaters | home_skaters | home_goalie."""
    if not code or len(code) != 4 or not code.isdigit():
        return (1, 5, 5, 1)
    return (int(code[0]), int(code[1]), int(code[2]), int(code[3]))


def time_to_seconds(t: str) -> int:
    if not t or ":" not in t:
        return 0
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def extract_plays(pbp: dict) -> pd.DataFrame:
    game_id = pbp["id"]
    season = pbp.get("season")
    home_id = pbp["homeTeam"]["id"]
    away_id = pbp["awayTeam"]["id"]
    home_abbr = pbp["homeTeam"].get("abbrev")
    away_abbr = pbp["awayTeam"].get("abbrev")
    rows = []
    home_score, away_score = 0, 0
    for p in pbp.get("plays", []):
        period = p.get("periodDescriptor", {}).get("number")
        period_type = p.get("periodDescriptor", {}).get("periodType")
        tip = p.get("timeInPeriod") or "00:00"
        trem = p.get("timeRemaining") or "00:00"
        seconds_in_period = time_to_seconds(tip)
        # Total elapsed across regulation: each period is 1200 s
        if period and period <= 3:
            total_elapsed = (period - 1) * 1200 + seconds_in_period
        else:
            total_elapsed = 3 * 1200 + max(0, seconds_in_period)
        ag, as_, hs, hg = parse_situation_code(p.get("situationCode"))
        type_key = p.get("typeDescKey")
        details = p.get("details", {}) or {}
        reason = details.get("reason")

        # Track score AFTER goal events so subsequent rows reflect new score
        scoring_team_id = details.get("eventOwnerTeamId") if type_key == "goal" else None

        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "event_id": p.get("eventId"),
                "sort_order": p.get("sortOrder"),
                "period": period,
                "period_type": period_type,
                "time_in_period_s": seconds_in_period,
                "time_remaining_in_period_s": time_to_seconds(trem),
                "total_elapsed_s": total_elapsed,
                "type_key": type_key,
                "reason": reason,
                "situation_code": p.get("situationCode"),
                "away_goalie": ag,
                "away_skaters": as_,
                "home_skaters": hs,
                "home_goalie": hg,
                "owner_team_id": details.get("eventOwnerTeamId"),
                "x_coord": details.get("xCoord"),
                "y_coord": details.get("yCoord"),
                "home_score_pre": home_score,
                "away_score_pre": away_score,
            }
        )
        if scoring_team_id == home_id:
            home_score += 1
        elif scoring_team_id == away_id:
            away_score += 1
    df = pd.DataFrame(rows)
    df["home_score_post"] = df["home_score_pre"]
    df["away_score_post"] = df["away_score_pre"]
    # Update post-score for goals only
    is_goal = df["type_key"] == "goal"
    is_home_goal = is_goal & (df["owner_team_id"] == home_id)
    is_away_goal = is_goal & (df["owner_team_id"] == away_id)
    df.loc[is_home_goal, "home_score_post"] = df.loc[is_home_goal, "home_score_pre"] + 1
    df.loc[is_away_goal, "away_score_post"] = df.loc[is_away_goal, "away_score_pre"] + 1
    return df


def scrape_season(season: str, sleep_s: float = 0.05) -> pd.DataFrame:
    """Scrape all regular-season games for a season. Returns full play-by-play frame."""
    seen_games: set[int] = set()
    games: list[dict] = []
    with httpx.Client(headers={"User-Agent": "nhl-timeouts-research/0.1"}) as client:
        logger.info("collecting %s schedules across %d teams...", season, len(TEAMS))
        for team in tqdm(TEAMS, desc=f"sched-{season}"):
            for g in fetch_schedule_for_team(client, team, season):
                gid = g.get("id")
                if gid and gid not in seen_games and g.get("gameState") in {"OFF", "FINAL"}:
                    seen_games.add(gid)
                    games.append(g)
        logger.info("found %d unique completed reg-season games for %s", len(games), season)

        all_frames: list[pd.DataFrame] = []
        for g in tqdm(games, desc=f"pbp-{season}"):
            pbp = fetch_play_by_play(client, g["id"])
            if pbp is None:
                continue
            try:
                frame = extract_plays(pbp)
            except (KeyError, TypeError) as exc:
                logger.warning("parse failed for game %s: %s", g["id"], exc)
                continue
            all_frames.append(frame)
            time.sleep(sleep_s)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def main() -> None:
    seasons = ["20222023", "20232024", "20242025"]
    for season in seasons:
        out = DATA / f"plays_{season}.parquet"
        if out.exists():
            logger.info("skipping %s — already scraped (%s)", season, out)
            continue
        df = scrape_season(season)
        if df.empty:
            logger.warning("no data for %s", season)
            continue
        df.to_parquet(out, index=False)
        logger.info("wrote %d rows to %s", len(df), out)

    # Build consolidated timeouts table
    frames = []
    for season in seasons:
        p = DATA / f"plays_{season}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            frames.append(df)
    if frames:
        all_plays = pd.concat(frames, ignore_index=True)
        timeouts = all_plays[
            all_plays["reason"].isin(["home-timeout", "away-timeout"])
        ].copy()
        timeouts["calling_side"] = timeouts["reason"].map(
            {"home-timeout": "home", "away-timeout": "away"}
        )
        timeouts["calling_team_id"] = timeouts.apply(
            lambda r: r["home_team_id"] if r["calling_side"] == "home" else r["away_team_id"],
            axis=1,
        )
        timeouts["calling_team_abbr"] = timeouts.apply(
            lambda r: r["home_abbr"] if r["calling_side"] == "home" else r["away_abbr"],
            axis=1,
        )
        timeouts["calling_team_score"] = timeouts.apply(
            lambda r: r["home_score_pre"] if r["calling_side"] == "home" else r["away_score_pre"],
            axis=1,
        )
        timeouts["opponent_score"] = timeouts.apply(
            lambda r: r["away_score_pre"] if r["calling_side"] == "home" else r["home_score_pre"],
            axis=1,
        )
        timeouts["score_diff"] = timeouts["calling_team_score"] - timeouts["opponent_score"]
        timeouts.to_parquet(DATA / "timeouts.parquet", index=False)
        logger.info("wrote %d timeout rows to %s", len(timeouts), DATA / "timeouts.parquet")


if __name__ == "__main__":
    main()
