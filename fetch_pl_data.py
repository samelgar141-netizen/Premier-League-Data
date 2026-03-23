"""
Premier League 2025/26 Season Data Fetcher
Fetches results, goals, shots, and xG stats from Understat.
"""

import json
import re
import sys
import requests
import pandas as pd
from pathlib import Path

SEASON = "2025"  # Understat uses the start year for season identification
LEAGUE = "EPL"
BASE_URL = "https://understat.com/league"
DATA_DIR = Path("data")


def fetch_understat_data(url: str) -> dict:
    """Fetch and parse JSON data embedded in an Understat page."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def extract_json_var(html: str, var_name: str) -> list | dict:
    """Extract a JSON variable embedded in Understat HTML."""
    pattern = rf"var {var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, html)
    if not match:
        raise ValueError(f"Could not find variable '{var_name}' in page HTML")
    raw = match.group(1)
    # Understat encodes the JSON string — decode unicode escapes
    decoded = raw.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
    return json.loads(decoded)


def fetch_matches() -> pd.DataFrame:
    """Fetch all match results with goals and xG for the 2025/26 season."""
    print(f"Fetching {LEAGUE} {SEASON}/26 match data...")
    url = f"{BASE_URL}/{LEAGUE}/{SEASON}"
    html = fetch_understat_data(url)
    matches_raw = extract_json_var(html, "datesData")

    rows = []
    for m in matches_raw:
        rows.append({
            "match_id": m.get("id"),
            "date": m.get("datetime", "")[:10],
            "home_team": m.get("h", {}).get("title"),
            "away_team": m.get("a", {}).get("title"),
            "home_goals": m.get("goals", {}).get("h"),
            "away_goals": m.get("goals", {}).get("a"),
            "home_xg": m.get("xG", {}).get("h"),
            "away_xg": m.get("xG", {}).get("a"),
            "forecast_w": m.get("forecast", {}).get("w"),
            "forecast_d": m.get("forecast", {}).get("d"),
            "forecast_l": m.get("forecast", {}).get("l"),
            "result": _result(m),
        })

    df = pd.DataFrame(rows)
    # Keep only played matches (goals field populated)
    df = df[df["home_goals"].notna()].copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["home_xg"] = df["home_xg"].astype(float).round(2)
    df["away_xg"] = df["away_xg"].astype(float).round(2)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _result(match: dict) -> str | None:
    """Return 'H' (home win), 'D' (draw), or 'A' (away win), or None if unplayed."""
    h = match.get("goals", {}).get("h")
    a = match.get("goals", {}).get("a")
    if h is None or a is None:
        return None
    h, a = int(h), int(a)
    if h > a:
        return "H"
    elif h < a:
        return "A"
    return "D"


def fetch_team_stats() -> pd.DataFrame:
    """Fetch per-team season totals (goals, shots, xG, etc.)."""
    print(f"Fetching {LEAGUE} {SEASON}/26 team stats...")
    url = f"{BASE_URL}/{LEAGUE}/{SEASON}"
    html = fetch_understat_data(url)
    teams_raw = extract_json_var(html, "teamsData")

    rows = []
    for team_id, team in teams_raw.items():
        history = team.get("history", [])
        if not history:
            continue
        # Aggregate across all played matches
        played = [h for h in history if h.get("result") in ("w", "d", "l")]
        if not played:
            continue

        wins = sum(1 for h in played if h["result"] == "w")
        draws = sum(1 for h in played if h["result"] == "d")
        losses = sum(1 for h in played if h["result"] == "l")
        goals_scored = sum(h.get("scored", 0) for h in played)
        goals_conceded = sum(h.get("missed", 0) for h in played)
        xg_for = round(sum(h.get("xG", 0) for h in played), 2)
        xg_against = round(sum(h.get("xGA", 0) for h in played), 2)
        shots = sum(h.get("npxG", 0) for h in played)  # shots proxy via npxG
        pts = wins * 3 + draws

        rows.append({
            "team": team.get("title"),
            "played": len(played),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_diff": goals_scored - goals_conceded,
            "points": pts,
            "xg_for": xg_for,
            "xg_against": xg_against,
            "xg_diff": round(xg_for - xg_against, 2),
        })

    df = pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based league position
    df.index.name = "position"
    return df


def fetch_shots(match_id: str) -> pd.DataFrame:
    """Fetch shot-level data for a single match."""
    url = f"https://understat.com/match/{match_id}"
    html = fetch_understat_data(url)
    shots_raw = extract_json_var(html, "shotsData")

    rows = []
    for side in ("h", "a"):
        for shot in shots_raw.get(side, []):
            rows.append({
                "match_id": match_id,
                "team_side": "home" if side == "h" else "away",
                "player": shot.get("player"),
                "minute": shot.get("minute"),
                "result": shot.get("result"),
                "x": shot.get("X"),
                "y": shot.get("Y"),
                "xg": shot.get("xG"),
                "situation": shot.get("situation"),
                "shot_type": shot.get("shotType"),
                "player_assisted": shot.get("player_assisted"),
            })
    return pd.DataFrame(rows)


def fetch_all_shots(matches_df: pd.DataFrame, max_matches: int | None = None) -> pd.DataFrame:
    """Fetch shot data for all (or a subset of) played matches."""
    ids = matches_df["match_id"].tolist()
    if max_matches:
        ids = ids[:max_matches]

    frames = []
    total = len(ids)
    for i, mid in enumerate(ids, 1):
        print(f"  Fetching shots for match {mid} ({i}/{total})...", end="\r")
        try:
            frames.append(fetch_shots(mid))
        except Exception as e:
            print(f"\n  Warning: could not fetch shots for match {mid}: {e}")
    print()  # newline after carriage-return loop

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def print_summary(matches_df: pd.DataFrame, team_df: pd.DataFrame) -> None:
    """Print a quick summary to stdout."""
    print("\n" + "=" * 60)
    print(f"PREMIER LEAGUE {SEASON}/{int(SEASON[2:]) + 1} SEASON SUMMARY")
    print("=" * 60)
    print(f"Matches played  : {len(matches_df)}")
    total_goals = matches_df["home_goals"].sum() + matches_df["away_goals"].sum()
    total_xg = matches_df["home_xg"].sum() + matches_df["away_xg"].sum()
    print(f"Total goals     : {total_goals}")
    print(f"Goals/match     : {total_goals / len(matches_df):.2f}")
    print(f"Total xG        : {total_xg:.1f}")
    print(f"xG/match        : {total_xg / len(matches_df):.2f}")
    print()
    print("CURRENT STANDINGS (top 10):")
    print(team_df[["team", "played", "wins", "draws", "losses",
                    "goals_scored", "goals_conceded", "xg_for", "xg_against",
                    "points"]].head(10).to_string())
    print()
    print("RECENT RESULTS (last 10):")
    recent = matches_df.tail(10)[["date", "home_team", "home_goals",
                                   "away_goals", "away_team",
                                   "home_xg", "away_xg"]].copy()
    recent.columns = ["date", "home", "hg", "ag", "away", "h_xg", "a_xg"]
    print(recent.to_string(index=False))


def main(fetch_shots_data: bool = False, max_shot_matches: int = 10) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    matches_df = fetch_matches()
    team_df = fetch_team_stats()

    # Save core files
    matches_path = DATA_DIR / "pl_2025_26_results.csv"
    team_path = DATA_DIR / "pl_2025_26_team_stats.csv"
    matches_df.to_csv(matches_path, index=False)
    team_df.to_csv(team_path)
    print(f"Saved: {matches_path}")
    print(f"Saved: {team_path}")

    if fetch_shots_data:
        print(f"\nFetching shot data for up to {max_shot_matches} matches...")
        shots_df = fetch_all_shots(matches_df, max_matches=max_shot_matches)
        shots_path = DATA_DIR / "pl_2025_26_shots.csv"
        shots_df.to_csv(shots_path, index=False)
        print(f"Saved: {shots_path}")

    print_summary(matches_df, team_df)


if __name__ == "__main__":
    # Pass --shots to also fetch shot-level data (makes many HTTP requests)
    fetch_shots_flag = "--shots" in sys.argv
    main(fetch_shots_data=fetch_shots_flag)
