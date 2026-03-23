# Premier League 2025/26 Data

Fetch and analyse Premier League 2025/26 results, goals, shots, and xG stats from [Understat](https://understat.com).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Fetch match results + team stats

```bash
python fetch_pl_data.py
```

Outputs:
| File | Contents |
|------|----------|
| `data/pl_2025_26_results.csv` | Every played match — date, teams, goals, xG, win probability |
| `data/pl_2025_26_team_stats.csv` | Season totals per team — W/D/L, goals, xG for/against |

### Also fetch shot-level data

```bash
python fetch_pl_data.py --shots
```

Adds `data/pl_2025_26_shots.csv` with one row per shot (player, minute, xG, position, situation).

> **Note:** `--shots` makes one HTTP request per match (380 total over the season). Expect several minutes of runtime for a full season.

## Analysis notebook

Open `analysis.ipynb` in Jupyter for interactive charts:
- League table with xG columns
- Goals vs xG per team (bar chart)
- xG difference ranking
- Goals & xG per match timeline
- Shot outcome distribution
- xG per shot histogram

```bash
jupyter notebook analysis.ipynb
```

## Data columns

### `pl_2025_26_results.csv`
| Column | Description |
|--------|-------------|
| `match_id` | Understat match ID |
| `date` | Match date (YYYY-MM-DD) |
| `home_team` / `away_team` | Team names |
| `home_goals` / `away_goals` | Full-time goals |
| `home_xg` / `away_xg` | Expected goals |
| `forecast_w/d/l` | Pre-match win/draw/loss probability |
| `result` | `H` / `D` / `A` |

### `pl_2025_26_team_stats.csv`
| Column | Description |
|--------|-------------|
| `position` | League position |
| `team` | Club name |
| `played` | Matches played |
| `wins/draws/losses` | Record |
| `goals_scored/conceded` | Season totals |
| `goal_diff` | Goal difference |
| `xg_for/against` | Season xG totals |
| `xg_diff` | xG for minus xG against |
| `points` | Points |

### `pl_2025_26_shots.csv`
| Column | Description |
|--------|-------------|
| `match_id` | Understat match ID |
| `team_side` | `home` or `away` |
| `player` | Shooter name |
| `minute` | Minute of shot |
| `result` | `Goal`, `SavedShot`, `MissedShots`, `BlockedShot`, `OwnGoal` |
| `x` / `y` | Pitch coordinates (0–1 scale) |
| `xg` | Expected goals value |
| `situation` | `OpenPlay`, `SetPiece`, `FromCorner`, `DirectFreekick`, `Penalty` |
| `shot_type` | `RightFoot`, `LeftFoot`, `Head` |
| `player_assisted` | Assister name (if applicable) |

## Sample data

The `data/` folder contains sample files (`sample_*.csv`) matching the exact schema above, so `analysis.ipynb` works out of the box before you run the fetcher.

## Source

All data sourced from [Understat](https://understat.com/league/EPL/2025).
