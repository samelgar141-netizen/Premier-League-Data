"""
Premier League 2025/26 Prediction Model
========================================
Implements a Dixon-Coles Poisson model with time-decay weighting to predict
match outcomes and expected goals for the upcoming gameweek.

The model treats home goals and away goals as independent Poisson random
variables, where each team's expected goals are determined by:
  - team attack rating
  - opponent defence rating
  - home advantage factor

A low-score correction (rho parameter) adjusts joint probabilities for
0-0, 0-1, 1-0, 1-1 scorelines, which Poisson over- or under-represents.

Usage:
    python prediction_model.py               # predict next gameweek
    python prediction_model.py --fetch       # fetch fresh data first
    python prediction_model.py --xg          # fit on xG instead of goals
    python prediction_model.py --xi 0.004    # custom time-decay constant
    python prediction_model.py --output predictions.csv
"""

import argparse
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DATA_DIR = Path("data")
RESULTS_FILE = DATA_DIR / "pl_2025_26_results.csv"
SAMPLE_RESULTS_FILE = DATA_DIR / "sample_results.csv"

# Time-decay constant: weight = exp(-XI * days_ago)
# XI=0.0065 gives a ~107-day half-life (recent form matters, old results fade)
DEFAULT_XI = 0.0065


# ── Data loading ─────────────────────────────────────────────────────────────

def load_results(use_xg: bool = False) -> pd.DataFrame:
    """Load played match results; falls back to sample data if live file absent."""
    path = RESULTS_FILE if RESULTS_FILE.exists() else SAMPLE_RESULTS_FILE
    if not path.exists():
        sys.exit(
            f"No data file found. Run `python fetch_pl_data.py` first, "
            f"or ensure {SAMPLE_RESULTS_FILE} exists."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)

    if use_xg:
        if "home_xg" not in df.columns:
            sys.exit("xG columns not found in data file.")
        df = df.dropna(subset=["home_xg", "away_xg"]).copy()
        df["home_goals"] = df["home_xg"].astype(float)
        df["away_goals"] = df["away_xg"].astype(float)

    return df.sort_values("date").reset_index(drop=True)


def load_upcoming_fixtures(played_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load unplayed fixtures. Tries the live results file first (rows with
    missing goals = unplayed), then falls back to a synthetic next gameweek.
    """
    if RESULTS_FILE.exists():
        full = pd.read_csv(RESULTS_FILE, parse_dates=["date"])
        upcoming = full[full["home_goals"].isna()][["date", "home_team", "away_team"]]
        if not upcoming.empty:
            return upcoming.sort_values("date").reset_index(drop=True)

    # Synthetic fallback: pair each team once for the next week
    teams = sorted(set(played_df["home_team"]) | set(played_df["away_team"]))
    next_gw = played_df["date"].max() + timedelta(days=7)
    fixtures, used = [], set()
    for i, home in enumerate(teams):
        if home in used:
            continue
        for away in teams[i + 1:]:
            if away not in used:
                fixtures.append({"date": next_gw, "home_team": home, "away_team": away})
                used.update([home, away])
                break

    return pd.DataFrame(fixtures)


# ── Dixon-Coles model ─────────────────────────────────────────────────────────

def _rho_correction(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Low-score correction factor (Dixon & Coles, 1997, Section 2)."""
    if hg == 0 and ag == 0:
        return 1.0 - lam_h * lam_a * rho
    if hg == 0 and ag == 1:
        return 1.0 + lam_h * rho
    if hg == 1 and ag == 0:
        return 1.0 + lam_a * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def _neg_log_likelihood(params: np.ndarray, teams: list, df: pd.DataFrame,
                         xi: float, l2: float = 0.01) -> float:
    """
    Negative log-likelihood for the Dixon-Coles model with time-decay weights
    and L2 regularisation to keep parameters bounded with sparse data.
    """
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    attack = params[:n]
    defence = params[n:2 * n]
    home_adv = params[2 * n]
    rho = params[2 * n + 1]

    ref_date = df["date"].max()
    total = 0.0

    for row in df.itertuples(index=False):
        ht, at = row.home_team, row.away_team
        if ht not in idx or at not in idx:
            continue

        hg = int(round(float(row.home_goals)))
        ag = int(round(float(row.away_goals)))
        days_ago = max((ref_date - row.date).days, 0)
        weight = np.exp(-xi * days_ago)

        exp_arg_h = attack[idx[ht]] - defence[idx[at]] + home_adv
        exp_arg_a = attack[idx[at]] - defence[idx[ht]]
        # Clip to avoid overflow
        lam_h = np.exp(np.clip(exp_arg_h, -10, 10))
        lam_a = np.exp(np.clip(exp_arg_a, -10, 10))

        rc = _rho_correction(hg, ag, lam_h, lam_a, rho)
        if rc <= 0:
            return 1e10

        total += weight * (
            np.log(rc)
            + poisson.logpmf(hg, lam_h)
            + poisson.logpmf(ag, lam_a)
        )

    # L2 penalty on attack and defence params (Gaussian prior ~ N(0, 1/l2))
    penalty = l2 * (np.sum(attack ** 2) + np.sum(defence ** 2))

    return -total + penalty


def fit_model(df: pd.DataFrame, xi: float = DEFAULT_XI) -> dict:
    """
    Fit Dixon-Coles parameters via maximum likelihood estimation.

    Uses L2 regularisation to stabilise estimates with limited data.
    Regularisation strength scales inversely with number of matches.

    Returns a dict with keys:
        teams, attack, defence, home_adv, rho
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)

    x0 = np.concatenate([
        np.zeros(n),    # attack
        np.zeros(n),    # defence
        [0.25],         # home advantage
        [-0.1],         # rho
    ])

    # L2 strength: lighter with more matches (full season ~380 games)
    l2 = max(0.001, 2.0 / max(len(df), 1))

    # Bounds: cap parameters to prevent numerical overflow
    param_bounds = (
        [(-3.0, 3.0)] * n      # attack
        + [(-3.0, 3.0)] * n    # defence
        + [(-0.5, 1.5)]        # home advantage
        + [(-0.5, 0.5)]        # rho
    )

    # Identifiability: fix sum of attack parameters to zero
    constraints = [{"type": "eq", "fun": lambda x, n=n: np.sum(x[:n])}]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(teams, df, xi, l2),
            method="SLSQP",
            bounds=param_bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-9},
        )

    if not result.success:
        print(f"  [warning] Optimiser: {result.message}", file=sys.stderr)

    p = result.x
    return {
        "teams": teams,
        "attack": dict(zip(teams, p[:n])),
        "defence": dict(zip(teams, p[n:2 * n])),
        "home_adv": float(p[2 * n]),
        "rho": float(p[2 * n + 1]),
    }


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_match(home: str, away: str, model: dict, max_goals: int = 10) -> dict | None:
    """
    Predict outcome probabilities for a single fixture.

    Returns W/D/L probabilities, expected goals, implied decimal odds,
    and the most likely correct score.
    """
    attack = model["attack"]
    defence = model["defence"]

    if home not in attack or away not in attack:
        return None

    exp_h = attack[home] - defence[away] + model["home_adv"]
    exp_a = attack[away] - defence[home]
    lam_h = np.exp(np.clip(exp_h, -10, 10))
    lam_a = np.exp(np.clip(exp_a, -10, 10))
    rho = model["rho"]

    # Score probability matrix [home_goals, away_goals]
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            rc = _rho_correction(i, j, lam_h, lam_a, rho)
            mat[i, j] = max(p * rc, 0.0)
    if mat.sum() == 0:
        return None
    mat /= mat.sum()

    # Outcome probabilities
    home_win = float(np.sum(np.tril(mat, -1)))   # i > j
    draw = float(np.trace(mat))                   # i == j
    away_win = float(np.sum(np.triu(mat, 1)))     # j > i

    # Most likely scoreline
    bi, bj = np.unravel_index(np.argmax(mat), mat.shape)

    def safe_odds(p: float) -> float | None:
        return round(1.0 / p, 2) if p > 0.001 else None

    return {
        "home_team": home,
        "away_team": away,
        "exp_home_goals": round(lam_h, 2),
        "exp_away_goals": round(lam_a, 2),
        "prob_home": round(home_win, 4),
        "prob_draw": round(draw, 4),
        "prob_away": round(away_win, 4),
        "odds_home": safe_odds(home_win),
        "odds_draw": safe_odds(draw),
        "odds_away": safe_odds(away_win),
        "predicted_score": f"{bi}-{bj}",
        "score_prob": round(float(mat[bi, bj]), 4),
    }


def predict_gameweek(fixtures_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Generate predictions for all fixtures in the given DataFrame."""
    rows = []
    for _, fix in fixtures_df.iterrows():
        pred = predict_match(fix["home_team"], fix["away_team"], model)
        if pred:
            pred["date"] = str(fix.get("date", ""))[:10]
            rows.append(pred)
    return pd.DataFrame(rows)


# ── Over/Under 2.5 goals ─────────────────────────────────────────────────────

def over_under_probs(home: str, away: str, model: dict, threshold: float = 2.5,
                     max_goals: int = 10) -> dict | None:
    """Compute P(total goals > threshold) using the fitted score matrix."""
    pred = predict_match(home, away, model, max_goals=max_goals)
    if pred is None:
        return None

    attack = model["attack"]
    defence = model["defence"]
    lam_h = np.exp(np.clip(attack[home] - defence[away] + model["home_adv"], -10, 10))
    lam_a = np.exp(np.clip(attack[away] - defence[home], -10, 10))
    rho = model["rho"]

    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            rc = _rho_correction(i, j, lam_h, lam_a, rho)
            mat[i, j] = max(p * rc, 0.0)
    if mat.sum() == 0:
        return None
    mat /= mat.sum()

    over = 0.0
    under = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if (i + j) > threshold:
                over += mat[i, j]
            else:
                under += mat[i, j]

    return {
        "home_team": home,
        "away_team": away,
        f"prob_over_{threshold}": round(over, 4),
        f"prob_under_{threshold}": round(under, 4),
        f"odds_over_{threshold}": round(1.0 / over, 2) if over > 0.001 else None,
        f"odds_under_{threshold}": round(1.0 / under, 2) if under > 0.001 else None,
    }


# ── Helper utilities ──────────────────────────────────────────────────────────

def recent_form(team: str, df: pd.DataFrame, n: int = 5) -> str:
    """Return last N result characters as a string, e.g. 'WDWLW'."""
    home = df[df["home_team"] == team].assign(
        rc=lambda d: d.apply(
            lambda r: "W" if r.home_goals > r.away_goals
            else ("D" if r.home_goals == r.away_goals else "L"),
            axis=1,
        )
    )[["date", "rc"]]
    away = df[df["away_team"] == team].assign(
        rc=lambda d: d.apply(
            lambda r: "W" if r.away_goals > r.home_goals
            else ("D" if r.away_goals == r.home_goals else "L"),
            axis=1,
        )
    )[["date", "rc"]]
    combined = pd.concat([home, away]).sort_values("date").tail(n)
    return "".join(combined["rc"].tolist())


def form_pts(form: str) -> int:
    return sum(3 if c == "W" else (1 if c == "D" else 0) for c in form)


def _signal(prob: float) -> str:
    """Simple betting signal based on model-implied probability."""
    if prob >= 0.55:
        return "★ STRONG"
    if prob >= 0.42:
        return "◆ CONSIDER"
    if prob >= 0.30:
        return "  POSSIBLE"
    return "  LOW"


def _bar(prob: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(prob * width))))
    return "█" * filled + "░" * (width - filled)


# ── Team strength table ───────────────────────────────────────────────────────

def team_strength_table(model: dict) -> pd.DataFrame:
    """Build a ranked team strength table from fitted parameters."""
    rows = []
    for team in model["teams"]:
        rows.append({
            "team": team,
            "attack": round(model["attack"][team], 3),
            "defence": round(model["defence"][team], 3),
            "net": round(model["attack"][team] - model["defence"][team], 3),
        })
    df = pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df


# ── Display ───────────────────────────────────────────────────────────────────

def print_team_ratings(model: dict) -> None:
    print("\n  TEAM STRENGTH RATINGS (attack / defence / net)")
    print("  " + "─" * 56)
    tbl = team_strength_table(model)
    print(f"  {'Rank':<5} {'Team':<28} {'Attack':>8} {'Defence':>9} {'Net':>7}")
    print("  " + "─" * 56)
    for rank, row in tbl.iterrows():
        print(f"  {rank:<5} {row['team']:<28} {row['attack']:>8.3f} {row['defence']:>9.3f} {row['net']:>7.3f}")


def print_predictions(preds_df: pd.DataFrame, results_df: pd.DataFrame,
                       ou_df: pd.DataFrame | None = None) -> None:
    sep = "=" * 74
    thin = "─" * 74

    print(f"\n{sep}")
    print("  PREMIER LEAGUE 2025/26 — UPCOMING GAMEWEEK PREDICTIONS")
    print(f"  Model: Dixon-Coles Poisson  |  {len(results_df)} matches fitted")
    print(sep)

    ou_map = {}
    if ou_df is not None:
        for _, r in ou_df.iterrows():
            ou_map[(r["home_team"], r["away_team"])] = r

    for _, row in preds_df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        # Use actual goals for form (not xG-substituted)
        actual_df = results_df.copy()
        ht_form = recent_form(ht, actual_df)
        at_form = recent_form(at, actual_df)

        date_str = str(row.get("date", ""))[:10] or "TBC"

        print(f"\n  {date_str}")
        print(f"  {ht}  vs  {at}")
        print(f"  Form (last 5):  {ht} [{ht_form or '—'}]   {at} [{at_form or '—'}]")
        print(f"  Expected goals: {ht} {row['exp_home_goals']}  —  {row['exp_away_goals']} {at}")
        print(f"  Predicted score: {row['predicted_score']}  (p = {row['score_prob']:.1%})")
        print()
        print(f"  {'Outcome':<22} {'Prob':>6}  {'Bar':<22} {'Impl.Odds':>9}  {'Signal'}")
        print(f"  {'─'*22}  {'─'*6}  {'─'*22}  {'─'*9}  {'─'*10}")

        for label, prob, odds in [
            (f"{ht} Win", row["prob_home"], row["odds_home"]),
            ("Draw",      row["prob_draw"], row["odds_draw"]),
            (f"{at} Win", row["prob_away"], row["odds_away"]),
        ]:
            bar = _bar(prob)
            sig = _signal(prob)
            odds_str = f"{odds:.2f}" if odds else " —"
            print(f"  {label:<22}  {prob:>5.1%}  {bar:<22}  {odds_str:>9}  {sig}")

        # Over/Under 2.5
        ou = ou_map.get((ht, at))
        if ou is not None:
            ov = ou["prob_over_2.5"]
            un = ou["prob_under_2.5"]
            ov_odds = ou["odds_over_2.5"]
            un_odds = ou["odds_under_2.5"]
            print()
            print(f"  Over/Under 2.5 goals:")
            print(f"  {'Over 2.5':<22}  {ov:>5.1%}  {_bar(ov):<22}  {ov_odds if ov_odds else '—':>9}")
            print(f"  {'Under 2.5':<22}  {un:>5.1%}  {_bar(un):<22}  {un_odds if un_odds else '—':>9}")

        print(f"  {thin}")

    print(f"\n  {'─'*74}")
    print("  DISCLAIMER: Predictions are for research/educational use only.")
    print("  These are model probabilities, not guaranteed outcomes.")
    print("  Please gamble responsibly. Never bet more than you can afford to lose.")
    print(f"  {'─'*74}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Premier League 2025/26 Dixon-Coles Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch fresh data from Understat before predicting",
    )
    parser.add_argument(
        "--xg", action="store_true",
        help="Fit model on xG values instead of actual goals (more stable, less 'noisy')",
    )
    parser.add_argument(
        "--xi", type=float, default=DEFAULT_XI,
        help="Time-decay constant (higher = more weight on recent matches)",
    )
    parser.add_argument(
        "--gameweek", type=str, default=None, metavar="YYYY-MM-DD",
        help="Filter upcoming fixtures to a 7-day window starting from this date",
    )
    parser.add_argument(
        "--output", type=str, default=None, metavar="PATH",
        help="Save predictions table to a CSV file",
    )
    parser.add_argument(
        "--ratings", action="store_true",
        help="Print full team attack/defence ratings table",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # Optionally refresh data
    if args.fetch:
        print("Fetching fresh data from Understat...")
        import fetch_pl_data
        fetch_pl_data.main()
        print()

    # Load data
    print("Loading match data...")
    df_fit = load_results(use_xg=args.xg)
    label = "xG" if args.xg else "goals"
    print(f"  {len(df_fit)} played matches  |  fitting on {label}")
    if len(df_fit) < 20:
        print(
            f"  [note] Only {len(df_fit)} matches found — using sample data."
            " Run `python fetch_pl_data.py` for full season data.",
            file=sys.stderr,
        )

    # Load actual goals separately for form (unaffected by --xg)
    df_actual = load_results(use_xg=False)

    # Fit model
    print(f"Fitting Dixon-Coles model (xi={args.xi})...")
    model = fit_model(df_fit, xi=args.xi)
    print(f"  Home advantage : {model['home_adv']:.4f}")
    print(f"  Rho (ρ)        : {model['rho']:.4f}")

    top_atk = sorted(model["attack"].items(), key=lambda x: -x[1])[:3]
    top_def = sorted(model["defence"].items(), key=lambda x: x[1])[:3]
    print(f"  Best attack    : {', '.join(f'{t} ({v:+.3f})' for t, v in top_atk)}")
    print(f"  Best defence   : {', '.join(f'{t} ({v:+.3f})' for t, v in top_def)}")

    if args.ratings:
        print_team_ratings(model)

    # Load fixtures
    print("\nLoading upcoming fixtures...")
    fixtures = load_upcoming_fixtures(df_actual)

    if args.gameweek:
        gw_start = pd.to_datetime(args.gameweek)
        fixtures = fixtures[
            (fixtures["date"] >= gw_start) &
            (fixtures["date"] <= gw_start + timedelta(days=7))
        ].reset_index(drop=True)

    if fixtures.empty:
        print("No upcoming fixtures found for the specified criteria.")
        return

    print(f"  {len(fixtures)} fixture(s) to predict.")

    # Generate predictions
    preds = predict_gameweek(fixtures, model)

    # Over/Under 2.5
    ou_rows = [
        over_under_probs(r["home_team"], r["away_team"], model)
        for _, r in preds.iterrows()
        if over_under_probs(r["home_team"], r["away_team"], model) is not None
    ]
    ou_df = pd.DataFrame(ou_rows) if ou_rows else None

    # Display
    print_predictions(preds, df_actual, ou_df)

    # Save
    if args.output:
        out_path = Path(args.output)
        save_df = preds.copy()
        if ou_df is not None:
            save_df = save_df.merge(
                ou_df, on=["home_team", "away_team"], how="left"
            )
        save_df.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
