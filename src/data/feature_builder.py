"""Shared feature engineering for ML-based prop projections.

Used at both training time (iterating over historical game logs) and inference
time (for today's matchups). Same function, same feature order — prevents
train/serve skew.

Conventions
-----------
- `pitcher_logs` / `batter_logs` are the player's past game_log rows sorted
  descending by date. At inference they are today's history; at training they
  are strictly games BEFORE the target row (caller is responsible for slicing).
- Feature vectors are numpy arrays in a fixed feature order, returned alongside
  the list of names so models persist the schema.
- All auxiliary DB lookups (platoon splits, bullpen factors, rest days) accept
  a sqlite connection via `db=None`; when None, safe defaults are used so the
  function works in tests without a database.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.park_factors import get_park_factor
from src.config import LEAGUE_AVG_K_RATE, LEAGUE_AVG_RUNS_PER_GAME

# Statcast league-average constants used for normalization / delta calculations.
_LEAGUE_AVG_WHIFF_PCT         = 0.255   # ~25.5% league-wide whiff rate
_LEAGUE_AVG_K_PER_9_PITCHER   = 8.8    # league avg K/9 per pitcher IP (2024 MLB)
_LEAGUE_AVG_SPIN_RATE_FF      = 2250.0  # rpm — 4-seam fastball league average
_LEAGUE_AVG_SPIN_RATE_STD     = 200.0   # rpm — std dev for z-scoring
_LEAGUE_AVG_EXIT_VELO         = 88.5   # mph — batter average exit velocity
_LEAGUE_AVG_SPRINT_SPEED      = 27.0   # ft/sec — runner sprint speed league avg


PITCHER_FEATURE_NAMES: List[str] = [
    "recent_k_per_9_l5",
    "recent_k_per_9_l10",
    "season_k_per_9",
    "recent_era_l10",
    "season_era",
    "recent_whip_l10",
    "recent_bb_per_9_l10",
    "pitches_per_ip_l5",
    "est_pitch_limit",
    "opp_team_rate",
    "ump_k_factor",
    "park_factor",
    "weather_temp",
    "weather_wind_mph",
    "rest_days",
    "is_home",
    "is_day_game",
    "month_of_season",
    # Pitch-mix + volatility proxies derived from existing game logs.
    # command: K/BB is the cheapest stand-in for stuff+command when we don't
    # have pitch-by-pitch data. flyball: HR/9 allowed correlates with FB%,
    # which interacts with wind/park. volatility: start-to-start stdev in
    # K and IP — two pitchers with identical means can have very different
    # over/under distributions. Books price the distribution, not the mean.
    "k_bb_ratio_l10",
    "hr_per_9_l10",
    "k_volatility_l10",
    "ip_volatility_l10",
    "h2h_k_per_9_delta",
    # Statcast stuff-quality metrics (Baseball Savant, no API key).
    # whiff_pct and chase_rate are the strongest K predictors in small samples.
    # barrel_pct_against and hard_hit_pct_against predict ER/HR allowed.
    # spin_rate_ff: high spin → more break → higher K rate.
    # delta_k9_vs_xk9: K/9 minus expected from whiff% — regression-to-mean signal.
    # velocity_delta_l5: fastball velocity trend in last 5 starts (fatigue/injury proxy).
    "statcast_whiff_pct",
    "statcast_chase_rate",
    "statcast_barrel_pct_against",
    "statcast_hard_hit_pct_against",
    "statcast_spin_rate_ff_z",
    "statcast_avg_exit_velo_against",
    "statcast_delta_k9_vs_xk9",
    "statcast_k9_momentum_l5",
]

BATTER_FEATURE_NAMES: List[str] = [
    "recent_rate_per_pa_l15",
    "season_rate_per_pa",
    "recent_rate_per_pa_l5",
    "platoon_rate_vs_hand",
    "is_same_hand_matchup",
    "is_switch_hitter",
    "lineup_position",
    "park_factor",
    "weather_temp",
    "weather_wind_mph",
    "rest_days",
    "is_home",
    "is_day_game",
    "month_of_season",
    "opp_starter_k_per_9",
    "opp_bullpen_era",
    # Batter volatility + strikeout-proneness. High-K batters have thinner
    # hits/TB distributions (more 0s) which matters at the under line.
    "stat_volatility_l15",
    "k_rate_l15",
    # Statcast contact-quality metrics (Baseball Savant).
    # barrel_pct and xwoba are the strongest HR/TB predictors in small samples.
    # exit_velocity_avg and hard_hit_pct capture overall contact quality.
    # launch_angle_avg: high launch → more HRs/XBH, low launch → grounders/singles.
    # whiff_pct_batter: K risk proxy (relevant for hit-under bets).
    # sprint_speed: infield hit rate booster for hit props.
    "statcast_barrel_pct",
    "statcast_xwoba",
    "statcast_exit_velocity_avg",
    "statcast_hard_hit_pct",
    "statcast_launch_angle_avg",
    "statcast_whiff_pct_batter",
    "statcast_sprint_speed",
]


# ---------------------------------------------------------------------------
# Rolling-window helpers
# ---------------------------------------------------------------------------

def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    if d is None or d == 0:
        return default
    return float(n) / float(d)


def _rolling_pitcher_stats(logs: List[Dict], window: int) -> Dict[str, float]:
    recent = logs[:window] if logs else []
    ip = sum(l.get('innings_pitched') or 0 for l in recent)
    k = sum(l.get('strikeouts') or 0 for l in recent)
    er = sum(l.get('earned_runs') or 0 for l in recent)
    bb = sum(l.get('walks') or 0 for l in recent)
    h = sum(l.get('hits_allowed') or 0 for l in recent)
    pitches = sum(l.get('pitches_thrown') or 0 for l in recent)
    return {
        "k_per_9": _safe_div(k * 9.0, ip),
        "era": _safe_div(er * 9.0, ip),
        "whip": _safe_div(h + bb, ip),
        "bb_per_9": _safe_div(bb * 9.0, ip),
        "pitches_per_ip": _safe_div(pitches, ip, default=16.5),
        "avg_pitches": _safe_div(pitches, max(1, len(recent))),
    }


def _rolling_batter_rate(logs: List[Dict], stat_key: str, window: int) -> float:
    recent = logs[:window] if logs else []
    stat = sum(l.get(stat_key) or 0 for l in recent)
    pa = sum((l.get('plate_appearances') or 0) or (l.get('at_bats') or 0) for l in recent)
    return _safe_div(stat, pa)


def _stdev(values: List[float]) -> float:
    """Population stdev with safe defaults (0.0 for <2 samples)."""
    if not values or len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=0))


def _rolling_pitcher_hr_per_9(logs: List[Dict], window: int) -> float:
    """HR allowed per 9 IP over the window. Proxy for flyball tendency."""
    recent = logs[:window] if logs else []
    ip = sum(l.get('innings_pitched') or 0 for l in recent)
    hr = sum(l.get('home_runs_allowed') or 0 for l in recent)
    return _safe_div(hr * 9.0, ip)


def _rolling_pitcher_k_bb_ratio(logs: List[Dict], window: int) -> float:
    """K/BB ratio — command proxy. BB floored at 1 to avoid div-by-zero inflation."""
    recent = logs[:window] if logs else []
    k = sum(l.get('strikeouts') or 0 for l in recent)
    bb = sum(l.get('walks') or 0 for l in recent)
    return _safe_div(k, max(bb, 1.0))


def _rolling_pitcher_start_stdev(logs: List[Dict], stat_key: str, window: int) -> float:
    """Start-to-start stdev of a pitcher stat (raw per-start count, not a rate).
    Captures volatility the mean hides."""
    recent = logs[:window] if logs else []
    vals = [float(l.get(stat_key) or 0) for l in recent]
    return _stdev(vals)


def _rolling_batter_game_stdev(logs: List[Dict], stat_key: str, window: int) -> float:
    """Game-to-game stdev of a batter stat (raw count per game)."""
    recent = logs[:window] if logs else []
    vals = [float(l.get(stat_key) or 0) for l in recent]
    return _stdev(vals)


# ---------------------------------------------------------------------------
# Context helpers (rest days, day/night, platoon, bullpen)
# ---------------------------------------------------------------------------

def compute_rest_days(logs: List[Dict], game_date: str) -> int:
    """Days between `game_date` and the most-recent prior game in logs."""
    if not logs or not game_date:
        return 4
    try:
        gd = datetime.fromisoformat(game_date[:10])
    except ValueError:
        return 4
    last = logs[0].get('date') if logs else None
    if not last:
        return 4
    try:
        ld = datetime.fromisoformat(str(last)[:10])
    except ValueError:
        return 4
    return max(0, (gd - ld).days)


_STAT_KEY_TO_MARKET = {
    "hits":        "batter_hits",
    "home_runs":   "batter_home_runs",
    "total_bases": "batter_total_bases",
}

# Bayesian prior strength: number of pseudo-PA to weight toward the prior.
# At 50 PA the empirical rate is used directly; below 50 we shrink.
_PLATOON_PRIOR_PA = 30


def compute_platoon_split(batter_logs: List[Dict], vs_hand: str, stat_key: str,
                          db=None, batter_id: int = None,
                          season: int = None) -> float:
    """Per-PA rate for a batter vs a given pitcher hand.

    Lookup order:
      1. batter_platoon_splits table (empirical, Bayesian-shrunk when n_pa < 50).
      2. Fallback JOIN on probable_pitchers (legacy path).
      3. Overall season rate from logs.
    """
    season_rate = _rolling_batter_rate(batter_logs, stat_key, window=max(30, len(batter_logs)))

    if db is not None and batter_id is not None and vs_hand and season:
        market = _STAT_KEY_TO_MARKET.get(stat_key)
        if market:
            try:
                row = db.execute(
                    "SELECT rate_per_pa, n_pa FROM batter_platoon_splits "
                    "WHERE player_id=? AND season=? AND vs_hand=? AND market=?",
                    (batter_id, season, vs_hand.upper(), market),
                ).fetchone()
                if row and row["n_pa"]:
                    n_pa = int(row["n_pa"])
                    empirical = float(row["rate_per_pa"])
                    if n_pa >= 50:
                        return empirical
                    # Bayesian shrinkage: blend empirical toward prior (season rate)
                    return (n_pa * empirical + _PLATOON_PRIOR_PA * season_rate) / (n_pa + _PLATOON_PRIOR_PA)
            except Exception:
                pass

    # Legacy: join on probable_pitchers game-by-game
    if db is not None and batter_id is not None and vs_hand:
        try:
            rows = db.execute(
                """
                SELECT SUM(bgl.""" + _sqlite_identifier(stat_key) + """) AS stat,
                       SUM(COALESCE(bgl.plate_appearances, bgl.at_bats)) AS pa
                FROM batter_game_logs bgl
                JOIN probable_pitchers pp
                  ON pp.game_id = bgl.game_id
                 AND pp.throws = ?
                 AND pp.team != (
                     SELECT team FROM daily_lineups dl
                      WHERE dl.game_id = bgl.game_id
                        AND dl.player_id = bgl.player_id
                      LIMIT 1
                 )
                WHERE bgl.player_id = ?
                """,
                (vs_hand.upper(), batter_id),
            ).fetchone()
            if rows and rows["pa"]:
                return _safe_div(rows["stat"] or 0, rows["pa"])
        except Exception:
            pass

    return season_rate


def _sqlite_identifier(ident: str) -> str:
    if not ident.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQL identifier: {ident!r}")
    return ident


def compute_bullpen_factor(opp_team_id: int, game_date: str, db=None) -> float:
    """Recent bullpen ERA (last 14 days) for the opposing team. Defaults to league avg."""
    if db is None or not opp_team_id:
        return LEAGUE_AVG_RUNS_PER_GAME
    try:
        rows = db.execute(
            """
            SELECT SUM(earned_runs) AS er, SUM(innings_pitched) AS ip
            FROM pitcher_game_logs pgl
            JOIN players p ON p.player_id = pgl.player_id
            WHERE p.team_id = ?
              AND pgl.innings_pitched < 3
              AND pgl.date >= date(?, '-14 days')
              AND pgl.date <= ?
            """,
            (opp_team_id, game_date, game_date),
        ).fetchone()
        if rows and rows['ip']:
            return _safe_div((rows['er'] or 0) * 9.0, rows['ip'], default=LEAGUE_AVG_RUNS_PER_GAME)
    except Exception:
        pass
    return LEAGUE_AVG_RUNS_PER_GAME


def compute_pitcher_h2h_vs_team(pitcher_id: int, opp_team_id: int, game_date: str, db=None) -> float:
    """
    Returns the difference in K/9 between a pitcher's historical performance vs 
    a specific opposing team and their overall season average prior to the game.
    """
    if db is None or not opp_team_id or not pitcher_id:
        return 0.0

    try:
        rows = db.execute(
            """
            SELECT SUM(pgl.strikeouts) AS k, SUM(pgl.innings_pitched) AS ip
            FROM pitcher_game_logs pgl
            JOIN games g ON pgl.game_id = g.bdl_game_id
            WHERE pgl.player_id = ?
              AND pgl.date < ?
              AND (g.home_team_id = ? OR g.away_team_id = ?)
            """,
            (pitcher_id, game_date, opp_team_id, opp_team_id),
        ).fetchone()

        # Minimum 5 IP vs this team to avoid extreme variance
        if not rows or not rows['ip'] or rows['ip'] < 5.0:
            return 0.0

        h2h_k_per_9 = _safe_div((rows['k'] or 0) * 9.0, rows['ip'])

        season_rows = db.execute(
            """
            SELECT SUM(strikeouts) AS k, SUM(innings_pitched) AS ip
            FROM pitcher_game_logs
            WHERE player_id = ? AND date < ?
            """,
            (pitcher_id, game_date)
        ).fetchone()
        
        if not season_rows or not season_rows['ip']:
            return 0.0
            
        season_k_per_9 = _safe_div((season_rows['k'] or 0) * 9.0, season_rows['ip'])
        
        return h2h_k_per_9 - season_k_per_9
    except Exception:
        pass
        
    return 0.0


def _extract_weather(weather: Optional[Dict]) -> Tuple[float, float]:
    if not weather:
        return 72.0, 5.0
    temp = weather.get('temp_f') if 'temp_f' in weather else weather.get('temperature', 72.0)
    wind = weather.get('wind_mph') if 'wind_mph' in weather else weather.get('wind_speed', 5.0)
    try:
        return float(temp or 72.0), float(wind or 5.0)
    except (TypeError, ValueError):
        return 72.0, 5.0


def _month_from_date(game_date: Optional[str]) -> int:
    if not game_date:
        return 6
    try:
        return datetime.fromisoformat(str(game_date)[:10]).month
    except ValueError:
        return 6


def _is_day_game(game_time: Optional[str]) -> int:
    """Day game if first pitch is before 5pm local (approximated from ISO hour)."""
    if not game_time:
        return 0
    try:
        dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
        return 1 if dt.hour < 21 else 0  # UTC hour < 21 ~= local first pitch < ~5pm ET
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Statcast DB lookups
# ---------------------------------------------------------------------------

def _load_statcast_pitcher(player_id: int, season: int, db) -> Dict[str, Optional[float]]:
    """Return a dict of Statcast pitcher metrics for (player_id, season), or all None."""
    blank: Dict[str, Optional[float]] = {
        "whiff_pct": None, "chase_rate": None, "barrel_pct_against": None,
        "hard_hit_pct_against": None, "spin_rate_ff": None,
        "avg_exit_velocity_against": None,
    }
    if db is None or not player_id:
        return blank
    try:
        row = db.execute(
            """SELECT whiff_pct, chase_rate, barrel_pct_against, hard_hit_pct_against,
                      spin_rate_ff, avg_exit_velocity_against
               FROM statcast_pitcher_stats
               WHERE player_id = ? AND season = ?""",
            (player_id, season),
        ).fetchone()
        if row:
            return {k: row[k] for k in blank}
    except Exception:
        pass
    return blank


def _load_statcast_batter(player_id: int, season: int, db) -> Dict[str, Optional[float]]:
    """Return a dict of Statcast batter metrics for (player_id, season), or all None."""
    blank: Dict[str, Optional[float]] = {
        "exit_velocity_avg": None, "launch_angle_avg": None, "barrel_pct": None,
        "xwoba": None, "sprint_speed": None, "whiff_pct": None, "hard_hit_pct": None,
    }
    if db is None or not player_id:
        return blank
    try:
        row = db.execute(
            """SELECT exit_velocity_avg, launch_angle_avg, barrel_pct,
                      xwoba, sprint_speed, whiff_pct, hard_hit_pct
               FROM statcast_batter_stats
               WHERE player_id = ? AND season = ?""",
            (player_id, season),
        ).fetchone()
        if row:
            return {k: row[k] for k in blank}
    except Exception:
        pass
    return blank


# ---------------------------------------------------------------------------
# Public: pitcher feature builder
# ---------------------------------------------------------------------------

def build_pitcher_features(pitcher_logs: List[Dict],
                           market: str,
                           opponent_rate: float,
                           venue: Optional[str],
                           weather: Optional[Dict],
                           ump_k_factor: float = 1.0,
                           projected_ip: float = 5.5,  # noqa: ARG001
                           extra: Optional[Dict] = None) -> np.ndarray:
    """
    Build a feature vector in the order of PITCHER_FEATURE_NAMES.

    `market` selects the park factor key used (so column is stable but its
    meaning tracks the model being trained).
    `extra` carries loosely-typed context from the caller — rest_days, is_home,
    game_time, month_of_season, opp_bullpen_era — populated by the training
    loop or by scan_props at inference.
    """
    extra = extra or {}
    logs = pitcher_logs or []

    l5 = _rolling_pitcher_stats(logs, 5)
    l10 = _rolling_pitcher_stats(logs, 10)
    season = _rolling_pitcher_stats(logs, len(logs)) if logs else _rolling_pitcher_stats([], 1)

    park_key = "so" if market == "pitcher_strikeouts" else "runs"
    park_adj = get_park_factor(venue, weather=weather).get(park_key, 1.0) if venue else 1.0

    temp, wind = _extract_weather(weather)
    game_date = extra.get("game_date")
    rest_days = extra.get("rest_days")
    if rest_days is None:
        rest_days = compute_rest_days(logs, game_date)

    est_pitch_limit = extra.get("est_pitch_limit")
    if est_pitch_limit is None:
        est_pitch_limit = int(min(110, max(100, l5["avg_pitches"] + 10))) if l5["avg_pitches"] else 100

    k_bb_ratio_l10 = _rolling_pitcher_k_bb_ratio(logs, 10)
    hr_per_9_l10 = _rolling_pitcher_hr_per_9(logs, 10)
    k_vol_l10 = _rolling_pitcher_start_stdev(logs, "strikeouts", 10)
    ip_vol_l10 = _rolling_pitcher_start_stdev(logs, "innings_pitched", 10)

    h2h_k_delta = extra.get("h2h_k_delta")
    if h2h_k_delta is None:
        h2h_k_delta = 0.0
        if extra.get("db") and extra.get("pitcher_id") and extra.get("opp_team_id"):
            h2h_k_delta = compute_pitcher_h2h_vs_team(
                extra["pitcher_id"], extra["opp_team_id"], game_date, extra["db"]
            )

    # Statcast metrics — loaded from DB when pitcher_id and season are available.
    _cur_year = datetime.now().year
    sc_season = extra.get("season") or extra.get("month_of_season") or _cur_year
    sc = _load_statcast_pitcher(
        extra.get("pitcher_id") or 0,
        sc_season if sc_season > 2000 else _cur_year,
        extra.get("db"),
    )
    sc_whiff     = sc["whiff_pct"]           or 0.0
    sc_chase     = sc["chase_rate"]           or 0.0
    sc_barrel    = sc["barrel_pct_against"]   or 0.0
    sc_hard_hit  = sc["hard_hit_pct_against"] or 0.0
    sc_spin_z    = ((sc["spin_rate_ff"] or _LEAGUE_AVG_SPIN_RATE_FF) - _LEAGUE_AVG_SPIN_RATE_FF) / _LEAGUE_AVG_SPIN_RATE_STD
    sc_exit_velo = sc["avg_exit_velocity_against"] or _LEAGUE_AVG_EXIT_VELO

    # delta_k9_vs_xk9: positive = outperforming stuff (regression expected).
    # xK/9 estimated from whiff% scaled to league average K/9 rate.
    if sc_whiff > 0:
        xk9 = (sc_whiff / _LEAGUE_AVG_WHIFF_PCT) * _LEAGUE_AVG_K_PER_9_PITCHER
        sc_delta_k9 = season["k_per_9"] - xk9
    else:
        sc_delta_k9 = 0.0

    # k9_momentum_l5: recent K/9 trend (l5 minus l10). Positive = pitcher gaining
    # K rate over last 5 starts; negative = declining. Explicit delta is more
    # useful for the GLM (linear on log scale) than the two raw values separately.
    sc_k9_momentum = l5["k_per_9"] - l10["k_per_9"]

    values = [
        l5["k_per_9"],
        l10["k_per_9"],
        season["k_per_9"],
        l10["era"],
        season["era"],
        l10["whip"],
        l10["bb_per_9"],
        l5["pitches_per_ip"],
        float(est_pitch_limit),
        float(opponent_rate if opponent_rate is not None else
              (LEAGUE_AVG_K_RATE if market == "pitcher_strikeouts" else LEAGUE_AVG_RUNS_PER_GAME)),
        float(ump_k_factor or 1.0),
        float(park_adj),
        float(temp),
        float(wind),
        float(rest_days),
        float(extra.get("is_home", 0)),
        float(extra.get("is_day_game") if extra.get("is_day_game") is not None
              else _is_day_game(extra.get("game_time"))),
        float(extra.get("month_of_season") or _month_from_date(game_date)),
        float(k_bb_ratio_l10),
        float(hr_per_9_l10),
        float(k_vol_l10),
        float(ip_vol_l10),
        float(h2h_k_delta),
        # Statcast features (8)
        float(sc_whiff),
        float(sc_chase),
        float(sc_barrel),
        float(sc_hard_hit),
        float(sc_spin_z),
        float(sc_exit_velo),
        float(sc_delta_k9),
        float(sc_k9_momentum),
    ]
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public: batter feature builder
# ---------------------------------------------------------------------------

_MARKET_TO_STAT_KEY = {
    "batter_hits": "hits",
    "batter_home_runs": "home_runs",
    "batter_total_bases": "total_bases",
}


def build_batter_features(batter_logs: List[Dict],
                          market: str,
                          pitcher_hand: Optional[str],
                          batter_hand: Optional[str],
                          venue: Optional[str],
                          weather: Optional[Dict],
                          lineup_position: Optional[int],
                          projected_pa: float = 4.0,  # noqa: ARG001
                          extra: Optional[Dict] = None) -> np.ndarray:
    """Build a feature vector in the order of BATTER_FEATURE_NAMES."""
    extra = extra or {}
    logs = batter_logs or []

    stat_key = _MARKET_TO_STAT_KEY.get(market, "hits")
    rate_l15 = _rolling_batter_rate(logs, stat_key, 15)
    rate_l5 = _rolling_batter_rate(logs, stat_key, 5)
    rate_season = _rolling_batter_rate(logs, stat_key, max(30, len(logs)))

    # Platoon split: if caller passed explicit rate in `extra`, use it; else
    # compute from batter_platoon_splits table (with Bayesian shrinkage) or fall back.
    platoon_rate = extra.get("platoon_rate_vs_hand")
    if platoon_rate is None:
        sc_season = extra.get("season") or extra.get("month_of_season") or 2025
        platoon_rate = compute_platoon_split(
            batter_logs=logs, vs_hand=pitcher_hand or "", stat_key=stat_key,
            db=extra.get("db"), batter_id=extra.get("batter_id"),
            season=sc_season if sc_season > 2000 else 2025,
        )

    is_switch = 1 if (batter_hand and batter_hand.upper() == 'S') else 0
    same_hand = 0
    if batter_hand and pitcher_hand and not is_switch:
        same_hand = 1 if batter_hand.upper() == pitcher_hand.upper() else 0

    park_key_map = {"batter_hits": "hits", "batter_home_runs": "hr", "batter_total_bases": "hr"}
    park_key = park_key_map.get(market, "runs")
    park_adj = get_park_factor(venue, weather=weather).get(park_key, 1.0) if venue else 1.0

    temp, wind = _extract_weather(weather)
    game_date = extra.get("game_date")
    rest_days = extra.get("rest_days")
    if rest_days is None:
        rest_days = compute_rest_days(logs, game_date)

    opp_starter_k = extra.get("opp_starter_k_per_9", 8.5)
    opp_bullpen_era = extra.get("opp_bullpen_era", LEAGUE_AVG_RUNS_PER_GAME)

    stat_vol_l15 = _rolling_batter_game_stdev(logs, stat_key, 15)
    k_rate_l15 = _rolling_batter_rate(logs, "strikeouts", 15)

    # Statcast contact-quality metrics
    _cur_year = datetime.now().year
    sc_season = extra.get("season") or extra.get("month_of_season") or _cur_year
    sc = _load_statcast_batter(
        extra.get("batter_id") or 0,
        sc_season if sc_season > 2000 else _cur_year,
        extra.get("db"),
    )
    sc_barrel      = sc["barrel_pct"]        or 0.0
    sc_xwoba       = sc["xwoba"]             or 0.0
    sc_exit_velo   = sc["exit_velocity_avg"] or _LEAGUE_AVG_EXIT_VELO
    sc_hard_hit    = sc["hard_hit_pct"]      or 0.0
    sc_launch_ang  = sc["launch_angle_avg"]  or 0.0
    sc_whiff       = sc["whiff_pct"]         or 0.0
    sc_sprint      = sc["sprint_speed"]      or _LEAGUE_AVG_SPRINT_SPEED

    values = [
        rate_l15,
        rate_season,
        rate_l5,
        float(platoon_rate),
        float(same_hand),
        float(is_switch),
        float(lineup_position or 5),
        float(park_adj),
        float(temp),
        float(wind),
        float(rest_days),
        float(extra.get("is_home", 0)),
        float(extra.get("is_day_game") if extra.get("is_day_game") is not None
              else _is_day_game(extra.get("game_time"))),
        float(extra.get("month_of_season") or _month_from_date(game_date)),
        float(opp_starter_k),
        float(opp_bullpen_era),
        float(stat_vol_l15),
        float(k_rate_l15),
        # Statcast features (7)
        float(sc_barrel),
        float(sc_xwoba),
        float(sc_exit_velo),
        float(sc_hard_hit),
        float(sc_launch_ang),
        float(sc_whiff),
        float(sc_sprint),
    ]
    return np.array(values, dtype=np.float64)
