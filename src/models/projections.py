import json
import math
import os
import time
from typing import List, Dict, Any, Optional
from src.utils.logging_utils import get_logger
from src.models.distributions import (
    get_probabilities, get_probabilities_mixture, compute_hr_pi0,
)
from src.models.pa_estimator import (
    estimate_pa_distribution, expected_pa, implied_team_total,
)
from src.data.park_factors import get_park_factor
from src.config import (
    LEAGUE_AVG_K_RATE, LEAGUE_AVG_RUNS_PER_GAME,
    LEAGUE_AVG_PITCHES_PER_IP, DEFAULT_PITCH_LIMIT,
    PLATOON_ADJUSTMENTS,
    PITCHER_RECENT_WEIGHT, PITCHER_SEASON_WEIGHT,
    BATTER_RECENT_WEIGHT, BATTER_SEASON_WEIGHT,
    LINEUP_PA_MAP, DEFAULT_PROJECTED_PA,
    LEAGUE_AVG_GAME_TOTAL, PA_ELASTICITY_TO_TOTAL,
    PA_ESTIMATOR_ENABLED,
    HR_ZINB_ENABLED,
    UMP_ER_K_DAMPENING,
)


def _batter_iso_from_logs(logs):
    """Recent + season blended ISO (slugging - avg) from batter game logs."""
    if not logs:
        return None
    total_2b = sum(l.get('doubles', 0) or 0 for l in logs)
    total_3b = sum(l.get('triples', 0) or 0 for l in logs)
    total_hr = sum(l.get('home_runs', 0) or 0 for l in logs)
    total_h = sum(l.get('hits', 0) or 0 for l in logs)
    total_ab = sum(l.get('at_bats', 0) or 0 for l in logs)
    if total_ab <= 0:
        return None
    slg = (total_h + total_2b + 2 * total_3b + 3 * total_hr) / total_ab
    avg = total_h / total_ab
    return max(0.0, slg - avg)


def _compute_hr_pi0(market, batter_logs, park_hr_factor, extra_features, weather):
    """Compute structural-zero probability for batter_home_runs; None for other markets."""
    if not HR_ZINB_ENABLED or market != "batter_home_runs":
        return None
    pitcher_hr9 = (extra_features or {}).get('opp_pitcher_hr9')
    wind_in_mph = (weather or {}).get('wind_in_mph')
    batter_iso = _batter_iso_from_logs(batter_logs)
    return compute_hr_pi0(
        pitcher_hr9=pitcher_hr9,
        batter_iso=batter_iso,
        park_hr_factor=park_hr_factor,
        wind_in_mph=wind_in_mph,
    )

logger = get_logger(__name__)


def _scale_pa_for_game_total(base_pa: float, game_total: float | None) -> float:
    """Scale projected plate appearances based on the game's over/under total.

    Higher game totals imply more baserunners, longer innings, more PAs.
    The elasticity is ~0.35: a 10% increase in expected runs yields ~3.5% more PAs.

    Returns base_pa unchanged when game_total is None.
    """
    if game_total is None:
        return base_pa
    pct_deviation = (game_total - LEAGUE_AVG_GAME_TOTAL) / LEAGUE_AVG_GAME_TOTAL
    scale_factor = 1.0 + PA_ELASTICITY_TO_TOTAL * pct_deviation
    scale_factor = max(0.90, min(1.15, scale_factor))
    return round(base_pa * scale_factor, 2)

_dispersion_cache: Dict[tuple, Dict] | None = None
_dispersion_cache_ts: float = 0.0
_DISPERSION_CACHE_TTL = 3600.0  # reload at most once per hour


def _load_dispersion() -> Dict[tuple, Dict]:
    """Load dispersion_params table into an in-memory dict, refreshed every hour."""
    global _dispersion_cache, _dispersion_cache_ts
    now = time.monotonic()
    if _dispersion_cache is not None and (now - _dispersion_cache_ts) < _DISPERSION_CACHE_TTL:
        return _dispersion_cache
    new_cache: Dict[tuple, Dict] = {}
    try:
        from src.data.db import get_db_connection
        with get_db_connection() as conn:
            try:
                rows = conn.execute(
                    "SELECT entity_id, market, alpha, sigma FROM dispersion_params"
                ).fetchall()
            except Exception:
                return _dispersion_cache if _dispersion_cache is not None else new_cache
            for r in rows:
                new_cache[(r["entity_id"], r["market"])] = {
                    "alpha": r["alpha"],
                    "sigma": r["sigma"],
                }
    except Exception:
        return _dispersion_cache if _dispersion_cache is not None else new_cache
    _dispersion_cache = new_cache
    _dispersion_cache_ts = now
    return _dispersion_cache


def get_dispersion(entity_id: str, market: str) -> Dict:
    """Get (alpha, sigma) for an entity+market, falling back to pool."""
    cache = _load_dispersion()
    hit = cache.get((str(entity_id), market))
    if hit:
        return hit
    return cache.get(("__pool__", market), {})

_calib_cache: Dict | None = None
_calib_cache_ts: float = 0.0
_CALIB_CACHE_TTL = 3600.0


def _load_calibration() -> Dict[str, Dict]:
    """Load active calibration_params rows as {market: {method, params}} with 1h TTL."""
    global _calib_cache, _calib_cache_ts
    now = time.monotonic()
    if _calib_cache is not None and (now - _calib_cache_ts) < _CALIB_CACHE_TTL:
        return _calib_cache
    new_cache: Dict[str, Dict] = {}
    try:
        from src.data.db import get_db_connection
        with get_db_connection() as conn:
            try:
                rows = conn.execute(
                    "SELECT market, method, params_json FROM calibration_params WHERE is_active=1"
                ).fetchall()
            except Exception:
                return _calib_cache if _calib_cache is not None else new_cache
            for r in rows:
                try:
                    new_cache[r["market"]] = {
                        "method": r["method"],
                        "params": json.loads(r["params_json"]),
                    }
                except Exception:
                    pass
    except Exception:
        return _calib_cache if _calib_cache is not None else new_cache
    _calib_cache = new_cache
    _calib_cache_ts = now
    return _calib_cache


def _apply_calibration(prob: float, market: str) -> float:
    """Apply the active calibration transform (Platt or isotonic) to a probability."""
    entry = _load_calibration().get(market)
    if not entry:
        return prob
    method = entry["method"]
    params = entry["params"]
    if method == "identity" or not params:
        return prob
    if method == "platt":
        a = params.get("a", 1.0)
        b = params.get("b", 0.0)
        p = max(1e-6, min(1 - 1e-6, prob))
        logit_p = math.log(p / (1.0 - p))
        return 1.0 / (1.0 + math.exp(-(a * logit_p + b)))
    if method == "isotonic":
        x = params.get("x_thresholds", [])
        y = params.get("y_thresholds", [])
        if not x or not y:
            return prob
        if prob <= x[0]:
            return float(y[0])
        if prob >= x[-1]:
            return float(y[-1])
        import bisect
        i = bisect.bisect_right(x, prob) - 1
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (prob - x0) / (x1 - x0)
    return prob


def _calibrate_result(result: Dict | None) -> Dict | None:
    """Apply calibration to prob_over / prob_under in a projection result dict."""
    if result is None:
        return None
    market = result.get("market", "")
    result["prob_over"] = _apply_calibration(result["prob_over"], market)
    result["prob_under"] = _apply_calibration(result["prob_under"], market)
    return result


_GLM_MARKETS = (
    "pitcher_strikeouts",
    "pitcher_earned_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
)

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


class ProjectionModel:
    def __init__(self, models_dir: str = None):
        self._models_dir = models_dir or _MODELS_DIR
        self._glm: Dict[str, Any] = {m: None for m in _GLM_MARKETS}
        self._load_glm_models()

    def _load_glm_models(self):
        """Load any serialized PoissonGLM models from disk; silently skip missing ones."""
        if not os.path.isdir(self._models_dir):
            return
        try:
            from src.models.ml_model import PoissonGLM
        except ImportError:
            return
        for market in _GLM_MARKETS:
            path = os.path.join(self._models_dir, f"mlb_poisson_{market}.pkl")
            if os.path.exists(path):
                try:
                    self._glm[market] = PoissonGLM.load(path)
                    logger.info(f"Loaded GLM model for {market}")
                except Exception as e:
                    logger.warning(f"Failed to load GLM model {path}: {e}")

    def project_pitcher_strikeouts(self, pitcher_logs: List[Dict], opponent_k_rate: float,
                                   venue: str, line: float,
                                   weather: dict = None,
                                   ump_k_factor: float = 1.0,
                                   extra_features: dict = None,
                                   player_id: int = None) -> Optional[Dict]:
        """
        Project pitcher strikeouts for a game.

        If a trained PoissonGLM model is loaded for pitcher_strikeouts, it takes
        precedence; otherwise the original weighted-average algorithm runs:
        1. L10 K/9 blended with season K/9
        2. Opponent team K% adjustment
        3. Park SO factor (weather-adjusted if available)
        4. Projected innings from pitch efficiency model
        """
        if not pitcher_logs or len(pitcher_logs) < 3:
            return None

        disp = get_dispersion(str(player_id), "pitcher_strikeouts") if player_id else {}

        glm_result = self._try_glm_pitcher(
            market="pitcher_strikeouts",
            pitcher_logs=pitcher_logs,
            opponent_rate=opponent_k_rate,
            venue=venue,
            line=line,
            weather=weather,
            ump_k_factor=ump_k_factor,
            extra_features=extra_features,
            dispersion=disp,
        )
        if glm_result is not None:
            return _calibrate_result(glm_result)

        # Sort by date descending
        logs = sorted(pitcher_logs, key=lambda x: x['date'], reverse=True)

        # Season K/9
        total_ip = sum(l['innings_pitched'] for l in logs)
        total_k = sum(l['strikeouts'] for l in logs)
        if total_ip == 0:
            return None
        season_k_per_9 = (total_k / total_ip) * 9.0

        # Recent L10 K/9
        recent = logs[:10]
        recent_ip = sum(l['innings_pitched'] for l in recent)
        recent_k = sum(l['strikeouts'] for l in recent)
        if recent_ip == 0:
            return None
        recent_k_per_9 = (recent_k / recent_ip) * 9.0

        # Blended K/9
        blended_k_per_9 = (PITCHER_RECENT_WEIGHT * recent_k_per_9 +
                           PITCHER_SEASON_WEIGHT * season_k_per_9)

        # Opponent K% adjustment
        opp_adj = opponent_k_rate / LEAGUE_AVG_K_RATE if LEAGUE_AVG_K_RATE > 0 else 1.0

        # Park factor (weather-adjusted if available)
        park = get_park_factor(venue, weather=weather)
        park_adj = park.get('so', 1.0)

        # Projected innings from pitch efficiency model
        ip_result = self._project_innings(logs)
        proj_ip = ip_result['proj_ip']

        # Final projection — ump_k_factor multiplies the raw K projection directly.
        # A large-zone umpire (factor > 1) boosts Ks; a tight zone (factor < 1) suppresses.
        projected_k = max(0.0, (blended_k_per_9 / 9.0) * proj_ip * opp_adj * park_adj * ump_k_factor)

        prob_over, prob_under = get_probabilities(
            projected_k, line, "pitcher_strikeouts", alpha=disp.get("alpha"),
        )

        return _calibrate_result({
            "player_name": None,  # set by caller
            "market": "pitcher_strikeouts",
            "line": line,
            "projected_mean": round(projected_k, 2),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "alpha": disp.get("alpha"),
            "sigma": disp.get("sigma"),
            "injury_status": "Healthy",
            "sample_size": len(logs),
            "context": {
                "model": "weighted_avg",
                "blended_k9": round(blended_k_per_9, 2),
                "opp_k_rate": round(opponent_k_rate, 3),
                "opp_adj": round(opp_adj, 3),
                "park_adj": round(park_adj, 3),
                "ump_k_factor": round(ump_k_factor, 3),
                "proj_ip": round(proj_ip, 2),
                "pitches_per_ip": ip_result['pitches_per_ip'],
                "est_pitch_limit": ip_result['est_pitch_limit'],
                "venue": venue,
                "weather": weather,
            },
        })

    def project_pitcher_earned_runs(self, pitcher_logs: List[Dict], opponent_runs_per_game: float,
                                    venue: str, line: float,
                                    weather: dict = None,
                                    ump_k_factor: float = 1.0,
                                    extra_features: dict = None,
                                    player_id: int = None,
                                    bp_weight: float = 0.10) -> Optional[Dict]:
        """Project pitcher earned runs for a game."""
        if not pitcher_logs or len(pitcher_logs) < 3:
            return None

        disp = get_dispersion(str(player_id), "pitcher_earned_runs") if player_id else {}

        glm_result = self._try_glm_pitcher(
            market="pitcher_earned_runs",
            pitcher_logs=pitcher_logs,
            opponent_rate=opponent_runs_per_game,
            venue=venue,
            line=line,
            weather=weather,
            ump_k_factor=ump_k_factor,
            extra_features=extra_features,
            dispersion=disp,
        )
        if glm_result is not None:
            return _calibrate_result(glm_result)

        logs = sorted(pitcher_logs, key=lambda x: x['date'], reverse=True)

        # Season ERA
        total_ip = sum(l['innings_pitched'] for l in logs)
        total_er = sum(l['earned_runs'] for l in logs)
        if total_ip == 0:
            return None
        season_era = (total_er / total_ip) * 9.0

        # Recent L10 ERA
        recent = logs[:10]
        recent_ip = sum(l['innings_pitched'] for l in recent)
        recent_er = sum(l['earned_runs'] for l in recent)
        if recent_ip == 0:
            return None
        recent_era = (recent_er / recent_ip) * 9.0

        blended_era = (PITCHER_RECENT_WEIGHT * recent_era +
                       PITCHER_SEASON_WEIGHT * season_era)

        # Opponent run scoring adjustment
        opp_adj = opponent_runs_per_game / LEAGUE_AVG_RUNS_PER_GAME if LEAGUE_AVG_RUNS_PER_GAME > 0 else 1.0

        # Park factor (weather-adjusted if available)
        park = get_park_factor(venue, weather=weather)
        park_adj = park.get('runs', 1.0)

        # Projected innings from pitch efficiency model
        ip_result = self._project_innings(logs)
        proj_ip = ip_result['proj_ip']

        # Umpire effect on ER is indirect: a large-zone ump generates more Ks → fewer
        # baserunners → slightly fewer ER. The relationship is dampened relative to the
        # direct K adjustment (UMP_ER_K_DAMPENING = 0.3 by default).
        ump_er_factor = 1.0 - (ump_k_factor - 1.0) * UMP_ER_K_DAMPENING
        
        # Bullpen effect: a bad bullpen allows more of the starter's inherited runners to score
        pitcher_bullpen_era = extra_features.get('pitcher_bullpen_era', LEAGUE_AVG_RUNS_PER_GAME) if extra_features else LEAGUE_AVG_RUNS_PER_GAME
        bp_adj = (1.0 + ((pitcher_bullpen_era / LEAGUE_AVG_RUNS_PER_GAME) - 1.0) * bp_weight) if LEAGUE_AVG_RUNS_PER_GAME > 0 else 1.0
        
        projected_er = max(0.0, (blended_era / 9.0) * proj_ip * opp_adj * park_adj * ump_er_factor * bp_adj)

        prob_over, prob_under = get_probabilities(
            projected_er, line, "pitcher_earned_runs", alpha=disp.get("alpha"),
        )

        return _calibrate_result({
            "player_name": None,
            "market": "pitcher_earned_runs",
            "line": line,
            "projected_mean": round(projected_er, 2),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "alpha": disp.get("alpha"),
            "sigma": disp.get("sigma"),
            "injury_status": "Healthy",
            "sample_size": len(logs),
            "context": {
                "model": "weighted_avg",
                "blended_era": round(blended_era, 2),
                "opp_runs_pg": round(opponent_runs_per_game, 2),
                "opp_adj": round(opp_adj, 3),
                "park_adj": round(park_adj, 3),
                "ump_k_factor": round(ump_k_factor, 3),
                "ump_er_factor": round(ump_er_factor, 3),
                "bp_adj": round(bp_adj, 3),
                "proj_ip": round(proj_ip, 2),
                "pitches_per_ip": ip_result['pitches_per_ip'],
                "est_pitch_limit": ip_result['est_pitch_limit'],
                "venue": venue,
                "weather": weather,
            },
        })

    def project_batter_stat(self, batter_logs: List[Dict], stat_type: str,
                            pitcher_hand: str, batter_hand: str,
                            venue: str, line: float,
                            lineup_position: int = None,
                            weather: dict = None,
                            extra_features: dict = None,
                            player_id: int = None,
                            game_total: float = None,
                            bp_weight: float = 0.10) -> Optional[Dict]:
        """
        Project a batter stat (hits, total_bases, home_runs).

        Algorithm:
        1. Compute per-PA rates from L15 and season game logs
        2. Project today's PAs from lineup position (leadoff ~4.5, 9-hole ~3.7)
        3. projected_stat = per_PA_rate * projected_PAs * platoon_adj * park_adj
        Park factors are weather-adjusted when weather data is available.
        """
        if not batter_logs or len(batter_logs) < 5:
            return None

        logs = sorted(batter_logs, key=lambda x: x['date'], reverse=True)

        stat_key = self._get_batter_stat_key(stat_type)
        if not stat_key:
            return None

        market_key = self._stat_type_to_market(stat_type)
        disp = get_dispersion(str(player_id), market_key) if player_id else {}

        glm_result = self._try_glm_batter(
            market=market_key,
            batter_logs=logs,
            pitcher_hand=pitcher_hand,
            batter_hand=batter_hand,
            venue=venue,
            line=line,
            lineup_position=lineup_position,
            weather=weather,
            extra_features=extra_features,
            dispersion=disp,
            game_total=game_total,
        )
        if glm_result is not None:
            return _calibrate_result(glm_result)

        # --- Per-PA rates (not per-game) ---
        # Season per-PA rate
        total_stat_season = sum(l.get(stat_key, 0) for l in logs)
        total_pa_season = sum(l.get('plate_appearances', 0) or l.get('at_bats', 0) for l in logs)
        if total_pa_season == 0:
            return None
        season_rate_per_pa = total_stat_season / total_pa_season

        # Recent L15 per-PA rate
        recent = logs[:15]
        recent_stat = sum(l.get(stat_key, 0) for l in recent)
        recent_pa = sum(l.get('plate_appearances', 0) or l.get('at_bats', 0) for l in recent)
        if recent_pa == 0:
            return None
        recent_rate_per_pa = recent_stat / recent_pa

        # Blended per-PA rate
        blended_per_pa = (BATTER_RECENT_WEIGHT * recent_rate_per_pa +
                          BATTER_SEASON_WEIGHT * season_rate_per_pa)

        # --- Projected plate appearances from lineup position, scaled by game total ---
        base_pa = LINEUP_PA_MAP.get(lineup_position, DEFAULT_PROJECTED_PA)
        if PA_ESTIMATOR_ENABLED:
            itt = implied_team_total(game_total)
            pa_dist = estimate_pa_distribution(lineup_position, itt)
            projected_pa = expected_pa(pa_dist)
        else:
            itt = None
            pa_dist = None
            projected_pa = _scale_pa_for_game_total(base_pa, game_total)

        # Platoon adjustment
        platoon_adj = self._get_platoon_adjustment(batter_hand, pitcher_hand, stat_type)

        # Park factor (weather-adjusted if available)
        park = get_park_factor(venue, weather=weather)
        park_key = self._stat_to_park_key(stat_type)
        park_adj = park.get(park_key, 1.0)

        # Bullpen effect: bad opposing bullpen gives batters more late-inning opportunities
        opp_bullpen_era = extra_features.get('opp_bullpen_era', LEAGUE_AVG_RUNS_PER_GAME) if extra_features else LEAGUE_AVG_RUNS_PER_GAME
        bp_adj = (1.0 + ((opp_bullpen_era / LEAGUE_AVG_RUNS_PER_GAME) - 1.0) * bp_weight) if LEAGUE_AVG_RUNS_PER_GAME > 0 else 1.0

        # Final projection: rate * opportunities * adjustments
        projected = max(0.0, blended_per_pa * projected_pa * platoon_adj * park_adj * bp_adj)

        market_key = self._stat_type_to_market(stat_type)
        pi0 = _compute_hr_pi0(market_key, logs, park_adj, extra_features, weather)
        if pa_dist is not None:
            prob_over, prob_under = get_probabilities_mixture(
                blended_per_pa, line, market_key, pa_dist,
                adjustments=platoon_adj * park_adj * bp_adj,
                alpha=disp.get("alpha"), sigma=disp.get("sigma"),
                pi0=pi0,
            )
        else:
            prob_over, prob_under = get_probabilities(
                projected, line, market_key,
                alpha=disp.get("alpha"), sigma=disp.get("sigma"),
                pi0=pi0,
            )

        return _calibrate_result({
            "player_name": None,
            "market": market_key,
            "line": line,
            "projected_mean": round(projected, 3),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "alpha": disp.get("alpha"),
            "sigma": disp.get("sigma"),
            "injury_status": "Healthy",
            "sample_size": len(logs),
            "context": {
                "model": "weighted_avg",
                "rate_per_pa": round(blended_per_pa, 4),
                "recent_rate_per_pa": round(recent_rate_per_pa, 4),
                "season_rate_per_pa": round(season_rate_per_pa, 4),
                "base_pa": base_pa,
                "projected_pa": projected_pa,
                "pa_distribution": pa_dist,
                "implied_team_total": itt,
                "game_total": game_total,
                "lineup_position": lineup_position,
                "platoon_adj": round(platoon_adj, 3),
                "park_adj": round(park_adj, 3),
                "bp_adj": round(bp_adj, 3),
                "hr_pi0": pi0,
                "batter_hand": batter_hand,
                "pitcher_hand": pitcher_hand,
                "venue": venue,
                "weather": weather,
            },
        })

    # ------------------------------------------------------------------
    # GLM branch: shared helpers for pitcher and batter markets
    # ------------------------------------------------------------------

    def _try_glm_pitcher(self, market, pitcher_logs, opponent_rate, venue, line,
                         weather, ump_k_factor, extra_features, dispersion=None):
        """Run the Poisson GLM path for a pitcher market. Returns None if unavailable."""
        glm = self._glm.get(market)
        if glm is None:
            return None
        from src.data.feature_builder import build_pitcher_features
        from src.models.monte_carlo import mc_prob_over
        disp = dispersion or {}

        ip_result = self._project_innings(
            sorted(pitcher_logs, key=lambda x: x['date'], reverse=True)
        )
        try:
            features = build_pitcher_features(
                pitcher_logs=pitcher_logs,
                market=market,
                opponent_rate=opponent_rate,
                venue=venue,
                weather=weather,
                ump_k_factor=ump_k_factor,
                projected_ip=ip_result['proj_ip'],
                extra=extra_features or {},
            )
        except Exception as e:
            logger.warning(f"GLM feature build failed for {market}: {e}; falling back.")
            return None

        try:
            projected_mean = glm.predict_mean(features, exposure=ip_result['proj_ip'])
        except Exception as e:
            logger.warning(f"GLM predict failed for {market}: {e}; falling back.")
            return None

        prob_over, prob_under = mc_prob_over(
            projected_mean, line, market,
            nb_alpha=disp.get("alpha"),
        )

        return {
            "player_name": None,
            "market": market,
            "line": line,
            "projected_mean": round(projected_mean, 3),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "alpha": disp.get("alpha"),
            "sigma": disp.get("sigma"),
            "injury_status": "Healthy",
            "sample_size": len(pitcher_logs),
            "context": {
                "model": "glm",
                "proj_ip": round(ip_result['proj_ip'], 2),
                "pitches_per_ip": ip_result['pitches_per_ip'],
                "est_pitch_limit": ip_result['est_pitch_limit'],
                "ump_k_factor": round(ump_k_factor, 3),
                "venue": venue,
                "weather": weather,
            },
        }

    def _try_glm_batter(self, market, batter_logs, pitcher_hand, batter_hand,
                        venue, line, lineup_position, weather, extra_features,
                        dispersion=None, game_total=None):
        """Run the Poisson GLM path for a batter market. Returns None if unavailable."""
        glm = self._glm.get(market)
        if glm is None:
            return None
        from src.data.feature_builder import build_batter_features
        from src.models.monte_carlo import mc_prob_over
        disp = dispersion or {}

        base_pa = LINEUP_PA_MAP.get(lineup_position, DEFAULT_PROJECTED_PA)
        if PA_ESTIMATOR_ENABLED:
            itt = implied_team_total(game_total)
            pa_dist = estimate_pa_distribution(lineup_position, itt)
            projected_pa = expected_pa(pa_dist)
        else:
            itt = None
            pa_dist = None
            projected_pa = _scale_pa_for_game_total(base_pa, game_total)

        try:
            features = build_batter_features(
                batter_logs=batter_logs,
                market=market,
                pitcher_hand=pitcher_hand,
                batter_hand=batter_hand,
                venue=venue,
                weather=weather,
                lineup_position=lineup_position,
                projected_pa=projected_pa,
                extra=extra_features or {},
            )
        except Exception as e:
            logger.warning(f"GLM feature build failed for {market}: {e}; falling back.")
            return None

        try:
            projected_mean = glm.predict_mean(features, exposure=projected_pa)
        except Exception as e:
            logger.warning(f"GLM predict failed for {market}: {e}; falling back.")
            return None

        park_hr_factor = get_park_factor(venue, weather=weather).get("hr", 1.0)
        pi0 = _compute_hr_pi0(market, batter_logs, park_hr_factor, extra_features, weather)

        if pa_dist is not None:
            prob_over = 0.0
            prob_under = 0.0
            for k, p_k in pa_dist.items():
                if p_k <= 0:
                    continue
                try:
                    mean_k = glm.predict_mean(features, exposure=k)
                except Exception as e:
                    logger.warning(f"GLM predict failed for {market} at PA={k}: {e}; falling back.")
                    prob_over, prob_under = mc_prob_over(
                        projected_mean, line, market,
                        nb_alpha=disp.get("alpha"),
                        tb_std=disp.get("sigma"),
                        zinb_pi0=pi0,
                    )
                    break
                over_k, under_k = mc_prob_over(
                    mean_k, line, market,
                    nb_alpha=disp.get("alpha"),
                    tb_std=disp.get("sigma"),
                    zinb_pi0=pi0,
                )
                prob_over += p_k * over_k
                prob_under += p_k * under_k
        else:
            prob_over, prob_under = mc_prob_over(
                projected_mean, line, market,
                nb_alpha=disp.get("alpha"),
                tb_std=disp.get("sigma"),
                zinb_pi0=pi0,
            )

        return {
            "player_name": None,
            "market": market,
            "line": line,
            "projected_mean": round(projected_mean, 3),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "alpha": disp.get("alpha"),
            "sigma": disp.get("sigma"),
            "injury_status": "Healthy",
            "sample_size": len(batter_logs),
            "context": {
                "model": "glm",
                "base_pa": base_pa,
                "projected_pa": projected_pa,
                "pa_distribution": pa_dist,
                "implied_team_total": itt,
                "hr_pi0": pi0,
                "game_total": game_total,
                "lineup_position": lineup_position,
                "batter_hand": batter_hand,
                "pitcher_hand": pitcher_hand,
                "venue": venue,
                "weather": weather,
            },
        }

    def _get_batter_stat_key(self, stat_type: str) -> Optional[str]:
        mapping = {
            "hits": "hits",
            "total_bases": "total_bases",
            "home_runs": "home_runs",
        }
        return mapping.get(stat_type)

    def _stat_to_park_key(self, stat_type: str) -> str:
        mapping = {
            "hits": "hits",
            "total_bases": "hr",  # TB heavily influenced by HR factor
            "home_runs": "hr",
        }
        return mapping.get(stat_type, "runs")

    def _stat_type_to_market(self, stat_type: str) -> str:
        mapping = {
            "hits": "batter_hits",
            "total_bases": "batter_total_bases",
            "home_runs": "batter_home_runs",
        }
        return mapping.get(stat_type, stat_type)

    def _project_innings(self, logs: List[Dict]) -> Dict:
        """
        Estimate today's projected innings from pitch efficiency.

        Pulls recent pitches-per-inning from the last 5 starts, anchors the
        starter pitch limit at DEFAULT_PITCH_LIMIT (+10 if a pitcher has been
        running deeper than that), and divides to get projected IP.
        """
        recent = logs[:5] if logs else []

        total_ip = sum(l.get('innings_pitched') or 0 for l in recent)
        total_pitches = sum(l.get('pitches_thrown') or 0 for l in recent)

        if total_ip > 0 and total_pitches > 0:
            pitches_per_ip = total_pitches / total_ip
        else:
            pitches_per_ip = LEAGUE_AVG_PITCHES_PER_IP

        # Allow a small bump for pitchers who've been averaging more pitches
        # than the league standard starter budget.
        if recent:
            recent_pitch_counts = [l.get('pitches_thrown') or 0 for l in recent]
            avg_pitches = sum(recent_pitch_counts) / len(recent_pitch_counts) if recent_pitch_counts else 0
            est_pitch_limit = int(min(110, max(DEFAULT_PITCH_LIMIT, avg_pitches + 10)))
        else:
            est_pitch_limit = DEFAULT_PITCH_LIMIT

        proj_ip = est_pitch_limit / pitches_per_ip if pitches_per_ip > 0 else 5.5
        # Safety clamp: starters almost never exceed 8 IP or go under 3 IP in a projection
        proj_ip = max(3.0, min(8.0, proj_ip))

        return {
            "proj_ip": proj_ip,
            "pitches_per_ip": round(pitches_per_ip, 2),
            "est_pitch_limit": est_pitch_limit,
        }

    def _get_platoon_adjustment(self, batter_hand: str, pitcher_hand: str, stat_type: str) -> float:
        """
        Returns platoon multiplier based on handedness matchup.
        Same-hand = disadvantage for batter, opposite-hand = advantage.
        Switch hitters (S) get no adjustment.
        """
        if not batter_hand or not pitcher_hand:
            return 1.0

        batter_hand = batter_hand.upper()
        pitcher_hand = pitcher_hand.upper()

        if batter_hand == 'S':
            return 1.0

        market_key = self._stat_type_to_market(stat_type)

        if batter_hand == pitcher_hand:
            return PLATOON_ADJUSTMENTS["same_hand"].get(market_key, 1.0)
        else:
            return PLATOON_ADJUSTMENTS["opposite_hand"].get(market_key, 1.0)
