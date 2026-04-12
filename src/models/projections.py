from typing import List, Dict, Any, Optional
from src.utils.logging_utils import get_logger
from src.models.distributions import get_probabilities
from src.data.park_factors import get_park_factor
from src.config import (
    LEAGUE_AVG_K_RATE, LEAGUE_AVG_RUNS_PER_GAME,
    LEAGUE_AVG_PITCHES_PER_IP, DEFAULT_PITCH_LIMIT,
    PLATOON_ADJUSTMENTS,
    PITCHER_RECENT_WEIGHT, PITCHER_SEASON_WEIGHT,
    BATTER_RECENT_WEIGHT, BATTER_SEASON_WEIGHT,
    LINEUP_PA_MAP, DEFAULT_PROJECTED_PA,
    UMP_ER_K_DAMPENING,
)

logger = get_logger(__name__)


class ProjectionModel:
    def __init__(self):
        pass

    def project_pitcher_strikeouts(self, pitcher_logs: List[Dict], opponent_k_rate: float,
                                   venue: str, line: float,
                                   weather: dict = None,
                                   ump_k_factor: float = 1.0) -> Optional[Dict]:
        """
        Project pitcher strikeouts for a game.

        Algorithm:
        1. L10 K/9 blended with season K/9
        2. Opponent team K% adjustment
        3. Park SO factor (weather-adjusted if available)
        4. Projected innings from pitch efficiency model
        """
        if not pitcher_logs or len(pitcher_logs) < 3:
            return None

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
        projected_k = (blended_k_per_9 / 9.0) * proj_ip * opp_adj * park_adj * ump_k_factor

        prob_over, prob_under = get_probabilities(projected_k, line, "pitcher_strikeouts")

        return {
            "player_name": None,  # set by caller
            "market": "pitcher_strikeouts",
            "line": line,
            "projected_mean": round(projected_k, 2),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "injury_status": "Healthy",
            "sample_size": len(logs),
            "context": {
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
        }

    def project_pitcher_earned_runs(self, pitcher_logs: List[Dict], opponent_runs_per_game: float,
                                    venue: str, line: float,
                                    weather: dict = None,
                                    ump_k_factor: float = 1.0) -> Optional[Dict]:
        """Project pitcher earned runs for a game."""
        if not pitcher_logs or len(pitcher_logs) < 3:
            return None

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
        projected_er = (blended_era / 9.0) * proj_ip * opp_adj * park_adj * ump_er_factor

        prob_over, prob_under = get_probabilities(projected_er, line, "pitcher_earned_runs")

        return {
            "player_name": None,
            "market": "pitcher_earned_runs",
            "line": line,
            "projected_mean": round(projected_er, 2),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "injury_status": "Healthy",
            "sample_size": len(logs),
            "context": {
                "blended_era": round(blended_era, 2),
                "opp_runs_pg": round(opponent_runs_per_game, 2),
                "opp_adj": round(opp_adj, 3),
                "park_adj": round(park_adj, 3),
                "ump_k_factor": round(ump_k_factor, 3),
                "ump_er_factor": round(ump_er_factor, 3),
                "proj_ip": round(proj_ip, 2),
                "pitches_per_ip": ip_result['pitches_per_ip'],
                "est_pitch_limit": ip_result['est_pitch_limit'],
                "venue": venue,
                "weather": weather,
            },
        }

    def project_batter_stat(self, batter_logs: List[Dict], stat_type: str,
                            pitcher_hand: str, batter_hand: str,
                            venue: str, line: float,
                            lineup_position: int = None,
                            weather: dict = None) -> Optional[Dict]:
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

        # --- Projected plate appearances from lineup position ---
        projected_pa = LINEUP_PA_MAP.get(lineup_position, DEFAULT_PROJECTED_PA)

        # Platoon adjustment
        platoon_adj = self._get_platoon_adjustment(batter_hand, pitcher_hand, stat_type)

        # Park factor (weather-adjusted if available)
        park = get_park_factor(venue, weather=weather)
        park_key = self._stat_to_park_key(stat_type)
        park_adj = park.get(park_key, 1.0)

        # Final projection: rate * opportunities * adjustments
        projected = blended_per_pa * projected_pa * platoon_adj * park_adj

        market_key = self._stat_type_to_market(stat_type)
        prob_over, prob_under = get_probabilities(projected, line, market_key)

        return {
            "player_name": None,
            "market": market_key,
            "line": line,
            "projected_mean": round(projected, 3),
            "prob_over": prob_over,
            "prob_under": prob_under,
            "injury_status": "Healthy",
            "sample_size": len(logs),
            "context": {
                "rate_per_pa": round(blended_per_pa, 4),
                "recent_rate_per_pa": round(recent_rate_per_pa, 4),
                "season_rate_per_pa": round(season_rate_per_pa, 4),
                "projected_pa": projected_pa,
                "lineup_position": lineup_position,
                "platoon_adj": round(platoon_adj, 3),
                "park_adj": round(park_adj, 3),
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
