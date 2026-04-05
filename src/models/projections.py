from typing import List, Dict, Any, Optional
from src.utils.logging_utils import get_logger
from src.models.distributions import get_probabilities
from src.data.park_factors import get_park_factor
from src.config import (
    LEAGUE_AVG_K_RATE, LEAGUE_AVG_RUNS_PER_GAME,
    PLATOON_ADJUSTMENTS,
    PITCHER_RECENT_WEIGHT, PITCHER_SEASON_WEIGHT,
    BATTER_RECENT_WEIGHT, BATTER_SEASON_WEIGHT,
)

logger = get_logger(__name__)


class ProjectionModel:
    def __init__(self):
        pass

    def project_pitcher_strikeouts(self, pitcher_logs: List[Dict], opponent_k_rate: float,
                                   venue: str, line: float) -> Optional[Dict]:
        """
        Project pitcher strikeouts for a game.

        Algorithm:
        1. L10 K/9 blended with season K/9
        2. Opponent team K% adjustment
        3. Park SO factor
        4. Projected innings from L5 average
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

        # Park factor
        park = get_park_factor(venue)
        park_adj = park.get('so', 1.0)

        # Projected innings (L5 average, capped 4.0-8.0)
        recent_5 = logs[:5]
        proj_ip = sum(l['innings_pitched'] for l in recent_5) / len(recent_5)
        proj_ip = max(4.0, min(8.0, proj_ip))

        # Final projection
        projected_k = (blended_k_per_9 / 9.0) * proj_ip * opp_adj * park_adj

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
                "proj_ip": round(proj_ip, 2),
                "venue": venue,
            },
        }

    def project_pitcher_earned_runs(self, pitcher_logs: List[Dict], opponent_runs_per_game: float,
                                    venue: str, line: float) -> Optional[Dict]:
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

        # Park factor
        park = get_park_factor(venue)
        park_adj = park.get('runs', 1.0)

        # Projected innings
        recent_5 = logs[:5]
        proj_ip = sum(l['innings_pitched'] for l in recent_5) / len(recent_5)
        proj_ip = max(4.0, min(8.0, proj_ip))

        projected_er = (blended_era / 9.0) * proj_ip * opp_adj * park_adj

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
                "proj_ip": round(proj_ip, 2),
                "venue": venue,
            },
        }

    def project_batter_stat(self, batter_logs: List[Dict], stat_type: str,
                            pitcher_hand: str, batter_hand: str,
                            venue: str, line: float) -> Optional[Dict]:
        """
        Project a batter stat (hits, total_bases, home_runs).

        Algorithm:
        1. L15 per-game rate blended with season per-game rate
        2. Platoon adjustment based on handedness matchup
        3. Park factor
        """
        if not batter_logs or len(batter_logs) < 5:
            return None

        logs = sorted(batter_logs, key=lambda x: x['date'], reverse=True)

        stat_key = self._get_batter_stat_key(stat_type)
        if not stat_key:
            return None

        # Season per-game rate
        total_stat = sum(l.get(stat_key, 0) for l in logs)
        season_rate = total_stat / len(logs)

        # Recent L15 per-game rate
        recent = logs[:15]
        recent_stat = sum(l.get(stat_key, 0) for l in recent)
        recent_rate = recent_stat / len(recent)

        # Blended rate
        blended = (BATTER_RECENT_WEIGHT * recent_rate +
                   BATTER_SEASON_WEIGHT * season_rate)

        # Platoon adjustment
        platoon_adj = self._get_platoon_adjustment(batter_hand, pitcher_hand, stat_type)

        # Park factor
        park = get_park_factor(venue)
        park_key = self._stat_to_park_key(stat_type)
        park_adj = park.get(park_key, 1.0)

        # Final projection
        projected = blended * platoon_adj * park_adj

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
                "recent_rate": round(recent_rate, 3),
                "season_rate": round(season_rate, 3),
                "platoon_adj": round(platoon_adj, 3),
                "park_adj": round(park_adj, 3),
                "batter_hand": batter_hand,
                "pitcher_hand": pitcher_hand,
                "venue": venue,
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
