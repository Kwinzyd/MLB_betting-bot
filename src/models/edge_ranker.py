from typing import Dict, Any
from src.config import EDGE_MIN
from src.models.kelly import fractional_kelly
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def rank_edge(projection: Dict[str, Any], odds: float, side: str, devigged_prob: float) -> Dict[str, Any]:
    """
    Ranks the edge of a projection against a book's odds.
    Integrates Kelly sizing for bet recommendations.
    """
    model_prob = projection['prob_over'] if side == 'over' else projection['prob_under']

    # Expected value per $1 wagered
    ev = (model_prob * odds) - 1.0

    # Percentage edge vs book implied probability
    book_implied = 1.0 / odds
    edge_pct = (model_prob - book_implied) * 100

    # Kelly bet sizing
    kelly = fractional_kelly(model_prob, odds)

    # Playability filters
    is_playable = True
    reasons = []

    if edge_pct < EDGE_MIN:
        is_playable = False
        reasons.append(f"Edge too small ({edge_pct:.1f}% < {EDGE_MIN}%)")

    injury_status = projection.get('injury_status', 'Healthy')
    if injury_status in ['IL', 'Out']:
        is_playable = False
        reasons.append(f"Injury status: {injury_status}")

    sample_size = projection.get('sample_size', 0)
    if sample_size < 5:
        is_playable = False
        reasons.append(f"Insufficient sample ({sample_size} < 5 games)")

    if kelly['kelly_fraction'] <= 0:
        is_playable = False
        reasons.append("Negative Kelly (no edge)")

    return {
        "edge_pct": round(edge_pct, 2),
        "ev": round(ev, 4),
        "model_prob": round(model_prob, 4),
        "book_implied": round(book_implied, 4),
        "devigged_prob": round(devigged_prob, 4),
        "is_playable": is_playable,
        "reasons": reasons,
        "kelly": kelly,
    }
