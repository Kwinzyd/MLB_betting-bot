from typing import Dict, Any
from src.config import (
    EDGE_MIN, MIN_ODDS, MIN_MODEL_PROB, MIN_SAMPLE_SIZE,
    SHARP_MODEL_AGREEMENT_TOL,
)
from src.models.kelly import fractional_kelly
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def rank_edge(projection: Dict[str, Any], odds: float, side: str,
              sharp_prob: float) -> Dict[str, Any]:
    """
    Rank a soft-book offer against the sharp-devigged "true" probability.

    sharp_prob is the devigged probability from Pinnacle/Circa for this side.
    It is the source of truth for the edge calculation:
        edge_pct = (sharp_prob - 1/odds) * 100

    The model's probability is a validation gate, not the edge generator:
    when |model_prob - sharp_prob| exceeds SHARP_MODEL_AGREEMENT_TOL, the
    prop is marked unplayable on the assumption that the disagreement means
    one of the two is wrong and we shouldn't bet blind.

    Kelly sizing also uses sharp_prob — betting your edge, not your model.
    """
    model_prob = projection['prob_over'] if side == 'over' else projection['prob_under']

    book_implied = 1.0 / odds
    edge_pct = (sharp_prob - book_implied) * 100
    ev = (sharp_prob * odds) - 1.0
    kelly = fractional_kelly(sharp_prob, odds)

    is_playable = True
    reasons = []

    if edge_pct < EDGE_MIN:
        is_playable = False
        reasons.append(f"Edge too small ({edge_pct:.1f}% < {EDGE_MIN}%)")

    if odds < MIN_ODDS:
        is_playable = False
        reasons.append(f"Odds too juicy ({odds:.2f} < {MIN_ODDS})")

    if sharp_prob < MIN_MODEL_PROB:
        is_playable = False
        reasons.append(f"Low sharp confidence ({sharp_prob:.2f} < {MIN_MODEL_PROB})")

    disagreement = abs(model_prob - sharp_prob)
    if disagreement > SHARP_MODEL_AGREEMENT_TOL:
        is_playable = False
        reasons.append(
            f"Model/sharp disagreement ({disagreement:.2f} > {SHARP_MODEL_AGREEMENT_TOL})"
        )

    injury_status = projection.get('injury_status', 'Healthy')
    if injury_status in ['IL', 'Out']:
        is_playable = False
        reasons.append(f"Injury status: {injury_status}")

    sample_size = projection.get('sample_size', 0)
    if sample_size < MIN_SAMPLE_SIZE:
        is_playable = False
        reasons.append(f"Insufficient sample ({sample_size} < {MIN_SAMPLE_SIZE} games)")

    if kelly['kelly_fraction'] <= 0:
        is_playable = False
        reasons.append("Negative Kelly (no edge)")

    return {
        "edge_pct": round(edge_pct, 2),
        "ev": round(ev, 4),
        "model_prob": round(model_prob, 4),
        "sharp_prob": round(sharp_prob, 4),
        "book_implied": round(book_implied, 4),
        "disagreement": round(disagreement, 4),
        "is_playable": is_playable,
        "reasons": reasons,
        "kelly": kelly,
    }
