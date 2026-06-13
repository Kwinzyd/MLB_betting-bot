from typing import Dict, Any
from src.config import (
    EDGE_MIN, MIN_ODDS, MIN_MODEL_PROB, MIN_SAMPLE_SIZE,
    SHARP_MODEL_AGREEMENT_TOL, KELLY_FRACTION
)
from src.models.kelly import fractional_kelly
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def line_movement_signal(current_prob: float, opening_prob: float) -> Dict[str, Any]:
    """
    Compare current sharp probability to opening sharp probability.
    Returns tracking info and Kelly multiplier.
    If sharp money moves against us, kill the bet or reduce Kelly.
    """
    if opening_prob is None:
        return {"is_killed": False, "multiplier": 1.0, "diff": 0.0}

    diff = current_prob - opening_prob
    kill = False

    if diff <= -0.015:
        kill = True # Sharp moved against us by 1.5%+
        multiplier = 0.0
    elif diff < 0.0:
        multiplier = 0.5 # Slight fade, reduce size
    elif diff >= 0.015:
        multiplier = 1.5 # Caught steam, bump size
    else:
        multiplier = 1.0 # Negligible

    return {"is_killed": kill, "multiplier": multiplier, "diff": round(diff, 4)}


def rank_edge(projection: Dict[str, Any], odds: float, side: str,
              sharp_prob: float, opening_prob: float = None,
              steam_detected: bool = False) -> Dict[str, Any]:
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
    lm_signal = line_movement_signal(sharp_prob, opening_prob)
    # Steam and favorable line movement are two views of the same price
    # signal — don't let them stack to 2.25x. Combined boost caps at 1.5x.
    combined_multiplier = lm_signal["multiplier"]
    if steam_detected and combined_multiplier > 0:
        combined_multiplier = max(combined_multiplier, 1.5)
    combined_multiplier = min(combined_multiplier, 1.5)
    adjusted_fraction = KELLY_FRACTION * combined_multiplier
    kelly = fractional_kelly(sharp_prob, odds, fraction=adjusted_fraction)

    is_playable = True
    reasons = []

    if lm_signal["is_killed"]:
        is_playable = False
        reasons.append(f"Line moved against us (steam diff: {lm_signal['diff']})")

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
        "line_movement": lm_signal,
        "steam_detected": steam_detected,
    }
