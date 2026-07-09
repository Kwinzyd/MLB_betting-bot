"""Pitch-type arsenal matchup multipliers.

Turns BDL per-(player, pitch_type) season stats (in pitch_type_stats) into a
bounded multiplier on a projected mean, capturing how a specific pitcher's mix
plays against a specific batter (or lineup).

Design
------
- Applied as a POST-model adjustment (like park / umpire factors), not a GLM
  feature: the training set has no per-game opposing-pitcher identity, so a
  matchup feature would be constant in training and variable at serve.
- Each factor isolates the matchup DEVIATION from the player's own baseline:
  factor = clamp((arsenal_weighted_metric / natural_metric) ** sens, lo, hi).
  Centered at 1.0, so it never double-counts overall skill the model already
  prices — it only tilts for "this arsenal vs this profile."
- Fail-safe: any missing data, thin samples, or low arsenal coverage returns a
  neutral 1.0. A matchup can only nudge, never fabricate.

Batter markets  → batter's per-pitch xwoba weighted by the STARTER's usage.
                  >1 means the starter throws pitches this batter hits harder.
Pitcher K       → each opposing hitter's per-pitch whiff weighted by the
                  pitcher's usage, averaged over the lineup. >1 means the
                  arsenal targets pitches this lineup whiffs on more than usual.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.config import (
    PITCH_MATCHUP_SENS, PITCH_MATCHUP_MIN, PITCH_MATCHUP_MAX,
    PITCH_MATCHUP_MIN_PITCHES, PITCH_MATCHUP_MIN_COVERAGE,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Module-level default so tests can override via args; mirrors the config value.
_MIN_PITCHES = PITCH_MATCHUP_MIN_PITCHES


# ---------------------------------------------------------------------------
# Pure profile builders
# ---------------------------------------------------------------------------

def _pitcher_usage(rows: List[Dict], min_pitches: int) -> Dict[str, float]:
    """{pitch_type: usage_fraction} from a pitcher's rows, renormalized to sum
    to 1 over pitch types clearing the sample floor. Empty when no usage data."""
    raw: Dict[str, float] = {}
    for r in rows:
        pt = r.get("pitch_type")
        cnt = r.get("pitch_count") or 0
        usage = r.get("usage_pct")
        if not pt or cnt < min_pitches or usage in (None, ""):
            continue
        raw[pt] = float(usage)
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {pt: u / total for pt, u in raw.items()}


def _metric_by_pitch(rows: List[Dict], metric: str, min_pitches: int) -> Dict[str, Dict[str, float]]:
    """{pitch_type: {'val': metric, 'count': pitch_count}} for rows with a
    non-null metric and enough sample. `metric` is a column name (whiff_pct/xwoba)."""
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        pt = r.get("pitch_type")
        cnt = r.get("pitch_count") or 0
        val = r.get(metric)
        if not pt or cnt < min_pitches or val in (None, ""):
            continue
        out[pt] = {"val": float(val), "count": float(cnt)}
    return out


def _natural_rate(profile: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Pitch-count-weighted average of a per-pitch metric — the player's rate
    against a league-typical mix. None when the profile is empty."""
    num = sum(d["val"] * d["count"] for d in profile.values())
    den = sum(d["count"] for d in profile.values())
    return (num / den) if den > 0 else None


def _arsenal_weighted(usage: Dict[str, float], profile: Dict[str, Dict[str, float]]):
    """(weighted_metric, covered_usage_fraction) for a metric over the pitch
    types present in BOTH the usage mix and the opponent profile. Usage is
    renormalized across the covered pitch types so the weighted metric stays on
    the metric's own scale; covered fraction (of the original usage) gates trust."""
    covered = {pt: usage[pt] for pt in usage if pt in profile}
    covered_frac = sum(covered.values())
    if covered_frac <= 0:
        return None, 0.0
    weighted = sum((covered[pt] / covered_frac) * profile[pt]["val"] for pt in covered)
    return weighted, covered_frac


def _clamp_factor(ratio: Optional[float]) -> float:
    """Apply sensitivity and clamp a matchup/baseline ratio to a bounded factor.
    Neutral 1.0 on missing/degenerate input."""
    if ratio is None or ratio <= 0:
        return 1.0
    factor = ratio ** PITCH_MATCHUP_SENS
    return max(PITCH_MATCHUP_MIN, min(PITCH_MATCHUP_MAX, factor))


# ---------------------------------------------------------------------------
# Pure factor computations
# ---------------------------------------------------------------------------

def batter_arsenal_factor(batter_rows: List[Dict], pitcher_rows: List[Dict],
                          min_pitches: int = None, min_coverage: float = None) -> float:
    """Multiplier on a batter's projected mean vs a specific starter's arsenal.

    Ratio of the batter's starter-usage-weighted xwoba to their natural xwoba.
    Neutral 1.0 unless the arsenal is well covered by the batter's per-pitch data.
    """
    min_pitches = _MIN_PITCHES if min_pitches is None else min_pitches
    min_coverage = PITCH_MATCHUP_MIN_COVERAGE if min_coverage is None else min_coverage

    usage = _pitcher_usage(pitcher_rows, min_pitches)
    profile = _metric_by_pitch(batter_rows, "xwoba", min_pitches)
    if not usage or not profile:
        return 1.0
    weighted, covered = _arsenal_weighted(usage, profile)
    if covered < min_coverage:
        return 1.0
    natural = _natural_rate(profile)
    if not natural:
        return 1.0
    return _clamp_factor(weighted / natural)


def pitcher_k_factor(pitcher_rows: List[Dict], lineup_rows: List[List[Dict]],
                     min_pitches: int = None, min_coverage: float = None) -> float:
    """Multiplier on a pitcher's projected strikeouts vs the opposing lineup.

    For each hitter, the pitcher-usage-weighted whiff over their per-pitch whiff
    profile, divided by that hitter's natural whiff; averaged across hitters
    with sufficient coverage. Neutral 1.0 when too few hitters have data.
    """
    min_pitches = _MIN_PITCHES if min_pitches is None else min_pitches
    min_coverage = PITCH_MATCHUP_MIN_COVERAGE if min_coverage is None else min_coverage

    usage = _pitcher_usage(pitcher_rows, min_pitches)
    if not usage:
        return 1.0
    ratios = []
    for hitter_rows in lineup_rows:
        profile = _metric_by_pitch(hitter_rows, "whiff_pct", min_pitches)
        if not profile:
            continue
        weighted, covered = _arsenal_weighted(usage, profile)
        if covered < min_coverage:
            continue
        natural = _natural_rate(profile)
        if not natural:
            continue
        ratios.append(weighted / natural)
    if not ratios:
        return 1.0
    return _clamp_factor(sum(ratios) / len(ratios))


# ---------------------------------------------------------------------------
# DB loaders + inference entry points
# ---------------------------------------------------------------------------

def _load_rows(conn, player_id: int, role: str, season: int) -> List[Dict]:
    if not conn or not player_id:
        return []
    try:
        rows = conn.execute(
            """SELECT pitch_type, pitch_count, usage_pct, whiff_pct, contact_pct,
                      xwoba, pa_count, strikeout_count
               FROM pitch_type_stats
               WHERE player_id = ? AND role = ? AND season = ?""",
            (player_id, role, season),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:  # noqa: BLE001 — matchup is optional; never break projection
        logger.debug("pitch_type_stats load failed (%s/%s): %s", player_id, role, e)
        return []


def batter_arsenal_factor_db(conn, batter_id: int, pitcher_id: int, season: int) -> float:
    """Load rows and compute the batter-vs-starter factor. Neutral on any gap."""
    if not batter_id or not pitcher_id:
        return 1.0
    batter_rows = _load_rows(conn, batter_id, "hitter", season)
    pitcher_rows = _load_rows(conn, pitcher_id, "pitcher", season)
    if not batter_rows or not pitcher_rows:
        return 1.0
    return batter_arsenal_factor(batter_rows, pitcher_rows)


def pitcher_k_factor_db(conn, pitcher_id: int, opp_batter_ids: List[int], season: int) -> float:
    """Load rows and compute the pitcher-vs-lineup K factor. Neutral on any gap."""
    if not pitcher_id or not opp_batter_ids:
        return 1.0
    pitcher_rows = _load_rows(conn, pitcher_id, "pitcher", season)
    if not pitcher_rows:
        return 1.0
    lineup_rows = [_load_rows(conn, bid, "hitter", season) for bid in opp_batter_ids]
    lineup_rows = [r for r in lineup_rows if r]
    if not lineup_rows:
        return 1.0
    return pitcher_k_factor(pitcher_rows, lineup_rows)
