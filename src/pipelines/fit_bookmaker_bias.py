"""Fit per-bookmaker systematic pricing bias from the last 90 days of prop_snapshots.

Definition
----------
bias(bookmaker, market, side) = mean over matched snapshots of:
    sharp_devigged_side  -  soft_devigged_side

Positive bias → the soft book prices that side CHEAPER than sharp (in devigged
probability units). scan_props uses this to apply a small edge boost (+0.5%) when
bias > BOOKMAKER_BIAS_THRESHOLD.

Matching strategy
-----------------
For each (game_id, player_name, market, line, timestamp-bucket) we find one sharp
and one soft snapshot, then take the difference in their devigged probs.
Timestamp buckets are 30-minute windows so sharp/soft snapshots collected in the
same scan cycle are treated as contemporaneous.

Requirements
------------
- At least MIN_OBS = 50 matched pairs per (bookmaker, market, side).
  Below that threshold the entry is written as NULL so scan_props ignores it.
- Runs Saturday 05:00 via scheduler.

Schema
------
    bookmaker_bias (bookmaker, market, side, avg_bias, n_observations, last_computed)
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

from src.data.db import get_db_connection
from src.config import SHARP_BOOKMAKERS
from src.utils.logging_utils import get_logger
from src.utils.time_utils import get_utc_now_iso

logger = get_logger(__name__)

MIN_OBS = 50
_BUCKET_MINUTES = 30  # snap timestamp rounding window


def _bucket(ts: str) -> str:
    """Round an ISO timestamp down to the nearest 30-minute bucket."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ts[:13]  # fall back to hour bucket
    floored = dt - timedelta(minutes=dt.minute % _BUCKET_MINUTES,
                              seconds=dt.second,
                              microseconds=dt.microsecond)
    return floored.isoformat()


def fit_bookmaker_bias() -> Dict[str, int]:
    """Compute and upsert bookmaker bias rows. Returns {bookmaker: rows_updated}."""
    ninety_days_ago = (
        datetime.now(timezone.utc) - timedelta(days=90)
    ).isoformat()

    sharp_set = set(SHARP_BOOKMAKERS)

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT game_id, player_name, market, line,
                   bookmaker, timestamp,
                   devigged_over, devigged_under
            FROM prop_snapshots
            WHERE timestamp >= ?
              AND devigged_over IS NOT NULL
              AND devigged_under IS NOT NULL
            ORDER BY game_id, player_name, market, line, timestamp
            """,
            (ninety_days_ago,),
        ).fetchall()

    if not rows:
        logger.warning("fit_bookmaker_bias: no prop_snapshots with devigged prices in last 90 days")
        return {}

    # Group by (game_id, player_name, market, line, bucket) → {bookmaker: (dev_over, dev_under)}
    buckets: dict = defaultdict(dict)
    for r in rows:
        key = (r["game_id"], r["player_name"], r["market"], r["line"], _bucket(r["timestamp"]))
        buckets[key][r["bookmaker"]] = (r["devigged_over"], r["devigged_under"])

    # Collect bias observations: (bookmaker, market, side) → [bias_value, ...]
    observations: dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    for key, book_map in buckets.items():
        _, _, market, _, _ = key

        # Find the best sharp quote in this bucket
        sharp_probs = None
        for sb in SHARP_BOOKMAKERS:
            if sb in book_map:
                sharp_probs = book_map[sb]
                break
        if sharp_probs is None:
            continue

        sharp_over, sharp_under = sharp_probs

        # Compare each soft book against the sharp reference
        for book, (soft_over, soft_under) in book_map.items():
            if book in sharp_set:
                continue
            if soft_over is None or soft_under is None:
                continue
            # bias = sharp_devigged - soft_devigged
            # Positive → soft book is cheaper (better value) for bettors on that side
            observations[(book, market, "over")].append(float(sharp_over) - float(soft_over))
            observations[(book, market, "under")].append(float(sharp_under) - float(soft_under))

    if not observations:
        logger.warning("fit_bookmaker_bias: no matched sharp/soft pairs found")
        return {}

    now = get_utc_now_iso()
    results: Dict[str, int] = defaultdict(int)

    with get_db_connection() as conn:
        for (book, market, side), diffs in observations.items():
            n = len(diffs)
            avg_bias = sum(diffs) / n if n >= MIN_OBS else None
            conn.execute(
                """
                INSERT INTO bookmaker_bias
                    (bookmaker, market, side, avg_bias, n_observations, last_computed)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(bookmaker, market, side) DO UPDATE SET
                    avg_bias       = excluded.avg_bias,
                    n_observations = excluded.n_observations,
                    last_computed  = excluded.last_computed
                """,
                (book, market, side, avg_bias, n, now),
            )
            if avg_bias is not None:
                results[book] += 1
        conn.commit()

    total = sum(results.values())
    logger.info(
        "fit_bookmaker_bias: upserted %d bias rows across %d bookmakers",
        total, len(results),
    )
    return dict(results)
