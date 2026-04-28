"""Camouflage stake rounding.

Soft-book risk teams flag accounts that bet exact fractional-Kelly amounts
like $18.42. Snap displayed and persisted stakes to a round increment
(default $5) so wagers look like rec-style sizing. Applied at the
persistence layer only — `fractional_kelly`'s raw output stays available
for logs, diagnostics, and backtests.
"""

from src.config import STAKE_ROUNDING_INCREMENT


def round_stake(stake: float, increment: float = None) -> float:
    """Round a positive stake to the nearest configured increment.

    - increment <= 0 disables rounding (passthrough at 2 decimals).
    - non-positive stake passes through (-1.0, 0.0 unchanged).
    - When raw stake is positive but rounds to 0 (sub-half-increment),
      snap up to one increment so a real edge isn't silently nuked.
    """
    inc = STAKE_ROUNDING_INCREMENT if increment is None else increment
    if inc <= 0 or stake <= 0:
        return round(stake, 2)
    rounded = round(stake / inc) * inc
    if rounded <= 0:
        rounded = inc
    return round(rounded, 2)
