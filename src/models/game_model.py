"""Team-level run model for game markets (moneyline / total / run line).

Game markets are MODEL-DRIVEN: BDL exposes only soft vendors with no sharp
anchor, so the edge comes from a team run projection compared to the devigged
book price. Each team's expected runs is a log5-style product of its offense, the
opposing pitching (innings-weighted starter + bullpen), and the park. Runs are
modeled as independent Poissons; the run margin is therefore Skellam-distributed,
which gives moneyline and run-line probabilities in closed form, and the run
total is the sum-Poisson.

Nothing here touches the prop money path. The LLM never feeds these numbers.
"""
from __future__ import annotations

import math
from typing import Dict

from scipy.stats import poisson, skellam

from src.config import LEAGUE_AVG_RUNS_PER_GAME, GAME_HOME_TIE_SPLIT


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def project_team_runs(off_rpg: float,
                      opp_starter_runs9: float,
                      opp_starter_ip: float,
                      opp_bullpen_era: float,
                      park_runs: float = 1.0,
                      league_avg: float = LEAGUE_AVG_RUNS_PER_GAME) -> float:
    """Expected runs (Poisson lambda) for a team in this matchup.

    log5-style: expected = league_avg * (team offense vs league)
                * (opposing pitching vs league) * park, where opposing pitching
    is the opposing STARTER for his projected innings and the BULLPEN for the
    rest. All factors are clamped so one noisy input can't blow up the estimate.

    off_rpg            team runs scored per game (team_stats.runs_per_game)
    opp_starter_runs9  opposing starter projected runs allowed per 9
    opp_starter_ip     opposing starter projected innings
    opp_bullpen_era    opposing bullpen ERA (recent)
    park_runs          park run factor (get_park_factor(...)['runs'])
    """
    if league_avg <= 0:
        league_avg = 4.5

    off_factor = _clamp(off_rpg / league_avg, 0.60, 1.60)
    starter_factor = _clamp((opp_starter_runs9 or league_avg) / league_avg, 0.50, 1.80)
    bp_factor = _clamp((opp_bullpen_era or league_avg) / league_avg, 0.60, 1.60)

    # Innings split: starter covers his projected IP out of 9, bullpen the rest.
    frac = _clamp((opp_starter_ip or 5.5) / 9.0, 0.10, 0.95)
    pitch_factor = frac * starter_factor + (1.0 - frac) * bp_factor

    park = _clamp(park_runs or 1.0, 0.80, 1.30)
    lam = league_avg * off_factor * pitch_factor * park
    return _clamp(lam, 1.5, 9.0)


def moneyline_probabilities(lam_home: float, lam_away: float,
                            home_tie_split: float = GAME_HOME_TIE_SPLIT) -> Dict[str, float]:
    """P(home win), P(away win). Ties (end of regulation) go to extra innings;
    that mass is split to the home team by `home_tie_split` (~0.52)."""
    p_home_reg = 1.0 - float(skellam.cdf(0, lam_home, lam_away))   # P(D >= 1)
    p_away_reg = float(skellam.cdf(-1, lam_home, lam_away))        # P(D <= -1)
    p_tie = float(skellam.pmf(0, lam_home, lam_away))              # P(D == 0)
    p_home = p_home_reg + home_tie_split * p_tie
    p_away = p_away_reg + (1.0 - home_tie_split) * p_tie
    total = p_home + p_away
    if total <= 0:
        return {"home": 0.5, "away": 0.5}
    return {"home": p_home / total, "away": p_away / total}


def total_probabilities(lam_home: float, lam_away: float, line: float) -> Dict[str, float]:
    """P(over), P(under), P(push) for the combined run total.

    Half-point lines have no push. Integer lines push on the exact total; the
    returned over/under are then push-conditional (matching how a book grades a
    push as a refund), so they sum to 1.
    """
    lam_sum = lam_home + lam_away
    floor_line = math.floor(line)
    p_over = 1.0 - float(poisson.cdf(floor_line, lam_sum))
    p_under_incl = float(poisson.cdf(floor_line, lam_sum))
    if float(line).is_integer():
        push = float(poisson.pmf(int(line), lam_sum))
        p_under = max(0.0, p_under_incl - push)
        denom = p_over + p_under
        if push > 0 and denom > 0:
            return {"over": p_over / denom, "under": p_under / denom, "push": push}
        return {"over": p_over, "under": p_under, "push": push}
    return {"over": p_over, "under": p_under_incl, "push": 0.0}


def run_line_cover_prob(lam_home: float, lam_away: float, side: str, signed_line: float) -> float:
    """Probability the given side covers a run line.

    `signed_line` is that side's line: a favorite laying the runs is negative
    (e.g. home -1.5), a dog taking them is positive (e.g. away +1.5). The home
    margin D = home - away is Skellam-distributed.

    home covers when D + signed_line > 0  -> D >  -signed_line
    away covers when -D + signed_line > 0 -> D <   signed_line
    Standard ±1.5 lines never push; integer lines treat the exact line as a loss
    (no push handling) — acceptable for the ±1.5 run line v1 supports.
    """
    if side == "home":
        thresh = -signed_line                 # D > thresh
        k = math.floor(thresh) + 1            # smallest integer D that covers
        return 1.0 - float(skellam.cdf(k - 1, lam_home, lam_away))
    thresh = signed_line                      # D < thresh
    k = math.ceil(thresh) - 1                 # largest integer D that covers
    return float(skellam.cdf(k, lam_home, lam_away))


def game_market_probabilities(lam_home: float, lam_away: float, total_line: float,
                              run_line: float = 1.5) -> Dict[str, Dict[str, float]]:
    """Bundle all game-market model probabilities for a matchup.

    run_line is the magnitude (1.5); the favorite (higher lambda) lays -run_line,
    the dog takes +run_line. Returns nested dicts keyed by market.
    """
    ml = moneyline_probabilities(lam_home, lam_away)
    total = total_probabilities(lam_home, lam_away, total_line)
    home_fav = lam_home >= lam_away
    home_line = -run_line if home_fav else run_line
    away_line = run_line if home_fav else -run_line
    rl = {
        "home": run_line_cover_prob(lam_home, lam_away, "home", home_line),
        "away": run_line_cover_prob(lam_home, lam_away, "away", away_line),
        "home_line": home_line,
        "away_line": away_line,
    }
    return {"moneyline": ml, "total": total, "run_line": rl}
