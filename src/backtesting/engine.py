"""
Backtesting engine for MLB prop projection models.

Replays historical prop snapshots against the projection model using only
pre-game data (strict no look-ahead guarantee), then scores results against
actual box-score outcomes.

Key design constraints:
  - All stat queries use `date < game_date` — never equal, never after.
  - game_id in prop_snapshots/games is the Odds API string. Game logs use the
    integer bdl_game_id. _get_actual_result() always resolves via games table.
  - No historical weather is stored, so static park factors are used throughout.
    Live projections will differ slightly from backtested ones for outdoor parks.

Truth source and look-ahead:
  - Edges are scored ONLY against a sharp two-way devigged price at the line
    (prop_snapshots.devigged_over/under, populated for sharp books). Snapshots
    without a devig are kept for calibration/Brier but never become bets — we
    never substitute a 0.5 "true probability", which would manufacture an edge
    against every soft-book line.
  - By default (`use_trained_models=False`) the engine runs the weighted-average
    projection with calibration OFF, because the trained GLM and the calibration
    params are fit on data that postdates the snapshots being replayed. Pass
    `use_trained_models=True` only to inspect the *current* champion's behavior,
    accepting that it is in-sample.
  - Residual, un-removed look-ahead (documented, not yet fixed): dispersion
    params (dispersion_params) and team offensive stats (team_stats) are global,
    slowly-varying values read as of "now", not as of game_date. They shift
    results slightly but do not manufacture edges the way the 0.5 substitution did.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

from src.config import KELLY_FRACTION, LEAGUE_AVG_K_RATE, LEAGUE_AVG_RUNS_PER_GAME
from src.data.db import get_db_connection
from src.models.projections import ProjectionModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# A models dir guaranteed empty so ProjectionModel loads no GLM — used by the
# default out-of-sample backtest path to force the weighted-average algorithm.
_NO_MODELS_DIR = os.path.join(os.path.dirname(__file__), "__no_models__")

# Markets the engine knows how to project and resolve.
SUPPORTED_MARKETS = frozenset({
    'pitcher_strikeouts',
    'pitcher_earned_runs',
    'batter_hits',
    'batter_total_bases',
    'batter_home_runs',
})


@dataclass
class BacktestRecord:
    """One resolved prop: model inputs, model outputs, and actual result."""
    # --- Identity ---
    game_id: str          # Odds API game ID
    bdl_game_id: int      # BallDontLie game ID (used to join game logs)
    game_date: str
    player_name: str
    player_id: int
    market: str
    bookmaker: str

    # --- Line & market odds (opening snapshot) ---
    line: float
    over_odds: float
    under_odds: float
    devigged_over: float   # book's true probability of over
    devigged_under: float  # book's true probability of under

    # --- Model output ---
    projected_mean: float
    model_prob_over: float
    model_prob_under: float
    sample_size: int       # number of historical game logs available

    # --- Derived edges ---
    edge_over: float       # model_prob_over  - devigged_over
    edge_under: float      # model_prob_under - devigged_under
    best_side: str         # 'over' or 'under'
    best_edge: float       # max(edge_over, edge_under)

    # --- Actual outcome (None when box score is missing) ---
    actual_value: Optional[float]
    actual_over: Optional[bool]   # actual_value > line

    # --- P&L (None when no bet or no result) ---
    flat_pnl: Optional[float]      # +/- $1 per bet
    kelly_fraction_bet: float      # fraction of bankroll sized by Kelly
    kelly_pnl: Optional[float]     # kelly_fraction_bet * outcome


class BacktestEngine:
    """
    Runs a backtest over a date range.

    Usage:
        engine = BacktestEngine(start_date='2025-04-01', end_date='2025-09-30')
        records = engine.run()
        summary = compute_summary(records, ...)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        min_edge: float = 0.05,
        kelly_fraction: float = KELLY_FRACTION,
        markets: Optional[List[str]] = None,
        use_trained_models: bool = False,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.markets = [m for m in (markets or SUPPORTED_MARKETS) if m in SUPPORTED_MARKETS]
        self.use_trained_models = use_trained_models
        if use_trained_models:
            # In-sample: the current champion GLM + active calibration params.
            self._proj = ProjectionModel()
        else:
            # Out-of-sample-safe: weighted-average algorithm, calibration off.
            self._proj = ProjectionModel(
                models_dir=_NO_MODELS_DIR, apply_calibration=False
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> List[BacktestRecord]:
        """
        Fetch all prop snapshots in the date window, project each one using
        only pre-game data, and return a BacktestRecord for every prop that
        had enough historical data to project.
        """
        snapshots = self._load_snapshots()
        mode = "TRAINED (in-sample)" if self.use_trained_models else "weighted-avg (calibration off)"
        logger.info(
            f"Backtesting {len(snapshots)} opening-line snapshots "
            f"({self.start_date} → {self.end_date}); model={mode}."
        )
        if self.use_trained_models:
            logger.warning(
                "Backtest is running with TRAINED models: GLM + calibration are "
                "fit on data inside this window — results are in-sample, not predictive."
            )

        records: List[BacktestRecord] = []
        with get_db_connection() as conn:
            for snap in snapshots:
                record = self._process_snapshot(conn, snap)
                if record is not None:
                    records.append(record)

        n_with_result = sum(1 for r in records if r.actual_over is not None)
        n_bets = sum(1 for r in records if r.flat_pnl is not None)
        logger.info(
            f"Backtest complete: {len(records)} projections built, "
            f"{n_with_result} with known results, {n_bets} bets at "
            f"≥{self.min_edge*100:.0f}% edge."
        )
        return records

    # ------------------------------------------------------------------
    # Snapshot loading
    # ------------------------------------------------------------------

    def _load_snapshots(self) -> List[dict]:
        """
        Return one row per (game_id, player_name, market, bookmaker) using
        the earliest timestamp — this simulates betting on the opening line.
        """
        market_placeholders = ",".join("?" * len(self.markets))
        params = [self.start_date, self.end_date] + self.markets

        query = f"""
            SELECT
                ps.game_id, ps.player_name, ps.market, ps.bookmaker,
                ps.line, ps.over_odds, ps.under_odds,
                ps.devigged_over, ps.devigged_under,
                g.date        AS game_date,
                g.bdl_game_id AS bdl_game_id,
                g.venue, g.home_team, g.away_team
            FROM prop_snapshots ps
            JOIN games g ON ps.game_id = g.game_id
            WHERE g.date BETWEEN ? AND ?
              AND ps.market IN ({market_placeholders})
              AND ps.timestamp = (
                  SELECT MIN(ps2.timestamp)
                  FROM prop_snapshots ps2
                  WHERE ps2.game_id    = ps.game_id
                    AND ps2.player_name = ps.player_name
                    AND ps2.market      = ps.market
                    AND ps2.bookmaker   = ps.bookmaker
              )
            ORDER BY g.date, ps.player_name, ps.market
        """
        with get_db_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Per-snapshot processing
    # ------------------------------------------------------------------

    def _process_snapshot(self, conn, snap: dict) -> Optional[BacktestRecord]:
        player = self._lookup_player(conn, snap['player_name'])
        if not player:
            return None

        player_id = player['player_id']
        game_date = snap['game_date']

        # Build projection using ONLY pre-game data
        projection = self._build_historical_projection(
            conn, player, snap['market'], snap['line'],
            snap['game_id'], game_date, snap['venue'],
            snap['home_team'], snap['away_team'],
        )
        if not projection:
            return None

        model_prob_over = projection['prob_over']
        model_prob_under = projection['prob_under']

        # Edge requires a sharp two-way devigged truth at this line. Without it
        # there is no defensible "true probability" — substituting 0.5 would
        # manufacture an edge against every soft-book line. Such rows are still
        # kept for calibration/Brier (lean = higher model prob) but can never
        # become bets: NaN edges never clear the min_edge gate below.
        has_devig = snap['devigged_over'] is not None and snap['devigged_under'] is not None
        if has_devig:
            devigged_over = snap['devigged_over']
            devigged_under = snap['devigged_under']
            edge_over = model_prob_over - devigged_over
            edge_under = model_prob_under - devigged_under
            if edge_over >= edge_under:
                best_side, best_edge = 'over', edge_over
                bet_odds, bet_prob = snap['over_odds'], model_prob_over
            else:
                best_side, best_edge = 'under', edge_under
                bet_odds, bet_prob = snap['under_odds'], model_prob_under
        else:
            devigged_over = devigged_under = math.nan
            edge_over = edge_under = math.nan
            if model_prob_over >= model_prob_under:
                best_side, best_edge = 'over', math.nan
                bet_odds, bet_prob = snap['over_odds'], model_prob_over
            else:
                best_side, best_edge = 'under', math.nan
                bet_odds, bet_prob = snap['under_odds'], model_prob_under

        # Actual result — requires translating to bdl_game_id
        actual_value = self._get_actual_result(
            conn, snap['bdl_game_id'], player_id, snap['market']
        )
        actual_over = (actual_value > snap['line']) if actual_value is not None else None

        # P&L — only computed when the edge clears the threshold AND result is known
        flat_pnl = kelly_fraction_bet = kelly_pnl = None
        kelly_fraction_bet = 0.0

        if best_edge >= self.min_edge and actual_over is not None:
            won = (best_side == 'over') == actual_over
            b = bet_odds - 1.0          # net decimal odds (profit per $1 risked)
            q = 1.0 - bet_prob

            flat_pnl = (b if won else -1.0)

            full_kelly = (bet_prob * b - q) / b if b > 0 else 0.0
            kelly_fraction_bet = max(0.0, full_kelly * self.kelly_fraction)
            kelly_pnl = kelly_fraction_bet * (b if won else -1.0)

        return BacktestRecord(
            game_id=snap['game_id'],
            bdl_game_id=snap['bdl_game_id'],
            game_date=game_date,
            player_name=snap['player_name'],
            player_id=player_id,
            market=snap['market'],
            bookmaker=snap['bookmaker'],
            line=snap['line'],
            over_odds=snap['over_odds'],
            under_odds=snap['under_odds'],
            devigged_over=devigged_over,
            devigged_under=devigged_under,
            projected_mean=projection['projected_mean'],
            model_prob_over=model_prob_over,
            model_prob_under=model_prob_under,
            sample_size=projection['sample_size'],
            edge_over=edge_over,
            edge_under=edge_under,
            best_side=best_side,
            best_edge=best_edge,
            actual_value=actual_value,
            actual_over=actual_over,
            flat_pnl=flat_pnl,
            kelly_fraction_bet=kelly_fraction_bet,
            kelly_pnl=kelly_pnl,
        )

    # ------------------------------------------------------------------
    # Historical projection (no look-ahead)
    # ------------------------------------------------------------------

    def _build_historical_projection(
        self, conn, player: dict, market: str, line: float,
        game_id: str, game_date: str, venue: str,
        home_team: str, away_team: str,
    ) -> Optional[dict]:
        """
        Run the projection model using only game logs strictly before game_date.
        No weather is applied — we don't store historical conditions.
        """
        player_id = player['player_id']

        if market in ('pitcher_strikeouts', 'pitcher_earned_runs'):
            logs = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM pitcher_game_logs WHERE player_id = ? AND date < ?"
                    " ORDER BY date DESC",
                    (player_id, game_date),
                ).fetchall()
            ]

            # The opponent is the team the pitcher faces: away_team for home
            # starters, home_team for away starters.
            opp_team = self._resolve_opposing_team(conn, player, home_team, away_team)

            if market == 'pitcher_strikeouts':
                opp_k_rate = self._historical_opp_k_rate(conn, opp_team, game_date)
                return self._proj.project_pitcher_strikeouts(logs, opp_k_rate, venue, line)
            else:
                opp_runs_pg = self._historical_opp_runs_pg(conn, opp_team, game_date)
                return self._proj.project_pitcher_earned_runs(logs, opp_runs_pg, venue, line)

        if market in ('batter_hits', 'batter_total_bases', 'batter_home_runs'):
            logs = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM batter_game_logs WHERE player_id = ? AND date < ?"
                    " ORDER BY date DESC",
                    (player_id, game_date),
                ).fetchall()
            ]

            stat_type = {
                'batter_hits': 'hits',
                'batter_total_bases': 'total_bases',
                'batter_home_runs': 'home_runs',
            }[market]
            pitcher_hand = self._historical_pitcher_hand(
                conn, game_id, home_team, away_team, player
            )
            lineup_pos = self._lookup_lineup_position(conn, player['name'], game_id)

            # Approximate historical game total based on both teams' recent run rates
            home_runs_pg = self._historical_opp_runs_pg(conn, home_team, game_date)
            away_runs_pg = self._historical_opp_runs_pg(conn, away_team, game_date)
            game_total = home_runs_pg + away_runs_pg

            return self._proj.project_batter_stat(
                logs, stat_type, pitcher_hand, player['bats'] or '',
                venue, line, lineup_position=lineup_pos,
                game_total=game_total,
            )

        return None

    # ------------------------------------------------------------------
    # Actual result lookup
    # ------------------------------------------------------------------

    def _get_actual_result(
        self, conn, bdl_game_id: Optional[int], player_id: int, market: str
    ) -> Optional[float]:
        """Fetch the player's actual box-score value for this game."""
        if not bdl_game_id:
            return None

        if market == 'pitcher_strikeouts':
            row = conn.execute(
                "SELECT strikeouts FROM pitcher_game_logs WHERE game_id = ? AND player_id = ?",
                (bdl_game_id, player_id),
            ).fetchone()
            return float(row['strikeouts']) if row else None

        if market == 'pitcher_earned_runs':
            row = conn.execute(
                "SELECT earned_runs FROM pitcher_game_logs WHERE game_id = ? AND player_id = ?",
                (bdl_game_id, player_id),
            ).fetchone()
            return float(row['earned_runs']) if row else None

        stat_col = {
            'batter_hits': 'hits',
            'batter_total_bases': 'total_bases',
            'batter_home_runs': 'home_runs',
        }.get(market)
        if stat_col:
            row = conn.execute(
                f"SELECT {stat_col} FROM batter_game_logs WHERE game_id = ? AND player_id = ?",
                (bdl_game_id, player_id),
            ).fetchone()
            return float(row[stat_col]) if row else None

        return None

    # ------------------------------------------------------------------
    # Historical opponent stat helpers
    # ------------------------------------------------------------------

    def _historical_opp_k_rate(self, conn, team_name: str, before_date: str) -> float:
        """Opponent strikeout rate computed from batter logs before game_date."""
        team = conn.execute(
            "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
            (f"%{team_name}%",),
        ).fetchone()
        if not team:
            return LEAGUE_AVG_K_RATE

        row = conn.execute(
            """
            SELECT CAST(SUM(b.strikeouts) AS REAL) / NULLIF(SUM(b.plate_appearances), 0)
                AS k_rate
            FROM batter_game_logs b
            JOIN players p ON b.player_id = p.player_id
            WHERE p.team_id = ? AND b.date < ?
            """,
            (team['team_id'], before_date),
        ).fetchone()
        return (row['k_rate'] if row and row['k_rate'] else LEAGUE_AVG_K_RATE)

    def _historical_opp_runs_pg(self, conn, team_name: str, before_date: str) -> float:
        """Opponent runs per game computed from batter logs before game_date."""
        team = conn.execute(
            "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
            (f"%{team_name}%",),
        ).fetchone()
        if not team:
            return LEAGUE_AVG_RUNS_PER_GAME

        row = conn.execute(
            """
            SELECT CAST(SUM(b.runs) AS REAL) / NULLIF(COUNT(DISTINCT b.game_id), 0)
                AS runs_pg
            FROM batter_game_logs b
            JOIN players p ON b.player_id = p.player_id
            WHERE p.team_id = ? AND b.date < ?
            """,
            (team['team_id'], before_date),
        ).fetchone()
        return (row['runs_pg'] if row and row['runs_pg'] else LEAGUE_AVG_RUNS_PER_GAME)

    # ------------------------------------------------------------------
    # Pitcher hand + lineup position helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_opposing_team(conn, player: dict, home_team: str, away_team: str) -> str:
        """Name of the team the player faces, based on their team_id."""
        home = conn.execute(
            "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
            (f"%{home_team}%",),
        ).fetchone()
        if home and player.get('team_id') == home['team_id']:
            return away_team
        return home_team

    def _historical_pitcher_hand(
        self, conn, game_id: str, home_team: str, away_team: str, batter: dict
    ) -> str:
        """
        Determine opposing pitcher's throwing hand.
        Priority: probable_pitchers table → most-recent pitcher on opposing team.
        """
        home = conn.execute(
            "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
            (f"%{home_team}%",),
        ).fetchone()
        opp_name = (
            away_team if (home and batter['team_id'] == home['team_id']) else home_team
        )

        row = conn.execute(
            "SELECT throws FROM probable_pitchers"
            " WHERE game_id = ? AND team LIKE ? COLLATE NOCASE",
            (game_id, f"%{opp_name}%"),
        ).fetchone()
        if row and row['throws']:
            return row['throws']

        # Fallback: latest pitcher logged for that team
        opp = conn.execute(
            "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
            (f"%{opp_name}%",),
        ).fetchone()
        if not opp:
            return 'R'

        row = conn.execute(
            """
            SELECT p.throws FROM players p
            JOIN pitcher_game_logs pgl ON p.player_id = pgl.player_id
            WHERE p.team_id = ? AND p.position = 'P'
            ORDER BY pgl.date DESC LIMIT 1
            """,
            (opp['team_id'],),
        ).fetchone()
        return (row['throws'] if row and row['throws'] else 'R')

    def _lookup_lineup_position(self, conn, player_name: str, game_id: str) -> Optional[int]:
        row = conn.execute(
            "SELECT lineup_position FROM daily_lineups"
            " WHERE player_name = ? AND game_id = ? COLLATE NOCASE",
            (player_name, game_id),
        ).fetchone()
        return (int(row['lineup_position']) if row and row['lineup_position'] else None)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_player(conn, player_name: str) -> Optional[dict]:
        row = conn.execute(
            "SELECT * FROM players WHERE name = ? COLLATE NOCASE", (player_name,)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM players WHERE name LIKE ? COLLATE NOCASE",
                (f"%{player_name}%",),
            ).fetchone()
        return dict(row) if row else None
