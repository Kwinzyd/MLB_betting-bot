import sqlite3
from unittest.mock import patch

import pytest


@pytest.fixture
def memory_db():
    """In-memory DB with the alerts_sent + sgp_candidates tables exercised by sizing."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE alerts_sent (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT, market TEXT, line REAL, side TEXT,
            edge REAL, ev REAL, kelly_stake REAL, bookmaker TEXT,
            odds REAL, opening_odds REAL,
            model_prob_over REAL, model_prob_under REAL,
            game_id TEXT, timestamp TEXT
        );
        CREATE TABLE sgp_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, legs_json TEXT, joint_prob REAL,
            naive_parlay_odds REAL, fair_odds REAL, edge_vs_naive REAL,
            kelly_stake REAL, bookmakers TEXT, timestamp TEXT,
            UNIQUE(game_id, legs_json)
        );
    ''')
    conn.commit()
    return conn


def _candidate(game_id='g1', stake=50.0, pct=0.025):
    return {
        'game_id': game_id,
        'matchup': 'Red Sox @ Yankees',
        'legs': [
            {'player_name': 'A', 'market': 'pitcher_strikeouts', 'side': 'over',
             'line': 6.5, 'odds': 1.9, 'bookmaker': 'draftkings'},
            {'player_name': 'B', 'market': 'batter_hits', 'side': 'under',
             'line': 1.5, 'odds': 1.85, 'bookmaker': 'fanduel'},
            {'player_name': 'C', 'market': 'batter_total_bases', 'side': 'under',
             'line': 1.5, 'odds': 1.95, 'bookmaker': 'fanduel'},
        ],
        'joint_prob': 0.30,
        'naive_parlay_odds': 6.85,
        'fair_odds': 3.33,
        'edge_vs_naive': 1.05,
        'kelly_stake': stake,
        'kelly_pct': pct,
    }


class TestFormat:
    def test_message_includes_stake_line(self):
        from src.pipelines.find_sgp import _format_sgp_message
        msg = _format_sgp_message(_candidate(stake=42.50, pct=0.0106))
        assert "Stake: <b>$42.50</b>" in msg
        assert "1.06% bankroll" in msg

    def test_message_omits_stake_when_zero(self):
        from src.pipelines.find_sgp import _format_sgp_message
        msg = _format_sgp_message(_candidate(stake=0.0, pct=0.0))
        assert "Stake:" not in msg


class TestExposureCap:
    @patch('src.pipelines.find_sgp.MAX_BETS_PER_GAME', 3)
    @patch('src.pipelines.find_sgp.get_current_bankroll', return_value=1000.0)
    @patch('src.pipelines.find_sgp.get_db_connection')
    def test_no_singles_keeps_full_stake(self, mock_get_db, mock_bank, memory_db):
        # bankroll 1000 * 5% = $50 per-bet cap; * 3 games = $150 game cap
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        from src.pipelines.find_sgp import _apply_game_exposure_cap
        kept = _apply_game_exposure_cap([_candidate(stake=40.0)])
        assert len(kept) == 1
        assert kept[0]['kelly_stake'] == 40.0

    @patch('src.pipelines.find_sgp.MAX_BETS_PER_GAME', 3)
    @patch('src.pipelines.find_sgp.get_current_bankroll', return_value=1000.0)
    @patch('src.pipelines.find_sgp.get_db_connection')
    def test_partial_singles_trim_stake(self, mock_get_db, mock_bank, memory_db):
        # game_cap = $150; seed two singles totaling $100 used today.
        memory_db.execute(
            "INSERT INTO alerts_sent (game_id, kelly_stake, timestamp) "
            "VALUES ('g1', 50.0, datetime('now'))"
        )
        memory_db.execute(
            "INSERT INTO alerts_sent (game_id, kelly_stake, timestamp) "
            "VALUES ('g1', 50.0, datetime('now'))"
        )
        memory_db.commit()
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        from src.pipelines.find_sgp import _apply_game_exposure_cap
        kept = _apply_game_exposure_cap([_candidate(stake=80.0)])
        assert len(kept) == 1
        assert kept[0]['kelly_stake'] == 50.0  # remaining cap = 150 - 100

    @patch('src.pipelines.find_sgp.MAX_BETS_PER_GAME', 3)
    @patch('src.pipelines.find_sgp.get_current_bankroll', return_value=1000.0)
    @patch('src.pipelines.find_sgp.get_db_connection')
    def test_exhausted_cap_drops_candidate(self, mock_get_db, mock_bank, memory_db):
        # game_cap = $150; seed singles totaling $150.
        memory_db.execute(
            "INSERT INTO alerts_sent (game_id, kelly_stake, timestamp) "
            "VALUES ('g1', 150.0, datetime('now'))"
        )
        memory_db.commit()
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        from src.pipelines.find_sgp import _apply_game_exposure_cap
        kept = _apply_game_exposure_cap([_candidate(stake=80.0)])
        assert kept == []

    @patch('src.pipelines.find_sgp.MAX_BETS_PER_GAME', 3)
    @patch('src.pipelines.find_sgp.get_current_bankroll', return_value=1000.0)
    @patch('src.pipelines.find_sgp.get_db_connection')
    def test_multiple_sgps_share_game_budget(self, mock_get_db, mock_bank, memory_db):
        """Two SGPs on the same game shouldn't both consume the full headroom."""
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        from src.pipelines.find_sgp import _apply_game_exposure_cap
        kept = _apply_game_exposure_cap([
            _candidate(stake=100.0),
            _candidate(stake=100.0),
        ])
        # game_cap=150: first takes 100, second gets trimmed to 50.
        assert len(kept) == 2
        assert kept[0]['kelly_stake'] == 100.0
        assert kept[1]['kelly_stake'] == 50.0


class TestOpposingLegEnrichment:
    """Verify _find_opposing_under_legs returns leg dicts carrying the joint-PA
    inputs (team, lineup_position, mean_count, implied_team_total)."""

    def _seed_db(self):
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.executescript('''
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
                venue TEXT, status TEXT, date TEXT
            );
            CREATE TABLE projections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT, player_name TEXT, market TEXT,
                projected_mean REAL, prob_over REAL, prob_under REAL,
                context_json TEXT, timestamp TEXT,
                UNIQUE(game_id, player_name, market)
            );
            CREATE TABLE prop_snapshots (
                snapshot_id TEXT PRIMARY KEY, game_id TEXT, player_name TEXT,
                market TEXT, line REAL, over_odds REAL, under_odds REAL,
                bookmaker TEXT, timestamp TEXT,
                devigged_over REAL, devigged_under REAL
            );
            CREATE TABLE daily_lineups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT, team TEXT, player_name TEXT, player_id INTEGER,
                lineup_position INTEGER, date TEXT
            );
            CREATE TABLE game_totals_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL, total REAL NOT NULL,
                source TEXT NOT NULL, timestamp TEXT NOT NULL
            );
        ''')
        conn.execute("INSERT INTO games VALUES ('g1','Yankees','Red Sox','x','SCHEDULED','2024-05-15')")
        conn.execute("INSERT INTO daily_lineups (game_id, team, player_name, lineup_position, date) "
                     "VALUES ('g1', 'Yankees', 'Aaron Judge', 3, '2024-05-15')")
        conn.execute("INSERT INTO projections (game_id, player_name, market, projected_mean, "
                     "prob_over, prob_under, context_json, timestamp) "
                     "VALUES ('g1','Aaron Judge','batter_hits',1.25,0.45,0.55,'{}','2024-05-15T12:00:00')")
        # Sharp row carries the devigged truth; soft row carries the odds we'd
        # actually bet. find_sgp joins the two on (player, line) — scan_props
        # never writes devigged_* for soft books.
        conn.execute("INSERT INTO prop_snapshots VALUES "
                     "('s0','g1','Aaron Judge','batter_hits',1.5,2.05,1.85,"
                     "'pinnacle','2024-05-15T12:00:00',0.45,0.55)")
        conn.execute("INSERT INTO prop_snapshots VALUES "
                     "('s1','g1','Aaron Judge','batter_hits',1.5,2.1,1.8,"
                     "'draftkings','2024-05-15T12:00:00',NULL,NULL)")
        conn.execute("INSERT INTO game_totals_history (game_id, total, source, timestamp) "
                     "VALUES ('g1', 9.5, 'pinnacle', '2024-05-15T12:00:00')")
        conn.commit()
        return conn

    def test_leg_carries_joint_pa_fields(self):
        from src.pipelines.find_sgp import _find_opposing_under_legs
        conn = self._seed_db()
        legs = _find_opposing_under_legs(conn, 'g1', 'Yankees')
        assert len(legs) == 1
        leg = legs[0]
        assert leg['team'] == 'Yankees'
        assert leg['lineup_position'] == 3
        assert leg['mean_count'] == 1.25
        # game_total=9.5, symmetric split → 4.75
        assert leg['implied_team_total'] == pytest.approx(4.75)

    def test_implied_team_total_falls_back_when_no_history(self):
        from src.pipelines.find_sgp import _find_opposing_under_legs
        from src.config import LEAGUE_AVG_GAME_TOTAL
        conn = self._seed_db()
        conn.execute("DELETE FROM game_totals_history")
        conn.commit()
        legs = _find_opposing_under_legs(conn, 'g1', 'Yankees')
        assert legs[0]['implied_team_total'] == pytest.approx(LEAGUE_AVG_GAME_TOTAL / 2.0)


class TestSizingFraction:
    def test_sgp_fraction_is_tighter_than_singles(self):
        """SGP stake should be exactly half the single-bet stake at the same edge."""
        from src.config import KELLY_FRACTION, SGP_KELLY_FRACTION_MULT
        from src.models.kelly import fractional_kelly

        # Use explicit bankroll so we don't depend on the live ledger.
        single = fractional_kelly(0.55, 2.0, fraction=KELLY_FRACTION, bankroll=1000.0)
        sgp = fractional_kelly(
            0.55, 2.0,
            fraction=KELLY_FRACTION * SGP_KELLY_FRACTION_MULT,
            bankroll=1000.0,
        )
        assert single['recommended_stake'] > 0
        assert sgp['recommended_stake'] == pytest.approx(
            single['recommended_stake'] * SGP_KELLY_FRACTION_MULT, rel=1e-3
        )
