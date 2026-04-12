from datetime import date, timedelta

from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_COMPLETED_STATUSES = frozenset({'Final', 'final', 'COMPLETED'})
_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# DB helpers — each owns exactly one table / concern
# ---------------------------------------------------------------------------

def _upsert_teams(conn, teams):
    for team in teams:
        conn.execute('''
            INSERT INTO teams (team_id, abbreviation, name, league, division)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                abbreviation=excluded.abbreviation,
                name=excluded.name
        ''', (
            team.get('id'),
            team.get('abbreviation', ''),
            team.get('full_name', ''),
            team.get('league', ''),
            team.get('division', ''),
        ))


def _upsert_player(conn, player_stat):
    """Upsert a player row and return the identifiers needed for log insertion.

    Returns (player_id, player_name, position, gid, game_date), or
    (None, ...) when the stat record carries no player data.
    """
    player = player_stat.get('player') or {}
    if not player:
        return None, None, None, None, None

    player_id = player.get('id')
    player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
    position = player.get('position', '')
    team_data = player_stat.get('team') or {}
    p_team_id = team_data.get('id') if isinstance(team_data, dict) else None

    conn.execute('''
        INSERT INTO players (player_id, name, team_id, position, bats, throws)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            name=excluded.name,
            team_id=excluded.team_id,
            position=excluded.position
    ''', (player_id, player_name, p_team_id, position,
          player.get('bats', ''), player.get('throws', '')))

    game_data = player_stat.get('game') or {}
    gid = game_data.get('id')
    game_date = game_data.get('date', '')

    return player_id, player_name, position, gid, game_date


def _insert_pitcher_log(conn, gid, player_id, game_date, player_stat, player_name):
    """Insert a pitcher game log row. Returns 1 on success, 0 on parse error."""
    ip = player_stat.get('innings_pitched') or player_stat.get('ip')
    try:
        conn.execute('''
            INSERT OR IGNORE INTO pitcher_game_logs
            (game_id, player_id, date, innings_pitched, hits_allowed,
             runs_allowed, earned_runs, walks, strikeouts,
             home_runs_allowed, pitches_thrown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            gid, player_id, game_date,
            float(ip) if ip else 0.0,
            int(player_stat.get('hits_allowed', 0) or 0),
            int(player_stat.get('runs_allowed', 0) or 0),
            int(player_stat.get('earned_runs', 0) or 0),
            int(player_stat.get('walks', 0) or player_stat.get('bb', 0) or 0),
            int(player_stat.get('strikeouts', 0) or player_stat.get('k', 0) or 0),
            int(player_stat.get('home_runs_allowed', 0) or 0),
            int(player_stat.get('pitches_thrown', 0) or player_stat.get('pitches', 0) or 0),
        ))
        return 1
    except (ValueError, TypeError) as e:
        logger.debug(f"Pitcher log parse error for {player_name}: {e}")
        return 0


def _insert_batter_log(conn, gid, player_id, game_date, player_stat, player_name):
    """Insert a batter game log row. Returns 1 on success, 0 on parse error."""
    try:
        hits = int(player_stat.get('hits', 0) or 0)
        doubles = int(player_stat.get('doubles', 0) or 0)
        triples = int(player_stat.get('triples', 0) or 0)
        home_runs = int(player_stat.get('home_runs', 0) or 0)
        singles = hits - doubles - triples - home_runs
        total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)

        conn.execute('''
            INSERT OR IGNORE INTO batter_game_logs
            (game_id, player_id, date, at_bats, hits, doubles, triples,
             home_runs, runs, rbis, walks, strikeouts, total_bases, plate_appearances)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            gid, player_id, game_date,
            int(player_stat.get('at_bats', 0) or player_stat.get('ab', 0) or 0),
            hits, doubles, triples, home_runs,
            int(player_stat.get('runs', 0) or player_stat.get('r', 0) or 0),
            int(player_stat.get('rbis', 0) or player_stat.get('rbi', 0) or 0),
            int(player_stat.get('walks', 0) or player_stat.get('bb', 0) or 0),
            int(player_stat.get('strikeouts', 0) or player_stat.get('k', 0) or 0),
            total_bases,
            int(player_stat.get('plate_appearances', 0) or player_stat.get('pa', 0) or 0),
        ))
        return 1
    except (ValueError, TypeError) as e:
        logger.debug(f"Batter log parse error for {player_name}: {e}")
        return 0


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def sync_stats():
    """Fetch pitcher and batter game logs from BallDontLie for teams in upcoming games."""
    logger.info("Executing pipeline: sync_stats")
    bdl_client = MLBStatsClient()

    # 1. Resolve which team IDs appear in upcoming games.
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT home_team_id, away_team_id FROM games WHERE status != 'COMPLETED'"
        ).fetchall()

    if not rows:
        logger.info("No upcoming games found for stats sync.")
        return

    teams_to_sync = {
        tid
        for row in rows
        for tid in (row['home_team_id'], row['away_team_id'])
        if tid
    }

    # 2. Sync the BDL teams table (cached 24 hr; rarely changes mid-season).
    all_teams = bdl_client.get_teams()
    with get_db_connection() as conn:
        _upsert_teams(conn, all_teams)
        conn.commit()
    logger.info(f"Synced {len(all_teams)} teams.")

    # 3. One date-range call for all completed games across every relevant team,
    #    instead of one per-team call for the full season.
    today = date.today()
    date_window = [(today - timedelta(days=d)).isoformat() for d in range(_LOOKBACK_DAYS)]
    all_recent_games = bdl_client.get_games(dates=date_window)

    recent_game_ids = {
        g['id']
        for g in all_recent_games
        if g.get('status') in _COMPLETED_STATUSES
        and (
            (g.get('home_team') or {}).get('id') in teams_to_sync
            or (g.get('away_team') or {}).get('id') in teams_to_sync
        )
    }
    logger.info(
        f"Found {len(recent_game_ids)} completed games in the last {_LOOKBACK_DAYS} days "
        f"for {len(teams_to_sync)} teams."
    )

    if not recent_game_ids:
        logger.info("No completed recent games to sync stats from.")
        return

    # 4. Fetch all player stats in one batched call (≤50 game IDs per HTTP request).
    all_stats = bdl_client.get_stats_batch(recent_game_ids)
    logger.info(f"Processing {len(all_stats)} player-game stat records.")

    # 5. Write everything in a single transaction.
    total_pitcher_logs = 0
    total_batter_logs = 0

    with get_db_connection() as conn:
        for player_stat in all_stats:
            player_id, player_name, position, gid, game_date = _upsert_player(conn, player_stat)
            if player_id is None:
                continue

            ip = player_stat.get('innings_pitched') or player_stat.get('ip')
            if ip is not None and (position == 'P' or str(ip) != '0'):
                total_pitcher_logs += _insert_pitcher_log(
                    conn, gid, player_id, game_date, player_stat, player_name
                )
            else:
                total_batter_logs += _insert_batter_log(
                    conn, gid, player_id, game_date, player_stat, player_name
                )
        conn.commit()

    logger.info(f"Synced stats: {total_pitcher_logs} pitcher logs, {total_batter_logs} batter logs.")
