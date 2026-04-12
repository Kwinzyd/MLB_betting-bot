from datetime import datetime
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_team_stats():
    """
    Pre-calculates and stores aggregate team-level offensive stats.
    This is a performance optimization to avoid expensive queries during live odds scanning.
    This pipeline should be run after `sync_stats`.
    """
    logger.info("Executing pipeline: calculate_team_stats")

    with get_db_connection() as conn:
        # Get all teams
        teams = conn.execute("SELECT team_id, name FROM teams").fetchall()

        if not teams:
            logger.warning("No teams found in DB. Run sync_events or sync_stats first.")
            return

        stats_calculated = 0
        for team in teams:
            team_id = team['team_id']

            # Find all players currently on this team
            batters = conn.execute(
                "SELECT player_id FROM players WHERE team_id = ?", (team_id,)
            ).fetchall()

            if not batters:
                continue

            batter_ids = [b['player_id'] for b in batters]
            placeholders = ','.join('?' * len(batter_ids))

            # Aggregate stats from all game logs for these players
            totals = conn.execute(f'''
                SELECT
                    SUM(strikeouts) as total_k,
                    SUM(plate_appearances) as total_pa,
                    SUM(runs) as total_runs,
                    COUNT(DISTINCT game_id) as games
                FROM batter_game_logs WHERE player_id IN ({placeholders})
            ''', batter_ids).fetchone()

            if not totals or not totals['total_pa'] or not totals['games']:
                continue

            k_rate = totals['total_k'] / totals['total_pa']
            runs_per_game = totals['total_runs'] / totals['games']

            conn.execute('''
                INSERT INTO team_stats (team_id, k_rate, runs_per_game, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    k_rate=excluded.k_rate,
                    runs_per_game=excluded.runs_per_game,
                    last_updated=excluded.last_updated
            ''', (team_id, k_rate, runs_per_game, datetime.utcnow().isoformat()))
            stats_calculated += 1

        conn.commit()

    logger.info(f"Successfully calculated and stored stats for {stats_calculated} teams.")