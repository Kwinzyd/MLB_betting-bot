from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.utils.time_utils import (
    get_eastern_local_date, get_utc_now_iso, eastern_date_utc_window,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _stamp_confirmed_if_both_teams(conn, game_id: str, home_team: str, away_team: str):
    """
    Set games.lineups_confirmed_at to now (UTC ISO) the first time both
    teams in this game have at least one row in daily_lineups. Idempotent —
    never overwrites an existing timestamp.
    """
    teams_with_lineups = {
        r['team'] for r in conn.execute(
            "SELECT DISTINCT team FROM daily_lineups WHERE game_id = ?",
            (game_id,)
        ).fetchall()
    }
    home_present = any(home_team and home_team.lower() in (t or '').lower() for t in teams_with_lineups)
    away_present = any(away_team and away_team.lower() in (t or '').lower() for t in teams_with_lineups)
    if not (home_present and away_present):
        return
    conn.execute(
        "UPDATE games SET lineups_confirmed_at = ? "
        "WHERE game_id = ? AND lineups_confirmed_at IS NULL",
        (get_utc_now_iso(), game_id),
    )


async def sync_lineups():
    """
    Fetch today's starting lineups and probable pitchers from BDL /lineups endpoint.

    Stores:
    - Batting order positions in daily_lineups (drives projected PAs)
    - Probable pitchers in probable_pitchers (drives platoon adjustments)

    Lineups typically appear 1-2 hours before first pitch.
    """
    logger.info("Executing pipeline: sync_lineups")
    bdl_client = MLBStatsClient()
    today_date = get_eastern_local_date()
    today = str(today_date)

    # Get today's games by the Eastern-day UTC window. Filtering on the UTC
    # `date` string with the Eastern date dropped late games (UTC rolls to the
    # next day), so their lineups never confirmed pregame and the scan only
    # fired in-play. The window captures every game on the Eastern slate.
    start_utc, end_utc = eastern_date_utc_window(today_date)
    with get_db_connection() as conn:
        games = [dict(r) for r in conn.execute(
            "SELECT game_id, bdl_game_id, home_team, away_team FROM games "
            "WHERE status != 'COMPLETED' AND date >= ? AND date < ?",
            (start_utc, end_utc)
        ).fetchall()]

    if not games:
        logger.info("No games today for lineup sync.")
        return

    lineup_count = 0
    pitcher_count = 0

    failed_games = 0
    for game in games:
        bdl_game_id = game['bdl_game_id']
        if not bdl_game_id:
            continue

        # Per-game isolation: a transient BDL 5xx on one game must not abort
        # the rest of the pipeline (scan_props/send_alerts/SGP run after this).
        try:
            lineup_data = await bdl_client.get_lineups(bdl_game_id)
        except Exception as e:
            failed_games += 1
            logger.warning(
                f"Lineup fetch failed for game {game['game_id']} "
                f"(bdl_game_id={bdl_game_id}): {e}. Skipping; pipeline continues."
            )
            continue
        if not lineup_data:
            logger.debug(f"No lineup data yet for game {game['game_id']}")
            continue

        with get_db_connection() as conn:
            for entry in lineup_data:
                player = entry.get('player', {})
                team_data = entry.get('team', {})
                if not player or not team_data:
                    continue

                player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                player_id = player.get('id')
                team_name = (
                    team_data.get('display_name')
                    or team_data.get('name')
                    or ''
                ) if isinstance(team_data, dict) else ''
                bats_throws = player.get('bats_throws', '') or ''
                throws = bats_throws.split('/', 1)[1].strip() if '/' in bats_throws else ''
                is_probable_pitcher = entry.get('is_probable_pitcher', False)
                batting_order = entry.get('batting_order') or entry.get('lineup_position')

                if is_probable_pitcher:
                    conn.execute('''
                        INSERT INTO probable_pitchers (game_id, team, player_name, player_id, throws, date)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(game_id, team) DO UPDATE SET
                            player_name=excluded.player_name,
                            player_id=excluded.player_id,
                            throws=excluded.throws,
                            date=excluded.date
                    ''', (game['game_id'], team_name, player_name, player_id, throws, today))
                    pitcher_count += 1
                    logger.info(f"Probable pitcher: {player_name} ({throws}) for {team_name}")

                if batting_order:
                    conn.execute('''
                        INSERT INTO daily_lineups (game_id, team, player_name, player_id, lineup_position, date)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(game_id, player_name) DO UPDATE SET
                            lineup_position=excluded.lineup_position,
                            date=excluded.date
                    ''', (game['game_id'], team_name, player_name, player_id, int(batting_order), today))
                    lineup_count += 1

            _stamp_confirmed_if_both_teams(
                conn, game['game_id'], game['home_team'], game['away_team']
            )
            conn.commit()

    logger.info(
        f"Synced {lineup_count} lineup entries and {pitcher_count} probable "
        f"pitchers for {today}. Failed games: {failed_games}."
    )
