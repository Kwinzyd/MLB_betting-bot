from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def sync_stats():
    """Fetch pitcher and batter game logs from BallDontLie for teams in upcoming games."""
    logger.info("Executing pipeline: sync_stats")
    bdl_client = MLBStatsClient()

    # 1. Get teams from upcoming games
    with get_db_connection() as conn:
        games = conn.execute(
            "SELECT DISTINCT home_team, away_team, bdl_game_id FROM games WHERE status != 'COMPLETED'"
        ).fetchall()

    if not games:
        logger.info("No upcoming games found for stats sync.")
        return

    # 2. Collect all BDL game IDs we need stats for
    bdl_game_ids = set()
    for game in games:
        if game['bdl_game_id']:
            bdl_game_ids.add(game['bdl_game_id'])

    # 3. Also fetch recent games for the season to build player history
    all_teams = bdl_client.get_teams()
    team_id_map = {}
    for team in all_teams:
        team_name = team.get('full_name', '')
        team_id = team.get('id')
        if team_name and team_id:
            team_id_map[team_name.lower()] = team_id

    # Sync teams into DB
    with get_db_connection() as conn:
        for team in all_teams:
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
        conn.commit()
    logger.info(f"Synced {len(all_teams)} teams.")

    # 4. For each team in upcoming games, fetch their recent games and box scores
    teams_to_sync = set()
    for game in games:
        for team_name in [game['home_team'], game['away_team']]:
            t_lower = team_name.lower()
            for key, tid in team_id_map.items():
                if t_lower in key or key in t_lower:
                    teams_to_sync.add(tid)
                    break

    total_pitcher_logs = 0
    total_batter_logs = 0

    for team_id in teams_to_sync:
        team_games = bdl_client.get_games(season=None)
        # Get game IDs for this team's recent games
        recent_game_ids = []
        for g in team_games:
            home = g.get('home_team', {})
            away = g.get('visitor_team', {}) or g.get('away_team', {})
            h_id = home.get('id') if isinstance(home, dict) else None
            a_id = away.get('id') if isinstance(away, dict) else None
            if team_id in (h_id, a_id) and g.get('status') in ('Final', 'final', 'COMPLETED'):
                recent_game_ids.append(g['id'])

        # Limit to last 20 games to avoid excessive API calls
        recent_game_ids = recent_game_ids[-20:]

        for gid in recent_game_ids:
            stats = bdl_client.get_game_stats(gid)
            if not stats:
                continue

            with get_db_connection() as conn:
                for player_stat in stats:
                    player = player_stat.get('player', {})
                    if not player:
                        continue

                    player_id = player.get('id')
                    player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                    position = player.get('position', '')
                    team_data = player_stat.get('team', {})
                    p_team_id = team_data.get('id') if isinstance(team_data, dict) else None

                    # Upsert player
                    conn.execute('''
                        INSERT INTO players (player_id, name, team_id, position, bats, throws)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(player_id) DO UPDATE SET
                            name=excluded.name,
                            team_id=excluded.team_id,
                            position=excluded.position
                    ''', (
                        player_id, player_name, p_team_id, position,
                        player.get('bats', ''), player.get('throws', ''),
                    ))

                    game_data = player_stat.get('game', {})
                    game_date = game_data.get('date', '') if isinstance(game_data, dict) else ''

                    # Determine if pitcher or batter based on position or stat availability
                    ip = player_stat.get('innings_pitched') or player_stat.get('ip')
                    if ip is not None and (position == 'P' or str(ip) != '0'):
                        try:
                            ip_val = float(ip) if ip else 0.0
                            conn.execute('''
                                INSERT OR IGNORE INTO pitcher_game_logs
                                (game_id, player_id, date, innings_pitched, hits_allowed,
                                 runs_allowed, earned_runs, walks, strikeouts,
                                 home_runs_allowed, pitches_thrown)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                gid, player_id, game_date, ip_val,
                                int(player_stat.get('hits_allowed', 0) or 0),
                                int(player_stat.get('runs_allowed', 0) or 0),
                                int(player_stat.get('earned_runs', 0) or 0),
                                int(player_stat.get('walks', 0) or player_stat.get('bb', 0) or 0),
                                int(player_stat.get('strikeouts', 0) or player_stat.get('k', 0) or 0),
                                int(player_stat.get('home_runs_allowed', 0) or 0),
                                int(player_stat.get('pitches_thrown', 0) or player_stat.get('pitches', 0) or 0),
                            ))
                            total_pitcher_logs += 1
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Pitcher log parse error for {player_name}: {e}")
                    else:
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
                                 home_runs, rbis, walks, strikeouts, total_bases, plate_appearances)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                gid, player_id, game_date,
                                int(player_stat.get('at_bats', 0) or player_stat.get('ab', 0) or 0),
                                hits, doubles, triples, home_runs,
                                int(player_stat.get('rbis', 0) or player_stat.get('rbi', 0) or 0),
                                int(player_stat.get('walks', 0) or player_stat.get('bb', 0) or 0),
                                int(player_stat.get('strikeouts', 0) or player_stat.get('k', 0) or 0),
                                total_bases,
                                int(player_stat.get('plate_appearances', 0) or player_stat.get('pa', 0) or 0),
                            ))
                            total_batter_logs += 1
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Batter log parse error for {player_name}: {e}")

                conn.commit()

    logger.info(f"Synced stats: {total_pitcher_logs} pitcher logs, {total_batter_logs} batter logs.")
