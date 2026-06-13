from datetime import date, datetime, timedelta, timezone

from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger
from src.utils.validators import parse_baseball_ip

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

    Handles BDL MLB API response structure:
      - player.full_name (not first_name + last_name)
      - player.bats_throws = "Right/Right" (split on '/')
      - game_id is top-level (not nested under 'game')
      - team is team_name string (not a team dict with id)

    Returns (player_id, player_name, position, gid, game_date), or
    (None, ...) when the stat record carries no player data.
    """
    player = player_stat.get('player') or {}
    if not player:
        return None, None, None, None, None

    player_id = player.get('id')
    # BDL MLB: full_name is provided directly
    player_name = (
        player.get('full_name')
        or f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
    )
    position = player.get('position', '')

    # bats_throws = "Right/Right" or "Left/Left" etc.
    bats_throws = player.get('bats_throws', '') or ''
    parts = bats_throws.split('/')
    bats = parts[0].strip() if parts else ''
    throws = parts[1].strip() if len(parts) > 1 else ''

    # team_id: BDL MLB has team_name string not a team dict at the stat level
    team_data = player_stat.get('team') or {}
    p_team_id = team_data.get('id') if isinstance(team_data, dict) else None

    conn.execute('''
        INSERT INTO players (player_id, name, team_id, position, bats, throws)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            name=excluded.name,
            team_id=COALESCE(excluded.team_id, players.team_id),
            position=excluded.position,
            bats=COALESCE(excluded.bats, players.bats),
            throws=COALESCE(excluded.throws, players.throws)
    ''', (player_id, player_name, p_team_id, position, bats, throws))

    # BDL MLB: game_id is a top-level field (int), not nested under 'game'
    gid = player_stat.get('game_id')
    if gid is None:
        game_data = player_stat.get('game') or {}
        gid = game_data.get('id')
    # The stat payload's date fields are unreliable (observed empty across the
    # entire feed) — callers override with the /games payload date when the
    # value comes back blank. Normalize to YYYY-MM-DD either way.
    game_date = player_stat.get('game_date') or (player_stat.get('game') or {}).get('date', '')
    game_date = str(game_date or '')[:10]

    return player_id, player_name, position, gid, game_date


def _insert_pitcher_log(conn, gid, player_id, game_date, player_stat, player_name):
    """Insert a pitcher game log row. Returns 1 on success, 0 on parse error.

    Handles both BDL MLB API field names (er, p_hits, p_runs, p_bb, p_k,
    p_hr, pitch_count) and legacy/fallback names.
    """
    ip = player_stat.get('ip') or player_stat.get('innings_pitched')
    try:
        conn.execute('''
            INSERT OR IGNORE INTO pitcher_game_logs
            (game_id, player_id, date, innings_pitched, hits_allowed,
             runs_allowed, earned_runs, walks, strikeouts,
             home_runs_allowed, pitches_thrown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            gid, player_id, game_date,
            # Baseball notation: "5.2" = 5 innings + 2 OUTS = 5.667 innings.
            parse_baseball_ip(ip),
            # BDL MLB uses p_hits; fallback to hits_allowed
            int(player_stat.get('p_hits') or player_stat.get('hits_allowed') or 0),
            # BDL MLB uses p_runs; fallback to runs_allowed
            int(player_stat.get('p_runs') or player_stat.get('runs_allowed') or 0),
            # BDL MLB uses er; fallback to earned_runs
            int(player_stat.get('er') or player_stat.get('earned_runs') or 0),
            # BDL MLB uses p_bb; fallback chains
            int(player_stat.get('p_bb') or player_stat.get('walks') or player_stat.get('bb') or 0),
            # BDL MLB uses p_k; fallback chains
            int(player_stat.get('p_k') or player_stat.get('strikeouts') or player_stat.get('k') or 0),
            # BDL MLB uses p_hr; fallback
            int(player_stat.get('p_hr') or player_stat.get('home_runs_allowed') or 0),
            # BDL MLB uses pitch_count; fallback chains
            int(player_stat.get('pitch_count') or player_stat.get('pitches_thrown') or player_stat.get('pitches') or 0),
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
        if hits < doubles + triples + home_runs:
            logger.warning(
                "Inconsistent hit breakdown for %s game %s — clamping singles to 0",
                player_name, player_stat.get('game_id', '?'),
            )
        singles = max(0, hits - doubles - triples - home_runs)
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

async def sync_stats():
    """Fetch pitcher and batter game logs from BallDontLie for teams in upcoming games."""
    logger.info("Executing pipeline: sync_stats")
    bdl_client = MLBStatsClient()

    # 1. Resolve which team IDs appear in upcoming games. Even when there are
    #    none, keep going: the completed-game status sweep below is what lets
    #    settle_results find finished games, and it must run regardless.
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT home_team_id, away_team_id FROM games WHERE status != 'COMPLETED'"
        ).fetchall()

    teams_to_sync = {
        tid
        for row in rows
        for tid in (row['home_team_id'], row['away_team_id'])
        if tid
    }

    # 2. Sync the BDL teams table (cached 24 hr; rarely changes mid-season).
    all_teams = await bdl_client.get_teams()
    with get_db_connection() as conn:
        _upsert_teams(conn, all_teams)
        conn.commit()
    logger.info(f"Synced {len(all_teams)} teams.")

    # 3. One date-range call for all completed games across every relevant team,
    #    instead of one per-team call for the full season.
    today = date.today()
    date_window = [(today - timedelta(days=d)).isoformat() for d in range(_LOOKBACK_DAYS)]
    all_recent_games = await bdl_client.get_games(dates=date_window)

    # Mark finished games COMPLETED. This is the only live-pipeline status
    # transition to COMPLETED, and settle_results depends on it — without it
    # bets never settle and the bankroll ledger never moves.
    completed_bdl_ids = [
        g['id'] for g in all_recent_games
        if g.get('id') is not None and g.get('status') in _COMPLETED_STATUSES
    ]
    if completed_bdl_ids:
        with get_db_connection() as conn:
            placeholders = ",".join("?" for _ in completed_bdl_ids)
            cur = conn.execute(
                f"UPDATE games SET status='COMPLETED' "
                f"WHERE bdl_game_id IN ({placeholders}) AND status != 'COMPLETED'",
                completed_bdl_ids,
            )
            conn.commit()
        if cur.rowcount:
            logger.info(f"Marked {cur.rowcount} finished games COMPLETED.")

    recent_game_ids = {
        g['id']
        for g in all_recent_games
        if g.get('status') in _COMPLETED_STATUSES
        and (
            (g.get('home_team') or {}).get('id') in teams_to_sync
            or (g.get('away_team') or {}).get('id') in teams_to_sync
        )
    }
    # Filter out games already fully synced (last_synced_at IS NOT NULL).
    with get_db_connection() as conn:
        already_synced = {
            row['bdl_game_id']
            for row in conn.execute(
                "SELECT bdl_game_id FROM games "
                "WHERE bdl_game_id IS NOT NULL AND last_synced_at IS NOT NULL"
            ).fetchall()
        }
    skipped = len(recent_game_ids & already_synced)
    recent_game_ids -= already_synced

    logger.info(
        f"Found {len(recent_game_ids)} completed games to sync "
        f"({skipped} already synced, {_LOOKBACK_DAYS}-day window)."
    )

    if not recent_game_ids:
        logger.info("No new completed games to sync stats from.")
        return

    # 4. Fetch all player stats in one batched call (≤50 game IDs per HTTP request).
    all_stats = await bdl_client.get_stats_batch(recent_game_ids)
    logger.info(f"Processing {len(all_stats)} player-game stat records.")

    # 5. Write everything in a single transaction; track which bdl game IDs got rows.
    total_pitcher_logs = 0
    total_batter_logs = 0
    synced_gids: set = set()

    # Authoritative gid -> date map from the /games payload. Game-log rows must
    # carry a real date: every recency window, rest-day calc, and as-of filter
    # in the model stack keys on it.
    game_dates = {
        g['id']: str(g.get('date') or '')[:10]
        for g in all_recent_games
        if g.get('id') is not None
    }

    with get_db_connection() as conn:
        for player_stat in all_stats:
            player_id, player_name, position, gid, game_date = _upsert_player(conn, player_stat)
            if player_id is None:
                continue
            if not game_date:
                game_date = game_dates.get(gid, '')

            ip = player_stat.get('ip') or player_stat.get('innings_pitched')
            is_pitcher = ip is not None and str(ip) not in ('0', '0.0', '')
            if is_pitcher:
                n = _insert_pitcher_log(conn, gid, player_id, game_date, player_stat, player_name)
                total_pitcher_logs += n
            else:
                n = _insert_batter_log(conn, gid, player_id, game_date, player_stat, player_name)
                total_batter_logs += n
            if n and gid is not None:
                synced_gids.add(gid)

        # Stamp last_synced_at so these games are skipped on future runs.
        now_iso = datetime.now(timezone.utc).isoformat()
        for gid in synced_gids:
            conn.execute(
                "UPDATE games SET last_synced_at=? WHERE bdl_game_id=?",
                (now_iso, gid),
            )
        conn.commit()

    logger.info(
        f"Synced stats: {total_pitcher_logs} pitcher logs, {total_batter_logs} batter logs "
        f"across {len(synced_gids)} games."
    )
