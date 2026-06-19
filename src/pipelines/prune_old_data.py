from datetime import timedelta
from src.utils.time_utils import utcnow
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Tables that are never pruned — reference/configuration data.
# teams, players, park_factors: small and required for every query.
_STATIC_TABLES = {'teams', 'players', 'park_factors'}


def prune_old_data(
    game_logs_days: int = 90,
    completed_games_days: int = 90,
    alerts_days: int = 90,
    hot_data_days: int = 3,
    snapshot_days: int = 90,
) -> dict:
    """
    Remove old rows from every time-bounded table.

    Retention windows (all configurable):
      hot_data_days       — daily_lineups, probable_pitchers,
                            injury_reports, bet_candidates              (default 3)
      snapshot_days       — prop_snapshots, projections                 (default 90)
                            Long retention on purpose: these are the raw
                            material for backtests, CLV on late-settling
                            bets, and steam baselines.
      game_logs_days      — pitcher_game_logs, batter_game_logs         (default 90)
      completed_games_days — games with status='COMPLETED'               (default 90)
      alerts_days         — alerts_sent + cascaded bet_results           (default 90)

    Static reference tables (teams, players, park_factors) are never touched.

    Returns a dict mapping table name → rows deleted, for logging and tests.
    """
    logger.info("Executing pipeline: prune_old_data")

    now = utcnow()

    hot_cutoff       = (now - timedelta(days=hot_data_days)).isoformat()
    snapshot_cutoff  = (now - timedelta(days=snapshot_days)).isoformat()
    logs_cutoff      = (now - timedelta(days=game_logs_days)).isoformat()
    games_cutoff     = (now - timedelta(days=completed_games_days)).isoformat()
    alerts_cutoff    = (now - timedelta(days=alerts_days)).isoformat()

    deleted: dict = {}

    with get_db_connection() as conn:
        # 1a. Hot / transient data
        hot_tables = {
            "daily_lineups":    "date",
            "probable_pitchers":"date",
            "injury_reports":   "date",
            "bet_candidates":   "created_at",
        }
        for table, col in hot_tables.items():
            cur = conn.execute(f"DELETE FROM {table} WHERE {col} < ?", (hot_cutoff,))
            deleted[table] = cur.rowcount

        # 1b. Odds history — long retention (backtests / CLV / steam baselines)
        for table in ("prop_snapshots", "projections"):
            cur = conn.execute(
                f"DELETE FROM {table} WHERE timestamp < ?", (snapshot_cutoff,)
            )
            deleted[table] = cur.rowcount

        # 2. Game logs — keep enough history for the 20-game projection lookback.
        # Rows whose game_id belongs to a historical backfill (games.historical=1)
        # are preserved so we don't wipe training data.
        for table in ("pitcher_game_logs", "batter_game_logs"):
            cur = conn.execute(
                f"""
                DELETE FROM {table}
                WHERE date < ?
                  AND game_id NOT IN (
                    SELECT bdl_game_id FROM games
                     WHERE historical = 1 AND bdl_game_id IS NOT NULL
                  )
                """,
                (logs_cutoff,),
            )
            deleted[table] = cur.rowcount

        # 3. Completed games only — never remove scheduled, in-progress, or historical-backfill games
        cur = conn.execute(
            "DELETE FROM games WHERE date < ? AND status = 'COMPLETED' AND COALESCE(historical, 0) = 0",
            (games_cutoff,)
        )
        deleted["games"] = cur.rowcount

        # 4. Settled bet records — cascade child rows first to respect the FK
        cur = conn.execute(
            "DELETE FROM bet_results WHERE alert_id IN "
            "(SELECT alert_id FROM alerts_sent WHERE timestamp < ?)",
            (alerts_cutoff,)
        )
        deleted["bet_results"] = cur.rowcount

        cur = conn.execute(
            "DELETE FROM alerts_sent WHERE timestamp < ?",
            (alerts_cutoff,)
        )
        deleted["alerts_sent"] = cur.rowcount

        conn.commit()

    total = sum(deleted.values())
    logger.info(
        f"prune_old_data complete — {total} rows removed. "
        f"Breakdown: { {k: v for k, v in deleted.items() if v > 0} }"
    )
    return deleted
