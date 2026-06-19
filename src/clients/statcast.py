"""Baseball Savant client — downloads season leaderboard CSVs (no API key required).

All endpoints are public CSV exports from baseballsavant.mlb.com. The `player_id`
column in Savant data is the MLB MLBAM ID, which we join against players.mlb_id in
the local DB.

Endpoints used:
  - /leaderboard/expected_statistics  → barrel%, hard-hit%, xwOBA, exit velocity
  - /leaderboard/percentile-rankings  → whiff%, chase rate (O-swing%)
  - /leaderboard/pitch-arsenal        → fastball spin rate
  - /leaderboard/sprint_speed         → sprint speed (batters only)
"""
from __future__ import annotations

import asyncio
import io
from typing import Optional

import httpx
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BASE = "https://baseballsavant.mlb.com"
_TIMEOUT = httpx.Timeout(connect=15.0, read=90.0, write=15.0, pool=15.0)
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mlb-prop-bot/1.0)"}


class StatcastClient:
    """Downloads Statcast season leaderboards from Baseball Savant."""

    async def _fetch_csv(self, url: str) -> Optional[pd.DataFrame]:
        """Download a CSV from Savant and return a DataFrame, or None on error."""
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT, follow_redirects=True, headers=_HEADERS
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                return df
        except Exception as e:
            logger.warning("Statcast CSV fetch failed url=%s err=%s", url, e)
            return None

    async def get_pitcher_expected_stats(self, season: int) -> pd.DataFrame:
        """xwOBA, barrel%, hard-hit%, avg exit velocity allowed — per pitcher."""
        url = (
            f"{_BASE}/leaderboard/expected_statistics"
            f"?type=pitcher&year={season}&min=1&csv=true"
        )
        df = await self._fetch_csv(url)
        return df if df is not None else pd.DataFrame()

    async def get_pitcher_pitch_mix(self, season: int) -> pd.DataFrame:
        """Fastball (FF) spin rate and velocity per pitcher."""
        url = (
            f"{_BASE}/leaderboard/pitch-arsenal-stats"
            f"?type=pitcher&pitchType=FF&year={season}&team=&min=1&csv=true"
        )
        df = await self._fetch_csv(url)
        return df if df is not None else pd.DataFrame()

    async def get_pitcher_percentile_rankings(self, season: int) -> pd.DataFrame:
        """Whiff%, chase rate (O-swing%), K% per pitcher."""
        url = (
            f"{_BASE}/leaderboard/percentile-rankings"
            f"?type=pitcher&year={season}&pos=all&csv=true"
        )
        df = await self._fetch_csv(url)
        return df if df is not None else pd.DataFrame()

    async def get_batter_expected_stats(self, season: int) -> pd.DataFrame:
        """xwOBA, barrel%, hard-hit%, avg exit velocity — per batter."""
        url = (
            f"{_BASE}/leaderboard/expected_statistics"
            f"?type=batter&year={season}&min=1&csv=true"
        )
        df = await self._fetch_csv(url)
        return df if df is not None else pd.DataFrame()

    async def get_batter_percentile_rankings(self, season: int) -> pd.DataFrame:
        """Whiff%, K% per batter."""
        url = (
            f"{_BASE}/leaderboard/percentile-rankings"
            f"?type=batter&year={season}&pos=all&csv=true"
        )
        df = await self._fetch_csv(url)
        return df if df is not None else pd.DataFrame()

    async def get_sprint_speed(self, season: int) -> pd.DataFrame:
        """Sprint speed (ft/sec) per batter/runner."""
        url = (
            f"{_BASE}/leaderboard/sprint_speed"
            f"?year={season}&team=&position=&min=0&csv=true"
        )
        df = await self._fetch_csv(url)
        return df if df is not None else pd.DataFrame()

    async def get_all_pitcher_stats(self, season: int) -> pd.DataFrame:
        """Fetch and merge all pitcher Statcast sources into one DataFrame.

        Keyed by player_id (MLB MLBAM ID). Columns from different sources are
        merged; missing sources produce NaN columns (safe for DB upserts).
        """
        exp, mix, pct = await asyncio.gather(
            self.get_pitcher_expected_stats(season),
            self.get_pitcher_pitch_mix(season),
            self.get_pitcher_percentile_rankings(season),
        )
        return _merge_on_player_id(exp, mix, pct)

    async def get_all_batter_stats(self, season: int) -> pd.DataFrame:
        """Fetch and merge all batter Statcast sources into one DataFrame."""
        exp, pct, speed = await asyncio.gather(
            self.get_batter_expected_stats(season),
            self.get_batter_percentile_rankings(season),
            self.get_sprint_speed(season),
        )
        return _merge_on_player_id(exp, pct, speed)


# ---------------------------------------------------------------------------
# Helper: merge DataFrames on MLBAM player_id
# ---------------------------------------------------------------------------

def _merge_on_player_id(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Left-join all non-empty DataFrames on the MLBAM player_id column.

    Savant may name the ID column 'player_id', 'mlbam_id', or 'id' depending
    on the endpoint and season. We normalise to 'player_id' before joining.
    """
    result: Optional[pd.DataFrame] = None
    for df in dfs:
        if df is None or df.empty:
            continue
        # Find the ID column
        id_col = next(
            (c for c in ['player_id', 'mlbam_id', 'id'] if c in df.columns),
            None,
        )
        if id_col is None:
            logger.debug("Statcast merge: no player_id column found in DataFrame with cols %s",
                         list(df.columns)[:10])
            continue
        if id_col != 'player_id':
            df = df.rename(columns={id_col: 'player_id'})
        df = df.copy()
        df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce')
        df = df.dropna(subset=['player_id'])
        df['player_id'] = df['player_id'].astype(int)

        if result is None:
            result = df
        else:
            # Only merge columns that don't already exist (avoid _x/_y suffixes)
            new_cols = ['player_id'] + [c for c in df.columns if c not in result.columns]
            result = result.merge(df[new_cols], on='player_id', how='left')

    return result if result is not None else pd.DataFrame()
