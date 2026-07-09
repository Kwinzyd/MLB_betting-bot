import httpx
import asyncio
from src.config import SPORTSDATAIO_API_KEY
from src.utils.logging_utils import get_logger
from src.utils.time_utils import get_eastern_local_date

logger = get_logger(__name__)

SPORTSDATAIO_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    # SportsDataIO uses "ATH" for the Athletics (confirmed live), not "OAK".
    "Oakland Athletics": "ATH",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

MARKET_KEY_MAP = {
    'pitcher_strikeouts': 'Pitching Strikeouts',
    'batter_hits': 'Hits',
    'batter_home_runs': 'Home Runs',
    'batter_total_bases': 'Total Bases',
    'batter_rbi': 'Runs Batted In',
    'batter_runs': 'Runs',
}

class SportsDataIOClient:
    def __init__(self):
        self.api_key = SPORTSDATAIO_API_KEY
        self.base_url = "https://api.sportsdata.io/v3/mlb/odds/json"
        
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Cache for daily consensus player props
        self._props_cache = []
        self._props_cache_date = None

    async def _populate_props_cache(self):
        # MLB slates key on the US/Eastern calendar day, not local machine time.
        today = str(get_eastern_local_date())
        if self._props_cache_date == today and self._props_cache:
            return

        url = f"{self.base_url}/PlayerPropsByDate/{today}?key={self.api_key}"
        logger.info(f"Fetching SportsDataIO consensus props for {today}")
        
        try:
            resp = await self.client.get(url)
            resp.raise_for_status()
            self._props_cache = resp.json()
            self._props_cache_date = today
            logger.info(f"Cached {len(self._props_cache)} props from SportsDataIO.")
        except Exception as e:
            logger.error(f"Failed to fetch SportsDataIO props: {e}")

    async def get_event_odds(self, bdl_game_id: str, markets: list, bust_cache: bool = False, home_team_full: str = None, away_team_full: str = None) -> dict:
        """
        Fetch consensus odds for multiple markets for a single game.
        Translates the SportsDataIO schema into the expected Odds API format.
        """
        if bust_cache:
            self._props_cache_date = None
            
        await self._populate_props_cache()
        
        if not home_team_full or not away_team_full:
            logger.warning("SportsDataIO client requires home_team_full and away_team_full.")
            return {}
            
        home_abbr = SPORTSDATAIO_TEAM_MAP.get(home_team_full, home_team_full)
        away_abbr = SPORTSDATAIO_TEAM_MAP.get(away_team_full, away_team_full)

        sportsdataio_markets = []
        
        for market in markets:
            if market == 'totals':
                # Game total is handled by BDL now, skip querying it from here
                continue
                
            sdata_desc = MARKET_KEY_MAP.get(market)
            if not sdata_desc:
                continue

            market_outcomes = []
            
            for prop in self._props_cache:
                team = prop.get('Team')
                opponent = prop.get('Opponent')
                
                # Check if this prop belongs to this game
                if not ((team == home_abbr and opponent == away_abbr) or 
                        (team == away_abbr and opponent == home_abbr)):
                    continue
                    
                if prop.get('Description') != sdata_desc:
                    continue
                    
                player_name = prop.get('Name', 'Unknown')
                over_under = prop.get('OverUnder')
                if over_under is None:
                    continue
                
                over_payout = prop.get('OverPayout')
                under_payout = prop.get('UnderPayout')
                
                if over_payout:
                    market_outcomes.append({
                        'name': 'Over',
                        'point': float(over_under),
                        'price': self._american_to_decimal(over_payout),
                        'description': player_name
                    })
                    
                if under_payout:
                    market_outcomes.append({
                        'name': 'Under',
                        'point': float(over_under),
                        'price': self._american_to_decimal(under_payout),
                        'description': player_name
                    })

            if market_outcomes:
                sportsdataio_markets.append({
                    'key': market,
                    'outcomes': market_outcomes
                })

        if not sportsdataio_markets:
            return {}
            
        return {'bookmakers': [{'key': 'sportsdataio', 'markets': sportsdataio_markets}]}
        
    def _american_to_decimal(self, american: int) -> float:
        if american > 0:
            return round(1 + (american / 100), 3)
        elif american < 0:
            return round(1 - (100 / american), 3)
        return 1.0

    async def close(self):
        await self.client.aclose()
