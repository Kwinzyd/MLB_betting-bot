import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # OpenWeatherMap (free tier)
BDL_API_KEY = os.getenv("BDL_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EDGE_MIN = float(os.getenv("EDGE_MIN", "5.0"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
BANKROLL = float(os.getenv("BANKROLL", "1000.0"))
MLB_SEASON = int(os.getenv("MLB_SEASON", "2026"))
ODDS_REGION = os.getenv("ODDS_REGION", "us")
BOOKMAKERS = os.getenv("BOOKMAKERS", "draftkings,fanduel,betmgm,caesars,pointsbetus,betrivers").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "data.db")

# Hardcoded mapping: Odds API team name → standard MLB abbreviation.
# This guarantees 100% accurate joins between Odds API and BallDontLie data.
# BDL teams table stores the same abbreviations, so we join on abbreviation.
ODDS_API_TEAM_ABBREV = {
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
    "Oakland Athletics": "OAK",
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

MARKETS_MAPPING = {
    "pitcher_strikeouts": "pitcher_strikeouts",
    "batter_total_bases": "batter_total_bases",
    "batter_hits": "batter_hits",
    "batter_home_runs": "batter_home_runs",
    "pitcher_earned_runs": "pitcher_earned_runs",
}

# League average constants for adjustment calculations
LEAGUE_AVG_K_RATE = 0.225       # ~22.5% strikeout rate
LEAGUE_AVG_RUNS_PER_GAME = 4.5  # ~4.5 runs per team per game
LEAGUE_AVG_HR_RATE = 0.035      # ~3.5% HR per PA
LEAGUE_AVG_PITCHES_PER_IP = 16.5  # ~16.5 pitches per inning, MLB average
DEFAULT_PITCH_LIMIT = 100         # Typical pitch count ceiling for starters

# Platoon adjustment multipliers (same-hand vs opposite-hand matchups)
PLATOON_ADJUSTMENTS = {
    "same_hand": {
        "batter_hits": 0.92,
        "batter_home_runs": 0.88,
        "batter_total_bases": 0.90,
    },
    "opposite_hand": {
        "batter_hits": 1.05,
        "batter_home_runs": 1.08,
        "batter_total_bases": 1.06,
    },
}

# Projection blending weights
PITCHER_RECENT_WEIGHT = 0.60    # L10 starts
PITCHER_SEASON_WEIGHT = 0.40    # full season
BATTER_RECENT_WEIGHT = 0.55     # L15 games
BATTER_SEASON_WEIGHT = 0.45     # full season

# Umpire model coefficients
# All values are tunable; see sync_umpires and projections for derivation.
LEAGUE_AVG_K_PER_GAME  = 13.5   # combined both teams, 2024-25 MLB average
LEAGUE_AVG_BB_PER_GAME = 6.5    # combined both teams, 2024-25 MLB average
UMP_STATS_LOOKBACK_DAYS = 7     # days of schedule history to maintain umpire stats from
UMP_MIN_GAMES = 5               # minimum games called before applying a non-neutral factor
UMP_K_WEIGHT  = 0.5             # partial-trust blend: effective = 1 + (raw_factor - 1) * weight
# Dampening factor: umpire's K zone effect on ER (smaller than direct K effect)
# High-K zone → fewer baserunners → slightly fewer ER. Rule of thumb: ~30% of K adjustment.
UMP_ER_K_DAMPENING = 0.3

# Projected plate appearances by lineup position (1-9)
# Source: historical MLB averages across full 9-inning games
# Leadoff sees ~4.5 PA, bottom of the order ~3.7-3.8
LINEUP_PA_MAP = {
    1: 4.50,  # Leadoff
    2: 4.40,
    3: 4.30,
    4: 4.20,  # Cleanup
    5: 4.10,
    6: 4.00,
    7: 3.90,
    8: 3.80,
    9: 3.70,  # 9-hole (pitcher spot in NL / weakest hitter)
}
DEFAULT_PROJECTED_PA = 4.0  # Fallback when lineup position is unknown
