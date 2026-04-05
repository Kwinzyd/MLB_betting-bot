import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
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
