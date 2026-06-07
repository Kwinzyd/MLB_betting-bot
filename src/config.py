import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # OpenWeatherMap (free tier)
BDL_API_KEY = os.getenv("BDL_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Polymarket L2 / CLOB API Auth
POLYMARKET_HOST = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY")
POLYMARKET_SECRET = os.getenv("POLYMARKET_SECRET")
POLYMARKET_PASSPHRASE = os.getenv("POLYMARKET_PASSPHRASE")
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
EDGE_MIN = float(os.getenv("EDGE_MIN", "5.0"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
BANKROLL = float(os.getenv("BANKROLL", "1000.0"))
# Global kill switch. When False, send_alerts still computes edges and logs them
# but skips Telegram delivery and does not record alerts_sent rows. Flip to True
# only after a paper-trading validation window.
BETTING_ENABLED = os.getenv("BETTING_ENABLED", "false").lower() in ("1", "true", "yes")
TELEGRAM_VENUE_ENABLED = os.getenv("TELEGRAM_VENUE_ENABLED", "true").lower() in ("1", "true", "yes")
PAPER_EXCHANGE_ENABLED = os.getenv("PAPER_EXCHANGE_ENABLED", "false").lower() in ("1", "true", "yes")
POLYMARKET_ENABLED = os.getenv("POLYMARKET_ENABLED", "false").lower() in ("1", "true", "yes")
MIN_ODDS = float(os.getenv("MIN_ODDS", "1.70"))
MIN_MODEL_PROB = float(os.getenv("MIN_MODEL_PROB", "0.55"))
MIN_SAMPLE_SIZE = int(os.getenv("MIN_SAMPLE_SIZE", "10"))
MAX_BETS_PER_GAME = int(os.getenv("MAX_BETS_PER_GAME", "3"))
MAX_BETS_PER_PLAYER = int(os.getenv("MAX_BETS_PER_PLAYER", "1"))
PREGAME_WINDOW_MINUTES = int(os.getenv("PREGAME_WINDOW_MINUTES", "45"))
MLB_SEASON = int(os.getenv("MLB_SEASON", "2026"))
ODDS_REGION = os.getenv("ODDS_REGION", "us")
BOOKMAKERS = os.getenv("BOOKMAKERS", "draftkings,fanduel,betmgm,caesars,pointsbetus,betrivers").split(",")
# Sharp books used as the source of truth for CLV. Priority order: first available wins.
# Pinnacle lives in the `eu` region feed but is reachable by listing it in the `bookmakers`
# param of a `us`-region call (Odds API accepts cross-region books via that filter).
SHARP_BOOKMAKERS = [b.strip() for b in os.getenv("SHARP_BOOKMAKERS", "pinnacle,circasports").split(",") if b.strip()]
# Max allowed gap between model_prob and sharp-devigged prob for a bet to be
# playable. Sharp market is near-efficient; a wide gap means the model is
# wrong. 0.05 = 5 percentage points.
SHARP_MODEL_AGREEMENT_TOL = float(os.getenv("SHARP_MODEL_AGREEMENT_TOL", "0.05"))

# Same-game-parlay tunables. edge_vs_naive compares our correlation-adjusted
# joint probability to the independent-multiply parlay price a book would
# charge if it didn't penalize correlation. Higher than single-bet bar
# because parlays compound variance.
SGP_MIN_EDGE = float(os.getenv("SGP_MIN_EDGE", "0.10"))
SGP_MAX_PER_GAME = int(os.getenv("SGP_MAX_PER_GAME", "1"))
SGP_MAX_PER_DAY = int(os.getenv("SGP_MAX_PER_DAY", "3"))
# Kelly fraction multiplier for SGPs. Effective Kelly = KELLY_FRACTION * this.
# Tighter than singles because parlay variance compounds even after correlation
# correction, and the empirical-r estimate from ~50 starts has its own noise.
SGP_KELLY_FRACTION_MULT = float(os.getenv("SGP_KELLY_FRACTION_MULT", "0.5"))

# Weather / umpire edge-trigger thresholds. Fire a targeted odds pull when an
# environmental input shifts enough that our projections meaningfully move
# before the books have re-priced.
TRIGGER_UMP_THRESHOLD = float(os.getenv("TRIGGER_UMP_THRESHOLD", "0.10"))
TRIGGER_WEATHER_HR_THRESHOLD = float(os.getenv("TRIGGER_WEATHER_HR_THRESHOLD", "0.05"))
TRIGGER_WEATHER_SO_THRESHOLD = float(os.getenv("TRIGGER_WEATHER_SO_THRESHOLD", "0.03"))
TRIGGER_DEDUP_HOURS = int(os.getenv("TRIGGER_DEDUP_HOURS", "6"))
# Total-shift trigger: fire when sharp-book consensus total moves by >= this
# (in runs) since the last persisted snapshot. Soft books typically lag the
# move by minutes, so we force a targeted re-scan to catch them sleeping.
TRIGGER_TOTAL_SHIFT_THRESHOLD = float(os.getenv("TRIGGER_TOTAL_SHIFT_THRESHOLD", "0.5"))
# Skip the totals re-poll if the latest history row is younger than this —
# avoids flapping right after a scan and saves quota.
TRIGGER_TOTAL_MIN_HISTORY_MINUTES = int(os.getenv("TRIGGER_TOTAL_MIN_HISTORY_MINUTES", "5"))

# Alt-line shopping: max distance (in line units) we'll evaluate alt-lines from
# the sharp-anchored line. Soft books frequently post off-consensus alts; the
# model is anchor-validated at the sharp line and re-priced at the alt-line via
# get_probabilities. The cap keeps us inside the regime where the projection's
# tails are well-behaved.
ALT_LINE_MAX_DISTANCE = float(os.getenv("ALT_LINE_MAX_DISTANCE", "1.0"))

# Camouflage stake rounding. Soft books fingerprint accounts that bet exact
# fractional-Kelly amounts ($18.42, $7.31). Snap persisted/displayed stakes
# to a round increment so they look like rec-style $15/$20 wagers. Set 0 to
# disable (e.g. for backtests where exact Kelly figures matter).
STAKE_ROUNDING_INCREMENT = float(os.getenv("STAKE_ROUNDING_INCREMENT", "5.0"))

# Portfolio-level Kelly: pairwise return correlations used in covariance matrix.
# Tune these if you have empirical data; defaults are conservative estimates.
# Same team, same side (e.g. two batter props for the same team):
PORTFOLIO_CORR_SAME_TEAM = float(os.getenv("PORTFOLIO_CORR_SAME_TEAM", "0.55"))
# Same game, pitcher prop vs batter prop on the same team (inverse relationship):
PORTFOLIO_CORR_PITCHER_BATTER = float(os.getenv("PORTFOLIO_CORR_PITCHER_BATTER", "-0.25"))
# Same game, opposite teams (run environment correlation):
PORTFOLIO_CORR_SAME_GAME_OPP_TEAM = float(os.getenv("PORTFOLIO_CORR_SAME_GAME_OPP_TEAM", "0.10"))
# Max total bankroll fraction deployed across all simultaneous bets.
# At 20% of $1000 = $200 max in play at once. Prevents correlated wipeout.
PORTFOLIO_TOTAL_EXPOSURE_CAP = float(os.getenv("PORTFOLIO_TOTAL_EXPOSURE_CAP", "0.20"))

# Bookmaker bias detection. avg_bias above this threshold (in devigged prob units)
# triggers a +0.5% edge credit in scan_props.
BOOKMAKER_BIAS_THRESHOLD = float(os.getenv("BOOKMAKER_BIAS_THRESHOLD", "0.02"))

# Dispersion fitting: per-entity NB alpha / Normal sigma with EB shrinkage.
DISPERSION_MIN_OBS = int(os.getenv("DISPERSION_MIN_OBS", "10"))
DISPERSION_PRIOR_K = int(os.getenv("DISPERSION_PRIOR_K", "30"))
DISPERSION_ALPHA_CAP = float(os.getenv("DISPERSION_ALPHA_CAP", "2.0"))

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
    "Athletics": "OAK",
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
LEAGUE_AVG_3B_RATE = 0.005      # ~0.5% triples per PA (too noisy to back out from anchors)
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
LEAGUE_AVG_GAME_TOTAL = LEAGUE_AVG_RUNS_PER_GAME * 2  # both teams combined
PA_ELASTICITY_TO_TOTAL = 0.35  # 10% more expected runs -> ~3.5% more PAs

PA_ESTIMATOR_ENABLED = True
PA_DIST_SUPPORT = (3, 4, 5, 6, 7)
PA_DIST_ANCHOR = 3

HR_ZINB_ENABLED = True
LEAGUE_AVG_HR9 = 1.30
LEAGUE_AVG_ISO = 0.165
HR_PI0_MAX = 0.40
HR_PI0_BETA_PITCHER = 0.45
HR_PI0_BETA_BATTER = 4.50
HR_PI0_BETA_PARK = 1.20
HR_PI0_BETA_WIND = 0.04

# Same-team batter-batter PA correlation. The TBF latent in joint_pa.py
# determines every slot's PA deterministically, so two same-team slots co-move
# strongly. The derived boost replaces the static BATTER_CORR_BOOST entry for
# same-team batter-under pairs. Cap is a model-error guardrail.
JOINT_PA_ENABLED = os.getenv("JOINT_PA_ENABLED", "true").lower() in ("1", "true", "yes")
JOINT_PA_BOOST_CAP = float(os.getenv("JOINT_PA_BOOST_CAP", "1.25"))

# Live State Machine (in-game prop sniping)
# Minimum EV to fire a live Telegram alert. Higher bar than singles because
# live lines move quickly and model latency adds uncertainty.
LIVE_MIN_EV = float(os.getenv("LIVE_MIN_EV", "0.08"))   # 8% EV default
# Seconds between BDL live-game polls. 20s is safe for BDL free tier.
LIVE_POLL_INTERVAL_SECONDS = int(os.getenv("LIVE_POLL_INTERVAL_SECONDS", "20"))

# Fail fast at import time if critical keys are missing rather than crashing
# deep inside an HTTP call with a confusing error.
_REQUIRED_ENV_VARS = ["ODDS_API_KEY", "BDL_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
_missing = [k for k in _REQUIRED_ENV_VARS if not globals().get(k)]
if _missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Set them in your .env file or shell before running."
    )
