import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # OpenWeatherMap (free tier)
BDL_API_KEY = os.getenv("BDL_API_KEY")
SPORTSDATAIO_API_KEY = os.getenv("SPORTSDATAIO_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- OpenRouter LLM layer (qualitative enrichment, research agent, alert
# rationale, name reconciliation). The LLM never produces projections,
# probabilities, or stake sizes — only structured context the quant core reads.
# The whole layer is OPTIONAL and fail-safe: with no key or LLM_ENABLED=false,
# every call no-ops and the bot falls back to its non-LLM behavior. ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Cheap + fast default for high-volume extraction/tool-use; override per taste.
# Must be a current OpenRouter model id (the catalog changes — verify with
# GET https://openrouter.ai/api/v1/models if calls 404).
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3.5-flash")
# Optional stronger model for the research agent's reasoning; defaults to the
# cheap model so a single key/model works out of the box.
OPENROUTER_RESEARCH_MODEL = os.getenv("OPENROUTER_RESEARCH_MODEL", OPENROUTER_MODEL)
# Master switch. Defaults to enabled only when a key is present.
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() in ("1", "true", "yes") and bool(OPENROUTER_API_KEY)
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# Cache identical prompts this long to bound cost (enrichment re-runs on a slate).
LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "21600"))  # 6h
# Conservative skip gate: only when a fresh LLM signal estimates a player's
# probability of appearing below this AND the raw status already flagged a
# problem do we skip the prop. The LLM can tighten (skip more) but never
# loosens a bet decision, and it never alters a projection number.
LLM_INJURY_SKIP_PROBABILITY = float(os.getenv("LLM_INJURY_SKIP_PROBABILITY", "0.25"))

# Polymarket L2 / CLOB API Auth
POLYMARKET_HOST = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY")
POLYMARKET_SECRET = os.getenv("POLYMARKET_SECRET")
POLYMARKET_PASSPHRASE = os.getenv("POLYMARKET_PASSPHRASE")
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
EDGE_MIN = float(os.getenv("EDGE_MIN", "5.0"))
# Sanity ceiling on claimed edge (pct points). Settled-bet analysis (through
# 2026-07-02) showed claimed edges of 15-42% won no more often than 5-10% ones:
# a huge model-vs-book gap is model error, not free money. Skip, don't bet bigger.
EDGE_MAX = float(os.getenv("EDGE_MAX", "15.0"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
BANKROLL = float(os.getenv("BANKROLL", "1000.0"))
# Count settled SGP/parlay ticket P&L in the Kelly bankroll ledger. Off by
# default: tickets are Telegram suggestions that may never have been placed,
# and one paper longshot win can inflate the ledger by thousands.
BANKROLL_INCLUDE_PARLAYS = os.getenv("BANKROLL_INCLUDE_PARLAYS", "false").lower() in ("1", "true", "yes")
# Global kill switch. When False, send_alerts still computes edges and logs them
# but skips Telegram delivery and does not record alerts_sent rows. Flip to True
# only after a paper-trading validation window.
BETTING_ENABLED = os.getenv("BETTING_ENABLED", "false").lower() in ("1", "true", "yes")
TELEGRAM_VENUE_ENABLED = os.getenv("TELEGRAM_VENUE_ENABLED", "true").lower() in ("1", "true", "yes")
PAPER_EXCHANGE_ENABLED = os.getenv("PAPER_EXCHANGE_ENABLED", "false").lower() in ("1", "true", "yes")
POLYMARKET_ENABLED = os.getenv("POLYMARKET_ENABLED", "false").lower() in ("1", "true", "yes")
MIN_ODDS = float(os.getenv("MIN_ODDS", "1.70"))
# Longshot cap. Settled bets through 2026-07-02: odds < 2.0 were +$123 while
# odds >= 2.0 were -$226 (overs at plus money were the single biggest leak).
# The Poisson/NB tails are overconfident on longshots; cap until calibration
# proves otherwise.
MAX_ODDS = float(os.getenv("MAX_ODDS", "2.20"))
MIN_MODEL_PROB = float(os.getenv("MIN_MODEL_PROB", "0.55"))
MIN_SAMPLE_SIZE = int(os.getenv("MIN_SAMPLE_SIZE", "10"))
MAX_BETS_PER_GAME = int(os.getenv("MAX_BETS_PER_GAME", "3"))
MAX_BETS_PER_PLAYER = int(os.getenv("MAX_BETS_PER_PLAYER", "1"))
# Require a confirmed lineup/starter before a prop is playable. Default True so a
# batter not in today's posted lineup, or an unconfirmed/scratched starting
# pitcher, is never bet on a fallback projection (DEFAULT_PROJECTED_PA, neutral
# platoon) — that risk was only recovered as a post-hoc DNP void. Set False to
# bet early markets before lineups post, at the cost of late-scratch exposure.
REQUIRE_CONFIRMED_LINEUP = os.getenv("REQUIRE_CONFIRMED_LINEUP", "true").lower() in ("1", "true", "yes")
# How far before first pitch a game becomes scan-eligible. 120 min gives a few
# hours of pregame coverage so picks land well before the game starts (player
# props usually post 2-4h out). Raise for earlier picks at the cost of more
# Odds API calls; the re-scan throttle below bounds that cost.
PREGAME_WINDOW_MINUTES = int(os.getenv("PREGAME_WINDOW_MINUTES", "120"))
# Inside the pregame window, re-scan a game at most once per this many minutes.
# Stops a wide window from burning quota by re-pulling odds on every scheduler
# tick; a lineup change still forces an immediate re-scan via its own trigger.
PREGAME_RESCAN_MINUTES = int(os.getenv("PREGAME_RESCAN_MINUTES", "60"))
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

# Weight given to the books' own devigged consensus at an alt-line when forming
# the truth probability there (the remainder comes from the sharp anchor shifted
# along the model CDF — see _shift_anchor_truth). Settled-bet calibration showed
# the model's tail shape is its weakest part, and alt-line tails are exactly
# where it priced the losing longshot overs. 0.0 = trust the shifted anchor
# fully (old behavior); 1.0 = trust the quoted market fully (no alt-line edges).
ALT_LINE_MARKET_SHRINK = float(os.getenv("ALT_LINE_MARKET_SHRINK", "0.7"))

# send_alerts only consumes bet_candidates younger than this. Candidates are
# written by scan_props; anything older reflects odds that have likely moved.
ALERT_CANDIDATE_MAX_AGE_MINUTES = int(os.getenv("ALERT_CANDIDATE_MAX_AGE_MINUTES", "15"))

# --- Cross-game parlay builder (find_parlays.py) ---
# Distinct from the SGP pipeline: these parlays combine the strongest single-bet
# edges across DIFFERENT games (one leg per game), so legs are independent and
# the joint price is an exact product — no within-game correlation modeling.
# Leg counts to build. Each is the top-N legs by per-leg EV (which maximizes
# parlay EV under independence); they nest (the 8-leg contains the 2/4-leg legs)
# and are presented as alternative tickets — pick one.
PARLAY_SIZES = [int(x.strip()) for x in os.getenv("PARLAY_SIZES", "2,4,8").split(",") if x.strip()]
# Minimum ticket EV to alert. +EV legs compound (ev = prod(1+leg_ev) - 1), so
# any parlay of playable legs clears this easily; it mainly drops marginal 2-leggers.
PARLAY_MIN_EV = float(os.getenv("PARLAY_MIN_EV", "0.10"))
# Only legs whose single-bet edge clears this (%) are eligible. Defaults to the
# single-bet bar so a parlay never includes a leg we wouldn't bet straight.
PARLAY_MIN_LEG_EDGE = float(os.getenv("PARLAY_MIN_LEG_EDGE", str(EDGE_MIN)))
# Cap on parlay tickets persisted/alerted per run (across all sizes).
PARLAY_MAX_PER_DAY = int(os.getenv("PARLAY_MAX_PER_DAY", "3"))
# Kelly haircut for parlays (effective Kelly = KELLY_FRACTION * this). Parlay
# variance compounds with every leg; the Kelly math already shrinks long-shot
# stakes, and this adds a further safety margin.
PARLAY_KELLY_FRACTION_MULT = float(os.getenv("PARLAY_KELLY_FRACTION_MULT", "0.5"))
# Legs are sourced from bet_candidates no older than this (same staleness logic
# as send_alerts — older quotes have likely moved).
PARLAY_CANDIDATE_MAX_AGE_MINUTES = int(
    os.getenv("PARLAY_CANDIDATE_MAX_AGE_MINUTES", str(ALERT_CANDIDATE_MAX_AGE_MINUTES))
)

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

# Model-Only Mode: When True, bypasses the sharp-book validation requirement.
# Treats the model projection as absolute truth and computes edges against the quoted lines.
# Default OFF: settled-bet calibration (model 0.4 → 16.7% realized, 0.8 → 63.2%)
# shows the model cannot be its own truth source. Only enable if no sharp-book
# odds source is available, and expect overstated edges.
MODEL_AS_TRUTH_MODE = os.getenv("MODEL_AS_TRUTH_MODE", "false").lower() in ("1", "true", "yes")
MODEL_ONLY_KELLY_MULT = float(os.getenv("MODEL_ONLY_KELLY_MULT", "0.5"))

# Pitch-type arsenal matchup multiplier (pitch_matchup.py). Applied as a bounded
# post-model adjustment to the projected mean — pitcher K vs the opposing
# lineup's per-pitch whiff, and batters vs the starter's arsenal (weighted xwoba).
# It is NOT a GLM feature: training has no per-game opposing-pitcher identity, so
# a matchup feature would be constant in train / variable at serve (skew). The
# factor isolates the matchup DEVIATION from each player's own baseline, so it
# doesn't double-count overall skill the model already prices.
PITCH_MATCHUP_ENABLED = os.getenv("PITCH_MATCHUP_ENABLED", "true").lower() in ("1", "true", "yes")
# Sensitivity exponent (<1 dampens): factor = clamp((matchup/baseline)**sens, lo, hi).
PITCH_MATCHUP_SENS = float(os.getenv("PITCH_MATCHUP_SENS", "0.5"))
PITCH_MATCHUP_MIN = float(os.getenv("PITCH_MATCHUP_MIN", "0.90"))
PITCH_MATCHUP_MAX = float(os.getenv("PITCH_MATCHUP_MAX", "1.10"))
# Per-pitch sample floor: ignore a pitch type whose count is below this (noisy).
PITCH_MATCHUP_MIN_PITCHES = int(os.getenv("PITCH_MATCHUP_MIN_PITCHES", "20"))
# Minimum share of the pitcher's arsenal (by usage) that must have opponent data
# for the factor to be trusted; below this the matchup returns neutral 1.0.
PITCH_MATCHUP_MIN_COVERAGE = float(os.getenv("PITCH_MATCHUP_MIN_COVERAGE", "0.5"))

# Batter-vs-pitcher head-to-head multiplier (bvp.py). BvP is a notoriously weak,
# small-sample signal, so it is heavily shrunk toward the batter's own baseline
# (BVP_PRIOR_PA pseudo-PA) and tightly bounded. Default ON but conservative; set
# BVP_ENABLED=false to disable. It partially overlaps the pitch-type arsenal
# factor (both describe this batter vs this pitcher), which is why the bounds
# here are tighter than the arsenal factor's.
BVP_ENABLED = os.getenv("BVP_ENABLED", "true").lower() in ("1", "true", "yes")
BVP_PRIOR_PA = float(os.getenv("BVP_PRIOR_PA", "40"))   # shrinkage strength (pseudo-PA)
BVP_MIN_AB = int(os.getenv("BVP_MIN_AB", "10"))          # below this sample, stay neutral
BVP_MIN = float(os.getenv("BVP_MIN", "0.95"))
BVP_MAX = float(os.getenv("BVP_MAX", "1.05"))

# SportsDataIO consensus player props. Merged into the scan as an extra book
# keyed 'sportsdataio'. Requires SPORTSDATAIO_API_KEY; auto-disabled without one.
SPORTSDATAIO_ENABLED = (
    bool(SPORTSDATAIO_API_KEY)
    and os.getenv("SPORTSDATAIO_ENABLED", "true").lower() in ("1", "true", "yes")
)
# Use SDIO's devigged consensus as a FALLBACK truth anchor for a (player, market)
# only when no true sharp book (Pinnacle/Circa) quotes it — the Odds API rarely
# carries sharp MLB player props, so this recovers props that would otherwise be
# skipped. Consensus is softer than a true sharp line, so consensus-anchored bets
# are tagged edge_source='sportsdataio_consensus'. Set false to require a real
# sharp anchor always.
SPORTSDATAIO_ANCHOR_FALLBACK = os.getenv(
    "SPORTSDATAIO_ANCHOR_FALLBACK", "true"
).lower() in ("1", "true", "yes")

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

# Market+side combinations to never bet on.
# Populated from diagnostic analysis. Format: (market, side) tuples.
MARKET_SIDE_BLACKLIST = [
    # pitcher_strikeouts OVERs: 5 bets, 40% win, -$102.05. Disabled 2026-06-28.
    # Kept until post-calibration data clears it; K projections graded biased high.
    ("pitcher_strikeouts", "over"),
    # batter_home_runs OVER ban removed 2026-07-06: n=2 is noise, and the real
    # longshot leak is now handled structurally by MAX_ODDS.
]

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
# Strength of moneyline favorite-tilt on the implied team total. The team's
# share of the game total = 0.5 + K*(win_prob - 0.5), bounded to [0.40, 0.60].
# 0.5 keeps it mild (a 65% favorite gets ~57.5% of runs). 0 = symmetric split.
PA_MONEYLINE_TILT_K = float(os.getenv("PA_MONEYLINE_TILT_K", "0.5"))

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

# --- Game markets (moneyline / total / run line) ---
# Sourced from BDL game odds (free, soft vendors only — NO sharp anchor), so
# these are MODEL-DRIVEN: a team Poisson run model vs the best BDL vendor's
# devigged price. Because there's no sharp validation, game bets get a higher
# edge bar and a Kelly haircut, and require confirmed starting pitchers.
GAME_MARKETS_ENABLED = os.getenv("GAME_MARKETS_ENABLED", "false").lower() in ("1", "true", "yes")
# Minimum model-vs-book edge (in %) to flag a game-market bet. Higher than the
# prop bar (EDGE_MIN) because there's no sharp anchor to validate the model.
GAME_EDGE_MIN = float(os.getenv("GAME_EDGE_MIN", "3.5"))
# Effective Kelly = KELLY_FRACTION * this. Halved by default: a model-driven
# edge with no sharp confirmation deserves a smaller stake.
GAME_KELLY_FRACTION_MULT = float(os.getenv("GAME_KELLY_FRACTION_MULT", "0.5"))
GAME_MAX_BETS_PER_GAME = int(os.getenv("GAME_MAX_BETS_PER_GAME", "2"))
# Standard MLB run line.
RUN_LINE_VALUE = float(os.getenv("RUN_LINE_VALUE", "1.5"))
# Guard: the model's per-team expected runs (lambda) may deviate from the book's
# implied team total by at most this many runs. Keeps a broken projection from
# manufacturing a huge phantom edge against the market.
GAME_LAMBDA_GUARD_RUNS = float(os.getenv("GAME_LAMBDA_GUARD_RUNS", "1.5"))
# Home team's share of the extra-innings (tie) probability mass. MLB home teams
# win ~52% of games that reach a tie at the end of 9.
GAME_HOME_TIE_SPLIT = float(os.getenv("GAME_HOME_TIE_SPLIT", "0.52"))
# Require both probable starters confirmed before projecting a game (the run
# model leans heavily on starter quality). Off => fall back to team-only lambda.
GAME_REQUIRE_CONFIRMED_STARTERS = os.getenv("GAME_REQUIRE_CONFIRMED_STARTERS", "true").lower() in ("1", "true", "yes")
# send_game_alerts only consumes game_bet_candidates younger than this.
GAME_CANDIDATE_MAX_AGE_MINUTES = int(os.getenv("GAME_CANDIDATE_MAX_AGE_MINUTES", "15"))

# Fail fast at import time if critical keys are missing rather than crashing
# deep inside an HTTP call with a confusing error.
_REQUIRED_ENV_VARS = ["ODDS_API_KEY", "BDL_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
_missing = [k for k in _REQUIRED_ENV_VARS if not globals().get(k)]
if _missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Set them in your .env file or shell before running."
    )
