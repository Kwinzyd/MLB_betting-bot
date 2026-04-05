# MLB Prop Betting Bot

## Tech Stack
- Python 3.10+
- SQLite (data.db)
- The Odds API ($30/month, sport key: `baseball_mlb`)
- BallDontLie API (Goat Tier, base: `https://api.balldontlie.io/mlb/v1`)
- Telegram for alerts

## Architecture
Modular pipeline architecture mirroring the NBA prop bot pattern:
- `src/clients/` - API clients (Odds API, BallDontLie, CBS injuries, Telegram)
- `src/data/` - SQLite schema, DB helpers, cache, park factors
- `src/models/` - Projections, distributions, devig, Kelly criterion, edge ranking
- `src/pipelines/` - sync_events, sync_injuries, sync_stats, scan_props, send_alerts, settle_results
- `src/utils/` - Logging, retry, time, validators

## CLI Commands
- `python main.py sync` - Pull games, injuries, and stats
- `python main.py scan` - Scan live odds and find edges
- `python main.py run` - Full pipeline (sync + scan + alerts)
- `python main.py settle` - Post-game settlement with CLV tracking

## Key Markets
pitcher_strikeouts, batter_total_bases, batter_hits, batter_home_runs, pitcher_earned_runs

## Projection Model
- Pitchers: weighted blend of L10 K/9 + season K/9, adjusted for opponent K%, park factor
- Batters: weighted blend of L15 + season rates, adjusted for platoon splits (L/R), park factor
- Distributions: Poisson for discrete counts (K, H, HR, ER), Normal for total bases
- Edge = model_prob - book_implied, minimum 5% threshold
- Bet sizing: Fractional Kelly (default 1/4 Kelly)

## API Quota
~1500 requests/month on $30 plan. Batch all 5 markets per odds call. Cache events 1hr, odds 5min, stats 6hr.
