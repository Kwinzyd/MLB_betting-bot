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
- `python main.py sgp` - Find/alert same-game parlays (correlated legs, one game)
- `python main.py parlay` - Find/alert cross-game 2/4/8-leg parlays (independent legs)
- `python main.py ask "<question>"` - LLM research agent over the BDL data
- `python main.py enrich` - LLM-normalize injury reports into availability signals
- `python main.py reconcile` - LLM-resolve unmatched prop player names

## LLM Layer (OpenRouter)
Optional, fail-safe qualitative layer alongside BDL. Set `OPENROUTER_API_KEY` in
`.env` to enable (also `LLM_ENABLED`, `OPENROUTER_MODEL`; cheap-fast default).
With no key the whole layer no-ops and the bot runs unchanged.
- **Hard rule:** the LLM never produces projections, probabilities, edges, or
  stakes — only structured context the deterministic quant core reads.
- LLM calls happen in async pipelines that write to DB tables
  (`player_injury_signals`, `player_name_resolutions`); the sync money-path only
  reads them. Client: `src/clients/llm.py` (circuit breaker + cache + JSON mode).
- Roles: research agent (`research_agent.py`), injury enrichment
  (`enrich_injuries.py`, conservative tightening-only gate in scan_props), alert
  rationale (`alert_rationale.py`), name reconciliation (`player_resolver.py` +
  `reconcile_names.py`).

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
