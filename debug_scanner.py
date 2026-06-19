import asyncio
import sqlite3
import pandas as pd
from src.clients.odds_api import OddsAPIClient
from src.models.projections import ProjectionModel
from src.models.devig import devig_multiplicative
from src.config import MARKETS_MAPPING, SHARP_BOOKMAKERS, SHARP_MODEL_AGREEMENT_TOL
from src.pipelines.scan_props import (
    _parse_odds_by_player, _group_by_player_market,
    _pick_anchor_line, _build_projection, _get_ump_k_factor, _parse_game_total
)
from src.data.db import get_db_connection
from src.data.park_factors import get_stadium_meta

async def debug_scan():
    conn = sqlite3.connect('data.db')
    conn.row_factory = sqlite3.Row
    # Get just one active game
    game = conn.execute("SELECT * FROM games WHERE status != 'COMPLETED' LIMIT 1").fetchone()
    if not game:
        print("No active games")
        return
    
    print(f"Scanning game: {game['away_team']} @ {game['home_team']}")
    
    odds_client = OddsAPIClient()
    proj_model = ProjectionModel()
    
    # We ignore weather and ump_k_factor for this quick debug
    weather = None
    ump_k_factor = 1.0
    
    api_markets = list(MARKETS_MAPPING.keys()) + ['totals']
    event_odds = await odds_client.get_event_odds(game['game_id'], api_markets, bust_cache=True)
    if not event_odds:
        print("No odds returned")
        return
        
    game_total = _parse_game_total(event_odds)
    player_lines = _parse_odds_by_player(event_odds)
    player_market_groups = _group_by_player_market(player_lines)
    
    print(f"Found {len(player_market_groups)} player/market combos")
    
    reasons = []
    
    for (player_name, market_key), lines_for_market in player_market_groups.items():
        anchor = _pick_anchor_line(lines_for_market, SHARP_BOOKMAKERS)
        if not anchor:
            reasons.append("No sharp anchor")
            continue
            
        anchor_line, sharp_over, sharp_under, sharp_book = anchor
        sharp_prob_over, sharp_prob_under = devig_multiplicative(sharp_over, sharp_under)
        if sharp_prob_over is None:
            reasons.append("Devig failed")
            continue
            
        proj = _build_projection(
            proj_model, player_name, market_key, anchor_line,
            game['game_id'], game['home_team'], game['away_team'], game['venue'],
            weather=weather, ump_k_factor=ump_k_factor, game_total=game_total, bdl_market=None
        )
        
        if not proj:
            reasons.append("No projection (missing stats/injury)")
            continue
            
        diff = abs(proj['prob_over'] - sharp_prob_over)
        if diff > SHARP_MODEL_AGREEMENT_TOL:
            reasons.append(f"Model disagreement (diff={diff:.3f}, model={proj['prob_over']:.3f}, sharp={sharp_prob_over:.3f})")
            continue
            
        reasons.append("Passed agreement gate")
        
    import collections
    print("Summary of what happened to props:")
    for k, v in collections.Counter(reasons).items():
        print(f"  {k}: {v}")
        
    for r in reasons:
        if "Model disagreement" in r:
            print(f"  Example: {r}")
            break

if __name__ == "__main__":
    asyncio.run(debug_scan())
