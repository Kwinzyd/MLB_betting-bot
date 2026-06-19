import asyncio
import sqlite3
import pandas as pd
from src.clients.odds_api import OddsAPIClient
from src.models.projections import ProjectionModel
from src.models.devig import devig_multiplicative
from src.models.edge_ranker import rank_edge
from src.config import MARKETS_MAPPING, SHARP_BOOKMAKERS, SHARP_MODEL_AGREEMENT_TOL, ALT_LINE_MAX_DISTANCE
from src.pipelines.scan_props import (
    _parse_odds_by_player, _group_by_player_market,
    _pick_anchor_line, _build_projection, _pick_best_soft_line, _parse_game_total
)

async def debug_scan():
    conn = sqlite3.connect('data.db')
    conn.row_factory = sqlite3.Row
    game = conn.execute("SELECT * FROM games WHERE status != 'COMPLETED' LIMIT 1").fetchone()
    
    odds_client = OddsAPIClient()
    proj_model = ProjectionModel()
    api_markets = list(MARKETS_MAPPING.keys()) + ['totals']
    event_odds = await odds_client.get_event_odds(game['game_id'], api_markets, bust_cache=True)
    
    game_total = _parse_game_total(event_odds)
    player_lines = _parse_odds_by_player(event_odds)
    player_market_groups = _group_by_player_market(player_lines)
    
    print("Checking why passed-gate props aren't edges...")
    for (player_name, market_key), lines_for_market in player_market_groups.items():
        anchor = _pick_anchor_line(lines_for_market, SHARP_BOOKMAKERS)
        if not anchor: continue
        anchor_line, sharp_over, sharp_under, sharp_book = anchor
        sharp_prob_over, sharp_prob_under = devig_multiplicative(sharp_over, sharp_under)
        if sharp_prob_over is None: continue
            
        proj = _build_projection(
            proj_model, player_name, market_key, anchor_line,
            game['game_id'], game['home_team'], game['away_team'], game['venue'],
            game_total=game_total
        )
        if not proj: continue
            
        diff = abs(proj['prob_over'] - sharp_prob_over)
        if diff > SHARP_MODEL_AGREEMENT_TOL: continue
        
        # Passed agreement gate! Let's check alt-lines
        for line, line_data in lines_for_market.items():
            if abs(line - anchor_line) > ALT_LINE_MAX_DISTANCE: continue
            
            soft_best = _pick_best_soft_line(line_data, SHARP_BOOKMAKERS)
            for side in ['over', 'under']:
                odds_val, book = soft_best[side]
                if not odds_val or odds_val <= 1.0: continue
                
                truth_prob = sharp_prob_over if side == 'over' else sharp_prob_under
                
                # Mock get_probabilities since we don't have the distribution imports handy
                line_proj = dict(proj)
                line_proj['prob_over'] = proj['prob_over']
                line_proj['prob_under'] = proj['prob_under']
                
                edge_result = rank_edge(line_proj, odds_val, side, truth_prob)
                if not edge_result['is_playable']:
                    print(f"KILLED {player_name} {market_key} {side} @ {odds_val} {book}: {edge_result['reasons']}")
                else:
                    print(f"PLAYABLE EDGE! {player_name} {market_key} {side} @ {odds_val} {book}")

if __name__ == "__main__":
    asyncio.run(debug_scan())
