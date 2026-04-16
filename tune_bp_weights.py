import numpy as np
from src.backtesting.engine import BacktestEngine
from src.models.projections import ProjectionModel

def brier_score(records):
    valid = [r for r in records if r.actual_over is not None]
    if not valid:
        return None, 0
    errs = []
    for r in valid:
        errs.append((r.model_prob_over - int(r.actual_over))**2)
    return np.mean(errs), len(valid)

def run_tuning():
    weights = [0.05, 0.10, 0.15, 0.20, 0.25]
    start_date = "2025-04-01"
    end_date = "2025-09-30"

    print("--- Tuning Batter bp_weight ---")
    best_b_weight = None
    best_b_brier = 999.0

    for w in weights:
        class TunedBatterModel(ProjectionModel):
            def project_batter_stat(self, logs, stat_type, pitcher_hand, batter_hand,
                                    venue, line, lineup_position=None, weather=None, 
                                    extra_features=None, player_id=None, game_total=None, bp_weight=w):
                                    
                return super().project_batter_stat(
                    logs, stat_type, pitcher_hand, batter_hand, venue, line, 
                    lineup_position, weather, extra_features, player_id, game_total, bp_weight=w
                )

        engine = BacktestEngine(start_date, end_date, markets=['batter_hits', 'batter_total_bases', 'batter_home_runs'])
        engine._proj = TunedBatterModel()
        records = engine.run()
        
        brier, n = brier_score(records)
        if brier is not None:
            print(f"Batter bp_weight={w:.2f} -> Brier = {brier:.5f} (N={n})")
            if brier < best_b_brier:
                best_b_brier = brier
                best_b_weight = w

    print(f"\nOptimal Batter bp_weight: {best_b_weight} with Brier {best_b_brier:.5f}\n")

    print("--- Tuning Pitcher ER bp_weight ---")
    best_p_weight = None
    best_p_brier = 999.0

    for w in weights:
        class TunedPitcherModel(ProjectionModel):
            def project_pitcher_earned_runs(self, logs, opponent_runs_pg, venue, line,
                                            weather=None, ump_k_factor=1.0, extra_features=None, 
                                            player_id=None, bp_weight=w):
                                            
                return super().project_pitcher_earned_runs(
                    logs, opponent_runs_pg, venue, line, weather, ump_k_factor, extra_features, 
                    player_id, bp_weight=w
                )

        engine = BacktestEngine(start_date, end_date, markets=['pitcher_earned_runs'])
        engine._proj = TunedPitcherModel()
        records = engine.run()
        
        brier, n = brier_score(records)
        if brier is not None:
            print(f"Pitcher bp_weight={w:.2f} -> Brier = {brier:.5f} (N={n})")
            if brier < best_p_brier:
                best_p_brier = brier
                best_p_weight = w

    print(f"\nOptimal Pitcher ER bp_weight: {best_p_weight} with Brier {best_p_brier:.5f}\n")

if __name__ == '__main__':
    run_tuning()
