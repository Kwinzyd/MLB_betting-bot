import sqlite3
import pandas as pd

db_path = r'c:\Users\louis\nba-props-in\nba-prop-bot\props.db'
conn = sqlite3.connect(db_path)

print("--- Model Health (Recent Snapshots) ---")
try:
    health = pd.read_sql("SELECT snapshot_date, window, market, brier, n_samples FROM model_health ORDER BY recorded_at DESC LIMIT 20", conn)
    print(health.to_string())
except Exception as e:
    print("Could not load model_health:", e)

print("\n--- Backtest Results Summary ---")
try:
    df = pd.read_sql("SELECT hit, edge, simulated_line, model_prob_over, run_timestamp FROM backtest_results ORDER BY run_timestamp DESC", conn)
    if not df.empty:
        print(f"Total Backtest Samples: {len(df)}")
        print(f"Overall Hit Rate: {df['hit'].mean():.2%}")
        
        # Check if we have positive edge plays (bets bot would have made)
        # Assuming the bot takes bets with edge > 0.05
        bets = df[df['edge'] >= 0.05]
        print(f"\nHits on Positive Edge (>= 5%):")
        print(f"Count: {len(bets)}")
        if len(bets) > 0:
            print(f"Win Rate: {bets['hit'].mean():.2%}")
            
        print("\nRecent 100 bets performance:")
        recent = df.head(100)
        recent_bets = recent[recent['edge'] >= 0.05]
        print(f"Count: {len(recent_bets)}")
        if len(recent_bets) > 0:
            print(f"Win Rate: {recent_bets['hit'].mean():.2%}")
            
    else:
        print("No backtest results found.")
except Exception as e:
    print("Could not load backtest_results:", e)

print("\n--- Placed Bets Summary ---")
try:
    bets = pd.read_sql("SELECT mode, stake, fill_odds, alerted_odds FROM placed_bets", conn)
    if not bets.empty:
        print(bets.groupby('mode').agg({'stake': ['count', 'sum'], 'fill_odds': 'mean'}))
    else:
        print("No placed bets found.")
except Exception as e:
    print("Could not load placed_bets:", e)


conn.close()
