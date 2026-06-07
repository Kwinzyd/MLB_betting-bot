import sqlite3
import pandas as pd

db_path = r'c:\Users\louis\nba-props-in\nba-prop-bot\props.db'
conn = sqlite3.connect(db_path)

query = """
SELECT a.market, b.won, a.odds as alert_odds, a.edge
FROM alerts_sent a
JOIN bet_results b ON a.id = b.alert_id
"""
df = pd.read_sql_query(query, conn)

if not df.empty:
    def calc_roi(group):
        units_won = 0.0
        for _, row in group.iterrows():
            odds = row['alert_odds'] if row['alert_odds'] else 2.0
            if row['won']:
                units_won += (odds - 1.0)
            else:
                units_won -= 1.0
        return units_won / len(group)

    stats = df.groupby('market').apply(lambda x: pd.Series({
        'count': len(x),
        'win_rate': x['won'].mean(),
        'roi': calc_roi(x),
        'avg_edge': x['edge'].mean()
    })).reset_index()
    
    print("\n--- Performance by Market ---")
    print(stats.sort_values('roi', ascending=False).to_string(index=False))
else:
    print("No data.")

conn.close()
