import pandas as pd
import numpy as np
from finlab import data
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def analyze_timing_events():
    print("Fetching data for Timing Victory Analysis...")
    signals_eco = data.get('tw_business_indicators:景氣對策信號(分)')
    score = signals_eco['tw_business_indicators']
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    
    sma_60 = p0050.rolling(60).mean()
    common_start = p00631L.dropna().index[0]
    dates = p00631L.loc[common_start:].index
    
    fund_score_daily = score.reindex(dates, method='ffill').fillna(38)
    p0050_daily = p0050.reindex(dates)
    sma_60_daily = sma_60.reindex(dates)
    
    # 1. Re-run Hybrid Signal Generation
    hybrid_signal = pd.Series(index=dates, dtype=bool)
    current_state = False 
    for d in dates:
        if (fund_score_daily.loc[d] <= 22) or (p0050_daily.loc[d] > sma_60_daily.loc[d] and fund_score_daily.loc[d] < 34):
            current_state = True
        if current_state == True:
            if p0050_daily.loc[d] < sma_60_daily.loc[d] or fund_score_daily.loc[d] >= 38:
                current_state = False
        hybrid_signal.loc[d] = current_state
        
    # 2. Identify Events
    # Entry: False -> True
    # Exit: True -> False
    events = []
    prev_state = False
    for i in range(1, len(hybrid_signal)):
        curr_state = hybrid_signal.iloc[i]
        d = hybrid_signal.index[i]
        if curr_state and not prev_state:
            events.append({'type': 'ENTRY', 'date': d})
        elif not curr_state and prev_state:
            events.append({'type': 'EXIT', 'date': d})
        prev_state = curr_state
        
    # 3. Analyze each Period
    print("\n--- TIMING EVENT ANALYSIS ---")
    
    successful_escapes = 0 # Avoided > 5% drop while in cash
    missed_rallies = 0     # Missed > 5% rise while in cash
    false_alarms = 0       # Exit but market stayed flat or rose
    
    for i, event in enumerate(events):
        if event['type'] == 'EXIT':
            exit_date = event['date']
            # Find next ENTRY
            next_entry = events[i+1]['date'] if i+1 < len(events) else dates[-1]
            
            # Market return while we were in CASH
            market_price_exit = p0050_daily.loc[exit_date]
            market_price_reentry = p0050_daily.loc[next_entry]
            
            cash_period_return = (market_price_reentry / market_price_exit - 1) * 100
            
            print(f"EXIT on {exit_date.date()} -> Re-entry on {next_entry.date()} | Market Return: {cash_period_return:6.2f}%")
            
            if cash_period_return < -5:
                successful_escapes += 1
                print("  => SUCCESSFUL ESCAPE (Avoided significant drop)")
            elif cash_period_return > 5:
                missed_rallies += 1
                print("  => MISSED RALLY (Incorrectly stayed out during rise)")
            else:
                false_alarms += 1
                print("  => FALSE ALARM (Sideways/Minor move)")
                
    print("\n--- FINAL SCOREBOARD ---")
    print(f"Total Exits: {len([e for e in events if e['type'] == 'EXIT'])}")
    print(f"Successful Escapes (Avoided > 5% Fall): {successful_escapes}")
    print(f"Missed Rallies (Lost > 5% Gain):       {missed_rallies}")
    print(f"False Alarms / Sideways:               {false_alarms}")

if __name__ == "__main__":
    analyze_timing_events()
