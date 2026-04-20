import pandas as pd
import numpy as np
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def hybrid_fete_timing_strategy():
    """
    Hybrid F.E.T.E. (Fundamental Entry, Technical Exit) Strategy for 00631L.
    Logic:
    - Entry (Signal Blue): Economic Signal <= 22 OR (Technical Recovery AND Fund Score < 34).
    - Exit (Escape the Top): Price < SMA 60 OR Economic Signal >= 38.
    Returns the FinLab backtest report object.
    """
    # 1. Fetch Data
    signals = data.get('tw_business_indicators:景氣對策信號(分)')
    score = signals['tw_business_indicators']
    
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    
    # 2. Indicators
    sma_60 = p0050.rolling(60).mean()
    
    # 3. Stateful Hybrid Logic
    common_start = p00631L.dropna().index[0]
    dates = p00631L.loc[common_start:].index
    
    fund_score_daily = score.reindex(dates, method='ffill').fillna(38)
    p0050_daily = p0050.reindex(dates)
    sma_60_daily = sma_60.reindex(dates)
    
    hybrid_signal = pd.Series(index=dates, dtype=bool)
    current_state = False 
    
    for d in dates:
        # A. ENTRY: Fundamental Blue Light OR Technical Recovery while economy is sane
        if (fund_score_daily.loc[d] <= 22) or (p0050_daily.loc[d] > sma_60_daily.loc[d] and fund_score_daily.loc[d] < 34):
            current_state = True
            
        # B. EXIT: Technical "Top Escaping" (逃頂) OR Fundamental Overheat
        if current_state == True:
            if p0050_daily.loc[d] < sma_60_daily.loc[d] or fund_score_daily.loc[d] >= 38:
                current_state = False
                
        hybrid_signal.loc[d] = current_state
        
    # 4. Define Position and Run Backtest
    pos = pd.DataFrame({'00631L': hybrid_signal}, index=dates)
    report = sim(pos, upload=False)
    
    return report

if __name__ == "__main__":
    report = hybrid_fete_timing_strategy()
    
    # Summary of metrics
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    print(f"\n--- HYBRID F.E.T.E. (TOP ESCAPE) REPORT ---")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
