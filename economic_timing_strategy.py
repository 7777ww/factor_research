import pandas as pd
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def economic_timing_strategy():
    """
    00631L (Leveraged 2X) Fundamental Timing Strategy.
    Logic: 'Buy on Blue, Sell on Red' using Economic Outlook Signals (景氣燈號).
    - Buy Trigger: Score <= 22 (Blue / Yellow-Blue)
    - Sell Trigger: Score >= 38 (Red / Yellow-Red)
    Returns the FinLab backtest report object.
    """
    # 1. Fetch Economic Signals (Monthly)
    signals = data.get('tw_business_indicators:景氣對策信號(分)')
    score = signals['tw_business_indicators']
    
    # 2. Stateful Logic: Persistent Buy/Sell zones
    fund_signal = pd.Series(index=score.index, dtype=bool)
    current_state = False
    for d in score.index:
        if score.loc[d] <= 22: # Economic trough
            current_state = True
        elif score.loc[d] >= 38: # Economic peak
            current_state = False
        fund_signal.loc[d] = current_state
        
    # 3. Map to Daily for 00631L
    adj_close = data.get('etl:adj_close')
    p00631L = adj_close['00631L']
    fund_signal_daily = fund_signal.reindex(p00631L.index, method='ffill').fillna(False)
    
    # 4. Define Position
    pos = pd.DataFrame({'00631L': fund_signal_daily}, index=fund_signal_daily.index)
    
    # 5. Run Backtest
    common_start = p00631L.dropna().index[0]
    report = sim(pos.loc[common_start:], upload=False)
    
    return report

if __name__ == "__main__":
    report = economic_timing_strategy()
    
    # Summary of metrics
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    print(f"\n--- ECONOMIC SIGNAL TIMING REPORT (00631L) ---")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
