import pandas as pd
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def leveraged_timing_strategy():
    """
    00631L (Leveraged 2X) Timing Strategy.
    Holds 00631L when 0050 is above SMA 120, elsestays in cash.
    Returns the FinLab backtest report object.
    """
    adj_close = data.get('etl:adj_close')
    
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    
    SMA_PERIOD = 120
    
    # 1. Timing Signal: Long 00631L when 0050 > SMA 120
    long_signal = (p0050 > p0050.rolling(SMA_PERIOD).mean())
    
    # Define Position: 1.0 (100% in 00631L) when signal is True, else 0 (Cash)
    # Using a DataFrame with Boolean values for sim()
    pos = pd.DataFrame({'00631L': long_signal}, index=long_signal.index)
    
    # 2. Run Backtest
    # We filter for dates where 00631L exists to avoid NaN issues at start
    common_start = p00631L.dropna().index[0]
    report = sim(pos.loc[common_start:], upload=False)
    
    return report

if __name__ == "__main__":
    report = leveraged_timing_strategy()
    
    # Display summary
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    print(f"\n[00631L Leveraged Timing Report]")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
