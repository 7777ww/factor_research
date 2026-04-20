import pandas as pd
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def ema_timing_strategy():
    """
    00631L (Leveraged 2X) EMA Cross Timing Strategy.
    Logic: Buy 00631L when EMA 20 > EMA 60 of 0050 Index, else Cash.
    Returns the FinLab backtest report object.
    """
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    
    # 1. Calculate EMAs for the Signal
    ema_20 = p0050.ewm(span=20, adjust=False).mean()
    ema_60 = p0050.ewm(span=60, adjust=False).mean()
    
    # 2. Define Signal: EMA 20 Cross Above EMA 60
    long_signal = (ema_20 > ema_60)
    
    # 3. Define Position for 00631L
    pos = pd.DataFrame({'00631L': long_signal}, index=long_signal.index)
    
    # 4. Filter for valid 00631L trading start date
    p00631L = adj_close['00631L']
    common_start = p00631L.dropna().index[0]
    
    # 5. Run Backtest
    report = sim(pos.loc[common_start:], upload=False)
    
    return report

if __name__ == "__main__":
    report = ema_timing_strategy()
    
    # Summary of winning metrics
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    print(f"\n--- EMA CROSS 20/60 TIMING REPORT ---")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
