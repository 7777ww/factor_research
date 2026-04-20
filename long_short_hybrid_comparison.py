import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def generate_return_report(report, name="Strategy"):
    """
    Generates a summarized return report.
    """
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    
    print(f"\n[{name}]")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
    return {"name": name, "ar": ar, "sr": sr, "md": md}

def analyze_long_short_hybrid():
    print("Fetching Fundamental, Technical, and Inverse ETF Data...")
    signals = data.get('tw_business_indicators:景氣對策信號(分)')
    score = signals['tw_business_indicators']
    
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    p00632R = adj_close['00632R']
    
    # 1. Indicators
    ema_60 = p0050.rolling(60).mean()
    fund_buy_zone = (score <= 22)
    
    # 2. Hybrid Logic (F.E.T.E. Sentinel)
    common_start = p00631L.dropna().index[0]
    dates = p00631L.loc[common_start:].index
    
    fund_score_daily = score.reindex(dates, method='ffill').fillna(38)
    p0050_daily = p0050.reindex(dates)
    ema_60_daily = ema_60.reindex(dates)
    
    long_signal = pd.Series(index=dates, dtype=bool)
    current_state = False 
    
    for d in dates:
        if (fund_score_daily.loc[d] <= 22) or (p0050_daily.loc[d] > ema_60_daily.loc[d] and fund_score_daily.loc[d] < 34):
            current_state = True
        if current_state == True:
            if p0050_daily.loc[d] < ema_60_daily.loc[d] or fund_score_daily.loc[d] >= 38:
                current_state = False
        long_signal.loc[d] = current_state
        
    # Short Signal: Inverse of Long Signal (Except when starting)
    short_signal = ~long_signal
    
    # 3. Position DataFrames
    # Scenario A: Long 00631L, else Cash
    pos_long_only = pd.DataFrame({'00631L': long_signal}, index=dates)
    
    # Scenario B: Long 00631L, else Long 00632R (Shorting 0050)
    pos_long_short = pd.DataFrame({
        '00631L': long_signal,
        '00632R': short_signal
    }, index=dates)
    
    # 4. Backtests
    print("\nRunning Backtest: Long-Only Hybrid (F.E.T.E.)...")
    report_long = sim(pos_long_only, upload=False)
    
    print("Running Backtest: Long-Short Hybrid (F.E.T.E. + 00632R)...")
    report_ls = sim(pos_long_short, upload=False)
    
    # Benchmark
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(dates)}, index=dates), upload=False)
    
    # Results Summary
    print("\n--- LONG-SHORT HYBRID COMPARISON RESULTS ---")
    generate_return_report(report_long, "Long-Only (Cash)")
    generate_return_report(report_ls, "Long-Short (00632R)")
    generate_return_report(report_0050, "0050 Benchmark")
    
    # 5. Plotting
    plt.figure(figsize=(14, 10))
    (report_long.creturn / report_long.creturn.iloc[0]).plot(label='Long-Only (Hold Cash during Exits)', color='crimson', linewidth=3)
    (report_ls.creturn / report_ls.creturn.iloc[0]).plot(label='Long-Short (Buy 00632R during Exits)', color='green', linewidth=2)
    (report_0050.creturn / report_0050.creturn.iloc[0]).plot(label='0050 B&H', color='black', alpha=0.3, linestyle='--')
    
    plt.title('Long-Only vs Long-Short Hybrid Strategy Migration')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('long_short_hybrid_plot.png')
    print("\nComparison visual saved to long_short_hybrid_plot.png")

if __name__ == "__main__":
    analyze_long_short_hybrid()
