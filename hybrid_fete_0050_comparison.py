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

def analyze_hybrid_0050_impact():
    print("Fetching Data for 5-Year Impact Analysis (2021-2026)...")
    signals_eco = data.get('tw_business_indicators:景氣對策信號(分)')
    score = signals_eco['tw_business_indicators']
    
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    
    START_DATE = '2021-04-01'
    dates = p0050.loc[START_DATE:].index
    
    # 1. Generate F.E.T.E. Logic Signals
    ema_60 = p0050.rolling(60).mean()
    fund_score_daily = score.reindex(dates, method='ffill').fillna(38)
    p0050_daily = p0050.reindex(dates)
    ema_60_daily = ema_60.reindex(dates)
    
    hybrid_signal = pd.Series(index=dates, dtype=bool)
    current_state = False 
    
    for d in dates:
        # ENTRY: Fund Blue OR Tech Recovery
        if (fund_score_daily.loc[d] <= 22) or (p0050_daily.loc[d] > ema_60_daily.loc[d] and fund_score_daily.loc[d] < 34):
            current_state = True
        # EXIT: Tech Escape OR Fund Overheat
        if current_state == True:
            if p0050_daily.loc[d] < ema_60_daily.loc[d] or fund_score_daily.loc[d] >= 38:
                current_state = False
        hybrid_signal.loc[d] = current_state
        
    # 2. Backtests
    print("\nRunning Backtest: F.E.T.E. on 00631L (Standard)...")
    pos_631L = pd.DataFrame({'00631L': hybrid_signal}, index=dates)
    report_631L = sim(pos_631L, upload=False)
    
    print("Running Backtest: F.E.T.E. on 0050 (Low Leverage)...")
    pos_0050 = pd.DataFrame({'0050': hybrid_signal}, index=dates)
    report_0050_timing = sim(pos_0050, upload=False)
    
    # Baseline
    print("Running Backtest: 0050 Buy & Hold...")
    report_0050_bh = sim(pd.DataFrame({'0050': [True]*len(dates)}, index=dates), upload=False)
    
    # 3. Results Summary
    print("\n--- TIMING IMPACT COMPARISON (2021-2026) ---")
    generate_return_report(report_631L, "Hybrid F.E.T.E. (00631L)")
    generate_return_report(report_0050_timing, "Hybrid F.E.T.E. (0050)")
    generate_return_report(report_0050_bh, "0050 Buy & Hold")
    
    # 4. Plotting
    plt.figure(figsize=(14, 10))
    (report_631L.creturn / report_631L.creturn.iloc[0]).plot(label='F.E.T.E. + 00631L', color='crimson', linewidth=3)
    (report_0050_timing.creturn / report_0050_timing.creturn.iloc[0]).plot(label='F.E.T.E. + 0050', color='blue', linewidth=2)
    (report_0050_bh.creturn / report_0050_bh.creturn.iloc[0]).plot(label='0050 B&H', color='black', alpha=0.3, linestyle='--')
    
    plt.title('Performance Contribution: Timing vs Leverage (2021-2026)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('timing_vs_leverage_plot.png')
    print("\nVisual saved to timing_vs_leverage_plot.png")

if __name__ == "__main__":
    analyze_hybrid_0050_impact()
