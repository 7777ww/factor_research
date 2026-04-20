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

def analyze_hybrid_fete_timing():
    print("Fetching Fundamental and Technical Data...")
    signals = data.get('tw_business_indicators:景氣對策信號(分)')
    score = signals['tw_business_indicators']
    
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    
    # 1. Indicators
    # Technical Exit Indicator: EMA 60
    ema_60 = p0050.rolling(60).mean() # Baseline SMA/EMA for exit
    ema_20 = p0050.rolling(20).mean()
    
    # Fundamental Entry Indicator: Score <= 22
    fund_buy_zone = (score <= 22)
    
    # 2. Stateful Hybrid Logic (F.E.T.E.)
    # We reindex all to the highest frequency (Daily)
    common_start = p00631L.dropna().index[0]
    dates = p00631L.loc[common_start:].index
    
    # Align indicators to daily
    fund_buy_daily = fund_buy_zone.reindex(dates, method='ffill').fillna(False)
    p0050_daily = p0050.reindex(dates)
    ema_60_daily = ema_60.reindex(dates)
    ema_20_daily = ema_20.reindex(dates)
    
    hybrid_signal = pd.Series(index=dates, dtype=bool)
    current_state = False # START in Cash
    fund_score_daily = score.reindex(dates, method='ffill').fillna(38)
    
    for d in dates:
        # A. ENTRY: Fundamental Signal (Blue Light) OR Technical Recovery (Price > SMA 60)
        # We only re-enter if the economy is NOT overheated (score < 34)
        if (fund_buy_daily.loc[d] == True) or (p0050_daily.loc[d] > ema_60_daily.loc[d] and fund_score_daily.loc[d] < 34):
            current_state = True
            
        # B. EXIT: Technical "Top Escaping" (逃頂) OR Fundamental Overheat (Red Light)
        if current_state == True:
            if p0050_daily.loc[d] < ema_60_daily.loc[d] or fund_score_daily.loc[d] >= 38:
                current_state = False
                
        hybrid_signal.loc[d] = current_state
        
    # 3. Backtests
    print("\nRunning Backtest: Hybrid F.E.T.E. (60D Exit)...")
    pos_hybrid = pd.DataFrame({'00631L': hybrid_signal}, index=dates)
    report_hybrid = sim(pos_hybrid, upload=False)
    
    # Compare with pure Fundamental (Economic)
    print("Running Backtest: Pure Fundamental Comparison...")
    # Pure fundamental: 1 when <= 22, 0 when >= 38
    fund_signal = pd.Series(index=score.index, dtype=bool)
    f_state = False
    for d in score.index:
        if score.loc[d] <= 22: f_state = True
        elif score.loc[d] >= 38: f_state = False
        fund_signal.loc[d] = f_state
    pos_fund = pd.DataFrame({'00631L': fund_signal.reindex(dates, method='ffill').fillna(False)}, index=dates)
    report_fund = sim(pos_fund, upload=False)
    
    # Compare with pure Technical (EMA 60 Baseline)
    print("Running Backtest: Pure Technical Comparison...")
    pos_tech = pd.DataFrame({'00631L': (p0050 > ema_60).reindex(dates).fillna(False)}, index=dates)
    report_tech = sim(pos_tech, upload=False)
    
    # Compare with 0050
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(dates)}, index=dates), upload=False)
    
    # 4. Results
    print("\n--- HYBRID TIMING (F.E.T.E.) RESULTS ---")
    generate_return_report(report_hybrid, "Hybrid F.E.T.E. (逃頂版)")
    generate_return_report(report_fund, "Pure Fundamental")
    generate_return_report(report_tech, "Pure Technical")
    generate_return_report(report_0050, "0050 Benchmark")
    
    # 5. Plotting
    plt.figure(figsize=(14, 10))
    (report_hybrid.creturn / report_hybrid.creturn.iloc[0]).plot(label='Hybrid (F.E.T.E.)', color='crimson', linewidth=3)
    (report_fund.creturn / report_fund.creturn.iloc[0]).plot(label='Pure Fundamental', color='blue', alpha=0.5)
    (report_tech.creturn / report_tech.creturn.iloc[0]).plot(label='Pure Technical', color='green', alpha=0.5)
    (report_0050.creturn / report_0050.creturn.iloc[0]).plot(label='0050 B&H', color='black', alpha=0.3, linestyle='--')
    
    plt.title('Hybrid Market Timing: Fundamental Entry + Technical Exit (逃頂)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('hybrid_fete_timing_plot.png')
    print("\nVisual saved to hybrid_fete_timing_plot.png")

if __name__ == "__main__":
    analyze_hybrid_fete_timing()
