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
    Generates a summarized return report for comparison.
    """
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    
    print(f"\n[{name}]")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
    return {"name": name, "ar": ar, "sr": sr, "md": md}

def analyze_fundamental_timing():
    print("Fetching Fundamental Data (Economic Signals & P/E)...")
    
    # 1. Economic Outlook Signal (景氣對策信號)
    # Buy on Blue/Yellow-Blue, Sell on Red/Yellow-Red
    signals = data.get('tw_business_indicators:景氣對策信號(分)')
    score = signals['tw_business_indicators']
    
    # Fundamental Signal Logic
    # Buy: score <= 18 (Blue/Yellow-Blue)
    # Sell: score >= 38 (Red/Yellow-Red)
    # Note: These are monthly signals, so we need to carry forward
    is_buy_zone = (score <= 22) # More aggressive "Bottom"
    is_sell_zone = (score >= 38)
    
    # Current Position Logic:
    # 1 when Buy triggered, 0 when Sell triggered (Path dependent)
    # For sim(), we'll create a stateful signal
    fund_signal = pd.Series(index=score.index, dtype=bool)
    current_state = False
    for d in score.index:
        if score.loc[d] <= 22:
            current_state = True
        elif score.loc[d] >= 38:
            current_state = False
        fund_signal.loc[d] = current_state
    
    # Reindex to Daily for mapping to 00631L
    adj_close = data.get('etl:adj_close')
    p00631L = adj_close['00631L']
    fund_signal_daily = fund_signal.reindex(p00631L.index, method='ffill').fillna(False)
    
    # 2. Market P/E Comparison (Median of Top 50)
    print("Calculating Market P/E Signal...")
    pe = data.get('price_earning_ratio:本益比')
    market_value = data.get('etl:market_value')
    top_50 = market_value.is_largest(50)
    market_pe = pe[top_50].median(axis=1) # Aggregate P/E proxy
    
    # Buy when PE < 15, Sell when PE > 22 (Taiwan standard)
    pe_signal = pd.Series(index=market_pe.index, dtype=bool)
    pe_state = False
    for d in market_pe.index:
        if market_pe.loc[d] < 15:
            pe_state = True
        elif market_pe.loc[d] > 22:
            pe_state = False
        pe_signal.loc[d] = pe_state
    
    pe_signal_daily = pe_signal.reindex(p00631L.index, method='ffill').fillna(False)
    
    # 3. Baseline: EMA 20/60 Timing (Technical)
    p0050 = adj_close['0050']
    ema_20 = p0050.ewm(span=20, adjust=False).mean()
    ema_60 = p0050.ewm(span=60, adjust=False).mean()
    ema_signal = (ema_20 > ema_60)
    
    # Comparison
    signals_map = {
        "Fundamental (Economic Signal)": fund_signal_daily,
        "Fundamental (Market P/E)": pe_signal_daily,
        "Technical (EMA 20/60)": ema_signal
    }
    
    common_start = p00631L.dropna().index[0]
    plot_data = {}
    
    print("\nStarting Fundamental Backtests...")
    for name, sig in signals_map.items():
        pos = pd.DataFrame({'00631L': sig}, index=sig.index)
        report = sim(pos.loc[common_start:], upload=False)
        generate_return_report(report, name)
        plot_data[name] = report.creturn / report.creturn.iloc[0]
        
    # 0050 Benchmark
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(p0050)}, index=p0050.index).loc[common_start:], upload=False)
    
    # Plotting
    plt.figure(figsize=(14, 10))
    for name, curve in plot_data.items():
        curve.plot(label=name, linewidth=2)
    
    (report_0050.creturn / report_0050.creturn.iloc[0]).plot(label='0050 B&H', color='black', alpha=0.5, linestyle='--')
    
    plt.title('Fundamental vs Technical Timing for 00631L')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('fundamental_timing_plot.png')
    print("\nVisual saved to fundamental_timing_plot.png")

if __name__ == "__main__":
    analyze_fundamental_timing()
