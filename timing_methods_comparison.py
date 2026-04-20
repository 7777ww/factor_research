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
    Generates a summarized return report in the format requested by the user.
    """
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    
    # Also get monthly sharpe and CAGR from stats for consistency with other reports
    stats = report.get_stats()
    
    print(f"\n[{name}]")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
    return {"name": name, "ar": ar, "sr": sr, "md": md}

def analyze_timing_methods():
    print("Fetching data for Timing Methods Comparison...")
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    
    # 1. Indicator Calculations
    # A. SMA 100
    sma_100 = p0050.rolling(100).mean()
    
    # B. EMA Cross (20/60)
    ema_20 = p0050.ewm(span=20, adjust=False).mean()
    ema_60 = p0050.ewm(span=60, adjust=False).mean()
    
    # C. MACD (12, 26, 9)
    exp12 = p0050.ewm(span=12, adjust=False).mean()
    exp26 = p0050.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # 2. Strategy Signals
    signals = {
        "SMA_100": (p0050 > sma_100),
        "EMA_Cross_20_60": (ema_20 > ema_60),
        "MACD_Signal": (macd > signal)
    }
    
    comparison_results = []
    plot_data = {}
    common_start = p00631L.dropna().index[0]
    
    print("\nStarting Backtests...")
    for name, sig in signals.items():
        # Long 00631L when signal is True, else Cash
        pos = pd.DataFrame({'00631L': sig}, index=sig.index)
        report = sim(pos.loc[common_start:], upload=False)
        res = generate_return_report(report, name)
        comparison_results.append(res)
        plot_data[name] = report.creturn / report.creturn.iloc[0]
        
    # Standard 0050 Benchmark
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(p0050)}, index=p0050.index).loc[common_start:], upload=False)
    res_0050 = generate_return_report(report_0050, "0050_Buy_and_Hold")
    comparison_results.append(res_0050)
    
    # Plotting
    plt.figure(figsize=(14, 10))
    for name, curve in plot_data.items():
        curve.plot(label=name)
    
    (report_0050.creturn / report_0050.creturn.iloc[0]).plot(label='0050 B&H', color='black', alpha=0.5, linestyle='--', linewidth=2)
    
    plt.title('00631L Timing Comparison: SMA vs EMA vs MACD')
    plt.yscale('log')
    plt.ylabel('Cumulative Return (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('timing_comparison_plot.png')
    print("\nComparison plot saved to timing_comparison_plot.png")

if __name__ == "__main__":
    analyze_timing_methods()
