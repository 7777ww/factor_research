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
    
    print(f"\n[{name}]")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
    return {"ar": ar, "sr": sr, "md": md}

def analyze_00631L_leveraged_timing():
    print("Fetching data for 00631L Leveraged Timing Strategy...")
    adj_close = data.get('etl:adj_close')
    
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    
    SMA_PERIOD = 120 # Slower, more stable timing to avoid whipsaws
    
    # 1. Timing Signal: Long 00631L when 0050 is above SMA 120
    long_signal = (p0050 > p0050.rolling(SMA_PERIOD).mean())
    
    # Define Position: 1.0 (100% in 00631L) when signal is True, else 0 (Cash)
    pos_00631L_timing = pd.DataFrame({'00631L': long_signal.map({True: 1.0, False: 0.0})}, index=long_signal.index)
    
    # 2. Backtest Scenarios
    print("Running Backtest: 00631L Timing...")
    # Since it's a single-ETF strategy, we use manual equity calc for precision or sim() with a trick
    # sim() expects a bool-like for stock IDs.
    report_timing = sim(pos_00631L_timing.astype(bool), upload=False)
    
    print("Running Backtest: 0050 Buy & Hold...")
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(p0050)}, index=p0050.index), upload=False)
    
    print("Running Backtest: 00631L Buy & Hold...")
    report_00631L_bh = sim(pd.DataFrame({'00631L': [True]*len(p00631L)}, index=p00631L.index), upload=False)
    
    # 3. Align Start Dates (00631L started around 2014)
    common_start = p00631L.dropna().index[0]
    
    # Results Summary
    print("\n--- 00631L LEVERAGED TIMING RESULTS ---")
    generate_return_report(report_timing, "00631L Timing (60D)")
    generate_return_report(report_0050, "0050 Buy & Hold")
    generate_return_report(report_00631L_bh, "00631L Buy & Hold")
    
    # 4. Plot Equity Curves
    plt.figure(figsize=(14, 10))
    # Note: sim().creturn is the cumulative equity
    (report_timing.creturn / report_timing.creturn.loc[common_start]).plot(label='00631L Timing (60D)', color='orange', linewidth=3)
    (report_0050.creturn / report_0050.creturn.loc[common_start]).plot(label='0050 Buy & Hold', color='black', alpha=0.5, linestyle='--')
    (report_00631L_bh.creturn / report_00631L_bh.creturn.loc[common_start]).plot(label='00631L Buy & Hold', color='blue', alpha=0.3)
    
    plt.title('00631L (Leveraged 2X) Timing Strategy vs Benchmark')
    plt.yscale('log')
    plt.ylabel('Cumulative Return (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig('00631L_timing_plot.png')
    print("\nVisual saved to 00631L_timing_plot.png")

if __name__ == "__main__":
    analyze_00631L_leveraged_timing()
