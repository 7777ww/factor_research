import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

# Bypass SSL if needed
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def analyze_0050_timing():
    print("Fetching 0050 adjusted data (including dividends)...")
    # Get adjusted price data
    close_adj = data.get('etl:adj_close')
    
    if '0050' not in close_adj.columns:
        print("Error: 0050 not found in adj_close table.")
        return
        
    p0050 = close_adj['0050']
    
    print("Calculating SMA factors...")
    sma20 = p0050.rolling(20).mean()
    sma60 = p0050.rolling(60).mean()
    
    # Define Strategy Signals (Shift 1 to avoid lookahead bias)
    pd.set_option('future.no_silent_downcasting', True)
    signal_sma60 = (p0050 > sma60).shift(1).fillna(False)
    signal_cross = (sma20 > sma60).shift(1).fillna(False)
    
    # Create position DataFrames
    position_sma60 = pd.DataFrame({'0050': signal_sma60})
    position_cross = pd.DataFrame({'0050': signal_cross})
    
    print("Running backtest for SMA 60 Timing...")
    report_sma60 = sim(position_sma60, resample="D", upload=False)
    
    print("\nRunning backtest for SMA 20/60 Cross Timing...")
    report_cross = sim(position_cross, resample="D", upload=False)
    
    # Calculate Benchmark (Buy & Hold 0050)
    # We can create a 100% position for 0050
    position_bh = pd.DataFrame({'0050': [True] * len(p0050)}, index=p0050.index)
    print("\nRunning backtest for 0050 Buy & Hold...")
    report_bh = sim(position_bh, resample="D", upload=False)

    # Metrics
    def print_report_stats(name, report):
        stats = report.get_stats()
        print(f"\n--- {name} Metrics ---")
        print(f"CAGR: {stats['cagr']:.2%}")
        print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {stats['monthly_sharpe']:.2f}")

    print_report_stats("SMA 60 Timing", report_sma60)
    print_report_stats("SMA 20/60 Cross", report_cross)
    print_report_stats("0050 Buy & Hold", report_bh)

    # Plotting
    plt.figure(figsize=(12, 8))
    report_sma60.creturn.plot(label=f'SMA 60 Timing (MDD: {report_sma60.get_stats()["max_drawdown"]:.1%})', color='blue')
    report_cross.creturn.plot(label=f'SMA 20/60 Cross (MDD: {report_cross.get_stats()["max_drawdown"]:.1%})', color='green')
    report_bh.creturn.plot(label=f'0050 Buy & Hold (MDD: {report_bh.get_stats()["max_drawdown"]:.1%})', color='gray', alpha=0.5, linestyle='--')
    
    plt.title('0050 ETF Timing Strategy Comparison (Adjusted Price)')
    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plot_path = '0050_timing_plot.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    analyze_0050_timing()
