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

def analyze_0050_shorting():
    print("Fetching adjusted data for 0050 and 00632R (50反1)...")
    # Get adjusted price data
    close_adj = data.get('etl:adj_close')
    
    if '0050' not in close_adj.columns or '00632R' not in close_adj.columns:
        print("Error: 0050 or 00632R not found in adj_close table.")
        return
        
    p0050 = close_adj['00632R'].notnull() & close_adj['0050'] # Align indices
    p0050 = close_adj['0050']
    p00632R = close_adj['00632R']
    
    print("Calculating SMA signals...")
    sma20 = p0050.rolling(20).mean()
    sma60 = p0050.rolling(60).mean()
    
    # Define primary signal: SMA 20/60 Cross (Shift 1 to avoid lookahead)
    pd.set_option('future.no_silent_downcasting', True)
    signal_bull = (sma20 > sma60).shift(1).fillna(False)
    signal_bear = ~signal_bull
    
    # --- Strategy 1: Long-only Timing (0050 vs Cash) ---
    pos_long_only = pd.DataFrame({'0050': signal_bull.astype(float)})
    
    # --- Strategy 2: Long-Short Switching (0050 vs 00632R) ---
    pos_switching = pd.DataFrame({
        '0050': signal_bull.astype(float),
        '00632R': signal_bear.astype(float)
    })
    
    # --- Strategy 3: Buy & Hold 0050 ---
    pos_bh = pd.DataFrame({'0050': [1.0] * len(p0050)}, index=p0050.index)
    
    print("Running backtests...")
    report_bh = sim(pos_bh, resample="D", upload=False)
    report_long_only = sim(pos_long_only, resample="D", upload=False)
    report_switching = sim(pos_switching, resample="D", upload=False)
    
    # Metrics
    def get_metrics(name, report):
        stats = report.get_stats()
        return {
            'Name': name,
            'CAGR': f"{stats['cagr']:.2%}",
            'MDD': f"{stats['max_drawdown']:.2%}",
            'Sharpe': f"{stats['monthly_sharpe']:.2f}"
        }

    results = []
    results.append(get_metrics("0050 Buy & Hold", report_bh))
    results.append(get_metrics("0050 Long-Only Timing", report_long_only))
    results.append(get_metrics("0050/00632R Switching", report_switching))
    
    print("\n" + pd.DataFrame(results).to_string(index=False))

    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Equity Curves
    ax1 = plt.subplot(2, 1, 1)
    report_bh.creturn.plot(ax=ax1, label='Buy & Hold 0050', color='gray', alpha=0.5, linestyle='--')
    report_long_only.creturn.plot(ax=ax1, label='Long-Only (vs Cash)', color='blue')
    report_switching.creturn.plot(ax=ax1, label='Long-Short (vs 00632R)', color='red', linewidth=2)
    
    ax1.set_title('0050 Timing Strategy Comparison (Including 50反1 Shorting)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    # Subplot 2: Drawdowns
    ax2 = plt.subplot(2, 1, 2)
    def plot_dd(report, label, color):
        dd = (report.creturn / report.creturn.cummax()) - 1
        dd.plot(ax=ax2, label=label, color=color)
        
    plot_dd(report_bh, 'Buy & Hold 0050', 'gray')
    plot_dd(report_long_only, 'Long-Only', 'blue')
    plot_dd(report_switching, 'Long-Short', 'red')
    
    ax2.set_title('Drawdown Comparison')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = '0050_shorting_plot.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    analyze_0050_shorting()
