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

def analyze_00631L_leveraged():
    print("Fetching adjusted data for 0050, 00631L (2x), and 00632R (Inverse)...")
    # Get adjusted price data
    close_adj = data.get('etl:adj_close')
    
    symbols = ['0050', '00631L', '00632R']
    for s in symbols:
        if s not in close_adj.columns:
            print(f"Error: {s} not found in adj_close table.")
            return
            
    p0050 = close_adj['0050']
    p00631L = close_adj['00631L']
    p00632R = close_adj['00632R']
    
    print("Calculating SMA signals on 00631L (Using 0050 as reference)...")
    # Testing a faster signal: Price > SMA 20
    sma20 = p0050.rolling(20).mean()
    
    pd.set_option('future.no_silent_downcasting', True)
    # Aggressive signal: Hold when close > SMA 20
    signal_bull = (p0050 > sma20).shift(1).fillna(False)
    signal_bear = ~signal_bull
    
    # --- Strategies ---
    pos_lev_timing = pd.DataFrame({'00631L': signal_bull.astype(float)})
    pos_lev_ls = pd.DataFrame({
        '00631L': signal_bull.astype(float),
        '00632R': signal_bear.astype(float)
    })
    pos_std_timing = pd.DataFrame({'0050': signal_bull.astype(float)})
    pos_bh = pd.DataFrame({'0050': [1.0] * len(p0050)}, index=p0050.index)
    pos_bh_631L = pd.DataFrame({'00631L': [1.0] * len(p0050)}, index=p0050.index)
    
    print("Running backtests...")
    report_bh = sim(pos_bh, resample="D", upload=False)
    report_bh_631L = sim(pos_bh_631L, resample="D", upload=False)
    report_std_timing = sim(pos_std_timing, resample="D", upload=False)
    report_lev_timing = sim(pos_lev_timing, resample="D", upload=False)
    report_lev_ls = sim(pos_lev_ls, resample="D", upload=False)
    
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
    results.append(get_metrics("00631L Buy & Hold", report_bh_631L))
    results.append(get_metrics("0050 Aggressive Timing", report_std_timing))
    results.append(get_metrics("00631L Aggressive Timing", report_lev_timing))
    results.append(get_metrics("00631L/00632R Leveraged LS", report_lev_ls))
    
    print("\n" + pd.DataFrame(results).to_string(index=False))

    # Plotting
    plt.figure(figsize=(14, 12))
    
    # Subplot 1: Equity Curves
    ax1 = plt.subplot(2, 1, 1)
    report_bh.creturn.plot(ax=ax1, label='Buy & Hold 0050', color='gray', alpha=0.5, linestyle='--')
    report_std_timing.creturn.plot(ax=ax1, label='0050 Timing (vs Cash)', color='blue')
    report_lev_timing.creturn.plot(ax=ax1, label='00631L Leveraged Timing', color='orange')
    report_lev_ls.creturn.plot(ax=ax1, label='00631L/00632R Leveraged LS', color='red', linewidth=2)
    
    ax1.set_title('Leveraged Strategy Comparison: Beating the 0050 Benchmark')
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
    plot_dd(report_std_timing, '0050 Timing', 'blue')
    plot_dd(report_lev_timing, '00631L Timing', 'orange')
    plot_dd(report_lev_ls, '00631L LS', 'red')
    
    ax2.set_title('Drawdown Comparison (Leverage Risk)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = '00631L_leveraged_plot.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    analyze_00631L_leveraged()
