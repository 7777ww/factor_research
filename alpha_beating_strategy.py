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

def analyze_alpha_beating():
    print("Fetching data for Multi-Factor Alpha Strategy...")
    # Get Factors
    pb = data.get('price_earning_ratio:股價淨值比')
    market_value = data.get('etl:market_value')
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    adj_close = data.get('etl:adj_close') # For 0050 benchmark
    
    print("Calculating Selection Alpha (Small-Cap Value)...")
    # 1. Liquidity Filter: Avg volume > 200k shares (Small caps need lower threshold)
    cond_liquidity = (vol.average(20) > 200000)
    
    # 2. Multi-Factor Score: Combine Value (Low PB) and Size (Small Market Cap)
    pb_rank = pb.rank(axis=1, pct=True)
    size_rank = market_value.rank(axis=1, pct=True)
    score = pb_rank + size_rank
    
    # Select Top 20 stocks
    pos_selection = score[cond_liquidity].is_smallest(20)
    
    print("Calculating Market Timing Filter (SMA 60)...")
    # 3. Market Timing (Optional protection)
    p0050 = adj_close['0050']
    sma60 = p0050.rolling(60).mean()
    mkt_bull = (p0050 > sma60).shift(1).fillna(False)
    
    # 4. Combined Strategy: Selection + Timing
    pos_combined = pos_selection.multiply(mkt_bull, axis=0)
    
    print("Running backtests...")
    # Strategy A: Just selection (Small-Cap Value)
    report_alpha = sim(pos_selection, resample="M", upload=False)
    
    # Strategy B: Alpha + Timing
    report_alpha_timed = sim(pos_combined, resample="M", upload=False)
    
    # Aligned Benchmark: 0050 Buy & Hold
    p0050 = adj_close['0050']
    start_date = report_alpha.creturn.index[0]
    end_date = report_alpha.creturn.index[-1]
    
    p0050_aligned = p0050.loc[start_date:end_date]
    pos_bh_aligned = pd.DataFrame({'0050': [True] * len(p0050_aligned)}, index=p0050_aligned.index)
    report_bh = sim(pos_bh_aligned, resample="M", upload=False)
    
    # Metrics
    def get_metrics(name, report):
        stats = report.get_stats()
        return {
            'Name': name,
            'CAGR': f"{stats['cagr']:.2%}",
            'MDD': f"{stats['max_drawdown']:.2%}",
            'Sharpe': f"{stats['monthly_sharpe']:.2f}",
            'Start': report.creturn.index[0].strftime('%Y-%m-%d'),
            'End': report.creturn.index[-1].strftime('%Y-%m-%d')
        }

    results = []
    results.append(get_metrics("0050 Buy & Hold (Bench)", report_bh))
    results.append(get_metrics("Multi-Factor Selection", report_alpha))
    results.append(get_metrics("Selection + Timing (Alpha-Timed)", report_alpha_timed))
    
    print("\n" + pd.DataFrame(results).to_string(index=False))

    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Unified Start Date for Plotting
    plot_start = '2014-01-01'
    
    def get_plot_ready(report, start_date):
        series = report.creturn.loc[start_date:]
        return series / series.iloc[0] # Re-index to 1.0
    
    # Prepare series for plotting
    bh_plot = get_plot_ready(report_bh, plot_start)
    alpha_plot = get_plot_ready(report_alpha, plot_start)
    alpha_timed_plot = get_plot_ready(report_alpha_timed, plot_start)
    
    # Subplot: Equity Curves
    ax1 = plt.subplot(1, 1, 1)
    bh_plot.plot(ax=ax1, label='0050 Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    alpha_plot.plot(ax=ax1, label='Multi-Factor Selection', color='blue')
    alpha_timed_plot.plot(ax=ax1, label='Selection + Timing', color='red', linewidth=2)
    
    ax1.set_title('Winning Strategy: Multi-Factor Alpha vs 0050 Benchmark (Aligned from 2014)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Return (Re-indexed to 1.0)')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    plot_path = 'alpha_beating_plot.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    analyze_alpha_beating()
