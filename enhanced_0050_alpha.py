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

def analyze_enhanced_0050_alpha():
    print("Fetching data for Enhanced 0050 Alpha Strategy...")
    # Get Data
    close = data.get('price:收盤價')
    market_value = data.get('etl:market_value')
    rev = data.get('monthly_revenue:當月營收')
    adj_close = data.get('etl:adj_close') # For 0050 benchmark
    
    # 0050 Universe Proxy (Top 50 Market Cap)
    mkt_cap_50 = market_value.is_largest(50)
    
    print("Calculating Factors (Momentum & Revenue Growth)...")
    # 1. Momentum: 60-day relative rise
    mom = close.rise(60)
    
    # 2. Revenue Growth: YoY Growth
    # Monthly revenue table has 1 row per month per stock
    rev_growth = (rev / rev.shift(12)) - 1
    
    # Rank within the Top 50 universe
    mom_rank = mom[mkt_cap_50].rank(axis=1, pct=True)
    rev_rank = rev_growth[mkt_cap_50].rank(axis=1, pct=True)
    
    # Combine scores (Higher is better for both)
    score = mom_rank + rev_rank
    
    # Selection: Top 10 stocks with highest combined rank
    pos_selection = score.is_largest(10)
    
    print("Running backtests...")
    # Strategy: Enhanced 0050 Alpha
    report_alpha = sim(pos_selection, resample="M", upload=False)
    
    # Strategy: 0050 Buy & Hold (Aligned)
    p0050 = adj_close['0050']
    
    # Align start dates: The strategy starts when it has its first position
    start_date = report_alpha.creturn.index[0]
    end_date = report_alpha.creturn.index[-1]
    
    # Filter 0050 to the same period
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
    results.append(get_metrics("Enhanced 0050 Alpha (Mom+Rev)", report_alpha))
    
    print("\n" + pd.DataFrame(results).to_string(index=False))

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Unified Start Date for Plotting
    plot_start = '2014-01-01'
    
    def get_plot_ready(report, start_date):
        series = report.creturn.loc[start_date:]
        return series / series.iloc[0] # Re-index to 1.0
        
    bh_plot = get_plot_ready(report_bh, plot_start)
    alpha_plot = get_plot_ready(report_alpha, plot_start)
    
    bh_plot.plot(label='0050 Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    alpha_plot.plot(label='Enhanced 0050 Alpha (Top 10)', color='blue', linewidth=2)
    
    plt.title('Enhanced 0050 Alpha Strategy: Aligned from 2014')
    plt.yscale('log')
    plt.ylabel('Cumulative Return (Re-indexed to 1.0)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'enhanced_0050_plot.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    analyze_enhanced_0050_alpha()
