import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def run_factor_comparison():
    print("Fetching data for Factor Comparison Analysis...")
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    adj_close = data.get('etl:adj_close')
    
    # Factor Data
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    rev = data.get('monthly_revenue:當月營收')
    roe = data.get('fundamental_features:ROE稅後')
    inst_buy = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
    
    # 1. Universe: Top 500 Market Cap
    universe = market_value.is_largest(500)
    liquidity = (vol.average(20) > 200000)
    base_mask = universe & liquidity
    
    # 2. Factor Definitions
    factors = {
        "Yield": yield_ratio,
        "Momentum (60D)": close.rise(60),
        "Rev YoY Growth": rev_yoy,
        "Rev Accel": rev_yoy - rev_yoy.rolling(3).mean().shift(1),
        "ROE (Quality)": roe,
        "Inst. Buying": inst_buy / vol.average(20) # Normalize by volume
    }
    
    results = []
    plot_data = {}
    
    print("\nStarting Factor Sifting (Picking Top 50 per factor)...")
    
    for name, factor_df in factors.items():
        print(f"Backtesting {name}...")
        # Selection: Top 50 by factor rank within the universe
        pos = factor_df[base_mask].is_largest(50)
        
        # Run Backtest
        report = sim(pos, resample="M", upload=False)
        stats = report.get_stats()
        
        results.append({
            "Factor": name,
            "CAGR": f"{stats['cagr']:.2%}",
            "MDD": f"{stats['max_drawdown']:.2%}",
            "Sharpe": f"{stats['monthly_sharpe']:.2f}"
        })
        
        # Save equity curve for plotting (Re-indexed to 1.0 from a common start)
        common_start = '2016-01-01'
        curve = report.creturn.loc[common_start:]
        plot_data[name] = curve / curve.iloc[0]
        
    # Align and Benchmark (0050)
    p0050 = adj_close['0050'].loc['2016-01-01':]
    report_bh = sim(pd.DataFrame({'0050': [True]*len(p0050)}, index=p0050.index), resample="M", upload=False)
    stats_bh = report_bh.get_stats()
    results.append({
        "Factor": "0050 Buy & Hold",
        "CAGR": f"{stats_bh['cagr']:.2%}",
        "MDD": f"{stats_bh['max_drawdown']:.2%}",
        "Sharpe": f"{stats_bh['monthly_sharpe']:.2f}"
    })
    
    # Output Table
    df_results = pd.DataFrame(results)
    print("\n--- FACTOR LEADERBOARD (Sorted by Sharpe) ---")
    df_results['Sharpe_Val'] = df_results['Sharpe'].astype(float)
    print(df_results.sort_values("Sharpe_Val", ascending=False).drop(columns="Sharpe_Val").to_string(index=False))
    
    # Plotting
    plt.figure(figsize=(14, 10))
    for name, curve in plot_data.items():
        curve.plot(label=name)
        
    (p0050 / p0050.iloc[0]).plot(label='0050 Benchmark', color='black', linewidth=2, linestyle='--')
    
    plt.title('Single Factor Performance Comparison (Aligned from 2016)')
    plt.yscale('log')
    plt.ylabel('Cumulative Return (Re-indexed to 1.0)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig('factor_comparison_plot.png')
    print("\nComparison plot saved to factor_comparison_plot.png")

if __name__ == "__main__":
    run_factor_comparison()
