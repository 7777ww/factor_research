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

def analyze_high_beta_alpha_king():
    print("Fetching data for High-Beta Alpha King (Extreme Concentration)...")
    # Basic Data
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    adj_close = data.get('etl:adj_close')
    
    # Factors
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    
    # Strategy Parameters (Extreme Concentration for Alpha)
    TOP_K = 8
    
    # Universe: Top 150 (The sweet spot for AI leaders)
    universe = market_value.is_largest(150)
    liquidity_cond = (vol.average(20) > 200000)
    
    # Multi-Factor Score (Pure Momentum + Growth)
    def z_score(df):
        return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
    
    mom_60 = close.rise(60)
    
    score = (
        z_score(mom_60).fillna(0) * 0.70 + 
        z_score(rev_yoy).fillna(0) * 0.30
    )
    
    conds = universe & liquidity_cond
    
    # Selection
    position = score[conds].is_largest(TOP_K)
    
    # Rebalancing: Monthly for High Frequency
    resample_dates = position.index[position.index > '2023-01-01'][::20]
    
    print(f"Running Alpha King Backtest (2023-Present)...")
    pos_res = position.loc['2023-01-01':].reindex(method='ffill').dropna(how='all')
    
    # Pure Alpha: No defensive switches
    report = sim(pos_res, upload=False)
    
    # Benchmark Alignment
    p0050 = adj_close['0050']
    p0050_aligned = p0050.loc[report.creturn.index[0]:]
    report_bh = sim(pd.DataFrame({'0050': [True]*len(p0050_aligned)}, index=p0050_aligned.index), upload=False)
    
    # Results
    print("\n--- HIGH-BETA ALPHA KING PERFORMANCE (2023-Present) ---")
    generate_return_report(report, "Alpha King Strategy")
    generate_return_report(report_bh, "0050 Benchmark")
    
    # Plot
    plt.figure(figsize=(12, 8))
    (report.creturn / report.creturn.iloc[0]).plot(label='Alpha King Strategy', color='red', linewidth=3)
    (report_bh.creturn / report_bh.creturn.iloc[0]).plot(label='0050 Benchmark', color='black', alpha=0.5, linestyle='--')
    plt.title('High-Beta Alpha King vs 0050 (2023-Present)')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('high_beta_alpha_king.png')
    print("\nVisual saved to high_beta_alpha_king.png")

if __name__ == "__main__":
    analyze_high_beta_alpha_king()
