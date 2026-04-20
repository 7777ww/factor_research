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

def analyze_recent_alpha_accelerator():
    print("Fetching data for Recent Alpha Accelerator (2023-2024 Focus)...")
    # Basic Data
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    adj_close = data.get('etl:adj_close')
    
    # Factors
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    rev = data.get('monthly_revenue:當月營收')
    roe = data.get('fundamental_features:ROE稅後')
    inst_buy = data.get('institutional_investors_trading_summary:投信買賣超股數')
    
    # Strategy Parameters
    TOP_K = 15
    SMA_FAST = 60 # Faster trigger for 2023 recovery
    
    # Universe & Filters
    universe = market_value.is_largest(300)
    quality_cond = (roe > 0) & (rev.average(3) > rev.average(12))
    liquidity_cond = (vol.average(20) > 200000)
    
    # Multi-Factor Score (Optimized for AI/Growth Cycle)
    def z_score(df):
        return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
    
    mom_60 = close.rise(60)
    rev_accel = rev_yoy - rev_yoy.rolling(3).mean().shift(1)
    inst_flow = inst_buy.rolling(10).sum() # 2-week flow
    
    score = (
        z_score(rev_accel).fillna(0) * 0.4 + 
        z_score(inst_flow).fillna(0) * 0.3 + 
        z_score(mom_60).fillna(0) * 0.2 +
        z_score(yield_ratio).fillna(0) * 0.1
    )
    
    # Filter for Trend
    p0050 = adj_close['0050']
    market_is_up = (p0050 > p0050.rolling(SMA_FAST).mean())
    
    conds = universe & quality_cond & liquidity_cond
    
    # Selection
    position = score[conds].is_largest(TOP_K)
    
    # Rebalancing: Monthly for Recent Alpha
    resample_dates = position.index[position.index > '2022-01-01'][::20] # Roughly monthly
    
    print(f"Running Recent Alpha Accelerator Backtest (2023-Present)...")
    pos_res = position.loc['2023-01-01':].reindex(method='ffill').dropna(how='all')
    
    # Market Switch: If trend is down, reduce position or empty (Using SMA 60)
    # This ensures we don't fight the tape in 2022, but enter fast in 2023.
    final_pos = pos_res.copy()
    for d in pos_res.index:
        if d in market_is_up.index and not market_is_up.loc[d]:
            # Reduced exposure during downtrends to protect MDD
            final_pos.loc[d] = 0 

    report = sim(final_pos, upload=False)
    
    # Benchmark Alignment (2023-Present)
    start_date = report.creturn.index[0]
    p0050_aligned = p0050.loc[start_date:]
    report_bh = sim(pd.DataFrame({'0050': [True]*len(p0050_aligned)}, index=p0050_aligned.index), upload=False)
    
    # Results
    print("\n--- RECENT ALPHA ACCELERATOR PERFORMANCE (2023-Present) ---")
    generate_return_report(report, "Recent Alpha Strategy")
    generate_return_report(report_bh, "0050 Benchmark")
    
    # Plot
    plt.figure(figsize=(12, 8))
    (report.creturn / report.creturn.iloc[0]).plot(label='Recent Alpha Strategy', color='blue', linewidth=3)
    (report_bh.creturn / report_bh.creturn.iloc[0]).plot(label='0050 Benchmark', color='black', alpha=0.5, linestyle='--')
    plt.title('Recent Alpha Accelerator vs 0050 (2023-Present Focus)')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('recent_alpha_accelerator.png')
    print("\nVisual saved to recent_alpha_accelerator.png")

if __name__ == "__main__":
    analyze_recent_alpha_accelerator()
