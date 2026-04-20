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

def analyze_concentrated_pro_strategy():
    print("Fetching data for Concentrated Pro Strategy (Top 10)...")
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
    
    # Advanced Filters
    # 1. Revenue Milestone: Revenue at 24-month high
    rev_high_24 = (rev == rev.rolling(24).max())
    
    # 2. Institutional Flow (Net buy over 20 days)
    inst_flow = inst_buy.rolling(20).sum()
    
    TOP_K = 10 
    SMA_PERIOD_MARKET = 200 # Faster market filter (Less restrictive)
    
    # Universe & Liquidity (Expand universe for more Alpha in 10 stocks)
    universe = market_value.is_largest(500)
    quality_cond = (roe > 0) & (rev.average(3) > rev.average(12))
    liquidity_cond = (vol.average(20) > 200000)
    
    # Market Trend (Hedge Trigger)
    p0050 = adj_close['0050']
    market_is_up = (p0050 > p0050.rolling(SMA_PERIOD_MARKET).mean())
    
    # Final Mask: High Quality & Large Cap
    conds = universe & quality_cond & liquidity_cond
    
    # Multi-Factor Score (Added Revenue High as a score component)
    def z_score(df):
        return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
    
    mom = close.rise(60)
    rev_accel = rev_yoy - rev_yoy.rolling(3).mean().shift(1)
    
    score = (
        z_score(yield_ratio).fillna(0) * 0.20 + 
        z_score(mom).fillna(0) * 0.20 + 
        z_score(rev_accel).fillna(0) * 0.20 +
        z_score(inst_flow).fillna(0) * 0.20 +
        z_score(rev_high_24.astype(int)).fillna(0) * 0.20
    )
    
    # Selection: Top 10 High-Conviction Stocks
    position = score[conds].is_largest(TOP_K)
    
    # Rebalancing
    dates = []
    p_index = position.index
    y_start, y_end = p_index[0].year, p_index[-1].year
    for y in range(y_start, y_end + 1):
        dates.append(pd.to_datetime(f"{y}-05-31"))
        dates.append(pd.to_datetime(f"{y}-12-27"))
        
    def align_to_trading_day(target_dates, trading_days):
        aligned = []
        for d in target_dates:
            valid_days = trading_days[trading_days <= d]
            if len(valid_days) > 0: aligned.append(valid_days[-1])
        return pd.DatetimeIndex(aligned).unique().sort_values()

    resample_dates = align_to_trading_day(dates, close.index)
    resample_dates = resample_dates[(resample_dates >= p_index[0]) & (resample_dates <= p_index[-1])]
    
    print(f"Running Concentrated Pro Backtest (Top {TOP_K})...")
    pos_res = position.reindex(resample_dates, method='ffill').dropna(how='all')
    
    # Refine positions with Market Filter (SMA 200 on 0050)
    final_pos = pos_res.copy()
    for d in resample_dates:
        if d in market_is_up.index and not market_is_up.loc[d]:
            final_pos.loc[d] = 0 # Sell to cash if market is bad
            
    report = sim(final_pos, resample=resample_dates, upload=False, stop_loss=0.12)
    
    # Benchmark
    p0050_aligned = p0050.loc[report.creturn.index[0]:]
    report_bh = sim(pd.DataFrame({'0050': [True]*len(p0050_aligned)}, index=p0050_aligned.index), resample="M", upload=False)
    
    # Report
    print("\n--- CONCENTRATED PRO PERFORMANCE (Top 10) ---")
    generate_return_report(report, "Concentrated Pro (Top 10)")
    generate_return_report(report_bh, "0050 Benchmark")
    
    # Plot
    plt.figure(figsize=(14, 10))
    (report.creturn / report.creturn.iloc[0]).plot(label='Concentrated Pro (Top 10)', color='crimson', linewidth=3)
    (report_bh.creturn / report_bh.creturn.iloc[0]).plot(label='0050 Benchmark', color='black', alpha=0.5, linestyle='--')
    plt.title('Concentrated Pro (Top 10) Stock Selection vs 0050')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('concentrated_pro_strategy.png')
    print("\nVisual saved to concentrated_pro_strategy.png")

if __name__ == "__main__":
    analyze_concentrated_pro_strategy()
