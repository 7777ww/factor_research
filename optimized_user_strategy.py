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

def analyze_optimized_user_strategy():
    print("Fetching data for Optimized Strategy...")
    # Basic Data
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    adj_close = data.get('etl:adj_close')
    
    # Growth & Value Factors
    rev = data.get('monthly_revenue:當月營收')
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    roe = data.get('fundamental_features:ROE稅後')
    
    # 1. Universe: Top 300 by Market Value (Expand slightly for more Alpha)
    universe = market_value.is_largest(300)
    
    # 2. Quality Filter (Replaces restrictive hard filters)
    # ROE > 0 and positive Revenue Growth (3m vs 12m)
    quality_cond = (roe > 0) & (rev.average(3) > rev.average(12))
    
    # 3. Market Trend Filter (Keeps the best of User's technical logic)
    # Price > SMA 120 (Slightly faster than 240 for better entry)
    trend_cond = (close > close.average(120))
    
    # 4. Liquidity
    liquidity_cond = (vol.average(20) > 200000)
    
    # 5. Combined Filter
    conds = universe & quality_cond & trend_cond & liquidity_cond
    
    # 6. Multi-Factor Score (Z-Score Ranking)
    # We want High Yield, High Momentum, and High Growth
    def z_score(df):
        return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
        
    # Relative Strength (60-day rise)
    mom = close.rise(60)
    
    # NEW: Revenue Acceleration (YoY Burst)
    # Current YoY Growth - Avg(Last 3 months YoY Growth)
    rev_accel = rev_yoy - rev_yoy.rolling(3).mean().shift(1)
    
    # Weights: 30% Yield, 20% Momentum, 20% Revenue YoY, 30% Revenue Accel
    score = (
        z_score(yield_ratio).fillna(0) * 0.3 + 
        z_score(mom).fillna(0) * 0.2 + 
        z_score(rev_yoy).fillna(0) * 0.2 +
        z_score(rev_accel).fillna(0) * 0.3
    )
    
    # 7. Final Selection: Top 20 from filtered universe
    position = score[conds].is_largest(20)
    
    # 8. Rebalancing: Use User's specific dates (May 31 & Dec 27)
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
    
    print(f"Running Backtest with {len(resample_dates)} rebalance points...")
    position_resampled = position.reindex(resample_dates, method='ffill').dropna(how='all')
    report = sim(position_resampled, resample=resample_dates, upload=False)
    
    # 9. Benchmark Alignment
    p0050 = adj_close['0050']
    start_date = report.creturn.index[0]
    p0050_aligned = p0050.loc[start_date:]
    report_bh = sim(pd.DataFrame({'0050': [True]*len(p0050_aligned)}, index=p0050_aligned.index), resample="M", upload=False)
    
    # Results
    print("\n--- PERFORMANCE SUMMARY ---")
    generate_return_report(report, "Accel-Optimized Strategy")
    generate_return_report(report_bh, "0050 Benchmark")
    print(f"Period: {start_date.date()} to {report.creturn.index[-1].date()}")

    # Plot
    plt.figure(figsize=(12, 8))
    (report.creturn / report.creturn.iloc[0]).plot(label='Optimized Strategy', color='green', linewidth=2)
    (report_bh.creturn / report_bh.creturn.iloc[0]).plot(label='0050 Benchmark', color='gray', alpha=0.5, linestyle='--')
    plt.title('Optimized Dividend-Growth Strategy vs 0050')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('user_strategy_optimized.png')
    print("\nVisual saved to user_strategy_optimized.png")

if __name__ == "__main__":
    analyze_optimized_user_strategy()
