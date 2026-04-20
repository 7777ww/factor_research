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

def analyze_final_champion_strategy():
    print("Fetching data for Final Champion Strategy...")
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
    
    # Winning Parameters from Optimization (Top 30 + Balanced Weights)
    TOP_K = 30
    SMA_PERIOD = 120
    W_YIELD = 0.25
    W_MOM = 0.25
    W_GROWTH = 0.25
    W_ACCEL = 0.25
    
    universe = market_value.is_largest(300)
    quality_cond = (roe > 0) & (rev.average(3) > rev.average(12))
    trend_cond = (close > close.average(SMA_PERIOD))
    liquidity_cond = (vol.average(20) > 200000)
    
    conds = universe & quality_cond & trend_cond & liquidity_cond
    
    # Multi-Factor Score (Z-Score Ranking)
    def z_score(df):
        return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
        
    mom = close.rise(60)
    rev_accel = rev_yoy - rev_yoy.rolling(3).mean().shift(1)
    
    score = (
        z_score(yield_ratio).fillna(0) * W_YIELD + 
        z_score(mom).fillna(0) * W_MOM + 
        z_score(rev_yoy).fillna(0) * W_GROWTH +
        z_score(rev_accel).fillna(0) * W_ACCEL
    )
    
    # Final Selection
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
    
    print(f"Running Final Champion Backtest (Top {TOP_K})...")
    pos_res = position.reindex(resample_dates, method='ffill').dropna(how='all')
    report = sim(pos_res, resample=resample_dates, upload=False)
    
    # Benchmark
    p0050 = adj_close['0050']
    start_date = report.creturn.index[0]
    p0050_aligned = p0050.loc[start_date:]
    report_bh = sim(pd.DataFrame({'0050': [True]*len(p0050_aligned)}, index=p0050_aligned.index), resample="M", upload=False)
    
    # Report
    print("\n--- FINAL CHAMPION PERFORMANCE ---")
    generate_return_report(report, "Final Champion Strategy")
    generate_return_report(report_bh, "0050 Benchmark")
    
    # Plot
    plt.figure(figsize=(14, 10))
    (report.creturn / report.creturn.iloc[0]).plot(label='Final Champion', color='gold', linewidth=3)
    (report_bh.creturn / report_bh.creturn.iloc[0]).plot(label='0050 Benchmark', color='black', alpha=0.5, linestyle='--')
    plt.title('FINAL CHAMPION STRATEGY vs 0050 (Optimized Parameters)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('final_champion_strategy.png')
    print("\nVisual saved to final_champion_strategy.png")

if __name__ == "__main__":
    analyze_final_champion_strategy()
