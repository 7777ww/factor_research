import pandas as pd
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def dividend_growth_accel_strategy():
    """
    Optimized Dividend-Growth Strategy with Revenue Acceleration.
    Returns the FinLab backtest report object.
    """
    # 1. Fetch Basic Data
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    
    # 2. Fetch Factor Data
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    rev = data.get('monthly_revenue:當月營收')
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    roe = data.get('fundamental_features:ROE稅後')
    
    # 3. Define Universe and Quality Filters
    universe = market_value.is_largest(300)
    quality_cond = (roe > 0) & (rev.average(3) > rev.average(12))
    liquidity_cond = (vol.average(20) > 200000)
    
    # 4. Define Factors
    def z_score(df):
        return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
        
    mom_60 = close.rise(60)
    rev_accel = rev_yoy - rev_yoy.rolling(3).mean().shift(1)
    
    # 5. Multi-Factor Score (Balanced weights)
    score = (
        z_score(yield_ratio).fillna(0) * 0.25 + 
        z_score(mom_60).fillna(0) * 0.25 + 
        z_score(rev_yoy).fillna(0) * 0.25 +
        z_score(rev_accel).fillna(0) * 0.25
    )
    
    # 6. Technical Trend Filter
    trend_cond = (close > close.average(120))
    
    # 7. Final Selection
    conds = universe & quality_cond & liquidity_cond & trend_cond
    position = score[conds].is_largest(20)
    
    # 8. Rebalancing logic (May 31 and Dec 27)
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
    
    # 9. Resample Position
    pos_resampled = position.reindex(resample_dates, method='ffill').dropna(how='all')
    
    # 10. Run Backtest and Return Report
    report = sim(pos_resampled, resample=resample_dates, upload=False)
    
    return report

if __name__ == "__main__":
    report = dividend_growth_accel_strategy()
    
    # Display summary
    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    print(f"\n[Dividend-Growth Accel Report]")
    print(f"年化報酬率:{ar:7.2f}%  夏普比率:{sr:7.2f}  最大回檔:{md:7.2f}%")
