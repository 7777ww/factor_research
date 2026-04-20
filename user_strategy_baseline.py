import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def analyze_user_strategy():
    print("Fetching data for User Strategy Baseline...")
    
    # 1. Price and Volume Data
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    adj_close = data.get('etl:adj_close')
    
    # 2. Market Value Factor
    market_value = data.get('etl:market_value')
    base_stocks = market_value.is_largest(150)
    
    # 3. Dividend Factors
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    cash_dividend = data.get('financial_statement:發放現金股利')
    # Yield Condition: Top 40% (Quantile 0.6)
    yield_cond = (yield_ratio >= yield_ratio.quantile(0.6, axis=1))
    # Consistency: Paid dividends in last 3 years
    dividend_3y_cond = (cash_dividend.rolling(3).apply(lambda x: (x > 0).all()).fillna(0) > 0)
    
    # 4. Revenue Factors
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    rev = data.get('monthly_revenue:當月營收')
    # Revenue Growth: 3m avg > 12m avg
    rev_growth_cond = (rev.average(3) > rev.average(12))
    
    # 5. Fundamental Factors
    gross_margin_growth = data.get('fundamental_features:營業毛利成長率')
    eps = data.get('financial_statement:每股盈餘')
    gross_margin_cond = (gross_margin_growth > 0)
    eps_cond = (eps > 0)
    
    # 6. Technical Indicators
    # Moving Average: Price > SMA 120 and Price > SMA 240
    sma_cond = (close > close.average(120)) & (close > close.average(240))
    
    # Volatility: Bottom 50% relative to market
    std_rank = close.pct_change().rolling(60).std().rank(axis=1, pct=True)
    volatility_cond = (std_rank < 0.5)
    
    # 7. Liquidity Condition (Default)
    liquidity_cond = (vol.average(20) > 200000)
    
    print("Applying Combined Conditions...")
    # Combined filters
    conds = (
        (close > 0) &
        base_stocks &
        liquidity_cond &
        eps_cond &
        dividend_3y_cond &
        gross_margin_cond &
        rev_growth_cond &
        sma_cond &
        volatility_cond &
        yield_cond
    )
    
    # 8. Ranking Logic
    # position = (cash_dividend_annual.rank + 現金股利殖利率.rank + 去年同月增減.rank)
    # Using available proxies:
    rank_score = (
        cash_dividend.rank(axis=1, pct=True) + 
        yield_ratio.rank(axis=1, pct=True) + 
        rev_yoy.rank(axis=1, pct=True)
    )
    
    # Select Top 20 stocks
    position = rank_score[conds].is_largest(20)
    
    print("Preparing Resampling Dates (May 31 and Dec 27)...")
    dates = []
    # Get range from position index
    p_index = position.index
    y_start = p_index[0].year
    y_end = p_index[-1].year
    for y in range(y_start, y_end + 1):
        dates.append(pd.to_datetime(f"{y}-05-31"))
        dates.append(pd.to_datetime(f"{y}-12-27"))
    
    # Align to nearest PREVIOUS trading day in close.index
    def align_to_trading_day(target_dates, trading_days):
        aligned = []
        for d in target_dates:
            # Find the latest trading day <= target d
            valid_days = trading_days[trading_days <= d]
            if len(valid_days) > 0:
                aligned.append(valid_days[-1])
        return pd.DatetimeIndex(aligned).unique().sort_values()

    resample_dates = align_to_trading_day(dates, close.index)
    resample_dates = resample_dates[(resample_dates >= p_index[0]) & (resample_dates <= p_index[-1])]
    
    if len(resample_dates) < 3:
        print(f"Warning: Too few resample dates ({len(resample_dates)}). Using monthly resample instead.")
        resample_arg = "M"
        position_resampled = position
    else:
        print(f"Using {len(resample_dates)} custom resample dates.")
        position_resampled = position.reindex(resample_dates, method='ffill').dropna(how='all')
        resample_arg = resample_dates

    print("Running Baseline Backtest...")
    report = sim(position_resampled, resample=resample_arg, upload=False)
    
    # Aligned Benchmark: 0050 Buy & Hold
    p0050 = adj_close['0050']
    
    # Start the benchmark from the same first trade date
    start_date = report.creturn.index[0]
    p0050_aligned = p0050.loc[start_date:]
    
    print(f"Strategy Period: {start_date} to {report.creturn.index[-1]}")
    
    # Use simple buy & hold for benchmark comparison
    report_bh = sim(pd.DataFrame({'0050': [True] * len(p0050_aligned)}, index=p0050_aligned.index), resample="M", upload=False)
    
    # Metrics
    stats = report.get_stats()
    stats_bh = report_bh.get_stats()
    print("\n--- User Strategy Baseline ---")
    print(f"CAGR: {stats['cagr']:.2%}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {stats['monthly_sharpe']:.2f}")
    
    print("\n--- 0050 Benchmark (Aligned) ---")
    print(f"CAGR: {stats_bh['cagr']:.2%}")
    print(f"Max Drawdown: {stats_bh['max_drawdown']:.2%}")

    # Plot
    plt.figure(figsize=(12, 8))
    (report.creturn / report.creturn.iloc[0]).plot(label='User Strategy', color='blue', linewidth=2)
    (report_bh.creturn / report_bh.creturn.iloc[0]).plot(label='0050 Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    plt.title('User Strategy Baseline vs 0050')
    plt.legend()
    plt.grid(True)
    plt.savefig('user_strategy_baseline.png')
    print("\nPlot saved to user_strategy_baseline.png")

if __name__ == "__main__":
    analyze_user_strategy()
