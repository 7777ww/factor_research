import pandas as pd
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def get_5y_metrics(report):
    """Extract metrics for the last 5 years."""
    ar = report.metrics.annual_return() * 100
    md = report.metrics.max_drawdown() * 100
    sr = report.metrics.sharpe_ratio()
    return {"ar": ar, "md": md, "sr": sr}

def sync_5y_performance():
    print("Fetching data for 5-Year Performance Sync (2021-2026)...")
    adj_close = data.get('etl:adj_close')
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    rev = data.get('monthly_revenue:當月營收')
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    roe = data.get('fundamental_features:ROE稅後')
    signals_eco = data.get('tw_business_indicators:景氣對策信號(分)')
    
    START_DATE = '2021-04-01'
    
    results = {}
    
    # 1. 0050 Benchmark
    print("Running 0050 (5Y)...")
    p0050 = adj_close['0050'].loc[START_DATE:]
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(p0050)}, index=p0050.index), upload=False)
    results['0050'] = get_5y_metrics(report_0050)
    
    # 2. Optimized Dividend-Growth (Accel)
    print("Running Div-Growth (5Y)...")
    def z_score(df): return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
    universe_300 = market_value.is_largest(300)
    quality = (roe > 0) & (rev.average(3) > rev.average(12))
    score = (z_score(yield_ratio).fillna(0)*0.25 + z_score(close.rise(60)).fillna(0)*0.25 + 
             z_score(rev_yoy).fillna(0)*0.25 + z_score(rev_yoy - rev_yoy.rolling(3).mean().shift(1)).fillna(0)*0.25)
    pos_dg = score[universe_300 & quality & (close > close.average(120))].is_largest(20)
    report_dg = sim(pos_dg.loc[START_DATE:], resample='M', upload=False)
    results['DG'] = get_5y_metrics(report_dg)
    
    # 3. Small-Cap Value
    print("Running Small-Cap (5Y)...")
    pb = data.get('price_earning_ratio:股價淨值比')
    universe_sc = (market_value.rank(axis=1, pct=True) > 0.5) & (market_value.rank(axis=1, pct=True) < 0.9)
    cond_sc = universe_sc & (pb < 1.5) & (roe > 0)
    pos_sc = pb[cond_sc].is_smallest(20)
    report_sc = sim(pos_sc.loc[START_DATE:], resample='Q', upload=False)
    results['SC'] = get_5y_metrics(report_sc)
    
    # 4. Hybrid F.E.T.E. (Sentinel)
    print("Running Hybrid F.E.T.E. (5Y)...")
    p00631L = adj_close['00631L']
    dates = p00631L.loc[START_DATE:].index
    fund_score = signals_eco['tw_business_indicators'].reindex(dates, method='ffill').fillna(38)
    ema_60 = p0050.rolling(60).mean().reindex(dates)
    
    hybrid_sig = pd.Series(index=dates, dtype=bool)
    curr = False
    for d in dates:
        if (fund_score.loc[d] <= 22) or (p0050.loc[d] > ema_60.loc[d] and fund_score.loc[d] < 34): curr = True
        if curr and (p0050.loc[d] < ema_60.loc[d] or fund_score.loc[d] >= 38): curr = False
        hybrid_sig.loc[d] = curr
    report_hybrid = sim(pd.DataFrame({'00631L': hybrid_sig}, index=dates), upload=False)
    results['Hybrid'] = get_5y_metrics(report_hybrid)
    
    print("\n--- 5-YEAR PERFORMANCE SUMMARY (2021-2026) ---")
    for k, v in results.items():
        print(f"{k:8}: CAGR {v['ar']:7.2f}% | MDD {v['md']:7.2f}% | Sharpe {v['sr']:7.2f}")

if __name__ == "__main__":
    sync_5y_performance()
