import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def analyze_portfolio_allocation():
    print("Fetching Strategy Components for Portfolio Optimization...")
    # Fetching signals/positions from our previous research
    adj_close = data.get('etl:adj_close')
    market_value = data.get('etl:market_value')
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    rev = data.get('monthly_revenue:當月營收')
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    roe = data.get('fundamental_features:ROE稅後')
    signals_eco = data.get('tw_business_indicators:景氣對策信號(分)')
    close = data.get('price:收盤價')
    pb = data.get('price_earning_ratio:股價淨值比')
    
    START_DATE = '2021-01-01'
    
    # helper
    def z_score(df): return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)

    # 1. Component A: Hybrid F.E.T.E. (L)
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    dates = p00631L.loc[START_DATE:].index
    fund_score = signals_eco['tw_business_indicators'].reindex(dates, method='ffill').fillna(38)
    ema_60 = p0050.rolling(60).mean().reindex(dates)
    curr = False
    hybrid_sig = pd.Series(index=dates, dtype=bool)
    for d in dates:
        if (fund_score.loc[d] <= 22) or (p0050.loc[d] > ema_60.loc[d] and fund_score.loc[d] < 34): curr = True
        if curr and (p0050.loc[d] < ema_60.loc[d] or fund_score.loc[d] >= 38): curr = False
        hybrid_sig.loc[d] = curr
    
    # 2. Component B: Optimized Dividend-Growth (DG)
    universe_300 = market_value.is_largest(300)
    quality = (roe > 0) & (rev.average(3) > rev.average(12))
    score_dg = (z_score(yield_ratio).fillna(0)*0.25 + z_score(close.rise(60)).fillna(0)*0.25 + 
                z_score(rev_yoy).fillna(0)*0.25 + z_score(rev_yoy - rev_yoy.rolling(3).mean().shift(1)).fillna(0)*0.25)
    pos_dg = score_dg[universe_300 & quality & (close > close.average(120))].is_largest(20)
    
    # 3. Component C: Small-Cap Value (SC)
    universe_sc = (market_value.rank(axis=1, pct=True) > 0.6) & (market_value.rank(axis=1, pct=True) < 0.95)
    cond_sc = universe_sc & (pb < 1.8) & (roe > 5)
    pos_sc = pb[cond_sc].is_smallest(15)

    # Simulation - Extract individual performance curves
    print("\nExtracting component curves...")
    report_hybrid = sim(pd.DataFrame({'00631L': hybrid_sig}, index=dates), upload=False)
    report_dg = sim(pos_dg.loc[START_DATE:], resample='M', upload=False)
    report_sc = sim(pos_sc.loc[START_DATE:], resample='Q', upload=False)
    
    c_hybrid = report_hybrid.creturn
    c_dg = report_dg.creturn
    c_sc = report_sc.creturn
    
    # Reindex all to common daily dates
    common_dates = c_hybrid.index.intersection(c_dg.index).intersection(c_sc.index)
    c_hybrid = c_hybrid.loc[common_dates] / c_hybrid.loc[common_dates[0]]
    c_dg = c_dg.loc[common_dates] / c_dg.loc[common_dates[0]]
    c_sc = c_sc.loc[common_dates] / c_sc.loc[common_dates[0]]
    
    # Test Portfolios
    # Weights: [Hybrid, DG, SmallCap]
    portfolios = {
        "Balanced (40/40/20)": [0.4, 0.4, 0.2],
        "Aggressive (60/30/10)": [0.6, 0.3, 0.1],
        "Safe-Growth (30/60/10)": [0.3, 0.6, 0.1]
    }
    
    print("\n--- PORTFOLIO MULTI-FACTOR ALLOCATION RESULTS (2021-2026) ---")
    plt.figure(figsize=(14, 10))
    
    for name, w in portfolios.items():
        # Weighted daily returns
        p_return = (c_hybrid.pct_change().fillna(0)*w[0] + 
                    c_dg.pct_change().fillna(0)*w[1] + 
                    c_sc.pct_change().fillna(0)*w[2])
        p_creturn = (1 + p_return).cumprod()
        
        # Simple stats
        cagr = (p_creturn.iloc[-1] ** (252/len(p_creturn)) - 1) * 100
        mdd = (p_creturn / p_creturn.cummax() - 1).min() * 100
        vol = p_return.std() * (252**0.5) * 100
        sharpe = (cagr - 2) / vol if vol != 0 else 0 # 2% risk free
        
        print(f"{name:25}: CAGR {cagr:7.2f}% | MDD {mdd:7.2f}% | Sharpe {sharpe:7.2f}")
        p_creturn.plot(label=name)
        
    # Benchmark
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(common_dates)}, index=common_dates), upload=False)
    (report_0050.creturn / report_0050.creturn.iloc[0]).plot(label='0050 Benchmark', color='black', alpha=0.3, linestyle='--')
    
    plt.title('Portfolio Optimization: Multi-Strategy Capital Allocation')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('portfolio_allocation_plot.png')
    print("\nComparison visual saved to portfolio_allocation_plot.png")

if __name__ == "__main__":
    analyze_portfolio_allocation()
