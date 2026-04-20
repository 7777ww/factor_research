import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def z_score(df):
    return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)

def fetch_strategy_curves(start_date='2021-01-01'):
    print("Running Component 1: Momentum Breakout...")
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    rev = data.get('monthly_revenue:當月營收')
    odd_vol = data.get('intraday_odd_lot_trade:成交股數')
    roe = data.get('fundamental_features:ROE稅後')
    full_cash_filter = data.get('etl:full_cash_delivery_stock_filter')
    benchmark_return = data.get('benchmark_return:發行量加權股價報酬指數')

    # MB Logic
    rs_period = 120
    stock_return = close.pct_change(rs_period) + 1
    market_return = benchmark_return.pct_change(rs_period) + 1
    market_return_series = market_return.iloc[:, 0].reindex(close.index, method='ffill')
    rs = stock_return.div(market_return_series, axis=0).fillna(1)
    
    breakout = close >= close.rolling(120).max()
    cond_rev = rev.average(3) > rev.average(12)
    cond_vol = vol.average(5) > vol.average(20) * 1.2
    cond_odd = odd_vol.average(10) > 1000
    cond_roe = roe.rank(axis=1, pct=True) > 0.54
    cond_rs = rs > 0.90
    base_mask_mb = (breakout & cond_rev & cond_vol & cond_odd & cond_roe & full_cash_filter & cond_rs)
    score_mb = rev.average(3) / rev.average(12)
    pos_mb = score_mb[base_mask_mb].is_largest(10)
    report_mb = sim(pos_mb.loc[start_date:], resample='M', resample_offset='11D', position_limit=0.35, upload=False, stop_loss=0.09)

    print("Running Component 2: Intention Factor...")
    volume = vol
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    current_ratio = data.get('fundamental_features:流動比率')
    per = data.get('price_earning_ratio:本益比')
    
    rs_intention = close / close.shift(15)
    s_intention = close / close.shift(50) - 1
    v_intention = close.pct_change().abs().rolling(50).sum()
    
    cond_rev_i = rev.average(3) > rev.average(12)
    cond_liq_i = (s_intention < 0.2) & (volume > 50000)
    cond_rs_i = rs_intention > rs_intention.quantile_row(0.9)
    cond_odd_i = odd_vol.average(10) > 150
    cond_yield_i = (yield_ratio >= yield_ratio.quantile(0.6, axis=1))
    cond_current_i = current_ratio > 100
    # roc = data.indicator("ROC", timeperiod=5)
    roc = close.pct_change(5) * 100
    cond_roc_i = roc > 0
    
    mask_i = cond_rev_i & cond_liq_i & cond_rs_i & cond_odd_i & cond_yield_i & cond_roc_i & cond_current_i
    per_inv_rank = (1 - per.rank(axis=1, pct=True)).fillna(0)
    score_i = (s_intention / v_intention / volume) * per_inv_rank
    pos_i = score_i[mask_i].is_largest(5)
    report_i = sim(pos_i.loc[start_date:], resample='Q', position_limit=0.33, stop_loss=0.15, upload=False)

    print("Running Component 3: F.E.T.E. Timing (00631L)...")
    signals_eco = data.get('tw_business_indicators:景氣對策信號(分)')
    score_eco = signals_eco['tw_business_indicators']
    adj_close = data.get('etl:adj_close')
    p0050 = adj_close['0050']
    p00631L = adj_close['00631L']
    dates = p00631L.loc[start_date:].index
    fund_score_daily = score_eco.reindex(dates, method='ffill').fillna(38)
    ema_60 = p0050.rolling(60).mean().reindex(dates)
    p0050_daily = p0050.reindex(dates)
    
    hybrid_sig = pd.Series(index=dates, dtype=bool)
    curr = False
    for d in dates:
        if (fund_score_daily.loc[d] <= 22) or (p0050_daily.loc[d] > ema_60.loc[d] and fund_score_daily.loc[d] < 34):
            curr = True
        if curr and (p0050_daily.loc[d] < ema_60.loc[d] or fund_score_daily.loc[d] >= 38):
            curr = False
        hybrid_sig.loc[d] = curr
    report_fete = sim(pd.DataFrame({'00631L': hybrid_sig}, index=dates), upload=False)

    return report_mb, report_i, report_fete

def run_allocation_simulation():
    r_mb, r_i, r_fete = fetch_strategy_curves()
    
    c_mb = r_mb.creturn
    c_i = r_i.creturn
    c_fete = r_fete.creturn
    
    common = c_mb.index.intersection(c_i.index).intersection(c_fete.index)
    returns = pd.DataFrame({
        'Mom_Breakout': c_mb.loc[common].pct_change().fillna(0),
        'Intention': c_i.loc[common].pct_change().fillna(0),
        'FETE_631L': c_fete.loc[common].pct_change().fillna(0)
    })
    
    # Allocations: [MB, Intention, FETE]
    allocs = {
        "High Alpha (40/40/20)": [0.4, 0.4, 0.2],
        "Momentum-Heavy (60/20/20)": [0.6, 0.2, 0.2],
        "Balanced (33.3% each)": [0.333, 0.333, 0.334]
    }
    
    plt.figure(figsize=(14, 10))
    print("\n--- CUSTOM PORTFOLIO ALLOCATION (2021-2026) ---")
    for name, w in allocs.items():
        p_ret = returns['Mom_Breakout']*w[0] + returns['Intention']*w[1] + returns['FETE_631L']*w[2]
        p_cum = (1 + p_ret).cumprod()
        
        ar = (p_cum.iloc[-1] ** (252/len(p_cum)) - 1) * 100
        md = (p_cum / p_cum.cummax() - 1).min() * 100
        print(f"{name:25}: CAGR {ar:7.2f}% | MDD {md:7.2f}%")
        p_cum.plot(label=name)
        
    plt.yscale('log')
    plt.legend()
    plt.title('Custom Portfolio: Momentum + Intention + FETE Timing')
    plt.savefig('custom_portfolio_plot.png')
    print("\nVisual saved to custom_portfolio_plot.png")

if __name__ == "__main__":
    run_allocation_simulation()
