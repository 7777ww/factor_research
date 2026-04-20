import pandas as pd
from finlab import data
from finlab.backtest import sim
from strategy_health_monitor import StrategyHealthMonitor
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def run_golden_trio_audit():
    print("Fetching data for Health Audit...")
    adj_close = data.get('etl:adj_close')
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    rev = data.get('monthly_revenue:當月營收')
    odd_vol = data.get('intraday_odd_lot_trade:成交股數')
    roe = data.get('fundamental_features:ROE稅後')
    full_cash_filter = data.get('etl:full_cash_delivery_stock_filter')
    benchmark_return = data.get('benchmark_return:發行量加權股價報酬指數')
    
    START_DATE = '2020-01-01' # Longer history for better quantile distribution
    
    # 0. Benchmark Report (0050)
    p0050 = adj_close['0050'].loc[START_DATE:]
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(p0050)}, index=p0050.index), upload=False)
    
    # 1. Momentum Breakout Audit
    print("\n[Auditing Component 1: Momentum Breakout]")
    rs_period = 120
    stock_return = close.pct_change(rs_period) + 1
    market_return = benchmark_return.pct_change(rs_period) + 1
    market_return_series = market_return.iloc[:, 0].reindex(close.index, method='ffill')
    rs = stock_return.div(market_return_series, axis=0).fillna(1)
    breakout = close >= close.rolling(120).max()
    mask_mb = (breakout & (rev.average(3) > rev.average(12)) & (vol.average(5) > vol.average(20)*1.2) & (odd_vol.average(10) > 1000) & (roe.rank(axis=1, pct=True) > 0.54) & full_cash_filter & (rs > 0.90))
    pos_mb = (rev.average(3) / rev.average(12))[mask_mb].is_largest(10)
    report_mb = sim(pos_mb.loc[START_DATE:], resample='M', resample_offset='11D', position_limit=0.35, upload=False, stop_loss=0.09)
    
    shm_mb = StrategyHealthMonitor(report_mb, report_0050)
    shm_mb.output_report()
    shm_mb.plot_health_timeline('mb_health_dashboard.png')

    # 2. Hybrid F.E.T.E. Audit
    print("\n[Auditing Component 2: Hybrid F.E.T.E. (00631L)]")
    signals_eco = data.get('tw_business_indicators:景氣對策信號(分)')
    p00631L = adj_close['00631L'].loc[START_DATE:]
    dates = p00631L.index
    fund_score = signals_eco['tw_business_indicators'].reindex(dates, method='ffill').fillna(38)
    ema_60 = p0050.rolling(60).mean().reindex(dates)
    curr = False
    sig = pd.Series(index=dates, dtype=bool)
    for d in dates:
        if (fund_score.loc[d] <= 22) or (p0050.loc[d] > ema_60.loc[d] and fund_score.loc[d] < 34): curr = True
        if curr and (p0050.loc[d] < ema_60.loc[d] or fund_score.loc[d] >= 38): curr = False
        sig.loc[d] = curr
    report_fete = sim(pd.DataFrame({'00631L': sig}, index=dates), upload=False)
    
    shm_fete = StrategyHealthMonitor(report_fete, report_0050)
    shm_fete.output_report()
    shm_fete.plot_health_timeline('fete_health_dashboard.png')

if __name__ == "__main__":
    run_golden_trio_audit()
