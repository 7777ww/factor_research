import pandas as pd
import json
import os
import datetime
from finlab import data
from finlab.backtest import sim
from strategy_health_monitor import StrategyHealthMonitor
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def generate_html_dashboard(results):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quant Portfolio Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; color: #333; }}
            h1 {{ text-align: center; color: #1a237e; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }}
            .card-title {{ font-size: 1.25rem; font-weight: bold; margin: 0; }}
            .status {{ padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9rem; }}
            .status.OPTIMAL {{ background-color: #e8f5e9; color: #2e7d32; }}
            .status.CAUTION {{ background-color: #fff3e0; color: #ef6c00; }}
            .status.FAILURE {{ background-color: #ffebee; color: #c62828; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
            .metric-item {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
            .metric-label {{ font-size: 0.85rem; color: #666; margin-bottom: 5px; }}
            .metric-value {{ font-size: 1.5rem; font-weight: bold; color: #1a237e; }}
            .positions {{ margin-top: 15px; }}
            .positions pre {{ background: #f1f3f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .update-time {{ text-align: center; color: #666; font-size: 0.9rem; margin-bottom: 20px;}}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Quant Portfolio Dashboard</h1>
            <div class="update-time">Last Updated: {update_time}</div>
            
            {cards}
        </div>
    </body>
    </html>
    """

    card_template = """
    <div class="card">
        <div class="card-header">
            <h2 class="card-title">{name}</h2>
            <span class="status {status}">{status} ({score}/100)</span>
        </div>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-label">Current Drawdown</div>
                <div class="metric-value">{curr_dd}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Alpha t-stat</div>
                <div class="metric-value">{t_stat}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">KS p-value</div>
                <div class="metric-value">{p_val}</div>
            </div>
        </div>
        <div class="positions">
            <strong>Current Positions (Target):</strong>
            <pre>{positions}</pre>
        </div>
    </div>
    """

    cards_html = ""
    for r in results:
        curr_dd = round(r["details"]["drawdown"]["current_dd"] * 100, 2)
        t_stat = round(r["details"]["alpha_t_stat"]["t_stat"], 2)
        p_val = round(r["details"]["drift"].get("p_val_ks", 0), 4)
        
        # format positions
        pos_str = "\n".join([f"{k}: {v}" for k, v in r["positions"].items()]) if r["positions"] else "None / Cash"

        cards_html += card_template.format(
            name=r["name"],
            status=r["status"],
            score=r["score"],
            curr_dd=curr_dd,
            t_stat=t_stat,
            p_val=p_val,
            positions=pos_str
        )

    final_html = html_template.format(
        update_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        cards=cards_html
    )

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("Dashboard generated at docs/index.html")


def run_daily_update():
    print("Fetching data for Daily Update...")
    adj_close = data.get('etl:adj_close')
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    rev = data.get('monthly_revenue:當月營收')
    odd_vol = data.get('intraday_odd_lot_trade:成交股數')
    roe = data.get('fundamental_features:ROE稅後')
    full_cash_filter = data.get('etl:full_cash_delivery_stock_filter')
    benchmark_return = data.get('benchmark_return:發行量加權股價報酬指數')
    
    START_DATE = '2020-01-01' 
    
    p0050 = adj_close['0050'].loc[START_DATE:]
    report_0050 = sim(pd.DataFrame({'0050': [True]*len(p0050)}, index=p0050.index), upload=False)
    
    results = []

    # 1. Momentum Breakout
    print("Running Momentum Breakout...")
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
    res_mb = shm_mb.get_health_score()
    res_mb['name'] = 'Momentum Breakout'
    # Get last row of pos_mb where there are True/values
    last_pos_mb = pos_mb.iloc[-1]
    res_mb['positions'] = last_pos_mb[last_pos_mb > 0].to_dict() if not last_pos_mb.empty else {}
    results.append(res_mb)

    # 2. Intention Factor
    print("Running Intention Factor...")
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
    roc_i = close.pct_change(5) * 100
    cond_roc_i = roc_i > 0
    
    mask_i = cond_rev_i & cond_liq_i & cond_rs_i & cond_odd_i & cond_yield_i & cond_roc_i & cond_current_i
    per_inv_rank = (1 - per.rank(axis=1, pct=True)).fillna(0)
    score_i = (s_intention / v_intention / volume) * per_inv_rank
    pos_i = score_i[mask_i].is_largest(5)
    report_i = sim(pos_i.loc[START_DATE:], resample='Q', position_limit=0.33, stop_loss=0.15, upload=False)

    shm_i = StrategyHealthMonitor(report_i, report_0050)
    res_i = shm_i.get_health_score()
    res_i['name'] = 'Intention Factor'
    last_pos_i = pos_i.iloc[-1]
    res_i['positions'] = last_pos_i[last_pos_i > 0].to_dict() if not last_pos_i.empty else {}
    results.append(res_i)

    # 3. Hybrid F.E.T.E.
    print("Running Hybrid F.E.T.E. (00631L)...")
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
    res_fete = shm_fete.get_health_score()
    res_fete['name'] = 'F.E.T.E.'
    last_sig = sig.iloc[-1]
    res_fete['positions'] = {"00631L": 1.0} if last_sig else {}
    results.append(res_fete)

    generate_html_dashboard(results)

if __name__ == "__main__":
    run_daily_update()
