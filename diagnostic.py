import pandas as pd
from finlab import data
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def diagnostic():
    print("Running diagnostic on user filters...")
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    cash_dividend = data.get('financial_statement:發放現金股利')
    rev = data.get('monthly_revenue:當月營收')
    gross_margin_growth = data.get('fundamental_features:營業毛利成長率')
    eps = data.get('financial_statement:每股盈餘')

    # Conditions
    c1 = market_value.is_largest(150)
    c2 = (yield_ratio >= yield_ratio.quantile(0.6, axis=1))
    c3 = (rev.average(3) > rev.average(12))
    c4 = (gross_margin_growth > 0)
    c5 = (eps > 0)
    c6 = (close > close.average(120)) & (close > close.average(240))
    c7 = (close.pct_change().rolling(60).std().rank(axis=1, pct=True) < 0.5)
    c8 = (vol.average(20) > 200000)
    c9 = (cash_dividend.rolling(3).apply(lambda x: (x > 0).all()).fillna(0) > 0)

    conds = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    names = ["Market Cap Top 150", "Yield > 0.6 Q", "Rev Growth (3m>12m)", "Gross Margin > 0", "EPS > 0", "SMA 120/240", "Low Volatility", "Volume > 200k", "Cash Dividend 3y"]

    all_cond = (close > 0)
    print(f"{'Condition':<25} | {'Stocks Passing (Avg)':<25}")
    print("-" * 55)
    for name, c in zip(names, conds):
        passing = c.sum(axis=1).mean()
        print(f"{name:<25} | {passing:<25.2f}")
        all_cond &= c.reindex_like(all_cond).fillna(False)

    print("-" * 55)
    print(f"{'ALL CONDITIONS COMBINED':<25} | {all_cond.sum(axis=1).mean():<25.2f}")

if __name__ == "__main__":
    diagnostic()
