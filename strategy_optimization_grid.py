import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def run_optimization_grid():
    print("Fetching data for Parameter Optimization...")
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    market_value = data.get('etl:market_value')
    rev = data.get('monthly_revenue:當月營收')
    rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    roe = data.get('fundamental_features:ROE稅後')
    
    # 1. Universe & Quality (Baseline)
    universe = market_value.is_largest(300)
    quality_cond = (roe > 0) & (rev.average(3) > rev.average(12))
    liquidity = (vol.average(20) > 200000)
    
    # Pre-calculate factors to save time
    def z_score(df):
        return (df - df.mean(axis=1, skipna=True)) / df.std(axis=1, skipna=True)
        
    z_yield = z_score(yield_ratio).fillna(0)
    z_mom = z_score(close.rise(60)).fillna(0)
    z_growth = z_score(rev_yoy).fillna(0)
    z_accel = z_score(rev_yoy - rev_yoy.rolling(3).mean().shift(1)).fillna(0)
    
    optimization_results = []
    
    # A. Test TOP_K (Concentration Sensitivity)
    print("\n--- Testing Concentration (Top K) ---")
    for k in [10, 20, 30]:
        # Balanced weights (0.25 each), SMA 120
        conds = universe & quality_cond & liquidity & (close > close.average(120))
        score = z_yield * 0.25 + z_mom * 0.25 + z_growth * 0.25 + z_accel * 0.25
        pos = score[conds].is_largest(k)
        
        report = sim(pos, resample="M", upload=False)
        stats = report.get_stats()
        optimization_results.append({"Scenario": f"Top_{k}", "CAGR": stats['cagr'], "MDD": stats['max_drawdown'], "Sharpe": stats['monthly_sharpe']})
        print(f"Top_{k} | CAGR: {stats['cagr']:.2%} | MDD: {stats['max_drawdown']:.2%} | Sharpe: {stats['monthly_sharpe']:.2f}")

    # B. Test SMA Period (Trend Sensitivity)
    print("\n--- Testing Trend Sensitivity (SMA) ---")
    for s_period in [60, 120, 240]:
        # Top 20, Balanced weights
        conds = universe & quality_cond & liquidity & (close > close.average(s_period))
        score = z_yield * 0.25 + z_mom * 0.25 + z_growth * 0.25 + z_accel * 0.25
        pos = score[conds].is_largest(20)
        
        report = sim(pos, resample="M", upload=False)
        stats = report.get_stats()
        optimization_results.append({"Scenario": f"SMA_{s_period}", "CAGR": stats['cagr'], "MDD": stats['max_drawdown'], "Sharpe": stats['monthly_sharpe']})
        print(f"SMA_{s_period} | CAGR: {stats['cagr']:.2%} | MDD: {stats['max_drawdown']:.2%} | Sharpe: {stats['monthly_sharpe']:.2f}")

    # C. Test Weights (Factor Style Sensitivity)
    print("\n--- Testing Style Sensitivity (Weights) ---")
    # Styles: [Yield, Mom, Growth, Accel]
    styles = {
        "Yield-Heavy": [0.5, 0.1, 0.2, 0.2],
        "Growth-Heavy": [0.2, 0.2, 0.3, 0.3],
        "Momentum-Heavy-Pro": [0.1, 0.4, 0.2, 0.3],
        "Balanced": [0.25, 0.25, 0.25, 0.25]
    }
    for name, w in styles.items():
        conds = universe & quality_cond & liquidity & (close > close.average(120))
        score = z_yield * w[0] + z_mom * w[1] + z_growth * w[2] + z_accel * w[3]
        pos = score[conds].is_largest(20)
        
        report = sim(pos, resample="M", upload=False)
        stats = report.get_stats()
        optimization_results.append({"Scenario": f"Style_{name}", "CAGR": stats['cagr'], "MDD": stats['max_drawdown'], "Sharpe": stats['monthly_sharpe']})
        print(f"{name} | CAGR: {stats['cagr']:.2%} | MDD: {stats['max_drawdown']:.2%} | Sharpe: {stats['monthly_sharpe']:.2f}")

    # Final Leaderboard
    df_opt = pd.DataFrame(optimization_results)
    print("\n--- ALL SCENARIOS RANKED BY SHARPE ---")
    print(df_opt.sort_values("Sharpe", ascending=False).to_string(index=False))

if __name__ == "__main__":
    run_optimization_grid()
