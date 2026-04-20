import pandas as pd
import matplotlib.pyplot as plt
from finlab import data
from finlab.backtest import sim
import ssl

# Bypass SSL if needed
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

def analyze_market_timing():
    print("Fetching index data...")
    # Get the Weighted Total Return Index
    index_data = data.get('benchmark_return:發行量加權股價報酬指數')
    
    # Renaming for sim() compatibility (requires numeric-looking IDs)
    dummy_id = '0000'
    close = index_data['發行量加權股價報酬指數'].rename(dummy_id)
    
    print("Calculating factors...")
    # Calculate Moving Averages
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    sma240 = close.rolling(240).mean()
    
    # Calculate RSI
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1+rs))
    
    rsi = calculate_rsi(close)
    
    # Define Strategy Signals
    # 1. SMA 60 Timing: Long when price > SMA 60
    # Use shift(1) to avoid lookahead bias
    pd.set_option('future.no_silent_downcasting', True)
    signal_sma60 = (close > sma60).shift(1).fillna(False)
    
    # 2. SMA 20/60 Cross: Long when SMA 20 > SMA 60
    signal_cross = (sma20 > sma60).shift(1).fillna(False)

    print("Fetching market breadth data (all stocks)...")
    # This might take a bit of memory, but it's powerful
    all_prices = data.get('price:收盤價')
    all_sma20 = all_prices.rolling(20).mean()
    
    # Breadth: Percentage of stocks above their SMA 20
    breadth = (all_prices > all_sma20).sum(axis=1) / all_prices.notnull().sum(axis=1)
    
    # 3. Breadth Timing: Long when breadth > 0.3 (30% of stocks are in uptrend)
    # This is a very sensitive indicator
    signal_breadth = (breadth > 0.3).shift(1).fillna(False)
    
    # Create position DataFrame for sim()
    position_sma60 = pd.DataFrame({dummy_id: signal_sma60})
    position_cross = pd.DataFrame({dummy_id: signal_cross})
    position_breadth = pd.DataFrame({dummy_id: signal_breadth})
    
    print("Running backtests (manual calculation)...")
    # Returns of the index
    returns = close.pct_change()
    
    # Calculate Strategy Returns
    strategy_returns_sma60 = signal_sma60 * returns
    strategy_returns_cross = signal_cross * returns
    strategy_returns_breadth = signal_breadth * returns
    
    # Cumulative Returns (Equity Curves)
    equity_sma60 = (1 + strategy_returns_sma60).cumprod()
    equity_cross = (1 + strategy_returns_cross).cumprod()
    equity_breadth = (1 + strategy_returns_breadth).cumprod()
    benchmark_creturn = (1 + returns).cumprod()

    # Helper function for metrics
    def calculate_stats(equity, returns_series, benchmark_returns=None):
        # CAGR
        days = (equity.index[-1] - equity.index[0]).days
        cagr = (equity.iloc[-1] ** (365.25 / days)) - 1
        
        # Max Drawdown
        drawdown = (equity / equity.cummax()) - 1
        mdd = drawdown.min()
        
        # Sharpe (Annualized)
        vol = returns_series.std() * (252 ** 0.5)
        sharpe = (returns_series.mean() * 252) / vol if vol != 0 else 0
        
        # Relative Metrics
        alpha, beta, info_ratio = None, None, None
        if benchmark_returns is not None:
            # Beta: cov(r, b) / var(b)
            # Alignment is important
            combined = pd.DataFrame({'strat': returns_series, 'bench': benchmark_returns}).dropna()
            if len(combined) > 1:
                cov = combined.cov().iloc[0, 1]
                var_bench = combined['bench'].var()
                beta = cov / var_bench if var_bench != 0 else 0
                
                # Alpha (Annualized excess return over beta-adjusted benchmark)
                # Alpha = Strat_Return - Beta * Bench_Return
                alpha = (returns_series.mean() * 252) - beta * (benchmark_returns.mean() * 252)
                
                # Information Ratio = Excess Return / Tracking Error
                excess_returns = returns_series - benchmark_returns
                tracking_error = excess_returns.std() * (252 ** 0.5)
                info_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0
        
        return cagr, mdd, sharpe, alpha, beta, info_ratio

    def print_metrics(name, stats):
        cagr, mdd, sharpe, alpha, beta, ir = stats
        print(f"\n--- Performance Metrics ({name}) ---")
        print(f"Annual Return (CAGR): {cagr:.2%}")
        print(f"Max Drawdown: {mdd:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        if alpha is not None:
            print(f"Alpha (Annualized): {alpha:.2%}")
            print(f"Beta: {beta:.2f}")
            print(f"Information Ratio: {ir:.2f}")

    # Calculate Benchmark Stats first
    stats_bench = calculate_stats(benchmark_creturn, returns)
    
    # Calculate Strategy Stats
    stats_60 = calculate_stats(equity_sma60, strategy_returns_sma60, returns)
    stats_cross = calculate_stats(equity_cross, strategy_returns_cross, returns)
    stats_breadth = calculate_stats(equity_breadth, strategy_returns_breadth, returns)

    # Print All
    print_metrics("SMA 60 Timing", stats_60)
    print_metrics("SMA 20/60 Cross", stats_cross)
    print_metrics("Market Breadth > 0.3", stats_breadth)
    print_metrics("Buy & Hold (Benchmark)", stats_bench)

    # Extract MDDs for plotting labels
    mdd60 = stats_60[1]
    mddc = stats_cross[1]
    mddb = stats_breadth[1]
    mdd_bench = stats_bench[1]

    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Grid of 2 subplots: Returns and Breadth level
    ax1 = plt.subplot(2, 1, 1)
    equity_sma60.plot(ax=ax1, label=f'SMA 60 Timing (MDD: {mdd60:.1%})', color='blue')
    equity_cross.plot(ax=ax1, label=f'SMA 20/60 Cross (MDD: {mddc:.1%})', color='green')
    equity_breadth.plot(ax=ax1, label=f'Breadth > 0.3 (MDD: {mddb:.1%})', color='red', linewidth=2)
    benchmark_creturn.plot(ax=ax1, label=f'Buy & Hold (MDD: {mdd_bench:.1%})', color='gray', alpha=0.5, linestyle='--')
    
    ax1.set_title('Market Timing Strategy Comparison (TAIEX Total Return)')
    ax1.set_ylabel('Cumulative Return (Log Scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    
    # Subplot 2: Market Breadth Level
    ax2 = plt.subplot(2, 1, 2)
    breadth.plot(ax=ax2, label='Market Breadth (% stocks > SMA20)', color='orange', alpha=0.7)
    ax2.axhline(0.3, color='red', linestyle='--', label='Threshold (0.3)')
    ax2.set_title('Market Breadth Factor')
    ax2.set_ylabel('Breadth Ratio')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = 'market_timing_plot.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    analyze_market_timing()
