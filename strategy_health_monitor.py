import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class StrategyHealthMonitor:
    """
    Advanced Strategy Health Monitor (SHM)
    Uses professional statistical detection methods to evaluate strategy regime shifts.
    """
    def __init__(self, strategy_report, benchmark_report=None):
        self.report = strategy_report
        self.benchmark = benchmark_report
        self.returns = strategy_report.creturn.pct_change().dropna()
        
        if benchmark_report is not None:
            self.benchmark_returns = benchmark_report.creturn.pct_change().dropna()
            # Align dates
            common_idx = self.returns.index.intersection(self.benchmark_returns.index)
            self.returns = self.returns.loc[common_idx]
            self.benchmark_returns = self.benchmark_returns.loc[common_idx]
        else:
            self.benchmark_returns = None

    def _get_rolling_dd(self, window=252):
        """Calculate rolling drawdown series."""
        creturn = (1 + self.returns).cumprod()
        peak = creturn.rolling(window=window, min_periods=1).max()
        dd = (creturn / peak) - 1
        return dd

    def analyze_drawdown_health(self, lookback_ratio=0.8):
        """
        Dimension 1: Check if current DD > 95%/99% quantile of historical DD.
        Also checks recovery time.
        """
        # Split data into "Historical" and "Recent"
        split_idx = int(len(self.returns) * lookback_ratio)
        hist_returns = self.returns.iloc[:split_idx]
        recent_returns = self.returns.iloc[split_idx:]
        
        def calculate_dd(ret):
            c = (1 + ret).cumprod()
            return (c / c.cummax() - 1)
        
        hist_dd = calculate_dd(hist_returns)
        recent_dd = calculate_dd(recent_returns)
        
        # Quantiles
        q95 = hist_dd.quantile(0.05) # 5% lowest (max DDs are negative)
        q99 = hist_dd.quantile(0.01)
        curr_dd = recent_dd.iloc[-1]
        
        # Scoring (0-100)
        score = 100
        if curr_dd < q99: score = 0
        elif curr_dd < q95: score = 50
        
        return {"current_dd": curr_dd, "q95": q95, "q99": q99, "score": score}

    def analyze_alpha_significance(self, lookback_days=120):
        """
        Dimension 2: t-stat(alpha). Checks if edge is statistically significant.
        """
        if self.benchmark_returns is None:
            return {"score": 100, "t_stat": np.nan}
            
        recent_ret = self.returns.tail(lookback_days)
        recent_bench = self.benchmark_returns.tail(lookback_days)
        
        # Excess returns
        alpha_series = recent_ret - recent_bench
        mean_alpha = alpha_series.mean()
        std_alpha = alpha_series.std()
        
        if std_alpha == 0 or np.isnan(std_alpha):
            t_stat = 0
        else:
            t_stat = mean_alpha / (std_alpha / np.sqrt(len(alpha_series)))
            
        # Scoring: t > 2 is great, t < 1 is warning, t < 0 is failure
        score = 100
        if t_stat < 0: score = 0
        elif t_stat < 1: score = 40
        elif t_stat < 1.5: score = 70
        
        return {"t_stat": t_stat, "score": score}

    def analyze_structural_drift(self, lookback_ratio=0.8, window_corr=60):
        """
        Dimension 3: Correlation Drift (2-sigma) and KS-test for distribution drift.
        """
        results = {"score": 100}
        
        # 1. KS-Test (Comparison of Distributions)
        split_idx = int(len(self.returns) * lookback_ratio)
        hist_ret = self.returns.iloc[:split_idx]
        recent_ret = self.returns.iloc[split_idx:]
        
        ks_stat, p_val = stats.ks_2samp(hist_ret, recent_ret)
        
        ks_score = 100
        if p_val < 0.01: ks_score = 0   # High confidence of drift
        elif p_val < 0.05: ks_score = 40 # Significant drift
        
        # 2. Correlation Shift
        if self.benchmark_returns is not None:
            rolling_corr = self.returns.rolling(window_corr).corr(self.benchmark_returns)
            hist_corr = rolling_corr.iloc[:split_idx].dropna()
            recent_corr = rolling_corr.iloc[-1]
            
            mean_corr = hist_corr.mean()
            std_corr = hist_corr.std()
            
            z_corr = (recent_corr - mean_corr) / std_corr if std_corr > 0 else 0
            
            corr_score = 100
            if abs(z_corr) > 3: corr_score = 0
            elif abs(z_corr) > 2: corr_score = 50
            
            results["score"] = (ks_score + corr_score) / 2
            results["p_val_ks"] = p_val
            results["z_corr"] = z_corr
        else:
            results["score"] = ks_score
            results["p_val_ks"] = p_val
            
        return results

    def get_health_score(self):
        """Aggregates all dimensions into a final Strategy Health Score."""
        dd_res = self.analyze_drawdown_health()
        alpha_res = self.analyze_alpha_significance()
        drift_res = self.analyze_structural_drift()
        
        # Weights: DD (40%), Alpha (30%), Drift (30%)
        final_score = (dd_res['score'] * 0.4 + 
                      alpha_res['score'] * 0.3 + 
                      drift_res['score'] * 0.3)
        
        status = "OPTIMAL"
        if final_score < 50: status = "FAILURE"
        elif final_score < 80: status = "CAUTION"
            
        return {
            "score": round(final_score, 2),
            "status": status,
            "details": {
                "drawdown": dd_res,
                "alpha_t_stat": alpha_res,
                "drift": drift_res
            }
        }

    def output_report(self):
        res = self.get_health_score()
        print(f"\n{'='*40}")
        print(f" STRATEGY HEALTH REPORT: {res['status']}")
        print(f"{'='*40}")
        print(f"Final Health Score: {res['score']} / 100")
        print(f"\n--- Dimensions ---")
        print(f"1. Drawdown Health   : {res['details']['drawdown']['score']}/100 (Curr DD: {res['details']['drawdown']['current_dd']:.2%})")
        print(f"2. Alpha Significance : {res['details']['alpha_t_stat']['score']}/100 (t-stat: {res['details']['alpha_t_stat']['t_stat']:.2f})")
        print(f"3. Structural Drift   : {res['details']['drift']['score']}/100 (KS p-val: {res['details']['drift'].get('p_val_ks', 0):.4f})")
        
        if res['status'] == "FAILURE":
            print(f"\n[⚠️ ALERT] Strategy shows significant failure signs. Manual review required.")
        elif res['status'] == "CAUTION":
            print(f"\n[!] Warning: Statistical drift detected. Monitor closely.")
        else:
            print(f"\n[✓] Strategy is operating within historical norms.")
    def plot_health_timeline(self, save_path='strategy_health_timeline.png'):
        """
        Visualizes the strategy's health over time.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        
        # 1. Price & Drawdown
        creturn = (1 + self.returns).cumprod()
        axes[0].plot(creturn, label='Strategy Equity', color='blue')
        axes[0].set_title('Strategy Equity (Normalized)')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Rolling Drawdown vs Quantiles
        dd = (creturn / creturn.cummax() - 1)
        dd_stats = self.analyze_drawdown_health()
        axes[1].plot(dd, label='Current Drawdown', color='red')
        axes[1].axhline(dd_stats['q95'], color='orange', linestyle='--', label='95% Quantile')
        axes[1].axhline(dd_stats['q99'], color='darkred', linestyle='--', label='99% Quantile')
        axes[1].set_title('Drawdown Analysis vs Historical Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Rolling Alpha t-stat (60 Day)
        if self.benchmark_returns is not None:
            alpha = self.returns - self.benchmark_returns
            rolling_t = alpha.rolling(60).mean() / (alpha.rolling(60).std() / np.sqrt(60))
            axes[2].plot(rolling_t, label='Rolling 60D t-stat(alpha)', color='green')
            axes[2].axhline(1, color='orange', linestyle='--')
            axes[2].axhline(2, color='darkgreen', linestyle=':', label='High Significance (2.0)')
            axes[2].set_title('Alpha Significance Timeline (t-stat)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Health timeline plot saved to {save_path}")
        plt.close()
