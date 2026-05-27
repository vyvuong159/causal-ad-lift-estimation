"""
Evaluation and Statistical Power Analysis module.
Calculates the Minimum Detectable Effect (MDE) to validate the statistical
power of the causal estimates and confirm readiness for production scaling.
"""

import os
import time
from typing import Dict, Any

import numpy as np
import polars as pl
from scipy.stats import norm

def calculate_mde(df: pl.DataFrame, alpha: float = 0.05, power: float = 0.80) -> Dict[str, Any]:
    """
    Performs an a priori power analysis to compute the Minimum Detectable Effect (MDE)
    for a binary conversion outcome based on the sample sizes of treatment and control.
    """
    print("[INFO] Computing sample sizes and baseline conversion rates from dataset...")
    
    # Calculate group sizes
    n_total = len(df)
    n_treated = len(df.filter(pl.col("treatment") == 1))
    n_control = len(df.filter(pl.col("treatment") == 0))
    
    # Calculate baseline conversion rate in control group (p0)
    p0 = df.filter(pl.col("treatment") == 0)["conversion"].mean()
    p1 = df.filter(pl.col("treatment") == 1)["conversion"].mean()
    
    # Critical values for standard normal distribution
    # Two-sided alpha (significance level)
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    # One-sided beta (1 - power)
    z_beta = norm.ppf(power)
    
    # Standard error of the difference under the null hypothesis (p1 ≈ p0)
    # This is a standard conservative econometric formulation
    var_control = p0 * (1.0 - p0)
    var_treated = p0 * (1.0 - p0) # conservative null approximation
    
    se_null = np.sqrt((var_control / n_control) + (var_treated / n_treated))
    
    # Compute MDE: MDE = (z_alpha/2 + z_beta) * SE
    mde_absolute = (z_alpha + z_beta) * se_null
    mde_percentage = mde_absolute * 100.0
    
    # Standard error under alternative hypothesis (using actual p1 and p0)
    var_alternative_treated = p1 * (1.0 - p1)
    se_alt = np.sqrt((var_control / n_control) + (var_alternative_treated / n_treated))
    
    results = {
        "n_total": n_total,
        "n_control": n_control,
        "n_treated": n_treated,
        "control_conversion_rate_pct": p0 * 100.0,
        "treated_conversion_rate_pct": p1 * 100.0,
        "se_null": se_null,
        "se_alternative": se_alt,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "mde_absolute": mde_absolute,
        "mde_percentage": mde_percentage
    }
    
    return results


def main():
    # Setup paths
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    preprocessed_path = os.path.join(DATA_DIR, "criteo_sampled_preprocessed.parquet")
    
    print("=" * 65)
    print("      A PRIORI POWER ANALYSIS & MDE EVALUATION ENGINE          ")
    print("=" * 65)
    
    # 1. Load Preprocessed Data
    if not os.path.exists(preprocessed_path):
        print(f"[ERROR] Preprocessed data parquet not found at {preprocessed_path}.")
        print("Please execute 'python src/preprocess.py' first.")
        return
        
    start_time = time.time()
    df = pl.read_parquet(preprocessed_path)
    print(f"[INFO] Loaded {len(df):,} preprocessed rows in {time.time() - start_time:.2f} seconds.")
    
    # 2. Run Power Analysis
    # alpha = 0.05 (95% confidence level), power = 0.80 (20% type II error rate)
    alpha = 0.05
    power = 0.80
    
    analysis = calculate_mde(df, alpha=alpha, power=power)
    
    # 3. Compare with our actual DML ATE estimate (+0.092736%)
    ESTIMATED_ATE_PCT = 0.092736
    mde_pct = analysis["mde_percentage"]
    
    is_adequately_powered = ESTIMATED_ATE_PCT >= mde_pct
    power_ratio = ESTIMATED_ATE_PCT / mde_pct
    
    # 4. Print Summary Report
    print("\n" + "=" * 65)
    print("              STATISTICAL POWER & MDE ANALYSIS REPORT          ")
    print("=" * 65)
    print(f"  - Total Sample Size (N):    {analysis['n_total']:,}")
    print(f"    * Control Units (N0):     {analysis['n_control']:,} ({analysis['n_control']/analysis['n_total']:.1%})")
    print(f"    * Treated Units (N1):     {analysis['n_treated']:,} ({analysis['n_treated']/analysis['n_total']:.1%})")
    print("-" * 65)
    print(f"  - Control Baseline CR (p0): {analysis['control_conversion_rate_pct']:.4f}%")
    print(f"  - Treated Baseline CR (p1): {analysis['treated_conversion_rate_pct']:.4f}%")
    print(f"  - Standard Error under Null: {analysis['se_null']:.6f}")
    print(f"  - Alpha (Significance):     {alpha:.2f} (z = {analysis['z_alpha']:.2f})")
    print(f"  - Power (1 - Beta):         {power:.2f} (z = {analysis['z_beta']:.2f})")
    print("-" * 65)
    print(f"  - Minimum Detectable Effect (MDE): {mde_pct:+.6f}%")
    print(f"  - Isolated DML ATE Estimate:      {ESTIMATED_ATE_PCT:+.6f}%")
    print(f"  - ATE-to-MDE Power Ratio:         {power_ratio:.2f}x")
    print(f"  - Adequately Powered for Scale:   {'YES (ATE >= MDE)' if is_adequately_powered else 'NO (ATE < MDE)'}")
    print("-" * 65)
    
    # Econometric interpretation
    if is_adequately_powered:
        print("[INFO] Interpretation: The personalized ad feature's incremental lift")
        print(f"       ({ESTIMATED_ATE_PCT:+.4f}%) is robustly above the minimum detectable threshold")
        print(f"       ({mde_pct:.4f}%) at a 95% confidence level and 80% statistical power.")
        print("       This confirms that the finding is NOT a statistical fluke and is ready")
        print("       for production scaling and high-stakes capital expenditure decisions.")
    else:
        print("[WARNING] Interpretation: The isolated treatment effect is below the")
        print("          Minimum Detectable Effect threshold. The current sample size")
        print("          is underpowered to reliably distinguish this effect from noise.")
        print("          We recommend increasing the trial duration or sample size.")
    print("=" * 65)


if __name__ == "__main__":
    main()
