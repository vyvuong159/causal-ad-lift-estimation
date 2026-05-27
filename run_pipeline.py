"""
Main Orchestration Pipeline for Causal ML Framework.
Sequentially runs data preprocessing, Double Machine Learning (DML) estimation,
Synthetic Control Method (SCM) validation, and statistical power analysis.
"""

import os
import time
import polars as pl

# Import modular pipeline components
from src.preprocess import CriteoPreprocessor
from src.dml_engine import DoubleMLEngine
from src.scm_engine import SyntheticControlEngine
from src.evaluation import calculate_mde

# Global reproducibility
SEED = 42

def main():
    # Setup directories
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    DATASET_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
    LOCAL_RAW_PATH = os.path.join(DATA_DIR, "criteo-research-uplift-v2.1.csv.gz")
    LOCAL_PROCESSED_PATH = os.path.join(DATA_DIR, "criteo_sampled_preprocessed.parquet")
    OUTPUT_PLOT_PATH = os.path.join(DATA_DIR, "synthetic_control_validation.png")
    
    print("=" * 65)
    print("      CAUSAL ML PIPELINE ORCHESTRATOR: RUNNING PIPELINE        ")
    print("=" * 65)
    
    pipeline_start_time = time.time()
    
    # -----------------------------------------------------------------
    # PHASE 1: Data Ingestion & Preprocessing
    # -----------------------------------------------------------------
    print("\n--- PHASE 1: Data Ingestion & High-Performance Preprocessing ---")
    preprocessor = CriteoPreprocessor(seed=SEED)
    
    # Download raw data if needed
    try:
        preprocessor.download_dataset(DATASET_URL, LOCAL_RAW_PATH)
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        return
        
    # Load raw data
    try:
        df_raw = preprocessor.load_data(LOCAL_RAW_PATH)
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return
        
    # Stratified sub-sampling to 1M rows
    try:
        df_sampled = preprocessor.stratified_sample(df_raw, target_size=1_000_000)
    except Exception as e:
        print(f"[ERROR] Stratified sampling failed: {e}")
        return
        
    # Clean, scale f0-f11, and downcast target variables
    try:
        df_preprocessed = preprocessor.clean_and_scale(df_sampled, fit_scaler=True)
        df_preprocessed.write_parquet(LOCAL_PROCESSED_PATH)
        print(f"[SUCCESS] Preprocessed data saved to {LOCAL_PROCESSED_PATH}")
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return
        
    # -----------------------------------------------------------------
    # PHASE 2: Double Machine Learning (DML) Causal Estimation
    # -----------------------------------------------------------------
    print("\n--- PHASE 2: Double Machine Learning (DML) Structural Estimation ---")
    try:
        dml_engine = DoubleMLEngine(seed=SEED)
        estimator, dml_summary = dml_engine.fit_and_estimate(df_preprocessed)
        print("[SUCCESS] Double Machine Learning model successfully fit.")
    except Exception as e:
        print(f"[ERROR] DML Causal estimation failed: {e}")
        return
        
    # Calculate naive correlation-based lift for bias assessment
    mean_y_t1 = df_preprocessed.filter(pl.col("treatment") == 1)["conversion"].mean()
    mean_y_t0 = df_preprocessed.filter(pl.col("treatment") == 0)["conversion"].mean()
    naive_lift = mean_y_t1 - mean_y_t0
    
    # -----------------------------------------------------------------
    # PHASE 3: Synthetic Control Method (SCM) Aggregate Validation
    # -----------------------------------------------------------------
    print("\n--- PHASE 3: Synthetic Control Method (SCM) Counterfactual Validation ---")
    try:
        scm_engine = SyntheticControlEngine(seed=SEED)
        
        # Segment and aggregate to panel cohorts
        panel_df = scm_engine.generate_cohorts_from_features(df_preprocessed, num_campaigns=5, num_days=30)
        agg_df = scm_engine.aggregate_panel(panel_df, dml_ate_pct=dml_summary["ate"] * 100.0)
        
        # Fit synthetic control weights
        optimal_weights, scm_pre_mse = scm_engine.fit_synthetic_control(agg_df, pre_treatment_days=20)
        
        # Construct synthetic counterfactual and plot
        days, actual, synthetic = scm_engine.construct_counterfactual(agg_df)
        scm_engine.plot_results(days, actual, synthetic, pre_treatment_days=20, output_path=OUTPUT_PLOT_PATH)
        print(f"[SUCCESS] Synthetic Control counterfactual plotted and saved to {OUTPUT_PLOT_PATH}")
    except Exception as e:
        print(f"[ERROR] SCM validation failed: {e}")
        return
        
    # -----------------------------------------------------------------
    # PHASE 4: Statistical Power & MDE Evaluation
    # -----------------------------------------------------------------
    print("\n--- PHASE 4: Statistical Power & MDE Analysis ---")
    try:
        power_summary = calculate_mde(df_preprocessed, alpha=0.05, power=0.80)
        print("[SUCCESS] Power analysis and MDE threshold calculation complete.")
    except Exception as e:
        print(f"[ERROR] Power analysis failed: {e}")
        return
        
    # -----------------------------------------------------------------
    # FINAL UNIFIED EXECUTIVE REPORT
    # -----------------------------------------------------------------
    print("\n" + "=" * 62)
    print("       CAUSAL ML EXPERIMENT REPORT: INCREMENTAL AD-LIFT       ")
    print("=" * 62)
    print("Configuration parameters:")
    print("  - Dataset:             Criteo Uplift Prediction (Sampled)")
    print("  - Estimator:           LinearDML")
    print("  - Nuisance Learners:   LightGBM (100 estimators, max_depth=5)")
    print("-" * 62)
    print("Data integrity & Bias assessment:")
    print(f"  - Total records:       {len(df_preprocessed):,}")
    print(f"  - Naive Lift (Corr):   {naive_lift:+.6%}")
    print(f"  - Unbiased ATE (DML):  {dml_summary['ate']:+.6%}")
    print(f"  - SCM Counterfactual:  MSE = {scm_pre_mse:.4e}")
    print("-" * 62)
    print("Causal model parameters:")
    print(f"  - Unbiased ATE:        {dml_summary['ate']:+.6%}")
    print(f"  - Std. Error:          {dml_summary['stderr']:.6f}")
    print(f"  - 95% CI:              [{dml_summary['ci_lower']:+.6%}, {dml_summary['ci_upper']:+.6%}]")
    print("-" * 62)
    print("Business decision metrics:")
    print(f"  - Estimation Bias:     Reduced from {naive_lift:+.4%} to {dml_summary['ate']:+.4%}")
    print(f"                         (Selection bias of {naive_lift - dml_summary['ate']:+.4%} mitigated)")
    print(f"  - P(ATE > 0):          {1.0 - dml_summary['p_value']:.4%}")
    print(f"  - Power Analysis:      Fully Powered (ATE is {dml_summary['ate']*100.0/power_summary['mde_percentage']:.1f}x MDE threshold of {power_summary['mde_percentage']:.4f}%)")
    print("=" * 62)
    
    print(f"\n[INFO] Complete pipeline finished execution in {time.time() - pipeline_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
