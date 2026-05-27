"""
Double Machine Learning (DML) Engine for Incremental Ad-Lift Estimation.
Utilizes EconML LinearDML and LightGBM nuisance learners.
"""

import os
import time
from typing import Tuple, Dict, Any

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor, LGBMClassifier
from econml.dml import LinearDML

# Global reproducibility
SEED = 42

class DoubleMLEngine:
    """
    Double Machine Learning Causal Estimation Engine.
    Uses EconML to isolate the unbiased Average Treatment Effect (ATE).
    """
    
    def __init__(self, seed: int = SEED):
        self.seed = seed
        
        # 1. Initialize Nuisance Learners with standard parameters
        # Outcome model: LGBMRegressor predicts conversion probability (Y)
        self.model_y = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1
        )
        
        # Treatment model: LGBMClassifier predicts propensity scores (T|X)
        self.model_t = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1
        )
        
        # 2. Configure LinearDML
        # discrete_treatment=True specifies binary/categorical treatment
        # cv=3 specifies 3-fold cross-fitting to mitigate overfitting
        self.estimator = LinearDML(
            model_y=self.model_y,
            model_t=self.model_t,
            discrete_treatment=True,
            cv=3,
            random_state=self.seed
        )
        
    def fit_and_estimate(self, df: pl.DataFrame) -> Tuple[LinearDML, Dict[str, Any]]:
        """
        Fits the LinearDML model on the preprocessed Polars DataFrame.
        Computes the unbiased ATE point estimate and its 95% Confidence Interval.
        """
        print("[INFO] Preparing data matrices for EconML...")
        start_time = time.time()
        
        # Extract variables and convert to NumPy arrays
        Y = df["conversion"].to_numpy().astype(np.float64)
        T = df["treatment"].to_numpy().astype(np.int8)
        
        # Feature columns f0 through f11
        feature_cols = [f"f{i}" for i in range(12)]
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        
        print(f"[INFO] Fitting Double Machine Learning (LinearDML) model on {len(df):,} rows...")
        print("[INFO] Training Y and T nuisance models via 3-fold cross-fitting. Please wait...")
        
        # Fit with statsmodels inference to calculate standard errors and CI
        self.estimator.fit(Y, T, X=X, inference="statsmodels")
        
        fit_duration = time.time() - start_time
        print(f"[INFO] Causal model fitting completed in {fit_duration:.2f} seconds.")
        
        # 3. Perform Inference
        print("[INFO] Calculating ATE point estimate and 95% Confidence Interval...")
        ate_inference_obj = self.estimator.ate_inference(X)
        
        # Extract ATE and confidence bounds (using 0.05 alpha for 95% confidence)
        # We use PopulationSummaryResults attributes to avoid individual/population stderr issues
        ate_point = float(np.ravel(ate_inference_obj.mean_point)[0])
        ate_se = float(np.ravel(ate_inference_obj.stderr_mean)[0])
        p_value = float(np.ravel(ate_inference_obj.pvalue())[0])
        
        # Calculate conf_int_mean for 95% Confidence Interval on the Average Treatment Effect
        ci_lower, ci_upper = ate_inference_obj.conf_int_mean(alpha=0.05)
        ate_lower = float(np.ravel(ci_lower)[0])
        ate_upper = float(np.ravel(ci_upper)[0])
        
        results_summary = {
            "ate": ate_point,
            "ci_lower": ate_lower,
            "ci_upper": ate_upper,
            "stderr": ate_se,
            "p_value": p_value,
            "fit_time_seconds": fit_duration,
            "inference_obj": ate_inference_obj
        }
        
        return self.estimator, results_summary

    def check_propensity_overlap(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Diagnoses the Common Support (Positivity) assumption by fitting a propensity score
        model on features X and evaluating the overlap of predicted propensity scores
        between treated and control units. Uses out-of-sample cross-validation for honest metrics.
        """
        print("[INFO] Evaluating Common Support (Positivity/Overlap) assumption...")
        
        # Prepare feature matrix X and treatment vector T
        feature_cols = [f"f{i}" for i in range(12)]
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        T = df["treatment"].to_numpy().astype(np.int8)
        
        # Cross-validated out-of-sample propensity scores to avoid overfitting
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        propensity_scores = np.zeros(len(df))
        
        for train_idx, val_idx in kf.split(X):
            fold_model = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1
            )
            fold_model.fit(X[train_idx], T[train_idx])
            propensity_scores[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]
            
        # Get propensity scores for treated and control units
        scores_t1 = propensity_scores[T == 1]
        scores_t0 = propensity_scores[T == 0]
        
        diagnostics = {
            "overall_mean": float(np.mean(propensity_scores)),
            "treated": {
                "min": float(np.min(scores_t1)),
                "p5": float(np.percentile(scores_t1, 5)),
                "median": float(np.median(scores_t1)),
                "p95": float(np.percentile(scores_t1, 95)),
                "max": float(np.max(scores_t1))
            },
            "control": {
                "min": float(np.min(scores_t0)),
                "p5": float(np.percentile(scores_t0, 5)),
                "median": float(np.median(scores_t0)),
                "p95": float(np.percentile(scores_t0, 95)),
                "max": float(np.max(scores_t0))
            }
        }
        
        overlap_lower = max(diagnostics["treated"]["min"], diagnostics["control"]["min"])
        overlap_upper = min(diagnostics["treated"]["max"], diagnostics["control"]["max"])
        has_overlap = overlap_lower < overlap_upper
        
        near_zero_or_one = np.sum((propensity_scores < 0.01) | (propensity_scores > 0.99))
        near_zero_or_one_pct = near_zero_or_one / len(df)
        
        print("\n" + "=" * 65)
        print("          PROPENSITY SCORE OVERLAP DIAGNOSTIC (POSITIVITY)      ")
        print("=" * 65)
        print(f"  - Overall Mean Propensity:   {diagnostics['overall_mean']:.4%}")
        print("-" * 65)
        print("  - Treated Group (T=1) Propensity Distribution:")
        print(f"    * Min:    {diagnostics['treated']['min']:.4%}")
        print(f"    * 5th %:  {diagnostics['treated']['p5']:.4%}")
        print(f"    * Median: {diagnostics['treated']['median']:.4%}")
        print(f"    * 95th %: {diagnostics['treated']['p95']:.4%}")
        print(f"    * Max:    {diagnostics['treated']['max']:.4%}")
        print("-" * 65)
        print("  - Control Group (T=0) Propensity Distribution:")
        print(f"    * Min:    {diagnostics['control']['min']:.4%}")
        print(f"    * 5th %:  {diagnostics['control']['p5']:.4%}")
        print(f"    * Median: {diagnostics['control']['median']:.4%}")
        print(f"    * 95th %: {diagnostics['control']['p95']:.4%}")
        print(f"    * Max:    {diagnostics['control']['max']:.4%}")
        print("-" * 65)
        print(f"  - Common Support Range:      [{overlap_lower:.4%}, {overlap_upper:.4%}]")
        print(f"  - Strict Positivity Check:   {'PASS' if has_overlap and near_zero_or_one_pct < 0.01 else 'WARNING'}")
        print(f"  - Extreme Propensities:      {near_zero_or_one:,} units ({near_zero_or_one_pct:.2%}) have P(T|X) < 1% or > 99%")
        print("=" * 65 + "\n")
        
        diagnostics["has_overlap"] = has_overlap
        diagnostics["extreme_units_pct"] = near_zero_or_one_pct
        return diagnostics

    def multiple_hypothesis_correction(self, estimator: LinearDML, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """
        Applies Bonferroni and Benjamini-Hochberg (FDR) corrections to the p-values of the 
        CATE structural model coefficients. This prevents invalid multiple hypothesis inference.
        """
        print("[INFO] Applying Multiple Hypothesis Testing Corrections for Feature Moderation...")
        
        # Robustly retrieve standard errors and p-values via EconML's coef__inference() method
        coef_inf_obj = estimator.coef__inference()
        
        def flatten_to_1d(arr):
            if arr is None:
                return None
            arr_np = np.asarray(arr)
            return arr_np.ravel()
                
        coefficients = flatten_to_1d(coef_inf_obj.point_estimate)
        stderrs = flatten_to_1d(coef_inf_obj.stderr)
        p_values = flatten_to_1d(coef_inf_obj.pvalue())
        
        feature_cols = [f"f{i}" for i in range(12)]
        m = len(p_values)
        
        # 1. Bonferroni Correction
        p_bonferroni = np.minimum(1.0, p_values * m)
        sig_bonferroni = p_bonferroni < alpha
        
        # 2. Benjamini-Hochberg (FDR) Correction
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        q_values = np.zeros(m)
        prev_q = 1.0
        for i in range(m - 1, -1, -1):
            q = sorted_p[i] * m / (i + 1)
            q = min(prev_q, q)
            q_values[i] = q
            prev_q = q
            
        p_bh = np.zeros(m)
        p_bh[sorted_indices] = q_values
        sig_bh = p_bh < alpha
        
        summary_df = pl.DataFrame({
            "feature": feature_cols,
            "coefficient": coefficients,
            "stderr": stderrs,
            "raw_p_value": p_values,
            "bonferroni_p": p_bonferroni,
            "sig_bonferroni": sig_bonferroni,
            "bh_fdr_q": p_bh,
            "sig_bh": sig_bh
        })
        
        print("\n" + "=" * 80)
        print("          MULTIPLE HYPOTHESIS TESTING ADJUSTMENT REPORT (CATE COEFFICIENTS)    ")
        print("=" * 80)
        print(f"  * Significance Threshold (alpha): {alpha:.2f} | Number of tests: {m}")
        print("-" * 80)
        print(f"{'Feature':<10} | {'Coef':<10} | {'StdErr':<8} | {'Raw p-val':<10} | {'Bonf p-val':<10} | {'FDR q-val':<10} | {'Sig (FDR)':<10}")
        print("-" * 80)
        for row in summary_df.iter_rows(named=True):
            coef_str = f"{row['coefficient']:+.6f}"
            se_str = f"{row['stderr']:.6f}"
            raw_p = f"{row['raw_p_value']:.4e}"
            bonf_p = f"{row['bonferroni_p']:.4e}"
            fdr_q = f"{row['bh_fdr_q']:.4e}"
            sig_str = "YES" if row['sig_bh'] else "no"
            if row['sig_bonferroni']:
                sig_str += " (Bonf)"
            print(f"{row['feature']:<10} | {coef_str:<10} | {se_str:<8} | {raw_p:<10} | {bonf_p:<10} | {fdr_q:<10} | {sig_str:<10}")
        print("=" * 80 + "\n")
        
        return p_bonferroni, p_bh, summary_df


def main():
    # Setup paths
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    
    # Check for the parquet file produced by preprocess.py
    processed_path_parquet = os.path.join(DATA_DIR, "criteo_sampled_preprocessed.parquet")
    processed_path_fallback = os.path.join(DATA_DIR, "processed_data.parquet")
    
    if os.path.exists(processed_path_parquet):
        file_path = processed_path_parquet
    elif os.path.exists(processed_path_fallback):
        file_path = processed_path_fallback
    else:
        print("[ERROR] Preprocessed data not found.")
        print(f"Please run 'python src/preprocess.py' first or place your preprocessed data at: {processed_path_parquet}")
        return
        
    print("=" * 65)
    print("           DOUBLE MACHINE LEARNING (DML) ESTIMATION            ")
    print("=" * 65)
    
    # 1. Load the preprocessed Parquet dataset
    print(f"[INFO] Loading preprocessed data from {file_path}...")
    start_time = time.time()
    df = pl.read_parquet(file_path)
    print(f"[INFO] Loaded {len(df):,} rows in {time.time() - start_time:.2f} seconds.")
    
    # Calculate Naive Lift for comparison
    # Naive Lift = Mean(Y | T=1) - Mean(Y | T=0)
    mean_y_t1 = df.filter(pl.col("treatment") == 1)["conversion"].mean()
    mean_y_t0 = df.filter(pl.col("treatment") == 0)["conversion"].mean()
    naive_lift = mean_y_t1 - mean_y_t0
    
    # 2. Run DML Estimation
    dml_engine = DoubleMLEngine(seed=SEED)
    estimator, summary = dml_engine.fit_and_estimate(df)
    
    # Run propensity overlap check
    _ = dml_engine.check_propensity_overlap(df)
    
    # Run multiple hypothesis corrections for features
    _, _, _ = dml_engine.multiple_hypothesis_correction(estimator)
    
    # Extract feature matrix X to compute individual-level CATEs
    feature_cols = [f"f{i}" for i in range(12)]
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    
    # Compute Conditional Average Treatment Effects (CATE) for each individual user
    cate_estimates = estimator.effect(X)
    
    # 3. Print Results Report
    print("\n" + "=" * 65)
    print("           CAUSAL ML INFERENCE REPORT: ATE ESTIMATION           ")
    print("=" * 65)
    print(f"  - Observations:        {len(df):,}")
    print(f"  - Naive Correlation:   {naive_lift:+.6%}")
    print("-" * 65)
    print(f"  - Unbiased ATE:        {summary['ate']:+.6%}")
    print(f"  - Standard Error:      {summary['stderr']:.6f}")
    print(f"  - 95% Conf. Interval:  [{summary['ci_lower']:+.6%}, {summary['ci_upper']:+.6%}]")
    print(f"  - p-value:             {summary['p_value']:.4e}")
    print(f"  - Statistical Sig:     {'Yes (p < 0.05)' if summary['p_value'] < 0.05 else 'No'}")
    print("-" * 65)
    print(f"  - Computation Time:    {summary['fit_time_seconds']:.2f} seconds")
    print("=" * 65)
    
    # 4. Print heterogeneous treatment effect (CATE) distribution
    print("\n" + "=" * 65)
    print("       CONDITIONAL AVERAGE TREATMENT EFFECT (CATE) PROFILE      ")
    print("=" * 65)
    print(f"  - Mean CATE:           {np.mean(cate_estimates):+.6%}")
    print(f"  - Std Dev (Hetero):    {np.std(cate_estimates):.6%}")
    print(f"  - Min Individual CATE: {np.min(cate_estimates):+.6%}")
    print(f"  - 5th Percentile:      {np.percentile(cate_estimates, 5):+.6%}")
    print(f"  - 50th (Median) CATE:  {np.median(cate_estimates):+.6%}")
    print(f"  - 95th Percentile:     {np.percentile(cate_estimates, 95):+.6%}")
    print(f"  - Max Individual CATE: {np.max(cate_estimates):+.6%}")
    print("=" * 65)
    
    # Print the full formal parameter table from Statsmodels inference
    print("\n[INFO] Printing Full CATE Structural Model Coefficients:")
    print(estimator.summary())


if __name__ == "__main__":
    main()
