"""
Preprocessing module for Causal ML Framework on Criteo Uplift dataset.
Optimized for high performance and low memory footprint using Polars.
"""

import os
import urllib.request
import time
from typing import List, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

# Ensure full reproducibility
SEED = 42
np.random.seed(SEED)

class CriteoPreprocessor:
    """
    A modular preprocessing pipeline for the Criteo Uplift dataset.
    Uses Polars for high-performance and scikit-learn for standardization.

    Causal Assumptions & Context for this dataset:
    1. Unconfoundedness (Conditional Ignorability): Y(0), Y(1) ⊥ T | X. Since the raw Criteo Uplift
       dataset is derived from a randomized controlled trial (RCT), treatment is randomly assigned,
       meaning unconfoundedness holds unconditionally (Y(0), Y(1) ⊥ T). In standard observational settings, 
       this requires that all variables that simultaneously affect treatment T and outcome Y are in X.
    2. Common Support (Positivity): 0 < P(T=1 | X) < 1. Every individual must have a non-zero probability
       of being in both treatment and control. In the Criteo RCT, this holds perfectly as treatment was
       randomly targeted to ~85% of the population.
    3. SUTVA (Stable Unit Treatment Value Assumption): The treatment assignment of one user does not
       affect another user's outcome, and there are no multiple versions of the treatment. This is
       assumed to hold at the individual level since user browsing/purchase decisions are independent.
    """
    
    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.scaler = StandardScaler()
        self.feature_cols = [f"f{i}" for i in range(12)]
        self.target_cols = ["treatment", "conversion", "visit", "exposure"]

    @staticmethod
    def get_df_memory_usage(df: pl.DataFrame) -> float:
        """
        Calculates and returns the memory usage of a Polars DataFrame in megabytes (MB).
        """
        # Estimated size in bytes of the dataframe in memory
        return df.estimated_size() / (1024 * 1024)

    def download_dataset(self, url: str, dest_path: str) -> str:
        """
        Downloads the Criteo Uplift dataset if it is not already present.
        """
        if os.path.exists(dest_path):
            print(f"[INFO] Dataset already exists at {dest_path}. Skipping download.")
            return dest_path
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"[INFO] Downloading dataset from {url}...")
        print(f"[INFO] This is a ~300MB compressed file (~13M rows). Please wait...")
        
        start_time = time.time()
        
        # Download with simple progress updates
        def report_progress(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = min(100, read_so_far * 100 / total_size)
                # Print progress update at intervals
                if block_num % 1000 == 0 or percent == 100:
                    mb_read = read_so_far / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"Downloaded {mb_read:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end="\r")
            else:
                if block_num % 1000 == 0:
                    mb_read = read_so_far / (1024 * 1024)
                    print(f"Downloaded {mb_read:.1f}MB", end="\r")
                    
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print(f"\n[INFO] Download completed in {time.time() - start_time:.2f} seconds.")
        return dest_path

    def load_data(self, file_path: str) -> pl.DataFrame:
        """
        Loads the Criteo dataset using Polars' fast CSV reader.
        Handles both compressed (.gz) and raw (.csv) files automatically.
        """
        print(f"[INFO] Loading dataset from {file_path}...")
        start_time = time.time()
        
        # Polars scan_csv can read gzipped CSV files directly, but we collect to load into memory
        # to demonstrate and track pre/post memory utilization.
        df = pl.read_csv(file_path)
        
        print(f"[INFO] Loaded {len(df):,} rows in {time.time() - start_time:.2f} seconds.")
        print(f"[INFO] Initial memory usage: {self.get_df_memory_usage(df):.2f} MB")
        return df

    def stratified_sample(self, df: pl.DataFrame, target_size: int = 1_000_000) -> pl.DataFrame:
        """
        Performs a stratified sub-sampling to target_size rows, maintaining the
        exact joint distribution of 'treatment' and 'conversion'.
        """
        print(f"[INFO] Running stratified sub-sampling to {target_size:,} rows...")
        start_time = time.time()
        
        strat_cols = ["treatment", "conversion"]
        
        # Verify columns exist
        for col in strat_cols:
            if col not in df.columns:
                raise ValueError(f"Required stratification column '{col}' is missing from the DataFrame.")
        
        # Compute stratum counts in the original dataset
        # Using a highly compatible Polars aggregation approach
        strata_counts = (
            df.group_by(strat_cols)
            .agg(pl.len().alias("count"))
            .sort(strat_cols)
        )
        
        total_rows = len(df)
        print("[INFO] Original Strata Proportions:")
        for row in strata_counts.iter_rows(named=True):
            prop = row["count"] / total_rows
            print(f"  Strata (T={row['treatment']}, Y={row['conversion']}): count={row['count']:,} ({prop:.4%})")
            
        # Compute exact target sample size for each stratum
        target_sizes = []
        for row in strata_counts.iter_rows(named=True):
            prop = row["count"] / total_rows
            target_sizes.append(round(prop * target_size))
            
        # Adjust for potential rounding discrepancies to sum exactly to target_size
        diff = target_size - sum(target_sizes)
        if diff != 0:
            # Adjust the largest stratum
            largest_idx = np.argmax(target_sizes)
            target_sizes[largest_idx] += diff
            
        # Draw samples for each stratum
        strata_samples = []
        for idx, row in enumerate(strata_counts.iter_rows(named=True)):
            t_val, c_val = row["treatment"], row["conversion"]
            n_samples = target_sizes[idx]
            
            # Filter the dataframe for this stratum
            strata_df = df.filter((pl.col("treatment") == t_val) & (pl.col("conversion") == c_val))
            
            # Deterministically sample from this stratum
            sampled_strata = strata_df.sample(n=n_samples, seed=self.seed)
            strata_samples.append(sampled_strata)
            
        # Concatenate strata and shuffle rows deterministically to avoid ordered blocks
        sampled_df = pl.concat(strata_samples)
        sampled_df = sampled_df.sample(fraction=1.0, shuffle=True, seed=self.seed)
        
        print(f"[INFO] Stratified sampling completed in {time.time() - start_time:.2f} seconds.")
        
        # Verify the target proportions in the sampled dataset
        sample_strata_counts = (
            sampled_df.group_by(strat_cols)
            .agg(pl.len().alias("count"))
            .sort(strat_cols)
        )
        
        print("[INFO] Sampled Strata Proportions:")
        for row in sample_strata_counts.iter_rows(named=True):
            prop = row["count"] / target_size
            print(f"  Strata (T={row['treatment']}, Y={row['conversion']}): count={row['count']:,} ({prop:.4%})")
            
        return sampled_df

    def clean_and_scale(self, df: pl.DataFrame, fit_scaler: bool = True) -> pl.DataFrame:
        """
        Cleans missing values, standardizes the features f0-f11, and downcasts numeric types.
        """
        print("[INFO] Running cleaning, scaling, and numeric downcasting...")
        start_time = time.time()
        
        # 1. Cleaning: Check and fill missing values for features f0-f11
        # The Criteo dataset typically does not contain missing values, but we implement median
        # imputation for robustness.
        expressions = []
        for col in self.feature_cols:
            if col in df.columns:
                median_val = df[col].median()
                expressions.append(pl.col(col).fill_null(median_val).alias(col))
        
        if expressions:
            df = df.with_columns(expressions)
            
        # 2. Scaling: Standardize f0 to f11 using scikit-learn's StandardScaler
        # Check that all features are in the dataframe
        available_features = [col for col in self.feature_cols if col in df.columns]
        if len(available_features) < 12:
            print(f"[WARNING] Only found {len(available_features)} feature columns instead of 12.")
            
        if available_features:
            # Extract features as NumPy array for StandardScaler
            features_np = df.select(available_features).to_numpy()
            
            if fit_scaler:
                scaled_np = self.scaler.fit_transform(features_np)
            else:
                scaled_np = self.scaler.transform(features_np)
                
            # Put scaled features back as Float32 to achieve memory downcasting
            scale_expressions = [
                pl.Series(col, scaled_np[:, i], dtype=pl.Float32) 
                for i, col in enumerate(available_features)
            ]
            df = df.with_columns(scale_expressions)
            
        # 3. Downcasting Targets: Compress binary and target columns to minimal types
        # Downcast binary variables (treatment, conversion, visit, exposure) to Int8
        downcast_targets = []
        for col in self.target_cols:
            if col in df.columns:
                downcast_targets.append(pl.col(col).cast(pl.Int8))
                
        if downcast_targets:
            df = df.with_columns(downcast_targets)
            
        print(f"[INFO] Cleaning, scaling, and downcasting completed in {time.time() - start_time:.2f} seconds.")
        print(f"[INFO] Preprocessed memory usage: {self.get_df_memory_usage(df):.2f} MB")
        return df


def main():
    # Setup paths
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    DATASET_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
    LOCAL_RAW_PATH = os.path.join(DATA_DIR, "criteo-research-uplift-v2.1.csv.gz")
    LOCAL_PROCESSED_PATH = os.path.join(DATA_DIR, "criteo_sampled_preprocessed.parquet")
    
    print("=" * 65)
    print("      CRITEO DATA PREPROCESSING PIPELINE (POLARS WORKER)       ")
    print("=" * 65)
    
    preprocessor = CriteoPreprocessor(seed=SEED)
    
    # 1. Download dataset if not exists
    try:
        preprocessor.download_dataset(DATASET_URL, LOCAL_RAW_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("[ERROR] Please manually download the dataset to data/criteo-research-uplift-v2.1.csv.gz and rerun.")
        return
        
    # 2. Load dataset
    try:
        df_raw = preprocessor.load_data(LOCAL_RAW_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return
        
    # Track original size and memory
    original_rows = len(df_raw)
    original_mem = preprocessor.get_df_memory_usage(df_raw)
    
    # 3. Perform Stratified Sub-sampling to 1M rows
    try:
        df_sampled = preprocessor.stratified_sample(df_raw, target_size=1_000_000)
    except Exception as e:
        print(f"[ERROR] Failed to sample dataset: {e}")
        return
        
    sampled_mem = preprocessor.get_df_memory_usage(df_sampled)
    
    # 4. Clean, Scale, and Downcast Features/Labels
    try:
        df_preprocessed = preprocessor.clean_and_scale(df_sampled, fit_scaler=True)
    except Exception as e:
        print(f"[ERROR] Failed to clean and scale dataset: {e}")
        return
        
    final_mem = preprocessor.get_df_memory_usage(df_preprocessed)
    
    # Save the processed dataset in high-performance parquet format
    print(f"[INFO] Saving preprocessed sample to {LOCAL_PROCESSED_PATH}...")
    df_preprocessed.write_parquet(LOCAL_PROCESSED_PATH)
    
    # Summary of savings
    print("\n" + "=" * 65)
    print("                     PIPELINE METRIC SUMMARY                    ")
    print("=" * 65)
    print(f"Original Dataset Size:   {original_rows:,} rows | {original_mem:.2f} MB")
    print(f"Sampled Dataset Size:    {len(df_preprocessed):,} rows | {sampled_mem:.2f} MB")
    print(f"Preprocessed Size (32b): {len(df_preprocessed):,} rows | {final_mem:.2f} MB")
    
    # Calculate savings
    mem_saved_pct = (1 - (final_mem / original_mem)) * 100
    print(f"Total RAM footprint reduction: {mem_saved_pct:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
