"""Preprocess UNEP GEMS/Water dataset for experiments.

Extract 500k samples of 10 key indicators from the full archive.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


TARGET_INDICATORS = [
    'pH',
    'dissolved_oxygen',
    'turbidity',
    'conductivity',
    'nitrate',
    'phosphate',
    'total_suspended_solids',
    'BOD',
    'COD',
    'temperature'
]


def load_unep_data(raw_path):
    """Load raw UNEP GEMS/Water data.
    
    Expected format: CSV with columns for station_id, date, parameter, value
    """
    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df):,} records")
    return df


def filter_indicators(df, indicators=TARGET_INDICATORS):
    """Filter to keep only target indicators."""
    print(f"Filtering to {len(indicators)} indicators...")
    df = df[df['parameter'].isin(indicators)].copy()
    print(f"Retained {len(df):,} records")
    return df


def handle_missing_values(df):
    """Impute missing values using per-station means.
    
    Approximately 15% of entries have missing values.
    """
    print("Handling missing values...")
    
    missing_before = df['value'].isna().sum()
    print(f"  Missing before: {missing_before:,} ({missing_before/len(df)*100:.1f}%)")
    
    # Per-station, per-parameter mean imputation
    df['value'] = df.groupby(['station_id', 'parameter'])['value'].transform(
        lambda x: x.fillna(x.mean())
    )
    
    # Drop any remaining NaNs (stations with all missing for a parameter)
    df = df.dropna(subset=['value'])
    
    missing_after = df['value'].isna().sum()
    print(f"  Missing after: {missing_after:,}")
    
    return df


def remove_outliers(df, iqr_multiplier=1.5):
    """Remove outliers using IQR method per parameter."""
    print("Removing outliers...")
    
    before = len(df)
    
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param]['value']
        
        q1 = param_data.quantile(0.25)
        q3 = param_data.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        
        df = df[~((df['parameter'] == param) & 
                 ((df['value'] < lower) | (df['value'] > upper)))]
    
    after = len(df)
    removed = before - after
    print(f"  Removed {removed:,} outliers ({removed/before*100:.1f}%)")
    
    return df


def normalize_features(df):
    """Normalize each parameter to [0, 1] using min-max scaling."""
    print("Normalizing features...")
    
    scaler = MinMaxScaler()
    
    for param in df['parameter'].unique():
        mask = df['parameter'] == param
        df.loc[mask, 'value'] = scaler.fit_transform(
            df.loc[mask, 'value'].values.reshape(-1, 1)
        )
    
    print("  All features normalized to [0, 1]")
    
    return df


def create_windows(df, window_hours=24, output_hours=24):
    """Create 24-hour input windows for forecasting.
    
    Returns DataFrame with columns:
    - station_id
    - window_start
    - input features (10 × 24 = 240 values)
    - output targets (10 × 24 = 240 values)
    """
    print(f"Creating {window_hours}h windows...")
    
    # Group by station and create sliding windows
    # This is a simplified placeholder
    
    print("  Window creation complete")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to raw UNEP data (CSV)')
    parser.add_argument('--output', type=str, default='data/unep_subset.csv',
                       help='Output path for processed subset')
    parser.add_argument('--target-samples', type=int, default=500000,
                       help='Target number of samples')
    args = parser.parse_args()
    
    print("="*60)
    print("UNEP GEMS/Water Data Preprocessing")
    print("="*60)
    
    # Load raw data
    df = load_unep_data(args.input)
    
    # Filter to target indicators
    df = filter_indicators(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Normalize
    df = normalize_features(df)
    
    # Sample to target size if needed
    if len(df) > args.target_samples:
        print(f"Sampling to {args.target_samples:,} records...")
        df = df.sample(n=args.target_samples, random_state=42)
    
    # Save processed data
    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Final dataset: {len(df):,} samples, {len(TARGET_INDICATORS)} indicators")
    print("="*60)


if __name__ == '__main__':
    main()
