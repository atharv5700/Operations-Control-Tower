import pandas as pd
import numpy as np

def calculate_supplier_reliability(df):
    """
    Calculates supplier reliability score based on lead time consistency and stockouts.
    Returns a dataframe with scores.
    """
    supplier_stats = df.groupby('Supplier_ID').agg({
        'Supplier_Lead_Time_Days': ['mean', 'std'],
        'Stockout_Flag': 'mean' # Stockout rate
    }).reset_index()
    
    supplier_stats.columns = ['Supplier_ID', 'Avg_LT', 'Std_LT', 'Stockout_Rate']
    
    # Normalize metrics to 0-1 scale (lower is better for LT var and Stockout)
    # Simple scoring: Score = 100 - (Stockout_Rate * 50) - (Normalized_Std_LT * 50)
    
    # Handle NaN std dev (single observation)
    supplier_stats['Std_LT'] = supplier_stats['Std_LT'].fillna(0)
    
    max_std = supplier_stats['Std_LT'].max()
    if max_std == 0: max_std = 1
    
    supplier_stats['Reliability_Score'] = 100 - (supplier_stats['Stockout_Rate'] * 100 * 0.5) - ((supplier_stats['Std_LT'] / max_std) * 100 * 0.5)
    
    return supplier_stats

def predict_late_delivery_prob(lead_time_series, threshold_days):
    """
    Simple probability of late delivery based on historical distribution.
    """
    if len(lead_time_series) == 0:
        return 0.0
    
    late_count = np.sum(lead_time_series > threshold_days)
    return late_count / len(lead_time_series)
