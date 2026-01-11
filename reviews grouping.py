"""
Simple Review Grouping Script
Outputs CSV with headers: bot_id,reviews
"""

import pandas as pd
import csv


# ============================================================================
# OPTION 1: Using Pandas (Fastest - Recommended)
# ============================================================================

def group_reviews_pandas(input_file='bot_reviews99.csv', output_file='bot_reviews_grouped.csv'):
    """
    Group reviews by bot_id using pandas.
    Output: CSV with columns [bot_id, reviews]
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Grouping reviews by bot_id...")
    grouped = df.groupby('bot_id')['text'].apply(list).reset_index()
    grouped.columns = ['bot_id', 'reviews']  # Exact headers you want
    
    print(f"Saving to {output_file}...")
    grouped.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✓ Done! Saved {len(grouped)} bots")
    print(f"\nFirst 3 rows:")
    print(grouped.head(3))

    
if __name__ == "__main__":
    # Choose one method:
    
    # Method 1: Pandas (RECOMMENDED - much faster!)
    group_reviews_pandas('bot_reviews7.csv', 'bot_reviews_grouped.csv')