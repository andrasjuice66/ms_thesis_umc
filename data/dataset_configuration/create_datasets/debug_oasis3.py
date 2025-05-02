import pandas as pd
import numpy as np

# Read the OASIS3_demographics.csv file
df = pd.read_csv('C:/Projects/thesis_project/Data/all_demographics/OASIS3_demographics.csv')
print(f"OASIS3 DataFrame shape: {df.shape}")
print(f"OASIS3 columns: {df.columns.tolist()}")

# Check for the OASISID column
if 'OASISID' in df.columns:
    print(f"OASISID column exists with {df['OASISID'].nunique()} unique values")
    print(f"First few OASISID values: {df['OASISID'].head().tolist()}")

# Debug the get_unique_participants function
def get_unique_participants(df):
    """Extract unique participant count from dataframe"""
    id_patterns = ['participant_id', 'subject', 'id', 'oasis', 'OASISID']
    
    print("Checking each pattern:")
    for pattern in id_patterns:
        matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
        print(f"  Pattern '{pattern}' matches columns: {matching_cols}")
        if matching_cols:
            unique_count = df[matching_cols[0]].nunique()
            print(f"  Found {unique_count} unique values in column {matching_cols[0]}")
            return unique_count
    
    print("No matching columns found, returning row count")
    return df.shape[0]

# Run the function
participant_count = get_unique_participants(df)
print(f"Result from get_unique_participants: {participant_count}")

# Test case-sensitivity
if 'OASISID' in df.columns:
    for pattern in ['oasisid', 'OASISID', 'oasis', 'OASIS']:
        print(f"Is '{pattern}' in 'OASISID'.lower()? {pattern.lower() in 'OASISID'.lower()}") 