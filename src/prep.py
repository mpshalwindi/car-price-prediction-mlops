
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--test_train_ratio", type=str, default="0.2")
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    args = parser.parse_args()
    
    print("🚀 STARTING DATA PREPARATION")
    print(f"Input data: {args.input_data}")
    print(f"Test/train ratio: {args.test_train_ratio}")
    
    # Load the data
    input_path = Path(args.input_data)
    print(f"Loading data from: {input_path}")
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Ensure target column is named 'Price'
    if 'price' in df.columns and 'Price' not in df.columns:
        df = df.rename(columns={'price': 'Price'})
        print("✅ Renamed 'price' column to 'Price'")
    
    # Split the data
    test_ratio = float(args.test_train_ratio)
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
    
    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Save the splits
    train_output = Path(args.train_data)
    test_output = Path(args.test_data)
    
    train_output.mkdir(parents=True, exist_ok=True)
    test_output.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV files
    train_df.to_csv(train_output / "data.csv", index=False)
    test_df.to_csv(test_output / "data.csv", index=False)
    
    print(f"✅ Training data saved to: {train_output}")
    print(f"✅ Test data saved to: {test_output}")
    print("🎯 DATA PREPARATION COMPLETED")

if __name__ == "__main__":
    main()
