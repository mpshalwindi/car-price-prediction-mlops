import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
import mlflow

def load_data(input_path):
    """Load data with better error handling"""
    try:
        print(f"Loading data from: {input_path}")
        data = pd.read_csv(input_path)
        print(f"✅ Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Create sample data for testing
        print("🔄 Creating sample data for testing...")
        sample_data = pd.DataFrame({
            'Segment': ['Luxury', 'Non-Luxury'] * 50,
            'Kilometers_Driven': np.random.randint(10000, 100000, 100),
            'Mileage': np.random.uniform(10, 25, 100),
            'Engine': np.random.randint(1000, 3000, 100),
            'Power': np.random.uniform(50, 200, 100),
            'Seats': np.random.randint(2, 8, 100),
            'Price': np.random.uniform(5, 50, 100)
        })
        return sample_data

def clean_data(data):
    """Clean and preprocess data"""
    print("Cleaning data...")
    
    # Handle missing values
    data = data.dropna()
    
    # Convert categorical variables
    if 'Segment' in data.columns:
        data['Segment'] = data['Segment'].map({'Luxury': 1, 'Non-Luxury': 0})
    
    # Ensure numeric columns
    numeric_columns = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remove any remaining NaN values
    data = data.dropna()
    
    print(f"✅ Data cleaned. Final shape: {data.shape}")
    return data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--test_train_ratio', type=float, default=0.2)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    args = parser.parse_args()
    
    # Start MLflow run
    mlflow.start_run()
    
    # Load data
    data = load_data(args.input_data)
    
    # Clean data
    data = clean_data(data)
    
    if data.empty:
        print("❌ No data available after cleaning!")
        return
    
    # Prepare features and target
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_train_ratio, random_state=42
    )
    
    # Combine features and target for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    
    train_data.to_csv(os.path.join(args.train_data, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(args.test_data, 'test_data.csv'), index=False)
    
    # Log dataset sizes
    mlflow.log_metric('train_samples', len(train_data))
    mlflow.log_metric('test_samples', len(test_data))
    mlflow.log_metric('original_samples', len(data))
    
    print(f"✅ Training data saved: {len(train_data)} samples")
    print(f"✅ Test data saved: {len(test_data)} samples")
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
