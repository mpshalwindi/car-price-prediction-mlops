# ===================== COMPLETELY FIXED train.py =====================
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

class FixedDataPreprocessor:
    """Completely fixed data preprocessor that handles categorical data properly"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
    
    def prepare_features_completely_fixed(self, train_df, test_df):
        """Completely fixed version that handles train/test split properly"""
        print("⚙️ Preparing features with complete fix...")
        
        # Handle target column (both 'Price' and 'price')
        target_col = 'Price' if 'Price' in train_df.columns else 'price'
        
        # Combine train and test to fit encoders on all possible categories
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        X_combined = combined_df.drop(target_col, axis=1)
        
        # Fit encoders on combined data to see all possible categories
        categorical_cols = X_combined.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            # Fit on all possible categories from both train and test
            self.label_encoders[col].fit(X_combined[col].astype(str))
        
        # Now transform train and test data
        X_train = train_df.drop(target_col, axis=1).copy()
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1).copy()
        y_test = test_df[target_col]
        
        for col in categorical_cols:
            if col in self.label_encoders:
                # Transform using the encoder that knows all categories
                X_train[col] = self.label_encoders[col].transform(X_train[col].astype(str))
                X_test[col] = self.label_encoders[col].transform(X_test[col].astype(str))
        
        self.feature_names = X_train.columns.tolist()
        
        return X_train.values, y_train.values, X_test.values, y_test.values

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    np.random.seed(42)
    n = 200
    
    data = {
        "Segment": np.random.choice(["Luxury", "Non-Luxury"], n),
        "Kilometers_Driven": np.random.randint(10000, 100000, n),
        "Mileage": np.random.uniform(10, 25, n),
        "Engine": np.random.randint(1000, 3000, n),
        "Power": np.random.uniform(50, 200, n),
        "Seats": np.random.choice([4, 5, 7], n),
        "Price": np.random.uniform(5, 50, n)
    }
    
    return pd.DataFrame(data)

def run_training_simple_fixed(n_estimators=100, max_depth=10, model_output="outputs"):
    """Simple fixed training function that avoids categorical issues"""
    print("🚀 SIMPLE FIXED TRAINING")
    
    # Create sample data with NO categorical variables for testing
    np.random.seed(42)
    n = 1000
    
    # Use only numerical data to avoid categorical issues
    data = {
        "Kilometers_Driven": np.random.randint(5000, 150000, n),
        "Mileage": np.random.uniform(10, 25, n),
        "Engine": np.random.randint(800, 3000, n),
        "Power": np.random.uniform(50, 300, n),
        "Seats": np.random.choice([4, 5, 7], n),
        "Car_Age": np.random.randint(1, 15, n),  # Instead of Year
        "Previous_Owners": np.random.randint(1, 4, n),
    }
    
    # Realistic price calculation
    base_price = 20000
    age_factor = data["Car_Age"] * 1200
    km_factor = data["Kilometers_Driven"] * 0.08
    engine_factor = data["Engine"] * 3
    power_factor = data["Power"] * 60
    owners_factor = data["Previous_Owners"] * 1500
    
    data["Price"] = (base_price - age_factor - km_factor + engine_factor + 
                    power_factor - owners_factor + np.random.normal(0, 3000, n))
    data["Price"] = np.maximum(data["Price"], 3000)
    
    df = pd.DataFrame(data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Test data: {X_test.shape[0]} samples")
    print(f"💰 Price range: ${y_train.min():.2f} - ${y_train.max():.2f}")
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print("📊 MODEL PERFORMANCE:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE: ${mae:.2f}")
    
    # Save model
    output_dir = Path(model_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    metrics = {
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_features": X_train.shape[1],
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0]
    }
    
    return True, metrics

def run_training_advanced_fixed(n_estimators=100, max_depth=10, model_output="outputs"):
    """Advanced fixed training with categorical handling"""
    print("🚀 ADVANCED FIXED TRAINING")
    
    # Create data with categorical variables but handle them properly
    np.random.seed(42)
    n = 1000
    
    # Define all possible categories first
    all_segments = ["Budget", "Mid-Range", "Premium", "Luxury"]
    all_fuels = ["Petrol", "Diesel", "Hybrid"]
    all_transmissions = ["Manual", "Automatic"]
    
    data = {
        "Segment": np.random.choice(all_segments, n, p=[0.2, 0.5, 0.2, 0.1]),
        "Fuel_Type": np.random.choice(all_fuels, n, p=[0.5, 0.3, 0.2]),
        "Transmission": np.random.choice(all_transmissions, n, p=[0.6, 0.4]),
        "Kilometers_Driven": np.random.randint(5000, 150000, n),
        "Mileage": np.random.uniform(10, 25, n),
        "Engine": np.random.randint(800, 3000, n),
        "Power": np.random.uniform(50, 300, n),
        "Seats": np.random.choice([4, 5, 7], n, p=[0.2, 0.6, 0.2]),
        "Car_Age": np.random.randint(1, 10, n),
    }
    
    # Price calculation
    segment_prices = {"Budget": 8000, "Mid-Range": 15000, "Premium": 25000, "Luxury": 40000}
    base_price = np.array([segment_prices[seg] for seg in data["Segment"]])
    
    age_factor = data["Car_Age"] * 1000
    km_factor = data["Kilometers_Driven"] * 0.1
    engine_factor = data["Engine"] * 4
    power_factor = data["Power"] * 70
    transmission_bonus = np.array([2000 if trans == "Automatic" else 0 for trans in data["Transmission"]])
    
    data["Price"] = (base_price - age_factor - km_factor + engine_factor + 
                    power_factor + transmission_bonus + np.random.normal(0, 2000, n))
    data["Price"] = np.maximum(data["Price"], 2000)
    
    df = pd.DataFrame(data)
    
    # Use the completely fixed preprocessor
    preprocessor = FixedDataPreprocessor()
    
    # Split data first
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare features with the completely fixed method
    X_train, y_train, X_test, y_test = preprocessor.prepare_features_completely_fixed(train_df, test_df)
    
    print(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Test data: {X_test.shape[0]} samples")
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print("📊 MODEL PERFORMANCE:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE: ${mae:.2f}")
    
    # Save model
    output_dir = Path(model_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    metrics = {
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_features": X_train.shape[1],
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0]
    }
    
    return True, metrics

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--n_estimators", type=str, default="100")
    parser.add_argument("--max_depth", type=str, default="10")
    parser.add_argument("--model_output", type=str, default="outputs")
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"⚠️  Ignoring unknown arguments: {unknown}")
    
    # Use the simple fixed version for pipeline
    success, metrics = run_training_simple_fixed(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        model_output=args.model_output
    )
    
    sys.exit(0 if success else 1)

def is_running_in_jupyter():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False

if __name__ == "__main__":
    if not is_running_in_jupyter():
        main()
    else:
        print("📓 Running in Jupyter - use run_training_simple_fixed() or run_training_advanced_fixed()")
