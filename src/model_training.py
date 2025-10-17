import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os
import mlflow
import mlflow.sklearn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--model_output', type=str, required=True)
    args = parser.parse_args()
    
    mlflow.start_run()
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.train_data, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(args.test_data, 'test_data.csv'))
    
    # Prepare features and target
    X_train = train_df.drop('Price', axis=1)
    y_train = train_df['Price']
    X_test = test_df.drop('Price', axis=1)
    y_test = test_df['Price']
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    
    # Save model
    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(model, args.model_output)
    
    print(f"Model trained: MSE={mse:.4f}, R2={r2:.4f}")
    mlflow.end_run()

if __name__ == "__main__":
    main()
