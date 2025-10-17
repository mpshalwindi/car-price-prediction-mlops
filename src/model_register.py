
import argparse, mlflow, joblib
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    args = ap.parse_args()

    model_dir = Path(args.model)
    model_path = next(model_dir.glob("*.joblib"))
    model = joblib.load(model_path)

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path="random_forest_price_regressor",
                                 registered_model_name="used_cars_price_prediction_model")

if __name__ == "__main__":
    main()
