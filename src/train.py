"""Train a regression model on the scikit-learn diabetes dataset and save it in MLflow format.

Usage:
    python src/train.py --output outputs/model [--n-estimators 200] [--learning-rate 0.05]

The MLflow model folder can be registered in Azure Machine Learning with
    az ml model create --name diabetes-regressor --type mlflow_model --path outputs/model
and deployed to a managed online endpoint without a scoring script.
"""
import argparse
import json
import os
import shutil

import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/model", help="folder for the MLflow model")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=40)

    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
    }
    print(json.dumps(metrics, indent=2))

    # Save in MLflow format with a signature so Azure ML can deploy it with no scoring script.
    # Azure ML's built-in MLflow scoring script imports azureml-ai-monitoring, so it goes into the model's environment.
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.save_model(
        model,
        args.output,
        signature=signature,
        input_example=X_train.head(2),
        extra_pip_requirements=["azureml-ai-monitoring==1.0.0"],
    )

    metrics_path = os.path.join(os.path.dirname(args.output) or ".", "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model saved to {args.output}; metrics in {metrics_path}")


if __name__ == "__main__":
    main()
