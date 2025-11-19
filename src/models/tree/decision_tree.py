from src.data.load_data import load_data, split_data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib



data = load_data()

data_encoded = pd.get_dummies(data, drop_first=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cleaned_data_one_hot_encoding.csv")
data_encoded.to_csv(output_path, sep=",", index=False)

X_train, X_test, y_train, y_test = split_data(data_encoded, test_size=0.2, random_state=42)

criterions = ["squared_error", "absolute_error"]


output_dir = "models/tree"
os.makedirs(output_dir, exist_ok=True)

for crit in criterions:
    model = DecisionTreeRegressor(
        criterion=crit,
        max_depth=6,
        random_state=42,
        min_samples_split=10,
        min_samples_leaf=5,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n Ergebnisse Decision Tree ({crit}):")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    importance_path = os.path.join(output_dir, f"decision_tree_{crit}_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)


    model_path = os.path.join(output_dir, f"decision_tree_{crit}_model.pkl")
    joblib.dump(model, model_path)


    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "metrics": {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        },
        "model_type": f"DecisionTree_{crit}"
    }

    metrics_path = os.path.join(output_dir, f"decision_tree_{crit}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n--- Overfitting Check ---")
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test  R²: {test_r2:.3f}")
    print(f"R²-Differenz (Train - Test): {train_r2 - test_r2:.3f}")

    print(f" Modell & Metriken gespeichert unter: {metrics_path}")

