from src.data.load_data import load_data, split_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib

data = load_data()

TARGET = "Preis"
X = data.drop(columns=[TARGET])
y = data[TARGET]

categorical = X.select_dtypes(include=['object']).columns.tolist()
numeric = X.select_dtypes(include=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical),
        ('num', 'passthrough', numeric)
    ]
)

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

output_dir = "models/linear/pipeline"
os.makedirs(output_dir, exist_ok=True)

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n✅ Ergebnisse {name} Pipeline-Modell:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")

    model_path = os.path.join(output_dir, f"{name.lower()}_pipeline.pkl")
    joblib.dump(pipe, model_path)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "metrics": {"MAE": mae, "MSE": mse, "R2": r2},
        "model_type": name
    }

    with open(os.path.join(output_dir, f"{name.lower()}_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"📦 Modell gespeichert: {model_path}")
