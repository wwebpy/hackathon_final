from src.data.load_data import split_data, load_data
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from catboost import CatBoostRegressor, Pool, cv
import os 
import json
import datetime
import numpy as np

# Daten laden und aufteilen
X_train, X_test, y_train, y_test = split_data(load_data(), test_size=0.2, random_state=42)

cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

train_pool = Pool(X_train, y_train, cat_features=cat_features)


# Optuna setup für Hyperparameter-Tuning
def objective(trial):
    params = {
        "depth": trial.suggest_int("depth", 6, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        "iterations": trial.suggest_int("iterations", 200, 800),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0, 5),
        "loss_function": "RMSE",
        "early_stopping_rounds": 50,
        "verbose": False
    }

    cv_results = cv(
            pool=train_pool,
            params=params,
            fold_count=8,                 
            shuffle=True,
            partition_random_seed=42,
            verbose=False
        )
    best_rmse = np.min(cv_results["test-RMSE-mean"])
    return best_rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Beste Hyperparameter:", study.best_params)

# Modell mit den besten Hyperparametern trainieren
best_model = CatBoostRegressor(**study.best_params, silent=True)
best_model.fit(X_train, y_train, cat_features=cat_features)

preds = best_model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

# Ergebnisse ausgeben

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
feature = best_model.get_feature_importance(prettified=True)
print(feature)


output_dir = "models/tree/boosting"
os.makedirs(output_dir, exist_ok=True)

# Modell speichern
model_path = os.path.join(output_dir, "catboost_model.cbm")
best_model.save_model(model_path)


# To load the model later, use:
# loaded_model = CatBoostRegressor()
# loaded_model.load_model("models/boosting/catboost_best.cbm")


# Metriken & Parameter speichern
results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_path": model_path,
    "metrics": {
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    },
    "best_params": study.best_params
}

metrics_path = os.path.join(output_dir, "catboost_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)