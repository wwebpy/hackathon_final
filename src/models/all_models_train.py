import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# === Daten laden ===
df_hist = pd.read_csv(
    "../../data/processed/data_feature_zeit_3_gesamt.csv",
    parse_dates=[
        "Auftragseingang","Auftragsende_SOLL","AFO_Start_SOLL","AFO_Ende_SOLL",
        "AFO_Start_IST","AFO_Ende_IST","Auftragsende_IST"
    ],
    low_memory=False
)

df_ids = pd.read_csv("../../data/raw/df_IDs_for_eval_2025-11-03.csv")

# === Aggregieren auf Auftragsebene ===
df_orders = (
    df_hist.sort_values(["AuftragsID", "AFO_Ende_IST"])
    .groupby("AuftragsID")
    .agg({
        "BauteilID": "first",
        "Bauteilbezeichnung": "first",
        "Priorität": "first",
        "Auftragseingang": "first",
        "Auftragsende_SOLL": "first",
        "Auftragsende_IST": "max",
        "Arbeitsschritt": "max",
        "AFO_Start_IST": "min",
        "AFO_Ende_IST": "max",
        "AFO_Dauer_IST_Stunde": "sum"
    })
    .reset_index()
)

# === Zielvariable ===
df_orders["target_days"] = (
    df_orders["Auftragsende_IST"] - df_orders["Auftragseingang"]
).dt.total_seconds() / 86400

df_train = df_orders.dropna(subset=["target_days"])

y = df_train["target_days"]
X = df_train.drop(columns=["target_days", "Auftragsende_IST"])

# === Kategorische Spalten bestimmen ===
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("category")

# === Datetime-Spalten in numerische Werte umwandeln ===
datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
for col in datetime_cols:
    X[col] = X[col].view("int64")
    X[col] = X[col].fillna(X[col].median())

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === LightGBM Modell ===
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    categorical_feature=cat_cols
)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE = {mae:.2f} Tage")

# === Submission vorbereiten ===
df_submit = df_ids.merge(df_orders, on="AuftragsID", how="left")
df_submit[cat_cols] = df_submit[cat_cols].astype("category")

submit_datetime_cols = df_submit.select_dtypes(include=["datetime64[ns]"]).columns
for col in submit_datetime_cols:
    df_submit[col] = df_submit[col].view("int64")
    df_submit[col] = df_submit[col].fillna(df_submit[col].median())

pred_days = model.predict(df_submit[X.columns])
df_submit["Auftragsende_PREDICTED"] = (
    df_submit["Auftragseingang"] + pd.to_timedelta(pred_days, unit="D")
).dt.strftime("%Y-%m-%d")

df_submit["ID"] = np.arange(1, len(df_submit)+1)

df_submit_final = df_submit[["ID","AuftragsID","Auftragsende_PREDICTED"]]
df_submit_final.to_csv("submission.csv", index=False)

print("✅ Fertig! Submission erzeugt.")