import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

os.makedirs("models", exist_ok=True)

# Update DATA_PATH and TARGET to match your dataset
DATA_PATH = "data/e_waste_profit_data.csv"
TARGET = "profit"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Place your CSV there or update DATA_PATH."
    )

# Load data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"Data loaded. Shape: {df.shape}")

# If your dataset has a profit_tier column and you want to train a regression model on profit,
# it will be dropped from features automatically below.
DROP_COLS = ["profit_tier"] if "profit_tier" in df.columns else []

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset columns: {list(df.columns)}")

X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET]

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Random Forest hyperparameter tuning
# -----------------------------
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None]
}

rf_search = GridSearchCV(
    rf_pipe,
    rf_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

print("Starting Random Forest tuning...")
rf_search.fit(X_train, y_train)
print("Random Forest done.")

best_rf = rf_search.best_estimator_
rf_pred = best_rf.predict(X_test)

print("Best RF params:", rf_search.best_params_)
print("RF RMSE:", mean_squared_error(y_test, rf_pred, squared=False))
print("RF R2:", r2_score(y_test, rf_pred))

joblib.dump(best_rf, "models/random_forest_best.pkl")

# -----------------------------
# Gradient Boosting hyperparameter tuning
# -----------------------------
gb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42))
])

gb_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__subsample": [0.8, 1.0]
}

gb_search = GridSearchCV(
    gb_pipe,
    g b_param_grid if False else g b_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

print("Starting Gradient Boosting tuning...")
gb_search.fit(X_train, y_train)
print("Gradient Boosting done.")

best_gb = gb_search.best_estimator_
gb_pred = best_gb.predict(X_test)

print("Best GB params:", gb_search.best_params_)
print("GB RMSE:", mean_squared_error(y_test, gb_pred, squared=False))
print("GB R2:", r2_score(y_test, gb_pred))

joblib.dump(best_gb, "models/gradient_boosting_best.pkl")
