import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Paths
DATA_PATH = "dataset/winequality-red.csv"
MODEL_PATH = "output/model/model.pkl"
RESULT_PATH = "output/results/metrics.json"

# Create directories
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Preprocessing (NO scaling)
X_scaled = X.values

# Train-test split (CHANGED to 0.3)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Model
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save model
joblib.dump(model, MODEL_PATH)

# Save metrics
metrics = {
    "MSE": mse,
    "R2_score": r2
}

with open(RESULT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
