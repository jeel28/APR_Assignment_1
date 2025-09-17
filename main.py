import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Define column names (Boston Housing dataset)
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# 2. Load dataset with correct column names
data = pd.read_csv("housing.csv", header=None, delim_whitespace=True, names=columns)

print("First 5 rows of dataset:")
print(data.head())
print("\nColumns:", list(data.columns))

# 3. Features (X) and target (y)
X = data.drop(columns=['MEDV'])   # features
y = data['MEDV']                  # target (Median house value)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# 6. Predict
y_pred = lr.predict(X_test)

# 7. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"R²: {r2:.4f}")

# 8. Plot
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()

# 9. Feature importance
coefs = pd.Series(lr.coef_, index=X.columns)
print("\nFeature coefficients (importance):")
print(coefs.sort_values(ascending=False))
