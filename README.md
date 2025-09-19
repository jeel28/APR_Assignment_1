# 🏡 Boston Housing Price Prediction using Linear Regression

This project demonstrates the implementation of a **Linear Regression model** using the **Boston Housing dataset**.  
The goal is to predict the median value of owner-occupied homes (`MEDV`) based on various housing and neighborhood features.

---

## 📌 Features of the Project
- Loads the **Boston Housing dataset** with appropriate column names.
- Splits the data into **features (X)** and **target (y)**.
- Performs a **train-test split** to evaluate performance.
- Trains a **Linear Regression model** using `scikit-learn`.
- Evaluates the model with:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² score**
- Visualizes **Actual vs Predicted values** using a scatter plot.
- Displays **feature importance** through model coefficients.

---

## 📂 Dataset
The dataset used is the **Boston Housing dataset**.  
It contains 506 rows and 14 columns:
- 13 features such as crime rate (`CRIM`), nitric oxide concentration (`NOX`), average number of rooms (`RM`), etc.
- Target variable: `MEDV` (Median house value in $1000s).

⚠️ Make sure to include the dataset file named **`housing.csv`** in the project directory.

---

## ⚙️ Requirements
To run this project, you need the following Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib

Run the Python script:

python main.py
