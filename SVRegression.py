import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)

data = data.drop(columns=["name"])
data = data.dropna()
data = pd.get_dummies(data, columns=["origin"], drop_first=True)

X = data.drop("mpg", axis=1)
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
svr_model = SVR(kernel="rbf", C=100, epsilon=0.1)

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_svr = svr_model.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(name)
    print("R2:", r2_score(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print()

evaluate("Linear Regression", y_test, y_pred_linear)
evaluate("Ridge Regression", y_test, y_pred_ridge)
evaluate("SVR", y_test, y_pred_svr)

error_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_svr,
    "Error": y_test.values - y_pred_svr
})

print(error_df.reindex(error_df["Error"].abs().sort_values(ascending=False).index).head())