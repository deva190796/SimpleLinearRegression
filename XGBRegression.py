import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"Supervised_Learning\xgboost_regression_dataset.csv")

X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

print("Train R2:", model.score(X_train, y_train))
print("Test R2:", model.score(X_test, y_test))

importance = model.feature_importances_

plt.barh(X.columns, importance)
plt.show()

for lr in [0.01, 0.05, 0.1, 0.2]:
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=lr,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("LR:", lr, "R2:", r2_score(y_test, pred))