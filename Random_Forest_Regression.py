# Random Forest Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Supervised_Learning/bike_rentals.csv")

print(data.head())
print(data.info())


# Drop datetime (string column)
data = data.drop('datetime', axis=1)

X = data.drop('count', axis=1)   # correct column name
y = data['count']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Performance ---")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


print("\n--- Overfitting Check ---")
print("Train R2:", model.score(X_train, y_train))
print("Test R2:", model.score(X_test, y_test))


importance = model.feature_importances_

plt.barh(X.columns, importance)
plt.title("Feature Importance")
plt.show()