# KNN Regression (simple and clean)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("Supervised_Learning/knn_reg.csv")

X = data[['x']]
y = data['y']

# split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# try one K value first
from sklearn.neighbors import KNeighborsRegressor

k = 5
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("For K =", k)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# check different K values
print("\nTrying different K values")

for k in range(1, 11):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = r2_score(y_test, pred)
    print("K =", k, "R2 =", score)

# weighted KNN
print("\nUsing weighted KNN")

model = KNeighborsRegressor(n_neighbors=5, weights='distance')
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Weighted R2:", r2_score(y_test, pred))

# visualization (fit on full data just for graph)
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X, y)

X_sorted = np.sort(X.values, axis=0)
y_line = model.predict(X_sorted)

plt.scatter(X, y)
plt.plot(X_sorted, y_line, color='red')
plt.title("KNN Regression")
plt.show()