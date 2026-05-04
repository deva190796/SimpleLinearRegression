import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_excel(r"underfitting_dataset.xlsx")
#print(data)
x = data[["Study_Hours"]]
y = data[["Exam_Score"]]

model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
print(y_pred)

from sklearn.metrics import mean_squared_error,r2_score
print("MSE:", mean_squared_error(y, y_pred))
print("R2 :", r2_score(y, y_pred))

import numpy as np
y_mean = np.mean(y)
y_base = [y_mean] * len(y)

mse_base = mean_squared_error(y, y_base)
print("Baseline MSE:", mse_base)


plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model2 = LinearRegression()
model2.fit(x_poly, y)

y_pred2 = model2.predict(x_poly)

print("Linear R2 :", r2_score(y, y_pred))
print("Poly R2   :", r2_score(y, y_pred2))

plt.scatter(x, y, color='blue')
plt.plot(x, y_pred2, color='green')
plt.show()