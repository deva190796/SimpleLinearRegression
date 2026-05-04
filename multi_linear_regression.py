#importing libraries

import warnings
warnings.filterwarnings("ignore")
import pandas as pd #for dataset operations
from sklearn.linear_model import LinearRegression #for model usage

#dataset creation
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8],
    "Sleep_Hours": [6,7,6,8,7,8,7,9],
    "Exam_Score": [35,45,50,60,65,75,80,90]
}

df = pd.DataFrame(data) # conversion of data

#x
X = df[["Study_Hours","Sleep_Hours"]]

#y
y = df["Exam_Score"]

model = LinearRegression()
model.fit(X,y)

print("Coefficient:",model.coef_)
print("Intercept:",model.intercept_)

print("Prediction:",model.predict([[6,8]]))

from sklearn.metrics import mean_squared_error

y_pred = model.predict(X)
mse_model = mean_squared_error(y, y_pred)
print("Model MSE:", mse_model)

import numpy as np

y_mean = np.mean(y)
y_baseline = [y_mean] * len(y)

mse_baseline = mean_squared_error(y, y_baseline)
print("Baseline MSE:", mse_baseline)

from sklearn.metrics import r2_score

r2 = r2_score(y, y_pred)
print("R2 Score:", r2)






