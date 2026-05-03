import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_excel(r"study_hours_dataset.xlsx")
print(dataset)
# Step 1: separate input (X) and output (y)
X = dataset[["Study_Hours"]]   # input must be 2D
y = dataset["Exam_Score"]      # output

# Step 2: import model
from sklearn.linear_model import LinearRegression

# Step 3: create model
model = LinearRegression()

# Step 4: train model
model.fit(X, y)

# Step 5: check learned values
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Step 6: predict
prediction = model.predict([[6.5]])
print("Predicted score for 6.5 hours:", prediction[0])