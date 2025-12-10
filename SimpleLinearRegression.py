# #code to predict salary based on experience using Simple Linear Regression
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# path = r"C:/Users/Lenovo/Desktop/ML/SimpleLinearRegression/Salary_Data (2).csv"
# df = pd.read_csv(path)
# print(df)

# x = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(x_train,y_train)

# y_pred = regressor.predict(x_test)

# plt.scatter(x_test,y_test, color = 'red')
# plt.plot(x_train, regressor.predict(x_train), color = 'blue')
# plt.title("Salary VS Experience (Test set)")
# plt.xlabel('Years Of Experience')
# plt.ylabel("Salary")
# plt.show()

# #slope = coefficient i.e, coef_

# m = regressor.coef_
# c = regressor.intercept_
# bias = regressor.score(x_train,y_train)
# variance = regressor.score(x_test,y_test)
# exp_12 = (m * 12) + c

# #Statistics
# print(df.mean())
# print(df['Salary'].mean())  
# print(df.median())
# print(df['Salary'].median())
# print(df['Salary'].mode())
# print(df.var())
# print(df['Salary'].var())
# print(df['Salary'].std())

# from scipy.stats import variation
# print(variation(df['Salary']))
# print(df.corr())
# print(df['Salary'].corr(df['YearsExperience']))
# print(df.skew())
# print(df['Salary'].skew())
# print(df.sem())
# print(df['Salary'].sem())
# import scipy.stats as stats
# print(df.apply(stats.zscore))



# #SSR
# y_mean = np.mean(y)
# SSR = np.sum((y-y_pred)**2)
# print(SSR)

# y = y[0:6]
# SSE = np.sum((y-y_pred)**2)
# print(SSE)

# mean_total = np.mean(df.values)
# SST = np.sum((df.values - mean_total)**2)
# print(SST)

# import pickle 
# filename = 'linear_regressio_model.pkl'
# with open (filename,'wb') as file:
#     pickle.dump(regressor, file)
# print("model has pickled and saved")
# import os
# print(os.getcwd())


import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
dataset = pd.read_csv(r'C:/Users/Lenovo/Desktop/ML/SimpleLinearRegression/Salary_Data (2).csv')

# Split the data into independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values 

# Split the dataset into training and testing sets (80-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set
y_pred = regressor.predict(X_test)

# Visualize the training set
plt.scatter(X_train, y_train, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
plt.scatter(X_test, y_test, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model performance
bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")
