#code to predict salary based on experience using Simple Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = r"C:/Users/Lenovo/Desktop/ML/SimpleLinearRegression/Salary_Data (2).csv"
df = pd.read_csv(path)
print(df)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary VS Experience (Test set)")
plt.xlabel('Years Of Experience')
plt.ylabel("Salary")
plt.show()

#slope = coefficient i.e, coef_

m = regressor.coef_
c = regressor.intercept_

exp_12 = (m * 12) + c