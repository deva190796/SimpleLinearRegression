import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv(r"C:/Users/Lenovo/Desktop/Spy/Salary_Data (2).csv")
print(data)
x = data[['YearsExperience']]
y = data[['Salary']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20,random_state = 0 )
model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x_test)

print(y_pred)

#checking the performance of the model
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print("Train R^2:",train_score)
print("Test R^2:",test_score)

m = model.intercept_
c = model.coef_

print("Slope:",m)
print("Coefficient:",c)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R^2 score:",r2)
print("MSE:",mse)



