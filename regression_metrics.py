#Evaluation metrics for regression

from sklearn.metrics import mean_absolute_error

y_true = [10, 20, 30, 40]
y_pred = [12, 18, 33, 37]

print("Mean Absolute Error: ",mean_absolute_error(y_true, y_pred))

from sklearn.metrics import mean_squared_error

print("Mean_Squared_Error: ",mean_squared_error(y_true,y_pred))

import numpy as np
RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE: ",RMSE)

from sklearn.metrics import r2_score
R2_Score = r2_score(y_true, y_pred)
print("R2_Score: ",R2_Score)
