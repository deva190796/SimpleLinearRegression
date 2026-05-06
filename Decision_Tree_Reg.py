import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_excel("Supervised_Learning\decision_tree_dataset.xlsx")

le = LabelEncoder()
for col in ["Assessment", "Assignment", "Project"]:
    data[col] = le.fit_transform(data[col])

X = data.drop("Result", axis=1)
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()

new_data = [[2, 1, 1]]  # example encoded input
print(model.predict(new_data))

print(data.head())