# A simple linear expression: y = mx + b

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=1/3)

linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

y_pred = linear_regressor.predict(x_test)

# Visualizing predicted and test data
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, linear_regressor.predict(x_train), color="blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, linear_regressor.predict(x_train), color="blue")

plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
