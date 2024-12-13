import numpy as np  # For numerical computations
import pandas as pd  # For handling datasets
import matplotlib.pyplot as plt  # For visualizations

data = pd.read_csv(r"D:\FSDS Material\Dataset\Non Linear emp_sal.csv")

# Extract independent variable (x) and dependent variable (y)
x = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                     splitter='random',
                                     max_depth=4,
                                     min_samples_split=5,
                                     random_state=0)
dt_regressor.fit(x,y)

dt_reg_pred = dt_regressor.predict([[6.5]])
print(dt_reg_pred)

plt.scatter(x, y, color = 'red')
plt.plot(x, dt_regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, dt_regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

