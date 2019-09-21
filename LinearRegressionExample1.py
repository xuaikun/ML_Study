print(__doc__)
# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print("diabetes =\n", diabetes)

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
# 多增加一个轴
print("np.newaxis =", np.newaxis)
# Slipt the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
print("diabetes_X_test =\n", diabetes_X_test)

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
print("diabetes_y_test =\n", diabetes_y_test)
# Create linear regression object
regr = linear_model.LinearRegression()
print("regr =\n", regr)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
print("diabetes_y_pred =\n", diabetes_y_pred)
# The coefficients
print('coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %0.2f"
	%mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('variance score: %.2f'%r2_score(diabetes_y_test, diabetes_y_pred))

# Plot output
plt.scatter(diabetes_X_test, diabetes_y_test, color = 'black')
plt.plot(diabetes_X_test, diabetes_y_pred, color = 'blue', linewidth = 3)

plt.xticks(())
plt.yticks(())

plt.show()