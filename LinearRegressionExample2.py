# encoding: utf-8
# %matplotlib inline

import numpy as np
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import sklearn

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
boston = load_boston()
print("boston.keys() =", boston.keys())
print("boston.feature_names =", boston.feature_names)
# print("boston.DESCR =", boston.DESCR)
bos = pd.DataFrame(boston.data)
print("bos.head() =\n", bos.head())
# change the number with feature name
bos.columns = boston.feature_names
print("bos.head() =\n", bos.head())

print("boston.target[:5] =",boston.target[:5])

bos['PRICE'] = boston.target
print("bos.head() =\n", bos.head())

X = bos.drop('PRICE', axis = 1)
# This creat a LinearRegression object
lm = LinearRegression()
lm.fit(X, bos.PRICE)
print("lm =", lm)
print("Estimated intercept coefficient:", lm.intercept_)
print("Number of coefficient:", len(lm.coef_))
print("lm.coef_ =", lm.coef_)

print("X.columns =", X.columns)
print("zip(X.columns, lm.coef_) =", zip(X.columns, lm.coef_))
# 关于zip使用发生错误，可以看解决方案，
# 如：https://stackoverflow.com/questions/45388800/python-data-argument-cant-be-an-iterator
print("pd.DataFrame(list(zip(X.columns, lm.coef_))) =\n", pd.DataFrame(list(zip(X.columns, lm.coef_))))
print("pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['features', 'estimatedCoefficients']) =\n", 
	pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['features', 'estimatedCoefficients']))

plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Prise")
plt.title("Relationship between RM and Price")
plt.show()

print("lm.predict(X)[0:5] =", lm.predict(X)[0:5])

plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted price: $\hat{Y}_i$")
plt.title("Price vs Predicted Price: $Y_i$ vs $\hat{Y}_i$")
plt.show()

# mean square error
mseFull = np.mean((bos.PRICE - lm.predict(X))**2)

print("mseFull =", mseFull)

lm = LinearRegression()
lm.fit(X[['PTRATIO']], bos.PRICE)
msePTRATIO = np.mean((bos.PRICE - lm.predict(X[['PTRATIO']]))**2)
print("msePTRATIO =", msePTRATIO)

# training&test
X_train = X[:-50]
X_test = X[-50:]
Y_train = bos.PRICE[:-50]
Y_test = bos.PRICE[-50:]

print("X_train.shape =", X_train.shape, "\n X_test =",X_test.shape, 
	"\n Y_train =", Y_train.shape, "\n Y_test =", Y_test.shape)
# 对取出的值进行交叉验证
X_train, X_test, Y_train, Y_test = train_test_split(
	X, bos.PRICE, test_size = 0.33, random_state =5)
print("X_train.shape =", X_train.shape, "\n X_test =",X_test.shape, 
	"\n Y_train =", Y_train.shape, "\n Y_test =", Y_test.shape)

lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

mseTrain = np.mean((Y_train - pred_train)**2)
mseTest = np.mean((Y_test - pred_test)**2)
print("mseTrain =", mseTrain)
print("mseTest =", mseTest)


plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c = 'b', s = 40, alpha = 0.5)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c = 'g', s = 40)
plt.hlines(y = 0, xmin = 0, xmax = 5)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residual')
plt.show()