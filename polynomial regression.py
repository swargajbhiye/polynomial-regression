import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#linear model -- linear algor (degree -1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#polynomical model (bydefeaut degree - 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#linear regression visualization
plt.scatter(x,y, color = 'blue')
plt.plot(x,lin_reg.predict(x),color = 'red')
plt.xlabel('Linear Regression graph')
plt.ylabel('Position level')
plt.show()

#poly nomial visualization
plt.scatter(x,y, color = 'blue')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'red')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#prediction 
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred
