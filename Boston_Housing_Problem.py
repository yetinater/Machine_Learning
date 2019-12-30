# Boston housing dataset from scikit worked upon using multivariate linear regression

from sklearn.datasets import load_boston

boston = load_boston()

X = boston.data
Y = boston.target

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

lr = LinearRegression(normalize = True)

lr.fit(X_train,Y_train)

print(lr.coef_)
print(lr.intercept_)
print(lr.score(X_train,Y_train))
