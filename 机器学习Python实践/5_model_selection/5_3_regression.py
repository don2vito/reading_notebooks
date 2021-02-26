import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',100)
pd.set_option('precision',3)

# 使用Pandas导入数据(回归算法)
file_name2 = os.path.join('./','housing.csv')
col_names2 = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATIO','B','LSTAT','MEDV']
data2 = pd.read_csv(file_name2,names=col_names2,delim_whitespace=True)
# data2['CHAS'] = data2['CHAS'].astype('float64')
# data2['RAD'] = data2['RAD'].astype('float64')
# data2['MEDV'] = data2['RAD'].astype('int')
array2 = data2.values
X = array2[:,0:13]
y = array2[:,13]

# 回归算法
# 线性算法
# 线性回归
n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits,random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('线性回归的10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 线性回归的10折交叉验证的MSE指标是： -34.71，标准差是： 45.57

# 岭回归
model = Ridge()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('岭回归的10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 岭回归的10折交叉验证的MSE指标是： -34.08，标准差是： 45.90

# 套索回归
model = Lasso()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('套索回归的10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 套索回归的10折交叉验证的MSE指标是： -34.46，标准差是： 27.89

# 弹性网络回归
model = ElasticNet()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('弹性网络回归的10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 弹性网络回归的10折交叉验证的MSE指标是： -31.16，标准差是： 22.71

# 非线性算法
# 使用Pandas导入数据(回归算法)
file_name2 = os.path.join('./','housing.csv')
col_names2 = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATIO','B','LSTAT','MEDV']
data2 = pd.read_csv(file_name2,names=col_names2,delim_whitespace=True)
# data2['CHAS'] = data2['CHAS'].astype('float64')
# data2['RAD'] = data2['RAD'].astype('float64')
data2['MEDV'] = data2['RAD'].astype('int')
array2 = data2.values
X = array2[:,0:13]
y = array2[:,13]
n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits,random_state=seed)

# K近邻算法（KNN）
model = KNeighborsClassifier()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('KNN的10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# KNN的10折交叉验证的MSE指标是： -7.73，标准差是： 11.56

# 分类与回归树
model = DecisionTreeClassifier()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('决策树的10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 决策树的10折交叉验证的MSE指标是： -0.03，标准差是： 0.06

# 支持向量机（SVM）
model = SVR()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('SVM的10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# SVM的10折交叉验证的MSE指标是： -97.80，标准差是： 147.95