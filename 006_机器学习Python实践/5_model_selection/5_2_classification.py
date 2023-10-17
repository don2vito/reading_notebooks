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
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',100)
pd.set_option('precision',3)

# 使用Pandas导入数据(分类算法)
file_name = os.path.join('./','pima_data.csv')
col_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(file_name,names=col_names)
array = data.values
X = array[:,0:8]
y = array[:,8]

# 线性算法
# 逻辑回归
n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits,random_state=seed)
model = LogisticRegression(solver='liblinear')
result = cross_val_score(model,X,y,cv=kfold)
print('逻辑回归的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100,result.std()))

# 逻辑回归的10折交叉验证的准确率评价指标是： 76.95%，标准差是： 0.05

# 线性判别分析（LDA）
model = LinearDiscriminantAnalysis()
result = cross_val_score(model,X,y,cv=kfold)
print('LDA的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100,result.std()))

# LDA的10折交叉验证的准确率评价指标是： 77.35%，标准差是： 0.05

# 非线性算法
# K近邻算法（KNN）
model = KNeighborsClassifier()
result = cross_val_score(model,X,y,cv=kfold)
print('KNN的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100,result.std()))

# KNN的10折交叉验证的准确率评价指标是： 72.66%，标准差是： 0.06

# 贝叶斯分类器
model = GaussianNB()
result = cross_val_score(model,X,y,cv=kfold)
print('高斯贝叶斯方法的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100,result.std()))

# 高斯贝叶斯方法的10折交叉验证的准确率评价指标是： 75.52%，标准差是： 0.04

# 分类与回归树
model = DecisionTreeClassifier()
result = cross_val_score(model,X,y,cv=kfold)
print('CART决策树的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100,result.std()))

# CART决策树的10折交叉验证的准确率评价指标是： 68.48%，标准差是： 0.06

# 支持向量机（SVM）
model = SVC()
result = cross_val_score(model,X,y,cv=kfold)
print('SVM的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100,result.std()))

# SVM的10折交叉验证的准确率评价指标是： 65.10%，标准差是： 0.07