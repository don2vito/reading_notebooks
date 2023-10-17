import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

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

# 评估算法
# 分离训练数据集和评估数据集
test_size = 0.33
seed = 4
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
result = model.score(X_test,y_test)
print('算法评估结果是： {:.2f}%'.format(result * 100))

# 算法评估结果是： 80.31%

# K折交叉验证
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)
model = LogisticRegression(solver='liblinear')
result = cross_val_score(model,X,y,cv=kfold)
print('10折交叉验证的评估结果是： {:.2f}%，标准差是： {:.2f}%'.format(result.mean() * 100,result.std() * 100))

# 10折交叉验证的评估结果是： 76.95%，标准差是： 4.84%

# 弃一交叉验证
loocv = LeaveOneOut()
model = LogisticRegression(solver='liblinear')
result = cross_val_score(model,X,y,cv=loocv)
print('弃一交叉验证的评估结果是： {:.2f}%，标准差是： {:.2f}%'.format(result.mean() * 100,result.std() * 100))

# 弃一交叉验证的评估结果是： 76.95%，标准差是： 42.11%

# 重复随机验证（洗牌）
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=seed)
model = LogisticRegression(solver='liblinear')
result = cross_val_score(model,X,y,cv=kfold)
print('重复随机验证的评估结果是： {:.2f}%，标准差是： {:.2f}%'.format(result.mean() * 100,result.std() * 100))

# 重复随机验证的评估结果是： 76.50%，标准差是： 1.70%

# 算法评估矩阵
# 分类算法矩阵
# 分类准确度（越大越好）
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)
model = LogisticRegression(solver='liblinear')
result = cross_val_score(model,X,y,cv=kfold)
print('10折交叉验证的分类准确度是： {:.2f}%，标准差是： {:.2f}%'.format(result.mean() * 100,result.std() * 100))

# 10折交叉验证的分类准确度是： 76.95%，标准差是： 4.84%

# 对数损失函数（越小越好）
scoring = 'neg_log_loss'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('10折交叉验证的对数损失函数是： {:.2f}%，标准差是： {:.2f}%'.format(result.mean() * 100,result.std() * 100))

# 10折交叉验证的对数损失函数是： -49.26%，标准差是： 4.71%

# AUC图（越大越好）
scoring = 'roc_auc'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('10折交叉验证的AUC指标是： {:.2f}%，标准差是： {:.2f}%'.format(result.mean() * 100,result.std() * 100))

# 10折交叉验证的AUC指标是： 82.36%，标准差是： 4.08%

# 混淆矩阵
test_size = 0.33
seed = 4
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(y_test,predicted)
classes = ['0','1']
dataframe = pd.DataFrame(data=matrix,index=classes,columns=classes)
print(f'混淆矩阵如下：\n{dataframe}')

# 混淆矩阵如下：
#      0   1
# 0  152  19
# 1   31  52

# 分类报告
# 精准率、召回率、F1值、样本数目
test_size = 0.33
seed = 4
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
predicted = model.predict(X_test)
report = classification_report(y_test,predicted)
print(f'分类报告如下：\n{report}')

# 分类报告如下：
#               precision    recall  f1-score   support
#
#          0.0       0.83      0.89      0.86       171
#          1.0       0.73      0.63      0.68        83
#
#    micro avg       0.80      0.80      0.80       254
#    macro avg       0.78      0.76      0.77       254
# weighted avg       0.80      0.80      0.80       254

# 回归算法矩阵
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
# print(data2.info())

# MAE（平均绝对误差）
n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits,random_state=seed)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_mean_absolute_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('10折交叉验证的MAE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 10折交叉验证的MAE指标是： -0.72，标准差是： 0.51

# MSE（均方误差）
scoring = 'neg_mean_squared_error'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('10折交叉验证的MSE指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 10折交叉验证的MSE指标是： -1.53，标准差是： 1.37

# R方（决定系数）
scoring = 'r2'
result = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
print('10折交叉验证的R方指标是： {:.2f}，标准差是： {:.2f}'.format(result.mean(),result.std()))

# 10折交叉验证的R方指标是： 0.13，标准差是： 0.96