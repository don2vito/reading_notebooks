import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 20)
pd.set_option('precision',3)

warnings.filterwarnings('ignore')

# 导入数据集
file_name = os.path.join('./','sonar.all-data.csv')
data = pd.read_csv(file_name,header=None)

# 数据理解
# 查看数据维度
print(f'数据集的维度有 {data.shape[0]}行，{data.shape[1]}列')

# 数据集的维度有 208行，61列

# 查看特征属性的字段类型
pd.set_option('display.max_rows', 70)
print((f'特征属性的字段类型如下：\n{data.dtypes}'))

# 特征属性的字段类型如下：
# 0     float64
# 1     float64
# 2     float64
# 3     float64
# 4     float64
# 5     float64
# 6     float64
# 7     float64
# 8     float64
# 9     float64
# 10    float64
# 11    float64
# 12    float64
# 13    float64
# 14    float64
# 15    float64
# 16    float64
# 17    float64
# 18    float64
# 19    float64
# 20    float64
# 21    float64
# 22    float64
# 23    float64
# 24    float64
# 25    float64
# 26    float64
# 27    float64
# 28    float64
# 29    float64
# 30    float64
# 31    float64
# 32    float64
# 33    float64
# 34    float64
# 35    float64
# 36    float64
# 37    float64
# 38    float64
# 39    float64
# 40    float64
# 41    float64
# 42    float64
# 43    float64
# 44    float64
# 45    float64
# 46    float64
# 47    float64
# 48    float64
# 49    float64
# 50    float64
# 51    float64
# 52    float64
# 53    float64
# 54    float64
# 55    float64
# 56    float64
# 57    float64
# 58    float64
# 59    float64
# 60     object
# dtype: object

# 查看数据集中前10行数据
pd.set_option('display.width',120)
print((f'数据集中前10行数据如下：\n{data.head(10)}'))

# 数据集中前10行数据如下：
#       0      1      2      3      4      5      6      7      8      9   ...     51     52     53     54     55  \
# 0  0.020  0.037  0.043  0.021  0.095  0.099  0.154  0.160  0.311  0.211  ...  0.003  0.006  0.016  0.007  0.017
# 1  0.045  0.052  0.084  0.069  0.118  0.258  0.216  0.348  0.334  0.287  ...  0.008  0.009  0.005  0.009  0.019
# 2  0.026  0.058  0.110  0.108  0.097  0.228  0.243  0.377  0.560  0.619  ...  0.023  0.017  0.009  0.018  0.024
# 3  0.010  0.017  0.062  0.021  0.021  0.037  0.110  0.128  0.060  0.126  ...  0.012  0.004  0.015  0.009  0.007
# 4  0.076  0.067  0.048  0.039  0.059  0.065  0.121  0.247  0.356  0.446  ...  0.003  0.005  0.011  0.011  0.002
# 5  0.029  0.045  0.028  0.017  0.038  0.099  0.120  0.183  0.210  0.304  ...  0.004  0.001  0.004  0.001  0.009
# 6  0.032  0.096  0.132  0.141  0.167  0.171  0.073  0.140  0.208  0.351  ...  0.020  0.025  0.013  0.007  0.014
# 7  0.052  0.055  0.084  0.032  0.116  0.092  0.103  0.061  0.146  0.284  ...  0.008  0.012  0.004  0.012  0.010
# 8  0.022  0.037  0.048  0.048  0.065  0.059  0.075  0.010  0.068  0.149  ...  0.015  0.013  0.015  0.006  0.005
# 9  0.016  0.017  0.035  0.007  0.019  0.067  0.106  0.070  0.096  0.025  ...  0.009  0.022  0.018  0.008  0.007
#
#       56     57     58     59  60
# 0  0.018  0.008  0.009  0.003   R
# 1  0.014  0.005  0.005  0.004   R
# 2  0.032  0.016  0.009  0.008   R
# 3  0.005  0.004  0.004  0.012   R
# 4  0.007  0.005  0.011  0.009   R
# 5  0.006  0.003  0.005  0.006   R
# 6  0.009  0.014  0.004  0.010   R
# 7  0.009  0.005  0.005  0.005   R
# 8  0.006  0.009  0.006  0.002   R
# 9  0.003  0.004  0.006  0.004   R
#
# [10 rows x 61 columns]

# 描述性统计信息
pd.set_option('precision',3)
print((f'数据集中的描述性统计信息如下：\n{data.describe()}'))

# 数据集中的描述性统计信息如下：
#             0          1        2        3        4        5        6        7        8        9   ...       50  \
# count  208.000  2.080e+02  208.000  208.000  208.000  208.000  208.000  208.000  208.000  208.000  ...  208.000
# mean     0.029  3.844e-02    0.044    0.054    0.075    0.105    0.122    0.135    0.178    0.208  ...    0.016
# std      0.023  3.296e-02    0.038    0.047    0.056    0.059    0.062    0.085    0.118    0.134  ...    0.012
# min      0.002  6.000e-04    0.002    0.006    0.007    0.010    0.003    0.005    0.007    0.011  ...    0.000
# 25%      0.013  1.645e-02    0.019    0.024    0.038    0.067    0.081    0.080    0.097    0.111  ...    0.008
# 50%      0.023  3.080e-02    0.034    0.044    0.062    0.092    0.107    0.112    0.152    0.182  ...    0.014
# 75%      0.036  4.795e-02    0.058    0.065    0.100    0.134    0.154    0.170    0.233    0.269  ...    0.021
# max      0.137  2.339e-01    0.306    0.426    0.401    0.382    0.373    0.459    0.683    0.711  ...    0.100
#
#               51         52       53         54         55         56         57         58         59
# count  2.080e+02  2.080e+02  208.000  2.080e+02  2.080e+02  2.080e+02  2.080e+02  2.080e+02  2.080e+02
# mean   1.342e-02  1.071e-02    0.011  9.290e-03  8.222e-03  7.820e-03  7.949e-03  7.941e-03  6.507e-03
# std    9.634e-03  7.060e-03    0.007  7.088e-03  5.736e-03  5.785e-03  6.470e-03  6.181e-03  5.031e-03
# min    8.000e-04  5.000e-04    0.001  6.000e-04  4.000e-04  3.000e-04  3.000e-04  1.000e-04  6.000e-04
# 25%    7.275e-03  5.075e-03    0.005  4.150e-03  4.400e-03  3.700e-03  3.600e-03  3.675e-03  3.100e-03
# 50%    1.140e-02  9.550e-03    0.009  7.500e-03  6.850e-03  5.950e-03  5.800e-03  6.400e-03  5.300e-03
# 75%    1.673e-02  1.490e-02    0.015  1.210e-02  1.058e-02  1.043e-02  1.035e-02  1.033e-02  8.525e-03
# max    7.090e-02  3.900e-02    0.035  4.470e-02  3.940e-02  3.550e-02  4.400e-02  3.640e-02  4.390e-02
#
# [8 rows x 60 columns]

# 查看数据的分类分布
print('数据集的分类分布情况如下：')
print(data.groupby(60).size())

# 数据集的分类分布情况如下：
# 60
# M    111
# R     97
# dtype: int64

# 数据可视化
# 单变量图
# 直方图
data.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
# plt.tight_layout()
pic_name = '直方图' + '_binary_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('直方图 保存成功！！')

# 密度图
data.plot(kind='density',subplots=True,layout=(8,8),sharex=False,legend=False,fontsize=1)
# plt.tight_layout()
pic_name = '密度图' + '_binary_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('密度图 保存成功！！')

# 多变量图
# 相关矩阵图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(),vmin=-1,vmax=1,interpolation='none')
fig.colorbar(cax)
# ticks = np.arange(0,61,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
plt.tight_layout()
pic_name = '相关矩阵图' + '_binary_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('相关矩阵图 保存成功！！')

# 分离测试数据集
array = data.values
X = array[:,0:60].astype('float')
y = array[:,60]
test_size = 0.2
seed = 7
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)

# 评估算法
# 评估算法 —— 原始数据
# 评估算法 —— 评估标准
num_folds = 10
seed = 7
scoring = 'accuracy'

# 评估算法 —— baseline
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['SVM'] = SVC()
models['NB'] = GaussianNB()

# 评估算法 —— 指标（accuracy越接近1越好）
results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    result = cross_val_score(models[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(result)
    print('{} 的10折交叉验证的准确率指标是： {:.2f}%，标准差是： {:.2f}'.format(key,result.mean() * 100, result.std()))

# LR 的10折交叉验证的准确率指标是： 78.27%，标准差是： 0.09
# LDA 的10折交叉验证的准确率指标是： 74.63%，标准差是： 0.12
# KNN 的10折交叉验证的准确率指标是： 80.81%，标准差是： 0.07
# CART 的10折交叉验证的准确率指标是： 72.35%，标准差是： 0.11
# SVM 的10折交叉验证的准确率指标是： 60.88%，标准差是： 0.12
# NB 的10折交叉验证的准确率指标是： 64.89%，标准差是： 0.14

# 评估算法 —— 箱线图
fig = plt.figure()
fig.suptitle('算法比较')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.tight_layout()
pic_name = '算法比较箱线图' + '_binary_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('算法比较箱线图 保存成功！！')

# 评估算法 —— 正态化数据
piplines = {}
piplines['ScalerLR'] = Pipeline([('Scaler',StandardScaler()),('LR',LogisticRegression())])
piplines['ScalerLDA'] = Pipeline([('Scaler',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])
piplines['ScalerKNN'] = Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])
piplines['ScalerCART'] = Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeClassifier())])
piplines['ScalerSVM'] = Pipeline([('Scaler',StandardScaler()),('SVM',SVC())])
piplines['ScalerNB'] = Pipeline([('Scaler',StandardScaler()),('NB',GaussianNB())])

cv_results = []
for key in piplines:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(piplines[key],X_train,y_train,cv=kfold,scoring=scoring)
    cv_results.append(cv_result)
    print('正态化数据后， {} 的10折交叉验证的准确率指标是： {:.2f}%，标准差是： {:.2f}'.format(key, cv_result.mean() * 100, cv_result.std()))

# 正态化数据后， ScalerLR 的10折交叉验证的准确率指标是： 73.42%，标准差是： 0.10
# 正态化数据后， ScalerLDA 的10折交叉验证的准确率指标是： 74.63%，标准差是： 0.12
# 正态化数据后， ScalerKNN 的10折交叉验证的准确率指标是： 82.57%，标准差是： 0.05
# 正态化数据后， ScalerCART 的10折交叉验证的准确率指标是： 74.08%，标准差是： 0.09
# 正态化数据后， ScalerSVM 的10折交叉验证的准确率指标是： 83.64%，标准差是： 0.09
# 正态化数据后， ScalerNB 的10折交叉验证的准确率指标是： 64.89%，标准差是： 0.14

# 评估算法 —— 箱线图
fig = plt.figure()
fig.suptitle('算法比较')
ax = fig.add_subplot(111)
plt.boxplot(cv_results)
ax.set_xticklabels(piplines.keys())
plt.tight_layout()
pic_name = '正态化数据后算法比较箱线图' + '_binary_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('正态化数据后算法比较箱线图 保存成功！！')

# 调参改善算法
# 调参改善算法 —— KNN
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
param_grid = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescalered_X,y=y_train)
print('调整后KNN模型的最优准确率是 {:.2f}%，使用的近邻个数k = {}'.format(grid_result.best_score_ * 100,grid_result.best_params_['n_neighbors']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举KNN算法中取不同k值时的情况如下：')
for mean,std,param in cv_results:
    print('KNN模型的准确率是 {:.2f}%，标准差是 {:.2f}，使用的近邻个数k = {}'.format(mean * 100,std,param['n_neighbors']))

# 调整后KNN模型的最优准确率是 84.94%，使用的近邻个数k = 1
# 穷举KNN算法中取不同k值时的情况如下：
# KNN模型的准确率是 84.94%，标准差是 0.06，使用的近邻个数k = 1
# KNN模型的准确率是 83.73%，标准差是 0.07，使用的近邻个数k = 3
# KNN模型的准确率是 83.73%，标准差是 0.04，使用的近邻个数k = 5
# KNN模型的准确率是 76.51%，标准差是 0.09，使用的近邻个数k = 7
# KNN模型的准确率是 75.30%，标准差是 0.09，使用的近邻个数k = 9
# KNN模型的准确率是 73.49%，标准差是 0.10，使用的近邻个数k = 11
# KNN模型的准确率是 73.49%，标准差是 0.11，使用的近邻个数k = 13
# KNN模型的准确率是 72.89%，标准差是 0.08，使用的近邻个数k = 15
# KNN模型的准确率是 71.08%，标准差是 0.08，使用的近邻个数k = 17
# KNN模型的准确率是 72.29%，标准差是 0.08，使用的近邻个数k = 19
# KNN模型的准确率是 71.08%，标准差是 0.11，使用的近邻个数k = 21

# 调参改善算法 —— SVM
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
param_grid = {}
param_grid['C'] = [0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0]
param_grid['kernel'] = ['linear','poly','rbf','sigmoid']
model = SVC()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescalered_X,y=y_train)
print('调整后SVM模型的最优准确率是 {:.2f}%，使用的惩罚系数C = {}，使用的径向基函数kernel是 {}'.format(grid_result.best_score_ * 100,grid_result.best_params_['C'],grid_result.best_params_['kernel']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举SVM算法中取不同惩罚系数和径向基函数时的情况如下：')
for mean,std,param in cv_results:
    print('SVM模型的准确率是 {:.2f}%，标准差是 {:.2f}，使用的惩罚系数C = {}，使用的径向基函数kernel是 {}'.format(mean * 100,std,param['C'],param['kernel']))

# 调整后SVM模型的最优准确率是 86.75%，使用的惩罚系数C = 1.5，使用的径向基函数kernel是 rbf
# 穷举SVM算法中取不同惩罚系数和径向基函数时的情况如下：
# SVM模型的准确率是 75.90%，标准差是 0.10，使用的惩罚系数C = 0.1，使用的径向基函数kernel是 linear
# SVM模型的准确率是 53.01%，标准差是 0.12，使用的惩罚系数C = 0.1，使用的径向基函数kernel是 poly
# SVM模型的准确率是 57.23%，标准差是 0.13，使用的惩罚系数C = 0.1，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 70.48%，标准差是 0.07，使用的惩罚系数C = 0.1，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 74.70%，标准差是 0.11，使用的惩罚系数C = 0.3，使用的径向基函数kernel是 linear
# SVM模型的准确率是 64.46%，标准差是 0.13，使用的惩罚系数C = 0.3，使用的径向基函数kernel是 poly
# SVM模型的准确率是 76.51%，标准差是 0.09，使用的惩罚系数C = 0.3，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 73.49%，标准差是 0.05，使用的惩罚系数C = 0.3，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 74.10%，标准差是 0.08，使用的惩罚系数C = 0.5，使用的径向基函数kernel是 linear
# SVM模型的准确率是 68.07%，标准差是 0.10，使用的惩罚系数C = 0.5，使用的径向基函数kernel是 poly
# SVM模型的准确率是 78.92%，标准差是 0.06，使用的惩罚系数C = 0.5，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 74.70%，标准差是 0.06，使用的惩罚系数C = 0.5，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 74.70%，标准差是 0.08，使用的惩罚系数C = 0.7，使用的径向基函数kernel是 linear
# SVM模型的准确率是 74.10%，标准差是 0.13，使用的惩罚系数C = 0.7，使用的径向基函数kernel是 poly
# SVM模型的准确率是 81.33%，标准差是 0.08，使用的惩罚系数C = 0.7，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 75.30%，标准差是 0.06，使用的惩罚系数C = 0.7，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 75.90%，标准差是 0.10，使用的惩罚系数C = 0.9，使用的径向基函数kernel是 linear
# SVM模型的准确率是 77.11%，标准差是 0.10，使用的惩罚系数C = 0.9，使用的径向基函数kernel是 poly
# SVM模型的准确率是 83.73%，标准差是 0.09，使用的惩罚系数C = 0.9，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 75.30%，标准差是 0.07，使用的惩罚系数C = 0.9，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 75.30%，标准差是 0.10，使用的惩罚系数C = 1.0，使用的径向基函数kernel是 linear
# SVM模型的准确率是 78.92%，标准差是 0.11，使用的惩罚系数C = 1.0，使用的径向基函数kernel是 poly
# SVM模型的准确率是 83.73%，标准差是 0.09，使用的惩罚系数C = 1.0，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 75.30%，标准差是 0.07，使用的惩罚系数C = 1.0，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 77.11%，标准差是 0.11，使用的惩罚系数C = 1.3，使用的径向基函数kernel是 linear
# SVM模型的准确率是 81.93%，标准差是 0.11，使用的惩罚系数C = 1.3，使用的径向基函数kernel是 poly
# SVM模型的准确率是 84.94%，标准差是 0.08，使用的惩罚系数C = 1.3，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 71.08%，标准差是 0.08，使用的惩罚系数C = 1.3，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 75.90%，标准差是 0.09，使用的惩罚系数C = 1.5，使用的径向基函数kernel是 linear
# SVM模型的准确率是 83.13%，标准差是 0.11，使用的惩罚系数C = 1.5，使用的径向基函数kernel是 poly
# SVM模型的准确率是 86.75%，标准差是 0.09，使用的惩罚系数C = 1.5，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 74.10%，标准差是 0.06，使用的惩罚系数C = 1.5，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 74.70%，标准差是 0.09，使用的惩罚系数C = 1.7，使用的径向基函数kernel是 linear
# SVM模型的准确率是 83.13%，标准差是 0.12，使用的惩罚系数C = 1.7，使用的径向基函数kernel是 poly
# SVM模型的准确率是 86.14%，标准差是 0.09，使用的惩罚系数C = 1.7，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 71.08%，标准差是 0.09，使用的惩罚系数C = 1.7，使用的径向基函数kernel是 sigmoid
# SVM模型的准确率是 75.90%，标准差是 0.09，使用的惩罚系数C = 2.0，使用的径向基函数kernel是 linear
# SVM模型的准确率是 83.13%，标准差是 0.11，使用的惩罚系数C = 2.0，使用的径向基函数kernel是 poly
# SVM模型的准确率是 86.75%，标准差是 0.09，使用的惩罚系数C = 2.0，使用的径向基函数kernel是 rbf
# SVM模型的准确率是 72.89%，标准差是 0.10，使用的惩罚系数C = 2.0，使用的径向基函数kernel是 sigmoid

# 集成算法
ensembles = {}
ensembles['ScalerAB'] = Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostClassifier())])
ensembles['ScalerGBM'] = Pipeline([('Scaler',StandardScaler()),('GBM',GradientBoostingClassifier())])
ensembles['ScalerRF'] = Pipeline([('Scaler',StandardScaler()),('RF',RandomForestClassifier())])
ensembles['ScalerET'] = Pipeline([('Scaler',StandardScaler()),('ET',ExtraTreesClassifier())])

cv_results = []
for key in ensembles:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(ensembles[key],X_train,y_train,cv=kfold,scoring=scoring)
    cv_results.append(cv_result)
    print('正态化数据后，集成算法 {} 的10折交叉验证的准确率是： {:.2f}%，标准差是： {:.2f}'.format(key, cv_result.mean() * 100, cv_result.std()))

# 正态化数据后，集成算法 ScalerAB 的10折交叉验证的准确率是： 81.40%，标准差是： 0.07
# 正态化数据后，集成算法 ScalerGBM 的10折交叉验证的准确率是： 84.78%，标准差是： 0.11
# 正态化数据后，集成算法 ScalerRF 的10折交叉验证的准确率是： 77.83%，标准差是： 0.07
# 正态化数据后，集成算法 ScalerET 的10折交叉验证的准确率是： 78.75%，标准差是： 0.13

# 集成算法 —— 箱线图
fig = plt.figure()
fig.suptitle('集成算法比较')
ax = fig.add_subplot(111)
plt.boxplot(cv_results)
ax.set_xticklabels(ensembles.keys())
plt.tight_layout()
pic_name = '正态化数据后集成算法比较箱线图' + '_binary_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('正态化数据后集成算法比较箱线图 保存成功！！')

# 集成算法调参
# 集成算法GBM —— 调参
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
param_grid = {'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
model = GradientBoostingClassifier()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescalered_X,y=y_train)
print('调整后GBM模型的最优准确率是 {:.2f}%，使用的估计量estimator = {}'.format(grid_result.best_score_ * 100,grid_result.best_params_['n_estimators']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举GBM算法中取不同估计量时的情况如下：')
for mean,std,param in cv_results:
    print('GBM模型的准确率是 {:.2f}%，标准差是 {:.2f}，使用的估计量estimator = {}'.format(mean * 100,std,param['n_estimators']))

# 调整后GBM模型的最优准确率是 86.14%，使用的估计量estimator = 500
# 穷举GBM算法中取不同估计量时的情况如下：
# GBM模型的准确率是 73.49%，标准差是 0.10，使用的估计量estimator = 10
# GBM模型的准确率是 83.13%，标准差是 0.08，使用的估计量estimator = 50
# GBM模型的准确率是 85.54%，标准差是 0.10，使用的估计量estimator = 100
# GBM模型的准确率是 84.34%，标准差是 0.14，使用的估计量estimator = 200
# GBM模型的准确率是 85.54%，标准差是 0.11，使用的估计量estimator = 300
# GBM模型的准确率是 85.54%，标准差是 0.11，使用的估计量estimator = 400
# GBM模型的准确率是 86.14%，标准差是 0.10，使用的估计量estimator = 500
# GBM模型的准确率是 85.54%，标准差是 0.10，使用的估计量estimator = 600
# GBM模型的准确率是 85.54%，标准差是 0.11，使用的估计量estimator = 700
# GBM模型的准确率是 85.54%，标准差是 0.10，使用的估计量estimator = 800
# GBM模型的准确率是 84.34%，标准差是 0.11，使用的估计量estimator = 900

# 确定最终模型
# 训练模型
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
svm = SVC(C=1.5,kernel='rbf')
svm.fit(X=rescalered_X,y=y_train)

# 评估算法模型
rescalered_X_test = scaler.transform(X_test)
predictions = svm.predict(rescalered_X_test)
accuracy = accuracy_score(y_test,predictions)
confusion = confusion_matrix(y_test,predictions)
classification = classification_report(y_test,predictions)
print('最终模型的预测准确率为 {:.2f}%'.format(accuracy * 100))
print('最终模型的预测混淆矩阵如下：\n{}'.format(confusion))
print('最终模型的预测分类指标如下：\n{}'.format(classification))

# 最终模型的预测准确率为 85.71%
# 最终模型的预测混淆矩阵如下：
# [[23  4]
#  [ 2 13]]
# 最终模型的预测分类指标如下：
#               precision    recall  f1-score   support
#
#            M       0.92      0.85      0.88        27
#            R       0.76      0.87      0.81        15
#
#    micro avg       0.86      0.86      0.86        42
#    macro avg       0.84      0.86      0.85        42
# weighted avg       0.86      0.86      0.86        42