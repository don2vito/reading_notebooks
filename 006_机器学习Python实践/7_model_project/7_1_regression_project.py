import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 20)
pd.set_option('precision',3)

# 使用Pandas导入数据(回归算法)
file_name2 = os.path.join('./','housing.csv')
col_names2 = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATIO','B','LSTAT','MEDV']
data2 = pd.read_csv(file_name2,names=col_names2,delim_whitespace=True)
# data2['CHAS'] = data2['CHAS'].astype('float64')
# data2['RAD'] = data2['RAD'].astype('float64')
# data2['MEDV'] = data2['RAD'].astype('int')

# 数据理解
# 查看数据维度
print(f'数据集的维度有 {data2.shape[0]}行，{data2.shape[1]}列')

# 数据集的维度有 506行，14列

# 查看特征属性的字段类型
print((f'特征属性的字段类型如下：\n{data2.dtypes}'))

# 特征属性的字段类型如下：
# CRIM       float64
# ZN         float64
# INDUS      float64
# CHAS         int64
# NOX        float64
# RM         float64
# AGE        float64
# DIS        float64
# RAD          int64
# TAX        float64
# PRTATIO    float64
# B          float64
# LSTAT      float64
# MEDV       float64
# dtype: object

# 查看数据集中前10行数据
print((f'数据集中前10行数据如下：\n{data2.head(10)}'))

# 数据集中前10行数据如下：
#     CRIM    ZN  INDUS  CHAS    NOX     RM    AGE    DIS  RAD    TAX  PRTATIO       B  LSTAT  MEDV
# 0  0.006  18.0   2.31     0  0.538  6.575   65.2  4.090    1  296.0     15.3  396.90   4.98  24.0
# 1  0.027   0.0   7.07     0  0.469  6.421   78.9  4.967    2  242.0     17.8  396.90   9.14  21.6
# 2  0.027   0.0   7.07     0  0.469  7.185   61.1  4.967    2  242.0     17.8  392.83   4.03  34.7
# 3  0.032   0.0   2.18     0  0.458  6.998   45.8  6.062    3  222.0     18.7  394.63   2.94  33.4
# 4  0.069   0.0   2.18     0  0.458  7.147   54.2  6.062    3  222.0     18.7  396.90   5.33  36.2
# 5  0.030   0.0   2.18     0  0.458  6.430   58.7  6.062    3  222.0     18.7  394.12   5.21  28.7
# 6  0.088  12.5   7.87     0  0.524  6.012   66.6  5.561    5  311.0     15.2  395.60  12.43  22.9
# 7  0.145  12.5   7.87     0  0.524  6.172   96.1  5.950    5  311.0     15.2  396.90  19.15  27.1
# 8  0.211  12.5   7.87     0  0.524  5.631  100.0  6.082    5  311.0     15.2  386.63  29.93  16.5
# 9  0.170  12.5   7.87     0  0.524  6.004   85.9  6.592    5  311.0     15.2  386.71  17.10  18.9

# 描述性统计信息
print((f'数据集中的描述性统计信息如下：\n{data2.describe()}'))

# 数据集中的描述性统计信息如下：
#           CRIM       ZN    INDUS     CHAS      NOX       RM      AGE      DIS      RAD      TAX  PRTATIO        B  \
# count  506.000  506.000  506.000  506.000  506.000  506.000  506.000  506.000  506.000  506.000  506.000  506.000
# mean     3.614   11.364   11.137    0.069    0.555    6.285   68.575    3.795    9.549  408.237   18.456  356.674
# std      8.602   23.322    6.860    0.254    0.116    0.703   28.149    2.106    8.707  168.537    2.165   91.295
# min      0.006    0.000    0.460    0.000    0.385    3.561    2.900    1.130    1.000  187.000   12.600    0.320
# 25%      0.082    0.000    5.190    0.000    0.449    5.886   45.025    2.100    4.000  279.000   17.400  375.377
# 50%      0.257    0.000    9.690    0.000    0.538    6.208   77.500    3.207    5.000  330.000   19.050  391.440
# 75%      3.677   12.500   18.100    0.000    0.624    6.623   94.075    5.188   24.000  666.000   20.200  396.225
# max     88.976  100.000   27.740    1.000    0.871    8.780  100.000   12.127   24.000  711.000   22.000  396.900
#
#          LSTAT     MEDV
# count  506.000  506.000
# mean    12.653   22.533
# std      7.141    9.197
# min      1.730    5.000
# 25%      6.950   17.025
# 50%     11.360   21.200
# 75%     16.955   25.000
# max     37.970   50.000

# 数据中的关联关系
pd.set_option('precision',2)
print(('数据集中的数据关联关系如下：'))
print(data2.corr(method='pearson'))

# 数据集中的数据关联关系如下：
#          CRIM    ZN  INDUS      CHAS   NOX    RM   AGE   DIS       RAD   TAX  PRTATIO     B  LSTAT  MEDV
# CRIM     1.00 -0.20   0.41 -5.59e-02  0.42 -0.22  0.35 -0.38  6.26e-01  0.58     0.29 -0.39   0.46 -0.39
# ZN      -0.20  1.00  -0.53 -4.27e-02 -0.52  0.31 -0.57  0.66 -3.12e-01 -0.31    -0.39  0.18  -0.41  0.36
# INDUS    0.41 -0.53   1.00  6.29e-02  0.76 -0.39  0.64 -0.71  5.95e-01  0.72     0.38 -0.36   0.60 -0.48
# CHAS    -0.06 -0.04   0.06  1.00e+00  0.09  0.09  0.09 -0.10 -7.37e-03 -0.04    -0.12  0.05  -0.05  0.18
# NOX      0.42 -0.52   0.76  9.12e-02  1.00 -0.30  0.73 -0.77  6.11e-01  0.67     0.19 -0.38   0.59 -0.43
# RM      -0.22  0.31  -0.39  9.13e-02 -0.30  1.00 -0.24  0.21 -2.10e-01 -0.29    -0.36  0.13  -0.61  0.70
# AGE      0.35 -0.57   0.64  8.65e-02  0.73 -0.24  1.00 -0.75  4.56e-01  0.51     0.26 -0.27   0.60 -0.38
# DIS     -0.38  0.66  -0.71 -9.92e-02 -0.77  0.21 -0.75  1.00 -4.95e-01 -0.53    -0.23  0.29  -0.50  0.25
# RAD      0.63 -0.31   0.60 -7.37e-03  0.61 -0.21  0.46 -0.49  1.00e+00  0.91     0.46 -0.44   0.49 -0.38
# TAX      0.58 -0.31   0.72 -3.56e-02  0.67 -0.29  0.51 -0.53  9.10e-01  1.00     0.46 -0.44   0.54 -0.47
# PRTATIO  0.29 -0.39   0.38 -1.22e-01  0.19 -0.36  0.26 -0.23  4.65e-01  0.46     1.00 -0.18   0.37 -0.51
# B       -0.39  0.18  -0.36  4.88e-02 -0.38  0.13 -0.27  0.29 -4.44e-01 -0.44    -0.18  1.00  -0.37  0.33
# LSTAT    0.46 -0.41   0.60 -5.39e-02  0.59 -0.61  0.60 -0.50  4.89e-01  0.54     0.37 -0.37   1.00 -0.74
# MEDV    -0.39  0.36  -0.48  1.75e-01 -0.43  0.70 -0.38  0.25 -3.82e-01 -0.47    -0.51  0.33  -0.74  1.00

# 数据可视化
# 单变量图
# 直方图
data2.hist(sharex=False,sharey=False,xlabelsize=6,ylabelsize=6)
plt.tight_layout()
pic_name = '直方图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('直方图 保存成功！！')

# 密度图
data2.plot(kind='density',subplots=True,layout=(4,4),sharex=False,fontsize=6)
plt.tight_layout()
pic_name = '密度图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('密度图 保存成功！！')

# 箱线图
data2.plot(kind='box',subplots=True,layout=(4,4),sharex=False,fontsize=6)
plt.tight_layout()
pic_name = '箱线图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('箱线图 保存成功！！')

# 多变量图
# 散点矩阵图
pd.plotting.scatter_matrix(data2)
# plt.tight_layout()
pic_name = '散点矩阵图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('散点矩阵图 保存成功！！')

# 相关矩阵图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data2.corr(),vmin=-1,vmax=1,interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_names2)
ax.set_yticklabels(col_names2)
plt.tight_layout()
pic_name = '相关矩阵图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('相关矩阵图 保存成功！！')

# 分离测试数据集
array2 = data2.values
X = array2[:,0:13]
y = array2[:,13]
test_size = 0.2
seed = 7
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)

# 评估算法
# 评估算法 —— 原始数据
# 评估算法 —— 评估标准
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# 评估算法 —— baseline
models = {}
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['KNN'] = KNeighborsRegressor()
models['CART'] = DecisionTreeRegressor()
models['SVM'] = SVR()
models['EN'] = ElasticNet()

# 评估算法 —— 指标（MES越接近0越好）
results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    result = cross_val_score(models[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(result)
    print('{} 的10折交叉验证的MSE评价指标是： {:.2f}，标准差是： {:.2f}'.format(key,result.mean(), result.std()))

# LR 的10折交叉验证的MSE评价指标是： -21.38，标准差是： 9.41
# LASSO 的10折交叉验证的MSE评价指标是： -26.42，标准差是： 11.65
# KNN 的10折交叉验证的MSE评价指标是： -41.90，标准差是： 13.90
# CART 的10折交叉验证的MSE评价指标是： -29.69，标准差是： 13.22
# SVM 的10折交叉验证的MSE评价指标是： -85.52，标准差是： 31.99
# EN 的10折交叉验证的MSE评价指标是： -27.50，标准差是： 12.31

# 评估算法 —— 箱线图
fig = plt.figure()
fig.suptitle('算法比较')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.tight_layout()
pic_name = '算法比较箱线图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('算法比较箱线图 保存成功！！')

# 评估算法 —— 正态化数据
piplines = {}
piplines['ScalerLR'] = Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])
piplines['ScalerLASSO'] = Pipeline([('Scaler',StandardScaler()),('LASSO',Lasso())])
piplines['ScalerKNN'] = Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsRegressor())])
piplines['ScalerCART'] = Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeRegressor())])
piplines['ScalerSVM'] = Pipeline([('Scaler',StandardScaler()),('SVM',SVR())])
piplines['ScalerEN'] = Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])

cv_results = []
for key in piplines:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(piplines[key],X_train,y_train,cv=kfold,scoring=scoring)
    cv_results.append(cv_result)
    print('正态化数据后， {} 的10折交叉验证的MES评价指标是： {:.2f}，标准差是： {:.2f}'.format(key, cv_result.mean(), cv_result.std()))

# 正态化数据后， ScalerLR 的10折交叉验证的MSE评价指标是： -21.38，标准差是： 9.41
# 正态化数据后， ScalerLASSO 的10折交叉验证的MSE评价指标是： -26.61，标准差是： 8.98
# 正态化数据后， ScalerKNN 的10折交叉验证的MSE评价指标是： -20.11，标准差是： 12.38
# 正态化数据后， ScalerCART 的10折交叉验证的MSE评价指标是： -27.67，标准差是： 14.58
# 正态化数据后， ScalerSVM 的10折交叉验证的MSE评价指标是： -29.63，标准差是： 17.01
# 正态化数据后， ScalerEN 的10折交叉验证的MSE评价指标是： -27.93，标准差是： 10.59

# 评估算法 —— 箱线图
fig = plt.figure()
fig.suptitle('算法比较')
ax = fig.add_subplot(111)
plt.boxplot(cv_results)
ax.set_xticklabels(piplines.keys())
plt.tight_layout()
pic_name = '正态化数据后算法比较箱线图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('正态化数据后算法比较箱线图 保存成功！！')

# 调参改善算法
# 调参改善算法 —— KNN
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
param_grid = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescalered_X,y=y_train)
print('调整后KNN模型的最优MSE指标是 {:.2f}，使用的近邻个数k = {}'.format(grid_result.best_score_,grid_result.best_params_['n_neighbors']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举KNN算法中取不同k值时的情况如下：')
for mean,std,param in cv_results:
    print('KNN模型的MSE指标是 {:.2f}，标准差是 {:.2f}，使用的近邻个数k = {}'.format(mean,std,param['n_neighbors']))

# 调整后KNN模型的最优MSE指标是 -18.17，使用的近邻个数k = 3
# 穷举KNN算法中取不同k值时的情况如下：
# KNN模型的MSE指标是 -20.21，标准差是 15.03，使用的近邻个数k = 1
# KNN模型的MSE指标是 -18.17，标准差是 12.95，使用的近邻个数k = 3
# KNN模型的MSE指标是 -20.13，标准差是 12.20，使用的近邻个数k = 5
# KNN模型的MSE指标是 -20.58，标准差是 12.35，使用的近邻个数k = 7
# KNN模型的MSE指标是 -20.37，标准差是 11.62，使用的近邻个数k = 9
# KNN模型的MSE指标是 -21.01，标准差是 11.61，使用的近邻个数k = 11
# KNN模型的MSE指标是 -21.15，标准差是 11.94，使用的近邻个数k = 13
# KNN模型的MSE指标是 -21.56，标准差是 11.54，使用的近邻个数k = 15
# KNN模型的MSE指标是 -22.79，标准差是 11.57，使用的近邻个数k = 17
# KNN模型的MSE指标是 -23.87，标准差是 11.34，使用的近邻个数k = 19
# KNN模型的MSE指标是 -24.36，标准差是 11.91，使用的近邻个数k = 21

# 集成算法
ensembles = {}
ensembles['ScalerAB'] = Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])
ensembles['ScalerAB-KNN'] = Pipeline([('Scaler',StandardScaler()),('ABKNN',AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)))])
ensembles['ScalerAB-LR'] = Pipeline([('Scaler',StandardScaler()),('ABLR',AdaBoostRegressor(base_estimator=LinearRegression()))])
ensembles['ScalerRFR'] = Pipeline([('Scaler',StandardScaler()),('RFR',RandomForestRegressor())])
ensembles['ScalerETR'] = Pipeline([('Scaler',StandardScaler()),('ETR',ExtraTreesRegressor())])
ensembles['ScalerGBR'] = Pipeline([('Scaler',StandardScaler()),('GBR',GradientBoostingRegressor())])

cv_results = []
for key in ensembles:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(ensembles[key],X_train,y_train,cv=kfold,scoring=scoring)
    cv_results.append(cv_result)
    print('正态化数据后，集成算法 {} 的10折交叉验证的MES评价指标是： {:.2f}，标准差是： {:.2f}'.format(key, cv_result.mean(), cv_result.std()))

# 正态化数据后，集成算法 ScalerAB 的10折交叉验证的MES评价指标是： -15.18，标准差是： 6.21
# 正态化数据后，集成算法 ScalerAB-KNN 的10折交叉验证的MES评价指标是： -16.34，标准差是： 10.17
# 正态化数据后，集成算法 ScalerAB-LR 的10折交叉验证的MES评价指标是： -23.85，标准差是： 9.12
# 正态化数据后，集成算法 ScalerRFR 的10折交叉验证的MES评价指标是： -13.05，标准差是： 6.61
# 正态化数据后，集成算法 ScalerETR 的10折交叉验证的MES评价指标是： -10.81，标准差是： 5.82
# 正态化数据后，集成算法 ScalerGBR 的10折交叉验证的MES评价指标是： -9.93，标准差是： 4.41

# 集成算法 —— 箱线图
fig = plt.figure()
fig.suptitle('集成算法比较')
ax = fig.add_subplot(111)
plt.boxplot(cv_results)
ax.set_xticklabels(ensembles.keys())
plt.tight_layout()
pic_name = '正态化数据后集成算法比较箱线图' + '_regression_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('正态化数据后集成算法比较箱线图 保存成功！！')

# 集成算法调参
# 集成算法GBR —— 调参
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
param_grid = {'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
model = GradientBoostingRegressor()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescalered_X,y=y_train)
print('调整后GBR模型的最优MSE指标是 {:.2f}，使用的估计量estimator = {}'.format(grid_result.best_score_,grid_result.best_params_['n_estimators']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举GBR算法中取不同估计量时的情况如下：')
for mean,std,param in cv_results:
    print('GBR模型的MSE指标是 {:.2f}，标准差是 {:.2f}，使用的估计量estimator = {}'.format(mean,std,param['n_estimators']))

# 调整后GBR模型的最优MSE指标是 -9.09，使用的估计量estimator = 600
# 穷举GBR算法中取不同估计量时的情况如下：
# GBR模型的MSE指标是 -25.28，标准差是 8.55，使用的估计量estimator = 10
# GBR模型的MSE指标是 -11.07，标准差是 4.98，使用的估计量estimator = 50
# GBR模型的MSE指标是 -10.29，标准差是 4.65，使用的估计量estimator = 100
# GBR模型的MSE指标是 -9.92，标准差是 4.44，使用的估计量estimator = 200
# GBR模型的MSE指标是 -9.63，标准差是 4.47，使用的估计量estimator = 300
# GBR模型的MSE指标是 -9.44，标准差是 4.43，使用的估计量estimator = 400
# GBR模型的MSE指标是 -9.50，标准差是 4.40，使用的估计量estimator = 500
# GBR模型的MSE指标是 -9.09，标准差是 4.20，使用的估计量estimator = 600
# GBR模型的MSE指标是 -9.52，标准差是 4.38，使用的估计量estimator = 700
# GBR模型的MSE指标是 -9.68，标准差是 4.48，使用的估计量estimator= 800
# GBR模型的MSE指标是 -9.55，标准差是 4.35，使用的估计量estimator = 900

# 集成算法ETR —— 调参
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
param_grid = {'n_estimators':[5,10,20,30,40,50,60,70,80,90,100]}
model = ExtraTreesRegressor()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescalered_X,y=y_train)
print('调整后ETR模型的最优MSE指标是 {:.2f}，使用的估计量estimator = {}'.format(grid_result.best_score_,grid_result.best_params_['n_estimators']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举ETR算法中取不同估计量时的情况如下：')
for mean,std,param in cv_results:
    print('ETR模型的MSE指标是 {:.2f}，标准差是 {:.2f}，使用的估计量estimator = {}'.format(mean,std,param['n_estimators']))

# 调整后ETR模型的最优MSE指标是 -9.09，使用的估计量estimator = 100
# 穷举ETR算法中取不同估计量时的情况如下：
# ETR模型的MSE指标是 -12.68，标准差是 6.13，使用的估计量estimator = 5
# ETR模型的MSE指标是 -11.02，标准差是 6.42，使用的估计量estimator = 10
# ETR模型的MSE指标是 -10.01，标准差是 6.43，使用的估计量estimator = 20
# ETR模型的MSE指标是 -9.94，标准差是 6.05，使用的估计量estimator = 30
# ETR模型的MSE指标是 -9.53，标准差是 5.44，使用的估计量estimator = 40
# ETR模型的MSE指标是 -9.29，标准差是 5.36，使用的估计量estimator = 50
# ETR模型的MSE指标是 -9.26，标准差是 5.40，使用的估计量estimator = 60
# ETR模型的MSE指标是 -9.28，标准差是 5.44，使用的估计量estimator = 70
# ETR模型的MSE指标是 -9.30，标准差是 5.50，使用的估计量estimator = 80
# ETR模型的MSE指标是 -9.21，标准差是 5.37，使用的估计量estimator = 90
# ETR模型的MSE指标是 -9.09，标准差是 5.64，使用的估计量estimator = 100

# 确定最终模型
# 训练模型
scaler = StandardScaler().fit(X_train)
rescalered_X = scaler.transform(X_train)
gbr = GradientBoostingRegressor(n_estimators=600)
gbr.fit(X=rescalered_X,y=y_train)

# 评估算法模型
rescalered_X_test = scaler.transform(X_test)
predictions = gbr.predict(rescalered_X_test)
result = mean_squared_error(y_test,predictions)
print('最终模型的预测MSE指标为 {:.2f}'.format(result))

# 最终模型的预测MSE指标为 12.39