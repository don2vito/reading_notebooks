import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',100)
pd.set_option('precision',3)

# 使用Pandas导入数据
file_name = os.path.join('./','pima_data.csv')
col_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(file_name,names=col_names)

# 数据预处理
# 调整数据尺度
array = data.values
X = array[:,0:8]
y = array[:,8]
transformer = MinMaxScaler(feature_range=(0,1))
new_X = transformer.fit_transform(X)
np.set_printoptions(precision=3)
print(f'调整后的数据尺度如下：\n{new_X}')

# 调整后的数据尺度为：
# [[0.353 0.744 0.59  ... 0.501 0.234 0.483]
#  [0.059 0.427 0.541 ... 0.396 0.117 0.167]
#  [0.471 0.92  0.525 ... 0.347 0.254 0.183]
#  ...
#  [0.294 0.608 0.59  ... 0.39  0.071 0.15 ]
#  [0.059 0.633 0.492 ... 0.449 0.116 0.433]
#  [0.059 0.467 0.574 ... 0.453 0.101 0.033]]

# 正态化数据
transformer = StandardScaler().fit(X)
new_X = transformer.transform(X)
np.set_printoptions(precision=3)
print(f'调整后的正态化数据如下：\n{new_X}')

# 调整后的正态化数据如下：
# [[ 0.64   0.848  0.15  ...  0.204  0.468  1.426]
#  [-0.845 -1.123 -0.161 ... -0.684 -0.365 -0.191]
#  [ 1.234  1.944 -0.264 ... -1.103  0.604 -0.106]
#  ...
#  [ 0.343  0.003  0.15  ... -0.735 -0.685 -0.276]
#  [-0.845  0.16  -0.471 ... -0.24  -0.371  1.171]
#  [-0.845 -0.873  0.046 ... -0.202 -0.474 -0.871]]

# 标准化数据
transformer = Normalizer().fit(X)
new_X = transformer.transform(X)
np.set_printoptions(precision=3)
print(f'调整后的标准化数据如下：\n{new_X}')

# 调整后的标准化数据如下：
# [[0.034 0.828 0.403 ... 0.188 0.004 0.28 ]
#  [0.008 0.716 0.556 ... 0.224 0.003 0.261]
#  [0.04  0.924 0.323 ... 0.118 0.003 0.162]
#  ...
#  [0.027 0.651 0.388 ... 0.141 0.001 0.161]
#  [0.007 0.838 0.399 ... 0.2   0.002 0.313]
#  [0.008 0.736 0.554 ... 0.241 0.002 0.182]]

# 二值化数据
transformer = Binarizer(threshold=0.0).fit(X)
new_X = transformer.transform(X)
np.set_printoptions(precision=3)
print(f'调整后的二值化数据如下：\n{new_X}')

# 调整后的二值化数据如下：
# [[1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]
#  ...
#  [1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]
#  [1. 1. 1. ... 1. 1. 1.]]

# 数据特征选定
# 单变量特征选定
test = SelectKBest(score_func=chi2,k=4)
fit = test.fit(X,y)
np.set_printoptions(precision=3)
print(f'卡方检验对每个特征的评分为：\n{fit.scores_}')
features = fit.transform(X)
print(f'得分最高的4个数据特征是：\n{features}')

# 卡方检验对每个特征的评分为：
# [ 111.52  1411.887   17.605   53.108 2175.565  127.669    5.393  181.304]
# 得分最高的4个数据特征是：
# [[148.    0.   33.6  50. ]
#  [ 85.    0.   26.6  31. ]
#  [183.    0.   23.3  32. ]
#  ...
#  [121.  112.   26.2  30. ]
#  [126.    0.   30.1  47. ]
#  [ 93.    0.   30.4  23. ]]

# 递归特征消除
model = LogisticRegression(solver='liblinear')
rfe = RFE(model,3)
fit = rfe.fit(X,y)
print(f'特征个数： {fit.n_features_}个')
print(f'被选定的特征：\n{fit.support_}')
print(f'特征排名：\n{fit.ranking_}')

# 特征个数： 3个
# 被选定的特征：
# [ True False False False False  True  True False]
# 特征排名：
# [1 2 3 5 6 1 1 4]

# 主成分分析（PCA）
pca = PCA(n_components=3)
fit = pca.fit(X)
print(f'所保留的3个成分各自的方差百分比如下：\n{fit.explained_variance_ratio_}')
print(f'具有最大方差的成分如下：')
print(fit.components_)

# 所保留的3个成分各自的方差百分比如下：
# [0.889 0.062 0.026]
# 具有最大方差的成分如下：
# [[-2.022e-03  9.781e-02  1.609e-02  6.076e-02  9.931e-01  1.401e-02
#    5.372e-04 -3.565e-03]
#  [-2.265e-02 -9.722e-01 -1.419e-01  5.786e-02  9.463e-02 -4.697e-02
#   -8.168e-04 -1.402e-01]
#  [-2.246e-02  1.434e-01 -9.225e-01 -3.070e-01  2.098e-02 -1.324e-01
#   -6.400e-04 -1.255e-01]]

# 特征重要性
model = ExtraTreesClassifier(n_estimators=10)
fit = model.fit(X,y)
print(f'每个特征数据的得分如下：\n{fit.feature_importances_}')

# 每个特征数据的得分如下：
# [0.12  0.22  0.098 0.077 0.075 0.139 0.123 0.148]