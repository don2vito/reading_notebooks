import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

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

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)

# 数据准备和生成模型的Pipeline
steps = []
steps.append(('Standardize',StandardScaler()))
steps.append(('lda',LinearDiscriminantAnalysis()))

model = Pipeline(steps)
result = cross_val_score(model,X,y,cv=kfold)
print('Pipeline的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# Pipeline的10折交叉验证的准确率评价指标是： 77.35%，标准差是： 0.05

# 特征选择和生成模型的Pipeline
features = []
features.append(('pca',PCA()))
features.append(('select_best',SelectKBest(k=6)))

steps = []
steps.append(('feature_union',FeatureUnion(features)))
steps.append(('logistic',LogisticRegression()))

model = Pipeline(steps)
result = cross_val_score(model,X,y,cv=kfold)
print('Pipeline的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# Pipeline的10折交叉验证的准确率评价指标是： 78.00%，标准差是： 0.05