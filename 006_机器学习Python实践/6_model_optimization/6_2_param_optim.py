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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import uniform

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

# 算法调参
# 网格搜索优化参数
model = Ridge()

param_grid = {'alpha':[1,0.1,0.01,0.001,0]}

grid = GridSearchCV(estimator=model,param_grid=param_grid)
grid.fit(X,y)

print('网格搜索的最高得分是： {:.2f}%'.format(grid.best_score_ * 100))
print('网格搜索的最优参数是： {}'.format(grid.best_estimator_.alpha))

# 网格搜索的最高得分是： 27.96%
# 网格搜索的最优参数是： 1

# 随机搜索优化参数
model = Ridge()

param_grid = {'alpha':uniform()}

grid = RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=100,random_state=7)
grid.fit(X,y)

print('随机搜索的最高得分是： {:.2f}%'.format(grid.best_score_ * 100))
print('随机搜索的最优参数是： {:.2f}'.format(grid.best_estimator_.alpha))

# 随机搜索的最高得分是： 27.96%
# 随机搜索的最优参数是： 0.98