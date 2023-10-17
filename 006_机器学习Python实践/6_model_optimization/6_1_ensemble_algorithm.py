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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

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

# 集成算法
# 装袋算法
# 装袋决策树
cart = DecisionTreeClassifier()
num_tree = 100
model = BaggingClassifier(base_estimator=cart,n_estimators=num_tree,random_state=seed)
result = cross_val_score(model,X,y,cv=kfold)
print('装袋决策树的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# 装袋决策树的10折交叉验证的准确率评价指标是： 77.07%，标准差是： 0.07

# 随机森林
num_tree = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_tree,random_state=seed,max_features=max_features)
result = cross_val_score(model,X,y,cv=kfold)
print('随机森林的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# 随机森林的10折交叉验证的准确率评价指标是： 77.34%，标准差是： 0.07

# 极端随机树
num_tree = 100
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_tree,random_state=seed,max_features=max_features)
result = cross_val_score(model,X,y,cv=kfold)
print('极端随机树的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# 极端随机树的10折交叉验证的准确率评价指标是： 76.30%，标准差是： 0.07

# 提升算法
# AdaBoost
num_tree = 30
model = AdaBoostClassifier(n_estimators=num_tree,random_state=seed)
result = cross_val_score(model,X,y,cv=kfold)
print('AdaBoost的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# AdaBoost的10折交叉验证的准确率评价指标是： 76.05%，标准差是： 0.05

# 随机梯度提升（GBM）
num_tree = 100
model = GradientBoostingClassifier(n_estimators=num_tree,random_state=seed)
result = cross_val_score(model,X,y,cv=kfold)
print('GBM的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# GBM的10折交叉验证的准确率评价指标是： 76.69%，标准差是： 0.06

# 投票算法
cart = DecisionTreeClassifier()
models = []
model_logistic = LogisticRegression(solver='liblinear')
models.append(('logistic',model_logistic))
model_cart = DecisionTreeClassifier()
models.append(('cart',model_cart))
model_svc = SVC()
models.append(('svm',model_svc))
ensemble_model = VotingClassifier(estimators=models)
result = cross_val_score(ensemble_model,X,y,cv=kfold)
print('投票算法的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(result.mean() * 100, result.std()))

# 投票算法的10折交叉验证的准确率评价指标是： 73.69%，标准差是： 0.07