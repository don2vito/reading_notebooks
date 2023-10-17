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

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)

models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['SVM'] = SVC()
models['NB'] = GaussianNB()

results = []
for key in models:
    result = cross_val_score(models[key],X,y,cv=kfold)
    results.append(result)
    print('{} 的10折交叉验证的准确率评价指标是： {:.2f}%，标准差是： {:.2f}'.format(key,result.mean() * 100, result.std()))

# LR 的10折交叉验证的准确率评价指标是： 76.95%，标准差是： 0.05
# LDA 的10折交叉验证的准确率评价指标是： 77.35%，标准差是： 0.05
# KNN 的10折交叉验证的准确率评价指标是： 72.66%，标准差是： 0.06
# CART 的10折交叉验证的准确率评价指标是： 70.44%，标准差是： 0.06
# SVM 的10折交叉验证的准确率评价指标是： 65.10%，标准差是： 0.07
# NB 的10折交叉验证的准确率评价指标是： 75.52%，标准差是： 0.04

# 可视化
fig = plt.figure()
fig.suptitle('算法比较')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.tight_layout()
pic_name = '算法比较箱线图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('算法比较箱线图 保存成功！！')