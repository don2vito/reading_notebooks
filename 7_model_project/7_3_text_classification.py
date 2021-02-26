import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
import warnings
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.naive_bayes import MultinomialNB
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

# 导入数据集的文件夹目录
categories = os.listdir('./20news-bydate-train')
print(f'数据集的文件夹目录列表如下：\n{np.array(categories).reshape(-1,1)}')

# 数据集的文件夹目录列表如下：
# [['alt.atheism']
#  ['comp.graphics']
#  ['comp.os.ms-windows.misc']
#  ['comp.sys.ibm.pc.hardware']
#  ['comp.sys.mac.hardware']
#  ['comp.windows.x']
#  ['misc.forsale']
#  ['rec.autos']
#  ['rec.motorcycles']
#  ['rec.sport.baseball']
#  ['rec.sport.hockey']
#  ['sci.crypt']
#  ['sci.electronics']
#  ['sci.med']
#  ['sci.space']
#  ['soc.religion.christian']
#  ['talk.politics.guns']
#  ['talk.politics.mideast']
#  ['talk.politics.misc']
#  ['talk.religion.misc']]

# 导入训练数据集
train_path = os.path.join('./','20news-bydate-train')
data_train = load_files(container_path=train_path,categories=categories)
print('导入训练数据集 成功！！')

# 导入测试数据集
test_path = os.path.join('./','20news-bydate-test')
data_test = load_files(container_path=test_path,categories=categories)
print('导入测试数据集 成功！！')

# 文本特征提取
# 数据准备与理解
# 计算词频
count_vector = CountVectorizer(stop_words='english',decode_error='ignore')
X_train_counts = count_vector.fit_transform(data_train.data)

# 查看词频的数据维度
print(f'词频的数据维度是 {X_train_counts.shape[0]}行，{X_train_counts.shape[1]}列')

# 词频的数据维度是 11314行，129782列

# 计算 TF-IDF
tf_transformer = TfidfVectorizer(stop_words='english',decode_error='ignore')
X_train_counts_tf = tf_transformer.fit_transform(data_train.data)

# 查看 TF-IDF 的数据维度
print(f'TF-IDF 的数据维度是 {X_train_counts_tf.shape[0]}行，{X_train_counts_tf.shape[1]}列')

# TF-IDF 的数据维度 11314行，129782列

# 评估算法
# 设置评估算法的基准
num_folds = 10
seed = 7
scoring = 'accuracy'

# 生成算法模型
models = {}
models['LR'] = LogisticRegression()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['SVM'] = SVC()
models['MNB'] = MultinomialNB()

# 比较算法
results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    result = cross_val_score(models[key],X_train_counts_tf,data_train.target,cv=kfold,scoring=scoring)
    results.append(result)
    print('{} 的10折交叉验证的准确率指标是： {:.2f}%，标准差是： {:.2f}'.format(key,result.mean() * 100, result.std()))

# LR 的10折交叉验证的准确率指标是： 90.01%，标准差是： 0.01
# KNN 的10折交叉验证的准确率指标是： 79.87%，标准差是： 0.01
# CART 的10折交叉验证的准确率指标是： 65.96%，标准差是： 0.01
# SVM 的10折交叉验证的准确率指标是： 5.06%，标准差是： 0.01
# MNB 的10折交叉验证的准确率指标是： 88.44%，标准差是： 0.01

# 评估算法 —— 箱线图
fig = plt.figure()
fig.suptitle('算法比较')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.tight_layout()
pic_name = '算法比较箱线图' + '_text_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('算法比较箱线图 保存成功！！')

# 算法调参
# LR（逻辑回归）调参
param_grid = {'C':[0.1,5,13,15]}
model = LogisticRegression()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=X_train_counts_tf,y=data_train.target)
print('逻辑回归模型的最优准确率是 {:.2f}%，使用的目标约束函数C = {}'.format(grid_result.best_score_ * 100,grid_result.best_params_['C']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举逻辑回归算法中取不同C值时的情况如下：')
for mean,std,param in cv_results:
    print('逻辑回归模型的准确率是 {:.2f}%，标准差是 {:.2f}，使用的目标约束函数C = {}'.format(mean * 100,std,param['C']))

# 逻辑回归模型的最优准确率是 92.30%，使用的目标约束函数C = 15
# 穷举逻辑回归算法中取不同C值时的情况如下：
# 逻辑回归模型的准确率是 83.75%，标准差是 0.01，使用的目标约束函数C = 0.1
# 逻辑回归模型的准确率是 91.97%，标准差是 0.01，使用的目标约束函数C = 5
# 逻辑回归模型的准确率是 92.26%，标准差是 0.01，使用的目标约束函数C = 13
# 逻辑回归模型的准确率是 92.30%，标准差是 0.01，使用的目标约束函数C = 15

# MNB（贝叶斯）调参
param_grid = {'alpha':[0.001,0.01,0.1,1.5]}
model = MultinomialNB()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=X_train_counts_tf,y=data_train.target)
print('贝叶斯模型的最优准确率是 {:.2f}%，使用的平滑参数α = {:.3f}'.format(grid_result.best_score_ * 100,grid_result.best_params_['alpha']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举贝叶斯算法中取不同α值时的情况如下：')
for mean,std,param in cv_results:
    print('贝叶斯模型的准确率是 {:.2f}%，标准差是 {:.2f}，使用的平滑参数α = {:.3f}'.format(mean * 100,std,param['alpha']))

# 贝叶斯模型的最优准确率是 91.67%，使用的平滑参数α = 0.010
# 穷举贝叶斯算法中取不同α值时的情况如下：
# 贝叶斯模型的准确率是 91.31%，标准差是 0.01，使用的平滑参数α = 0.001
# 贝叶斯模型的准确率是 91.67%，标准差是 0.01，使用的平滑参数α = 0.010
# 贝叶斯模型的准确率是 91.01%，标准差是 0.00，使用的平滑参数α = 0.100
# 贝叶斯模型的准确率是 87.63%，标准差是 0.01，使用的平滑参数α = 1.500

# 集成算法
ensembles = {}
ensembles['ScalerAB'] = AdaBoostClassifier()
ensembles['ScalerRF'] = RandomForestClassifier()

# 比较集成算法
cv_results = []
for key in ensembles:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(ensembles[key],X_train_counts_tf,data_train.target,cv=kfold,scoring=scoring)
    cv_results.append(cv_result)
    print('集成算法 {} 的10折交叉验证的准确率是： {:.2f}%，标准差是： {:.2f}'.format(key, cv_result.mean() * 100, cv_result.std()))

# 集成算法 ScalerAB 的10折交叉验证的准确率是： 54.44%，标准差是： 0.02
# 集成算法 ScalerRF 的10折交叉验证的准确率是： 74.29%，标准差是： 0.01

# 集成算法 —— 箱线图
fig = plt.figure()
fig.suptitle('集成算法比较')
ax = fig.add_subplot(111)
plt.boxplot(cv_results)
ax.set_xticklabels(ensembles.keys())
plt.tight_layout()
pic_name = '集成算法比较箱线图' + '_text_classification_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('集成算法比较箱线图 保存成功！！')

# 集成算法调参
# RF 调参
param_grid = {'n_estimators':[10,100,150,200]}
model = RandomForestClassifier()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=X_train_counts_tf,y=data_train.target)
print('随机森林算法模型的最优准确率是 {:.2f}%，使用的估计量estimator = {}'.format(grid_result.best_score_ * 100,grid_result.best_params_['n_estimators']))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
print('穷举随机森林算法中取不同估计量时的情况如下：')
for mean,std,param in cv_results:
    print('随机森林算法模型的准确率是 {:.2f}%，标准差是 {:.2f}，使用的估计量estimator = {}'.format(mean * 100,std,param['n_estimators']))

# 随机森林算法模型的最优准确率是 87.08%，使用的估计量estimator = 200
# 穷举随机森林算法中取不同估计量时的情况如下：
# 随机森林算法模型的准确率是 74.54%，标准差是 0.02，使用的估计量estimator = 10
# 随机森林算法模型的准确率是 86.28%，标准差是 0.01，使用的估计量estimator = 100
# 随机森林算法模型的准确率是 86.59%，标准差是 0.01，使用的估计量estimator = 150
# 随机森林算法模型的准确率是 87.08%，标准差是 0.01，使用的估计量estimator = 200

# 确定最终模型
# 训练模型
model = LogisticRegression(C=13)
model.fit(X_train_counts_tf,data_train.target)

# 评估算法模型
X_test_counts = tf_transformer.transform(data_test.data)
predictions = model.predict(X_test_counts)
accuracy = accuracy_score(data_test.target,predictions)
confusion = confusion_matrix(data_test.target,predictions)
classification = classification_report(data_test.target,predictions)
print('最终模型的预测准确率为 {:.2f}%'.format(accuracy * 100))
print('最终模型的预测混淆矩阵如下：\n{}'.format(confusion))
print('最终模型的预测分类指标如下：\n{}'.format(classification))

# 最终模型的预测准确率为 84.77%
# 最终模型的预测混淆矩阵如下：
# [[245   1   0   1   0   2   1   0   1   1   0   1   1   7   7  24   0   4
#     1  22]
#  [  1 313  10   7   9  15   4   2   0   7   1   3  10   0   3   1   0   1
#     0   2]
#  [  0  20 294  41   8   9   2   2   1   4   0   1   2   2   5   0   0   0
#     1   2]
#  [  0  14  19 295  18   3  15   2   1   1   0   0  21   0   2   0   0   0
#     0   1]
#  [  0   4   4  14 328   2  12   1   0   3   1   0  12   0   1   0   2   0
#     1   0]
#  [  1  36  36   6   4 304   2   0   1   0   0   0   1   1   3   0   0   0
#     0   0]
#  [  0   3   1  10  10   0 350   5   1   1   1   0   7   1   0   0   0   0
#     0   0]
#  [  0   2   1   6   1   1  11 356   3   2   0   0   7   1   1   0   2   0
#     2   0]
#  [  0   0   0   1   0   0   5  10 379   1   0   0   2   0   0   0   0   0
#     0   0]
#  [  0   0   1   0   1   1   4   1   1 374  13   0   0   0   0   0   1   0
#     0   0]
#  [  0   1   0   0   4   1   1   0   0   4 387   0   0   0   0   0   1   0
#     0   0]
#  [  1   3   1   0   2   4   5   2   1   1   0 367   2   1   0   1   3   0
#     2   0]
#  [  1   6   6  27   9   1   6   5   2   3   0   5 311   5   5   1   0   0
#     0   0]
#  [  4   7   1   1   2   3   6   0   1   5   1   1  13 345   0   2   1   1
#     2   0]
#  [  1   9   0   0   3   3   1   1   0   1   0   0   4   5 364   0   0   0
#     2   0]
#  [  4   0   2   1   0   0   0   0   0   2   0   0   3   1   3 371   0   0
#     0  11]
#  [  0   1   1   2   1   0   2   1   0   1   0   2   0   5   1   0 333   1
#     9   4]
#  [  8   2   0   1   0   8   0   1   1   3   1   0   0   2   2   5   1 331
#    10   0]
#  [  2   1   0   0   2   1   1   2   0   0   0   3   1   5   6   4  86   0
#   189   7]
#  [ 34   1   1   1   0   0   3   0   0   1   0   0   0   3   3  32  12   2
#     9 149]]
# 最终模型的预测分类指标如下：
#               precision    recall  f1-score   support
#
#            0       0.81      0.77      0.79       319
#            1       0.74      0.80      0.77       389
#            2       0.78      0.75      0.76       394
#            3       0.71      0.75      0.73       392
#            4       0.82      0.85      0.83       385
#            5       0.85      0.77      0.81       395
#            6       0.81      0.90      0.85       390
#            7       0.91      0.90      0.90       396
#            8       0.96      0.95      0.96       398
#            9       0.90      0.94      0.92       397
#           10       0.96      0.97      0.96       399
#           11       0.96      0.93      0.94       396
#           12       0.78      0.79      0.79       393
#           13       0.90      0.87      0.88       396
#           14       0.90      0.92      0.91       394
#           15       0.84      0.93      0.88       398
#           16       0.75      0.91      0.83       364
#           17       0.97      0.88      0.92       376
#           18       0.83      0.61      0.70       310
#           19       0.75      0.59      0.66       251
#
#    micro avg       0.85      0.85      0.85      7532
#    macro avg       0.85      0.84      0.84      7532
# weighted avg       0.85      0.85      0.85      7532