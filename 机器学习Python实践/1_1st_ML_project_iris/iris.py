import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
file_name = os.path.join('./','iris.data.csv')
column_name = ['separ_length','separ_width','petal_length','petal_width','class']
data = pd.read_csv(file_name,names=column_name)

# 概述数据
# 查看数据的维度
print(f'数据维度：{data.shape[0]}行，{data.shape[1]}列')

# 查看数据自身
print(f'查看数据的前10行：\n{data.head(10)}')

# 统计描述信息
print(f'描述性统计信息：\n{data.describe()}')

# 数据分类分布
print(f'数据分类分布的情况：')
print(data.groupby('class').size())

# 数据可视化
# 单变量图
# 箱线图
data.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.tight_layout()
pic_name = '箱线图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('箱线图 保存成功！！')

# 直方图
data.hist()
plt.tight_layout()
pic_name = '直方图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('直方图 保存成功！！')

# 多变量图
# 散点矩阵图
pd.plotting.scatter_matrix(data)
plt.tight_layout()
pic_name = '散点矩阵图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('散点矩阵图 保存成功！！')

# 评估算法
# 分离出评估数据集
array = data.values
X = array[:,0:4]
Y = array[:,-1]
validation_size = 0.2
seed = 7
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)

# 创建模型
# 算法审查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

# 评估算法
results = []
print('算法名称：   10折交叉验证后的均值   10折交叉验证后的标准差')
for key in models:
    kfold = KFold(n_splits=10,random_state=seed)
    cv_results = cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    print(f'{key}:   {cv_results.mean()}   ({cv_results.std()})')

# 算法名称：   10折交叉验证后的均值   10折交叉验证后的标准差
# LR:   0.9666666666666666   (0.04082482904638632)
# LDA:   0.975   (0.03818813079129868)
# KNN:   0.9833333333333332   (0.03333333333333335)
# CART:   0.9833333333333332   (0.03333333333333335)
# NB:   0.975   (0.053359368645273735)
# SVM:   0.9916666666666666   (0.025000000000000012)

# 选择最优模型
# 箱线图比较算法
fig =plt.figure()
fig.suptitle('箱线图比较算法')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.tight_layout()
pic_name = '箱线图比较算法' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('箱线图比较算法 保存成功！！')

# 实施预测
# SVM准确度最高
# 使用评估数据集评估算法
svm = SVC()
svm.fit(X=X_train,y=Y_train)
predictions = svm.predict(X_validation)
print(f'模型的准确率是：\n{accuracy_score(Y_validation,predictions)}')
print(f'模型的混淆矩阵如下：\n{confusion_matrix(Y_validation,predictions)}')
print(f'模型的指标如下：\n{classification_report(Y_validation,predictions)}')

# 模型的准确率是：
# 0.9333333333333333
# 模型的混淆矩阵如下：
# [[ 7  0  0]
#  [ 0 10  2]
#  [ 0  0 11]]
# 模型的指标如下：
#                  precision    recall  f1-score   support
#
#     Iris-setosa       1.00      1.00      1.00         7
# Iris-versicolor       1.00      0.83      0.91        12
#  Iris-virginica       0.85      1.00      0.92        11
#
#       micro avg       0.93      0.93      0.93        30
#       macro avg       0.95      0.94      0.94        30
#    weighted avg       0.94      0.93      0.93        30