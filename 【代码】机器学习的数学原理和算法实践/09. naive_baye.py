# 导入天气情况的数据集
import numpy as np
x = np.array([[0,1,0,1],[1,1,1,1],[1,1,1,0],[0,1,1,0],[0,1,0,0],[0,1,0,1],[1,1,0,1],[1,0,0,1],[1,1,0,1],[0,0,0,0]])
y = np.array([1,1,1,1,0,1,0,1,1,0])

# 导入伯努利贝叶斯分类器并训练数据
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(x,y)
day_pre = [[0,0,1,0]]
pre = bnb.predict(day_pre)
print('预测结果如下所示：')
print('*'*50)
print('结果为：',pre)
print('*'*50)


# 当天气情况为 day_pre=[[0,0,1,0]] 时，进一步查看是否下雨的概率分布情况
pre_pro = bnb.predict_proba(day_pre)
print('预测概率情况如下所示：')
print('*'*50)
print('结果为：',pre_pro)
print('*'*50)



# 导入数据集生成工具
from sklearn.datasets import make_blobs

# 生成样本数量为 800、分类数量为 6 的数据集 
x,y = make_blobs(n_samples=800,centers=6,random_state=6)

# 导入数据拆分工具，将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)

# 导入高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
print('*'*50)
print('高斯朴素贝叶斯准确率：',gnb.score(x_test,y_test))
print('*'*50)



# 导入数据集生成工具
from sklearn.datasets import make_blobs

# 生成样本数量为 800、分类数量为 6 的数据集
x,y = make_blobs(n_samples=800,centers=6,random_state=6)

# 导入数据拆分工具，将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)

# 数据预处理（标准化）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

# 导入多项式朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train_s,y_train)
print('*'*50)
print('多项式朴素贝叶斯准确率：',mnb.score(x_test_s,y_test))
print('*'*50)



# 从 sklearn.datasets 中在线导入 20 类新闻文本采集器
from sklearn.datasets import fetch_20newsgroups

# 下载全部文本并存储
newsgroups = fetch_20newsgroups(subset='all')

# 将数据集划分为特征变量与目标变量
x = newsgroups.data
y = newsgroups.target

# 查看目标变量名称
print(newsgroups.target_names)

# 查看目标变量名称
print('目标变量名称：\n',newsgroups.target_names)

# 查看特征变量情况
print('特征变量示例：\n',x[0])

# 查看目标变量情况
print('目标变量：\n',y)

# 将数据分割为训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=33,test_size=0.3)

# 从 sklearn.feature_extraction.text 中导入 CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# 采用默认配置对 CountVectorizer 进行初始化
vec = CountVectorizer()

# 将原始训练和测试文本转化为特征向量
x_vec_train = vec.fit_transform(x_train)
x_vec_test = vec.transform(x_test)

# 使用朴素贝叶斯分类器训练数据
from sklearn.naive_bayes import MultinomialNB

# 初始化朴素贝叶斯分类器
mnb = MultinomialNB()

# 训练模型
mnb.fit(x_vec_train,y_train)

# 使用训练好的朴素贝叶斯模型对数据进行预测 
mnb_y_predict = mnb.predict(x_vec_test)

from sklearn.metrics import classification_report
print('Accuracy:',mnb.score(x_vec_test,y_test))
print(classification_report(y_test,mnb_y_predict))