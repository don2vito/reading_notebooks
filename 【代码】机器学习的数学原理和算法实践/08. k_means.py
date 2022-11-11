# 导入 pandas、NumPy 以及 Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 从 sklearn.datasets 中导入手写数字数据集
from sklearn.datasets import load_digits
digits = load_digits()
x = digits.data
y = digits.target

# 将数据集分割为特征数据与目标数据
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.25)
# 从 sklearn.cluster 中导入 KMeans 模块
from sklearn.cluster import KMeans
from sklearn import metrics

# 设置聚类函数，分类数量为输入数据
def julei(n_clusters):    
# 把数据和对应的分类数放入聚类函数中进行聚类    
    cls = KMeans(n_clusters)    
    cls.fit(x_train)    
    # 判断每个测试图像所属聚类中心    
    y_pre=cls.predict(x_test)    
    # 当数据本身带有正确的类别信息时，使用 sklearn 中的 metrics 的 ARI 指标进行性能评估    
    print('聚类数为%d时，ARI指标为：'%(n_clusters),metrics.adjusted_rand_score(y_test,y_pre))
    
    
julei(5)
julei(8)
julei(10)
julei(11)
julei(12)
julei(13)
julei(14)
julei(20)




from sklearn.metrics import silhouette_score
def lunkuo(n):    
# 当评估数据没有类别或者不考虑其本身的类别时，习惯上会使用轮廓系数来度量    
    lunkuo_clusters = list(range(2,n+1))    
    sc_scores = []    
    for i in  lunkuo_clusters:        
        cls=KMeans(i)        
        cls.fit(x_train)                
    # 绘制轮廓系数与不同类簇数量的关系图
        sc_score = silhouette_score(x_train,cls.fit(x_train).labels_,metric='euclidean')        
        sc_scores.append(sc_score)  
    fig = plt.figure(figsize=(8, 8),dpi=800)
    ax = plt.subplot(111) 
    fig.add_axes(ax)
    plt.figure(dpi=800)    
    plt.plot(lunkuo_clusters,sc_scores,"*-")    
    plt.xlabel('Number of Clusters')
    # 聚类数    
    plt.ylabel('Silhouette Coefficient Score')
    # 轮廓系数    
    plt.show()
    
    
lunkuo(20)