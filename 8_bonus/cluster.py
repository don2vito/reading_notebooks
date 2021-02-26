import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 20)
pd.set_option('precision',3)

warnings.filterwarnings('ignore')

# 导入数据集
file_name = os.path.join('./','wine.data')
column_name = ['class','Alcohol','MalicAcid','Ash','AlclinityOfAsh','Magnesium','TotalPhenols','Flavanoids','NonflayanoidPhenols','Proanthocyanins','ColorIntensity','Hue','OD280/OD315','Proline']
data = pd.read_csv(file_name,names=column_name)
data['class'] = data['class'].replace(to_replace=[1,2,3],value=[0,1,2])
array = data.values
X = array[:,1:13]
y = array[:,0]

# 数据降维
pca = PCA(n_components=3)
X_scale = StandardScaler().fit_transform(X)
X_reduce = pca.fit_transform(X_scale)

# 模型训练
model = KMeans(n_clusters=3)
model.fit(X_reduce)
labels = model.labels_
print(f'聚类后的类别标签如下：\n{labels}')

# 聚类后的类别标签如下：
# [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2
#  2 0 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 2 0 0 2 0 0 2 0 2 0 2
#  0 0 0 0 2 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 2 0 0 2 0 0 0 0 0 0 0 0 0 0 0 2
#  0 0 0 0 0 0 0 1 0 0 2 0 0 2 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

# 模型的准确率指标
homogeneity_score = metrics.homogeneity_score(y,labels)
completeness_score = metrics.completeness_score(y,labels)
v_measure_score = metrics.v_measure_score(y,labels)
adjusted_rand_score = metrics.adjusted_rand_score(y,labels)
adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(y,labels)
silhouette_score = metrics.silhouette_score(X_reduce,labels)
print('模型的准确度指标如下：')
print('同质性得分为 {:.2f}%'.format(homogeneity_score * 100))
print('完整性得分为 {:.2f}%'.format(completeness_score * 100))
print('v测度得分为 {:.2f}%'.format(v_measure_score * 100))
print('调整后的随机得分为 {:.2f}%'.format(adjusted_rand_score * 100))
print('调整后的相互信息得分为 {:.2f}%'.format(adjusted_mutual_info_score * 100))
print('轮廓得分为 {:.2f}%'.format(silhouette_score * 100))

# 模型的准确度指标如下：
# 同质性得分为 71.99%
# 完整性得分为 71.51%
# v测度得分为 71.75%
# 调整后的随机得分为 72.52%
# 调整后的相互信息得分为 71.21%
# 轮廓得分为 41.34%

# 绘制模型的分布图
fig = plt.figure()
ax = Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
ax.scatter(X_reduce[:,0],X_reduce[:,1],X_reduce[:,2],c=labels.astype(np.float))
plt.tight_layout()
pic_name = '3D散点图' + '_cluster_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('3D散点图 保存成功！！')