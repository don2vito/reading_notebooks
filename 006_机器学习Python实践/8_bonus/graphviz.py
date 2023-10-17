import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import pydotplus
from matplotlib.image import imread

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 20)
pd.set_option('precision',3)

warnings.filterwarnings('ignore')

# 导入数据集
file_name = os.path.join('./','iris.data.csv')
column_name = ['separ_length','separ_width','petal_length','petal_width','class']
data = pd.read_csv(file_name,names=column_name)
# data['class'] = data['class'].replace(to_replace=['Iris-setosa','Iris-versicolor','Iris-virginica'],value=['A','B','C'])
array = data.values
X = array[:,0:4]
y = array[:,4]
# y = y.astype(np.str)

# 分离数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=7)

# 决策树模型训练
model = DecisionTreeClassifier()
model.fit(X=X_train,y=y_train)

# 决策树图形化
dot_data = export_graphviz(model,out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
path = os.getcwd() + '/'
tree_file = path + 'graphviz_DTC_output.png'

try:
    os.remove(tree_file)
except:
    print('此处没有文件可以被删除！！')
finally:
    graph.write(tree_file,format='png')

print('决策树图 保存成功！！')

# 显示图像
image_data = imread(tree_file)
plt.imshow(image_data)
plt.axis('off')
# plt.tight_layout()
# pic_name = 'graphviz_DTC_output.png'
# plt.savefig(os.path.join('./', pic_name), dpi=400)
# print('决策树图 保存成功！！')
# plt.show()

# 评估算法
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test,prediction)
print('模型的准确率是 {:.2f}%'.format(accuracy * 100))

# 模型的准确率是 90.00%