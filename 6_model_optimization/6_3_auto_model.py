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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

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
test_size = 0.33
seed = 4
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)

model = LogisticRegression()
model.fit(X_train,y_train)

# 持久化加载模型
# 通过 pickle 序列化和反序列化机器学习的模型
# 保存模型
model_file = './finalized_model.sav'
with open(model_file,'wb') as model_f:
    # 模型序列化
    dump(model,model_f)

# 加载模型
with open(model_file,'rb') as model_f:
    # 模型反序列化
    loaded_model = load(model_f)
    result = loaded_model.score(X_test,y_test)
    print('逻辑回归的评估结果是： {:.2f}%'.format(result * 100))

# 逻辑回归的评估结果是： 80.31%

# 通过 joblib 序列化和反序列化机器学习的模型
# 保存模型
model_file = './finalized_model_joblib.sav'
with open(model_file,'wb') as model_f:
    # 模型序列化
    dump(model,model_f)

# 加载模型
with open(model_file,'rb') as model_f:
    # 模型反序列化
    loaded_model = load(model_f)
    result = loaded_model.score(X_test,y_test)
    print('逻辑回归的评估结果是： {:.2f}%'.format(result * 100))

# 逻辑回归的评估结果是： 80.31%