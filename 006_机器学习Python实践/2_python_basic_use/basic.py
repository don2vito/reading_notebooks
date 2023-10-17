import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

# Numpy基础
# 创建数组：np.array([……])
# 查看数组的维度：array.shape
# 数组之间可以进行加、减、乘、除的广播原理

# Matplotlib基础
# 绘图使用的数据是numpy的ndarray多维数组数据类型

# Pandas基础
# array --> pd.Series(array,index=index)
# ndarray --> pd.Dataframe(data=ndarray,index=index,columns=columns)

# 数据导入
# 导入CSV文件
# 注意点：有无文件头、有无注释、分隔符号、有无引号、数据从第几行开始

# 使用csv标准库导入数据
file_name = os.path.join('./','pima_data.csv')
with open(file_name,'rt') as f:
    readers = csv.reader(f,delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(f'数据共有 {data.shape[0]}行,{data.shape[1]}列')

# 使用Numpy导入数据
with open(file_name,'rt') as f:
    data = np.loadtxt(f,delimiter=',')
    print(f'数据共有 {data.shape[0]}行,{data.shape[1]}列')

# 使用Pandas导入数据
col_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(file_name,names=col_names)
print(f'数据共有 {data.shape[0]}行,{data.shape[1]}列')