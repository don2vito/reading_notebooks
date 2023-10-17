import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import csv
import os

plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width',100)
pd.set_option('precision',2)

# 使用Pandas导入数据
file_name = os.path.join('./','pima_data.csv')
col_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(file_name,names=col_names)

# 查看数据前10行
print(f'数据的前10行如下：\n{data.head(10)}')

# 数据的前10行如下：
#    preg  plas  pres  skin  test  mass   pedi  age  class
# 0     6   148    72    35     0  33.6  0.627   50      1
# 1     1    85    66    29     0  26.6  0.351   31      0
# 2     8   183    64     0     0  23.3  0.672   32      1
# 3     1    89    66    23    94  28.1  0.167   21      0
# 4     0   137    40    35   168  43.1  2.288   33      1
# 5     5   116    74     0     0  25.6  0.201   30      0
# 6     3    78    50    32    88  31.0  0.248   26      1
# 7    10   115     0     0     0  35.3  0.134   29      0
# 8     2   197    70    45   543  30.5  0.158   53      1
# 9     8   125    96     0     0   0.0  0.232   54      1

# 查看数据的维度
print(f'数据共有 {data.shape[0]}行,{data.shape[1]}列')

# 数据共有 768行,9列

# 查看数据的属性和类型
print(f'数据的属性和类型如下：\n{data.dtypes}')

# 数据的属性和类型如下：
# preg       int64
# plas       int64
# pres       int64
# skin       int64
# test       int64
# mass     float64
# pedi     float64
# age        int64
# class      int64
# dtype: object

# 查看数据各字段的描述性统计信息
print(f'数据各字段的描述性统计信息如下：\n{data.describe()}')

# 数据各字段的描述性统计信息如下：
#          preg    plas    pres    skin    test    mass    pedi     age   class
# count  768.00  768.00  768.00  768.00  768.00  768.00  768.00  768.00  768.00
# mean     3.85  120.89   69.11   20.54   79.80   31.99    0.47   33.24    0.35
# std      3.37   31.97   19.36   15.95  115.24    7.88    0.33   11.76    0.48
# min      0.00    0.00    0.00    0.00    0.00    0.00    0.08   21.00    0.00
# 25%      1.00   99.00   62.00    0.00    0.00   27.30    0.24   24.00    0.00
# 50%      3.00  117.00   72.00   23.00   30.50   32.00    0.37   29.00    0.00
# 75%      6.00  140.25   80.00   32.00  127.25   36.60    0.63   41.00    1.00
# max     17.00  199.00  122.00   99.00  846.00   67.10    2.42   81.00    1.00
#
# [8 rows x 9 columns]

# 查看数据分类的分组分布情况
print('数据分类的分组情况如下：')
print(data.groupby('class').size())

# 数据分类的分组情况如下：
# class
# 0    500
# 1    268
# dtype: int64

# 查看数据属性的相关性
print('数据属性的相关性如下：')
print(data.corr(method='pearson'))

# 数据属性的相关性如下：
#        preg  plas  pres  skin  test  mass  pedi   age  class
# preg   1.00  0.13  0.14 -0.08 -0.07  0.02 -0.03  0.54   0.22
# plas   0.13  1.00  0.15  0.06  0.33  0.22  0.14  0.26   0.47
# pres   0.14  0.15  1.00  0.21  0.09  0.28  0.04  0.24   0.07
# skin  -0.08  0.06  0.21  1.00  0.44  0.39  0.18 -0.11   0.07
# test  -0.07  0.33  0.09  0.44  1.00  0.20  0.19 -0.04   0.13
# mass   0.02  0.22  0.28  0.39  0.20  1.00  0.14  0.04   0.29
# pedi  -0.03  0.14  0.04  0.18  0.19  0.14  1.00  0.03   0.17
# age    0.54  0.26  0.24 -0.11 -0.04  0.04  0.03  1.00   0.24
# class  0.22  0.47  0.07  0.07  0.13  0.29  0.17  0.24   1.00

# 查看数据的分布（偏离）情况
print(f'数据分布的偏离情况如下：\n{data.skew()}')

# 数据分布的偏离情况如下：
# preg     0.90
# plas     0.17
# pres    -1.84
# skin     0.11
# test     2.27
# mass    -0.43
# pedi     1.92
# age      1.13
# class    0.64
# dtype: float64

# 单变量图
# 直方图
data.hist(grid=False)
plt.tight_layout()
pic_name = '直方图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('直方图 保存成功！！')

# 密度图
data.plot(kind='density',subplots=True,layout=(3,3),sharex=False,grid=False)
plt.tight_layout()
pic_name = '密度图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('密度图 保存成功！！')

# 箱线图
data.plot(kind='box',subplots=True,layout=(3,3),sharex=False,grid=False)
plt.tight_layout()
pic_name = '箱线图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('箱线图 保存成功！！')

# 多变量图
# 相关矩阵图
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks= np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_names)
ax.set_yticklabels(col_names)
plt.tight_layout()
pic_name = '相关矩阵图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('相关矩阵图 保存成功！！')

# 散点矩阵图
pd.plotting.scatter_matrix(data)
plt.tight_layout()
pic_name = '散点矩阵图' + '_output.png'
plt.savefig(os.path.join('./', pic_name), dpi=400)
# plt.show()
print('散点矩阵图 保存成功！！')