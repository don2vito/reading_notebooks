# 从 sklearn.datasets 自带的数据中读取波士顿房价数据并将其存储在变量 bostonHouse 中
from sklearn.datasets import load_boston
bostonHouse=load_boston()

# 明确特征变量与目标变量
x=bostonHouse.data
y=bostonHouse.target

# 从 sklearn.model_selection 中导入数据分割器
from sklearn.model_selection import train_test_split

# 使用数据分割器将样本数据分割为训练数据和测试数据，其中测试数据占比为 25%；数据分割是为了获得训练集和测试集；训练集用来训练模型，测试集用来评估模型性能
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.25)

# 从 sklearn.linear_model 中选用线性回归模型 LinearRegression 来学习数据；波士顿房价数据的特征变量与目标变量之间可能存在某种线性关系，这种线性关系可以用线性回归模型 LinearRegression 来表达，所以选择该算法进行学习
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化线性回归器
lr = LinearRegression()

# 使用训练数据来估计参数，也就是通过训练数据的学习，为线性回归器找到一组合适的参数，从而获得一个带有参数的、具体的线性回归模型
lr.fit(x_train,y_train)

# 对测试数据进行预测；利用上述训练数据学习得到的带有参数的、具体的线性回归模型对测试数据进行预测，即将测试数据中每一条记录的特征变量（例如房间数、不动产税率等）输入该线性回归模型中，得到一个该条记录的预测值
lr_y_predict = lr.predict(x_test)

# 模型性能评估；可以通过比较测试数据的模型预测值与真实值之间的差距来评估，此处使用 MSE 来评估
from sklearn.metrics import mean_squared_error 
print('MSE:',mean_squared_error(y_test,lr_y_predict))