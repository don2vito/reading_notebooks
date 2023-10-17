# 导入模块
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-5,5,500)

# 定义函数
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(x,0)

# 画出各函数的图形
fig = plt.figure(figsize=(8, 8),dpi=800)
ax = plt.subplot(111) 
fig.add_axes(ax)
plt.plot(x,sigmoid,label='sigmoid')
plt.plot(x,tanh,label='tanh')
plt.plot(x,relu,label='relu')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-5, 5)
plt.ylim(-1, 2)
plt.show()



# 初始化操作，对单个神经元进行建模，该神经元含有 3 个输入和 1 个输出
from numpy import exp, array, random, dot
class NeuralNetwork():    
#使用__init__(self) 初始化
    def __init__(self):        
    # 设置随机数发生器种子，保证每次结果相同，数值范围是[0,1]        
        random.seed(0)        
        # 表达式的数值范围为 2 * [0,1] -1，即 [-1,1]        
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        
    # 定义 sigmoid 函数
    # 用 sigmoid 函数对输入的加权和进行压缩，使其数值范围为 0～1    
    def sigmoid(self, x):        
        return 1 / (1 + exp(-x))        
        
    # 定义 sigmoid 曲线的梯度    
    def sigmoid_derivative(self, x):        
        return x * (1 - x)
        
    # 定义输入函数，将输入值传给神经网络    
    def shuru(self, inputs):        
        return self.sigmoid(dot(inputs, self.synaptic_weights))
        
    # 神经网络数据训练
    def train(self, inputs, outputs, iterations):    
        for iteration in range(iterations):        
            # 将训练集导入神经网络        
            output = self.shuru(inputs)        
            # 计算输出的预测值与真实值的差距        
            error = outputs - output                 
            change = dot(inputs.T, error * self.sigmoid_derivative(output))                  # 调整权重        
            self.synaptic_weights += change
            
            
if __name__ == "__main__":    
    # 对神经网络进行初始化操作    
    neural_network = NeuralNetwork()    
    print("随机设定的初始权重为：")    
    print(neural_network.synaptic_weights)
    # 输入训练样本数据    
    inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])    
    outputs = array([[0, 1, 1, 0]]).T    
    # 用训练集数据对神经网络进行训练，训练 20000 次    
    neural_network.train(inputs,outputs, 20000)    
    print("样本训练后的权重为：")    
    print(neural_network.synaptic_weights)    
    # 用新数据测试神经网络    
    print("新输入值 [1, 0, 0]的输出值为多少? ")    
    print(neural_network.shuru(array([1, 0, 0])))
    
    
    
    
# 从 sklearn.datasets 中导入手写体数字加载器
from sklearn.datasets import load_digits

# 获取手写体数字的数码图像并存储在变量 digits 中
digits=load_digits()
x = digits.data
y = digits.target

# 导入数据分割模块
from sklearn.model_selection import train_test_split

# 将数据分割为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=33)

# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler

# 对特征数据进行标准化操作
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

from sklearn.neural_network import MLPClassifier

# 初始化神经网络分类器
mclf = MLPClassifier()

# 进行模型训练
mclf.fit(x_train,y_train)

# 进行预测
y_predict = mclf.predict(x_test)

# 使用模型自带的评估函数进行准确率评估
print('准确率为：',mclf.score(x_test,y_test))

# 使用 sklearn.metrics 中的 classification_report 模块对预测结果进行详细评估
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))