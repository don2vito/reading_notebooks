from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
print(breast_cancer)

# 分离出特征变量与目标变量
x = breast_cancer.data
y = breast_cancer.target

# 从 sklearn.model_selection 中导入数据分割器
from sklearn.model_selection import train_test_split

# 使用数据分割器将样本数据分割为训练数据和测试数据，其中测试数据占比为 30%；数据分割是为了获得训练集和测试集；训练集用来训练模型，测试集用来评估模型性能
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.3)

# 对数据进行标准化处理，使得每个特征维度的均值为 0，方差为 1，防止受到某个维度特征数值较大的影响
from sklearn.preprocessing import StandardScaler
breast_cancer_ss = StandardScaler()
x_train = breast_cancer_ss.fit_transform(x_train)
x_test = breast_cancer_ss.transform(x_test)

# 使用支持向量机分类器
# 从 sklearn.svm 中导入支持向量机分类器 LinearSVC
from sklearn.svm import LinearSVC

# 使用默认配置初始化支持向量机分类器
lsvc = LinearSVC()

# 使用训练数据来估计参数，也就是通过训练数据的学习，找到一组合适的参数，从而获得一个带有参数的、具体的算法模型
lsvc.fit(x_train,y_train)

# 对测试数据进行预测，利用上述训练数据学习得到的带有参数的具体模型对测试数据进行预测，即将测试数据中每一条记录的特征变量输入该模型中，得到一个该条记录的预测分类值
lsvc_y_predict = lsvc.predict(x_test)

# 性能评估，使用自带的评分函数score获取预测准确率数据，并使用 sklearn.metrics 的 classification_report 模块对预测结果进行全面评估
from sklearn.metrics import classification_report
print('Accuracy:',lsvc.score(x_test,y_test))
print(classification_report(y_test,lsvc_y_predict,target_names=['benign','malignant']))