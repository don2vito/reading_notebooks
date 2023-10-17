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

# 对率回归算法：默认配置
# 从 sklearn.linear_model 中选用逻辑回归模型 LogisticRegression 来学习数据；肿瘤分类数据的特征变量与目标变量之间可能存在某种线性关系，这种线性关系可以用对率回归模型 LogisticRegression 来表达，所以选择该算法进行学习
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train,y_train)

# 使用训练好的逻辑回归模型对数据进行预测 
lr_y_predict = lr.predict(x_test)

# 性能评估，使用对率回归自带的评分函数 score 获取预测准确率数据，并使用 sklearn.metrics  的 classification_report 模块对预测结果进行全面评估
from sklearn.metrics import classification_report
print('Accuracy:',lr.score(x_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['benign','malignant']))