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

# 从 sklearn.tree 中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier

# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()

# 使用训练数据来训练算法模型，确定算法模型参数
dtc.fit(x_train,y_train)

# 使用算法模型（参数确定的）来对测试数据进行预测
dtc_y_predict = dtc.predict(x_test)

# 从 sklearn.metrics 中导入 classification_report 来进行评估
from sklearn.metrics import classification_report
print(dtc.score(x_test,y_test))

# 查看详细评估结果，包括精确率、召回率及 f1 指标，输出混淆矩阵
print(classification_report(y_test,dtc_y_predict))