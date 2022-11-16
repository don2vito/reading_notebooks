# 定义一组字典列表，用来表示多个数据样本，其中每个字典表示一个数据样本
samples = [{'name':'李大龙','age':33},{'name':'张小小','age':32},{'name':'大牛','age':40}]

# 从 sklearn.feature_extraction 中导入 DictVectorizer
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

# 输出转化之后的特征矩阵内容
print(vec.fit_transform(samples).toarray())

# 输出各个维度的特征含义
print(vec.get_feature_names())



from sklearn.feature_extraction.text import CountVectorizer

# 创建语料库 "Dalong is good at playing basketball."，"Dalong likes playing football.Dalong  also likes music."
corpus = ["Dalong is good at playing basketball.","Dalong likes playing football.Dalong also likes music."]

# 创建词袋数据结构，按照默认配置
vectorizer = CountVectorizer() 
# 拟合模型，并返回文本矩阵
features = vectorizer.fit_transform(corpus)  

# 以列表的形式，展示所有文本的词汇
print("文本词汇列表：\n",vectorizer.get_feature_names()) 

# 词汇表是字典类型，其中 key 表示词语，value 表示对应的词典序号
print("词汇表中 key 表示词语，value 表示对应的词典序号:\n",vectorizer.vocabulary_)

# 文本矩阵，显示第 i 个字符串中对应词典序号为j的词汇的词频
print("第 i 个字符串中对应词典序号为 j 的词汇的词频:\n",features)   

# .toarray() 用于将结果转化为稀疏矩阵形式，展示各个字符串中对应词典序号的词汇的词频
print("字符串中对应词典序号的词汇的词频:\n",features.toarray())  

# 统计每个词汇在所有文档中的词频
print("每个词汇在所有文档中的词频:\n",features.toarray().sum(axis=0))



from sklearn.feature_extraction.text import TfidfVectorizer

# 创建语料库 "Dalong  is good at playing basketball."，"Dalong  is good at being a teacher."
corpus = ["Dalong  is  good at playing basketball.","Dalong  likes playing football.Dalong also likes music."]

# 创建词袋数据结构，按照默认配置
vectorizer = TfidfVectorizer() 

# 拟合模型，并返回文本矩阵
features = vectorizer.fit_transform(corpus) 
 
# 以列表的形式，展示所有文本的词汇
print("文本词汇列表：\n",vectorizer.get_feature_names()) 

# 词汇表是字典类型，其中 key 表示词汇，value 表示对应的词典序号
print("词汇表中 key 表示词汇，value 表示对应的词典序号:\n",vectorizer.vocabulary_)  

# 文本矩阵，显示第 i 个字符串中对应词典序号为j的词汇的 TF-IDF 值
print("第 i 个字符串中对应词典序号为 j 的词汇的TF-IDF值:\n",features)



# 从 sklearn 中导入自带的鸢尾花数据集
from sklearn.datasets import load_iris

# 从 sklearn 中导入特征筛选器 SelectKBest
from sklearn.feature_selection import SelectKBest

# 导入 chi2（卡方检验）
from sklearn.feature_selection import chi2
import pandas as pd
iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target)

# 初始化特征筛选器 SelectKBest，指定 k 值为 2，采用卡方检验 chi2
selector = SelectKBest(chi2, k=2)

# 使用特征筛选器 SelectKBest 处理数据
selector.fit(x, y)
x_new = selector.transform(x)

# 对比现实数据特征维度变化
print('原始数据特征形态：',x.shape)
print('特征选择后的新数据特征形态：',x_new.shape)

# 显示特征选择后的 k 个最有价值的特征名称
print('筛选保留的特征为：',x.columns[selector.get_support(indices=True)])



# 导入鸢尾花数据集
from sklearn.datasets import load_iris
iris = load_iris()
x,y = iris.data,iris.target

# 导入交叉验证的工具
from sklearn.model_selection import cross_val_score

# 导入用于分类的 SVC 分类器
from sklearn.svm import SVC

# 初始化 SVC，设置核函数为 linear
svc = SVC(kernel='linear')

# 使用交叉验证法对 svc 进行评分，这里设置 k = 5
scores = cross_val_score(svc,x,y,cv=5)

# 显示交叉验证得分
print('交叉验证得分：{}'.format(scores))

# 用得分均值作为最终得分
print('取平均值作为交叉验证最终得分：{}'.format(scores.mean()))



from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()

# 分离出特征变量与目标变量
x = breast_cancer.data
y = breast_cancer.target

# 从 sklearn.model_selection 中导入数据分割器
from sklearn.model_selection import train_test_split 

# 使用数据分割器将样本数据分割为训练数据和测试数据，其中测试数据占比为 30%，数据分割是为了获得训练集和测试集，训练集用来训练模型，测试集用来评估模型性能
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.3)

# 对数据进行标准化处理，使得每个特征维度的均值为 0，方差为 1，防止受到某个维度特征数值较大的影响
from sklearn.preprocessing import StandardScaler
breast_cancer_ss = StandardScaler()
x_train = breast_cancer_ss.fit_transform(x_train)
x_test = breast_cancer_ss.transform(x_test)

#（1）默认配置：采用默认配置的决策树模型
from sklearn.tree import DecisionTreeClassifier

# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()

# 训练数据
dtc.fit(x_train,y_train)

# 数据预测
dtc_y_predict = dtc.predict(x_test)

# 性能评估
from sklearn.metrics import classification_report
print('Accuracy:',dtc.score(x_test,y_test))
print(classification_report(y_test,dtc_y_predict,target_names =['benign','malignant']))



#（2）使用网格搜索工具来寻找模型最优配置
from sklearn.tree import DecisionTreeClassifier

# 导入网格搜索工具
from sklearn.model_selection import GridSearchCV

# 设置网格搜索的超参数组合范围
params_tree = {'max_depth':range(5,15),'criterion':['entropy']}

# 导入 K 折交叉验证工具
from sklearn.model_selection import KFold

# 初始化决策树分类器
dtc = DecisionTreeClassifier()

# 设置 10 折交叉验证
kf = KFold(n_splits=10,shuffle=False)

# 定义网格搜索中使用的模型和参数
grid_search_dtc = GridSearchCV(dtc,params_tree,cv=kf)

# 使用网格搜索模型拟合数据
grid_search_dtc.fit(x_train,y_train)

# 数据预测
grid_dtc_y_predict = grid_search_dtc.predict(x_test)

# 性能评估
from sklearn.metrics import classification_report
print('Accuracy:',grid_search_dtc.score(x_test,y_test))
print(classification_report(y_test,grid_dtc_y_predict,target_names=['benign','malignant']))
print('超参数配置情况:',grid_search_dtc.best_params_)



from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
print(breast_cancer)

# 分离出特征变量与目标变量
x = breast_cancer.data
y = breast_cancer.target

# 从 sklearn.model_selection 中导入数据分割器
from sklearn.model_selection import train_test_split

# 使用数据分割器将样本数据分割为训练数据和测试数据，其中测试数据占比为30%，数据分割是为了获得训练集和测试集。训练集用来训练模型，测试集用来评估模型性能
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=33,test_size=0.3)

# 对数据进行标准化处理，使得每个特征维度的均值为 0，方差为 1，防止受到某个维度特征数值较大的影响
from sklearn.preprocessing import StandardScaler
breast_cancer_ss=StandardScaler()
x_train = breast_cancer_ss.fit_transform(x_train)
x_test = breast_cancer_ss.transform(x_test)

# 对率回归算法
# 从 sklearn.linear_model 中选用对率回归模型 LogisticRegression 来学习数据，认为肿瘤分类数据的特征变量与目标变量之间可能存在某种线性关系，这种线性关系可以用对率回归模型，LogisticRegression，来表达，所以选择该算法进行学习
from sklearn.linear_model import LogisticRegression

# 使用默认配置初始化线性回归器
lr = LogisticRegression()

# 使用训练数据来估计参数，通过训练数据的学习，找到一组合适的参数，从而获得一个带有参数的、具体的算法模型
lr.fit(x_train,y_train)

# 对测试数据进行预测，利用上述训练数据学习得到的带有参数的、具体的线性回归模型对测试数据进行预测，即将测试数据中每一条记录的特征变量输入该模型中，得到一个该条记录的预测分类值
lr_y_predict = lr.predict(x_test)

# 性能评估，使用对率回归自带的评分函数 score 获取预测准确率数据，并使用 sklearn.metrics 的 classification_report 模块对预测结果进行全面评估
from sklearn.metrics import classification_report
print('Accuracy:',lr.score(x_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['benign','malignant']))



# 导入对率回归分类器
from sklearn.linear_model import LogisticRegression

# 导入网格搜索工具
from sklearn.model_selection import GridSearchCV

# 设置网格搜索的超参数组合范围
params_lr = {'C':  [0.0001,0.001,0.01,0.1,1,10],'penalty': ['l2'],'tol': [1e-4,1e-5,1e-6]}

# 导入 K 折交叉验证工具
from sklearn.model_selection import KFold

# 初始化决策树分类器
lr = LogisticRegression()

# 设置 10 折交叉验证
kf = KFold(n_splits=10,shuffle=False)

# 定义网格搜索中使用的模型和参数
grid_search_lr = GridSearchCV(lr,params_lr,cv=kf)

# 使用网格搜索模型拟合数据
grid_search_lr.fit(x_train,y_train)

# 数据预测
grid_lr_y_predict = grid_search_lr.predict(x_test)

# 性能评估
from sklearn.metrics import classification_report
print('Accuracy:',grid_search_lr.score(x_test,y_test))
print(classification_report(y_test,grid_lr_y_predict,target_names=['benign','malignant']))
print('超参数配置情况:',grid_search_lr.best_params_)



# 使用支持向量机分类器
# 从 sklearn.svm 中导入支持向量机分类器 linearSVC
from sklearn.svm import LinearSVC

# 使用默认配置初始化支持向量机分类器
lsvc=LinearSVC()

# 使用训练数据来估计参数，也就是通过训练数据的学习，找到一组合适的参数，从而获得一个带有参数的、具体的算法模型
lsvc.fit(x_train,y_train)

# 对测试数据进行预测，利用上述训练数据学习得到的带有参数的具体模型对测试数据进行预测，即将测试数据中每一条记录的特征变量输入该模型中，得到一个该条记录的预测分类值
lsvc_y_predict = lsvc.predict(x_test)

# 性能评估，使用自带的评分函数 score 获取预测准确率数据，并使用 sklearn.metrics 的 classification_repor t模块对预测结果进行全面评估
from sklearn.metrics import classification_report
print('Accuracy:',lsvc.score(x_test,y_test))
print(classification_report(y_test,lsvc_y_predict,target_names=['benign','malignant']))



from sklearn.svm import SVC

# 导入网格搜索工具
from sklearn.model_selection import GridSearchCV

# 设置网格搜索的超参数组合范围
params_svc = {'C':[4.5,5,5.5,6],'gamma':[0.0009,0.001,0.0011,0.002]}

# 导入 K 折交叉验证工具
from sklearn.model_selection import KFold

# 初始化决策树分类器
svc = SVC()

# 设置 10 折交叉验证
kf = KFold(n_splits=10,shuffle=False)

# 定义网格搜索中使用的模型和参数
grid_search_svc = GridSearchCV(svc,params_svc,cv=kf)

# 使用网格搜索模型拟合数据
grid_search_svc.fit(x_train,y_train)

# 数据预测
grid_svc_y_predict = grid_search_svc.predict(x_test)

# 性能评估
from sklearn.metrics import classification_report
print('Accuracy:',grid_search_svc.score(x_test,y_test))
print(classification_report(y_test,grid_svc_y_predict,target_names=['benign','malignant']))
print('超参数配置情况:',grid_search_svc.best_params_)



# 使用决策树分类器
# 从 sklearn.tree 中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier

# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()

# 训练数据
dtc.fit(x_train,y_train)

# 数据预测
dtc_y_predict = dtc.predict(x_test)

# 性能评估
from sklearn.metrics import classification_report
print('Accuracy:',dtc.score(x_test,y_test))
print(classification_report(y_test,dtc_y_predict,target_names=['benign','malignant']))



# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier

# 导入网格搜索工具
from sklearn.model_selection import GridSearchCV

# 设置网格搜索的超参数组合范围
params_tree = {'max_depth':range(5,15),'criterion':['entropy']}

# 导入 K 折交叉验证工具
from sklearn.model_selection import KFold

# 初始化决策树分类器
dtc = DecisionTreeClassifier()
# 设置 10 折交叉验证
kf = KFold(n_splits=10,shuffle=False)

# 定义网格搜索中使用的模型和参数
grid_search_dtc = GridSearchCV(dtc,params_tree,cv=kf)

# 使用网格搜索模型拟合数据
grid_search_dtc.fit(x_train,y_train)

# 数据预测
grid_dtc_y_predict = grid_search_dtc.predict(x_test)

# 性能评估
from sklearn.metrics import classification_report
print('Accuracy:',grid_search_dtc.score(x_test,y_test))
print(classification_report(y_test,grid_dtc_y_predict,target_names=['benign','malignant']))
print('超参数配置情况:',grid_search_dtc.best_params_)



# 对比上述各种算法模型性能，不难发现网格搜索下的支持向量机分类器具有最优的预测性能，考虑使用该模型来预测数据；为了简单，选取原始数据集中的某条记录来进行演示
# 使用原始数据集中的某条数据记录（第 30 条）来验证预测值与真实值的比较情况
x_try = x[30].reshape(1, -1)

# 使用网格搜索下的支持向量机分类器对某条记录进行预测
grid_search_svc_y_try = grid_search_svc.predict(x_try)
print('*'*50)
print('尝试预测结果为：',grid_search_svc_y_try)
print('真实结果为:',y[30])
print('*'*50)