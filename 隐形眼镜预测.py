import random
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
random.seed(10)
class Decision_Tree:
    ##################################初始化类，传入数据集参数和模式##############################
    def __init__(self,DataSet_Path=r'./lenses.csv',class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       random_state=2, splitter='best',radio=0.3):
        self.DataSet_Path=DataSet_Path
        self.criterion=criterion
        self.class_weight=class_weight
        self.max_depth=max_depth
        self.max_feature=max_features
        self.max_leaf_nods=max_leaf_nodes
        self.min_impurity_decrease=min_impurity_decrease
        self.min_samples_leaf=min_samples_leaf
        self.min_samples_split=min_samples_split
        self.min_weight_fraction_leaf=min_weight_fraction_leaf
        self.random_state=random_state
        self.splitter=splitter
        self.radio=radio



    ################################对传入的数据进行处理#######################################
    def Create_DataSet(self):

        self.DataSet=open(self.DataSet_Path).read()
        self.DataSet=self.DataSet.split('\n')
        for i in range(len(self.DataSet)):
            self.DataSet[i]=self.DataSet[i].split(',')
        self.DataSet=self.DataSet[0:len(self.DataSet)-1]
        self.feature_name=self.DataSet[0][0:len(self.DataSet[0])-1]   ##创建特征值名称向量
        del self.DataSet[0]
        self.label_name=sorted(set(np.array(self.DataSet)[:,-1].tolist()))   ##标签的名称创建
        DataSet_f=sorted(list(set([item for value in self.DataSet for item in value])) )##将数据展开成一维张量
        X_sample=self.DataSet[0][0:len(self.DataSet[0])-1]
        debox=[]
        ####将数据转化为int类型##############################################################################
        dictor={}
        count=0
        for item in DataSet_f:
            dictor.update({item:count})
            bebox=[]
            count=count+1
            for value in range(len(np.where(np.array(self.DataSet)==item)[0].tolist())):
                listd = []
                x=np.where(np.array(self.DataSet)==item)[0].tolist()[value]
                y=np.where(np.array(self.DataSet) == item)[1].tolist()[value]
                listd.append(x)
                listd.append(y)
                bebox.append(listd)
            debox.append(bebox)
        count=0

        for item in debox:
            count=count+1
            for value in item:
                self.DataSet[value[0]][value[1]]=count
        ######################################################################################################
        length=len(self.DataSet[0])
        target=np.array(self.DataSet).T.tolist()[-1]
        data=np.array(self.DataSet).T[0:length-1].T.tolist()
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=self.radio,random_state=self.random_state)
        self.Xtrain,self.Xtest,self.Ytrain,self.Ytest=Xtrain, Xtest, Ytrain, Ytest
        return Xtrain,Xtest,Ytrain,Ytest,self.feature_name,self.label_name,X_sample,dictor,DataSet_f

    def Create_Tree(self):
        Xtrain, Xtest, Ytrain, Ytest, feature_name, label_name,_,_,_=Decision_Tree.Create_DataSet(self)
        model=tree.DecisionTreeClassifier(class_weight=self.class_weight, criterion=self.criterion, max_depth=self.max_depth,
                       max_features=self.max_feature, max_leaf_nodes=self.max_leaf_nods,
                       min_impurity_decrease=self.min_impurity_decrease,
                       min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split,
                       min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                       random_state=self.random_state, splitter=self.splitter)
        model=model.fit(Xtrain,Ytrain)
        return model,feature_name,label_name

    def plot(self,width=15,height=9):
        model,feature_name,label_name=Decision_Tree.Create_Tree(self)
        plt.figure(figsize=(width, height))
        plot_tree(model, filled=True, feature_names=feature_name, class_names=label_name)
        plt.show()

    def get_score(self,Xtest=None,Ytest=None):
        _, X, _, Y, _, _,_,_,_=Decision_Tree.Create_DataSet(self)
        model, _, _ = Decision_Tree.Create_Tree(self)
        if Xtest==None:
            Xtest=X
        if Ytest==None:
            Ytest=Y
        score = model.score(Xtest, Ytest)
        print("模型得分为：{}".format(score))
        return score

    def prediect(self,X=None):
        model, _, _ = Decision_Tree.Create_Tree(self)
        _, _, _, _, _, _, X_sample, dictor,NAMES = Decision_Tree.Create_DataSet(self)
        if X==None:
            print("请输入要预测的值,例如（{}）".format([X_sample]))
        else:
            Xl=[]
            for item in X:
                if item in dictor:
                    Xl.append(dictor[item])
                else:
                    print("数据格式错误,请重新输入，例如（{}）".format([X_sample]))
            print("样本:{},的预测类别是:{}".format(X,[NAMES[model.predict([Xl])[0]-1]]))
            return [NAMES[model.predict([Xl])[0]-1]]

A=Decision_Tree()
A.plot(5,4)
A.get_score()
A.prediect(['young', 'myope', 'yes', 'normal'])
