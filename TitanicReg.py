# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:21:26 2020


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Reading Dataset'''
dataset = pd.read_csv("C:/Users/user/Downloads/ChanDarren_RaiTaran_Lab2a.csv")

'''Dimension of dataset'''
print(dataset.shape)


'''Check if any data is missing'''
print(dataset.isnull().sum())

'''Unique Values of Age'''
print(dataset["Age"].unique())

'''Unique Values of Cabin'''
print(dataset["Cabin"].unique())

'''Unique Values of Embarked'''
print(dataset["Embarked"].unique())

''' Filling the missing value of Age column'''
mean_Age = dataset["Age"].mean()
std_Age = dataset["Age"].std()

'''Using random select to fill the Value'''

import random
ran_list = np.random.randint(mean_Age-std_Age,mean_Age+std_Age)
dataset["Age"][np.isnan(dataset["Age"])] = ran_list

'''barplot of Age'''
sns.barplot(dataset["Survived"],dataset["Age"], color = "red")
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()

'''Filling the missing values as '0' of cabin column'''
def cabin(col):
    Cabin = col[0]
    if type(Cabin) == str:
        return 1
    else:
        return 0
dataset["Cabin"] = dataset[["Cabin"]].apply(cabin,axis = 1) 
 
'''Missing values of Embarked'''
print(dataset["Embarked"].nunique())
dataset["Embarked"] = dataset["Embarked"].fillna("S")

print(dataset.isnull().sum())

'''Age Description'''
print(dataset["Age"].describe())

'''male as 1 and female as 0'''
scale_mapper = {"male" : 1, "female" :0}
dataset["Sex"] = dataset["Sex"].replace(scale_mapper)

'''Embarked Encoding'''
embark_mapper = {"S":0,"C":1,"Q":2}
dataset["Embarked"] = dataset["Embarked"].replace(embark_mapper)

'''X ,y dataset'''
X = dataset.iloc[:,[0,2,4,5,6,7,9,10,11]]
y = dataset.iloc[:,1]

print(X.describe(include = "all"))
print(y.describe(include = "all"))

'''Spliting X_train, Y_train,X_test,y_test'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

'''Normalizing
from sklearn.preprocessing import StandardScaler
Sc_x = StandardScaler()
X_train = Sc_x.fit_transform(X_train)
X_test = Sc_x.transform(X_test)'''

'''Random Forest'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
parameters = {'criterion' : ['gini','entropy'],
               'max_depth' :[9,10,11,12],
               'min_samples_split' : [2,3,4,5],
               'class_weight' :['balanced',None],
               'max_features':['auto','sqrt','log2'],
             
               }
tr = RandomForestClassifier()
gsearch = GridSearchCV(tr,parameters,cv = 5)
gsearch.fit(X_train,y_train)

model = gsearch.best_estimator_
print(model)

'''Getting best Parameters'''

params = gsearch.best_params_
print(params)

fr = RandomForestClassifier(class_weight = 'balanced', criterion= 'entropy',
                            max_depth = 12, max_features= 'log2', min_samples_split= 5, n_estimators = 100,
                            )
fr.fit(X_train,y_train)
'''Predicting'''
y_pred = fr.predict(X_test) 
print(model.score(X_test,y_test))

'''Confusion Matrix'''

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
ranforcm = confusion_matrix(y_test,y_pred)
ranforcr = classification_report(y_test,y_pred)
print(ranforcm)
print(ranforcr)

print("Accuracy-Score",accuracy_score(y_test,y_pred))
'''Frame the result'''
op = pd.DataFrame(X_test['PassengerId'])
op['Survived'] = y_pred
