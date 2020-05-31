# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:52:00 2019

@author: Admin
"""

from sklearn.datasets.base import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
 
# 从 sklearn的datasets模块载入数据集加载酒的数据集
wineDataSet=load_wine()
#print(wineDataSet['data'],wineDataSet['target'])
 
# 将数据集拆分为训练数据集和测试数据集
X_train,X_test,y_train,y_test=train_test_split(wineDataSet['data'],wineDataSet['target'],random_state=0)
print(len(X_train),wineDataSet['target_names'])
 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
 
# 评估模型的准确率
print('测试数据集得分：{:.2f}'.format(knn.score(X_test,y_test)))
 
# 使用建好的模型对新酒进行分类预测
X_new = np.array([[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]])
prediction = knn.predict(X_new)
print(prediction)
print("预测新酒的分类为：{}".format(wineDataSet['target_names'][prediction]))     

