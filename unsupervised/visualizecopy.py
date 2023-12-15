#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 19:20:42 2022

@author: soham
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:58:37 2022

@author: soham
"""

#Importing required modules

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

#Load Data
from cluster import unpairLists, readData

data = readData(1000, 2)
trainData, testData, testLabels = data[0]
trainDataPCA, testDataPCA = data[1]
trainDataACF, testDataACF, testLabelsACF = data[2]
trainDataACFPCA, testDataACFPCA = data[3]

data = testData
pca = PCA(2)

dfTrain = pca.fit_transform(trainData)
xTrain, yTrain = unpairLists(dfTrain)

plt.figure(dpi=300)

labelt = testLabels
u_labelst = np.unique(testLabels)
#Transform the data
pca = PCA(2)
df = dfTrain

kmeans = KMeans(n_clusters=2)
label = kmeans.fit_predict(df)
u_labels = np.unique(label)
for i in u_labels:
    cn = 'gray'
    if i == 0:
        cn = 'aqua'
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i, color=cn, alpha=0.1)

pca = PCA(2)
df = pca.fit_transform(data)
#plt.scatter(xTrain, yTrain, color='gray', alpha=0.1)
for i in u_labelst:
    cn = 'C1'
    if i >= 4:
        cn = 'C2'
    
    plt.scatter(df[labelt == i, 0], df[labelt == i, 1], label=i, color=cn, alpha=0.7)
plt.xlim([-1500, 3500])
plt.legend()
plt.show()

# plt.figure(dpi=300)
# for i in u_labels:
#     cn = 'C1'
#     if i >= 4:
#         cn = 'C2'
#     plt.scatter(df[label == i, 0], df[label == i, 1], label=i, color=cn)
# plt.xlim([-1500, 3500])
# plt.legend()
# plt.show()
    

# kmeans = KMeans(n_clusters=2)
# label = kmeans.fit_predict(df)
 
# #Getting unique labels
# u_labels = np.unique(label)
# plt.figure(dpi=300)
# #plotting the results:
# for i in u_labels:
#     cn = 'C1'
#     if i == 0:
#         cn = 'C2'
#     plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i, color=cn)
# plt.xlim([-1500, 3500])
# plt.legend()
# plt.show()