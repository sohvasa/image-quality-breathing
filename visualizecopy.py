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

data = readData(1000)
trainData, testData, testLabels = data[0]
trainDataPCA, testDataPCA = data[1]
trainDataACF, testDataACF, testLabelsACF = data[2]
trainDataACFPCA, testDataACFPCA = data[3]

data = testDataACF
pca = PCA(2)

dfTrain = pca.fit_transform(trainDataACF)
xTrain, yTrain = unpairLists(dfTrain)

label = testLabelsACF
u_labels = np.unique(testLabelsACF)
#Transform the data
pca = PCA(2)
df = pca.fit_transform(data)

for i in u_labels:
    cn = 'C1'
    if i >= 4:
        cn = 'C2'
    plt.scatter(xTrain, yTrain, color='aqua', alpha=0.01)
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i, color=cn)
plt.legend()
plt.show()
    

#Initialize the class object
kmeans = KMeans(n_clusters=2)
 
#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
#Getting unique labels
u_labels = np.unique(label)
 
#plotting the results:
for i in u_labels:
    cn = 'C1'
    if i == 0:
        cn = 'C2'
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i, color=cn)
plt.legend()
plt.show()

dfTrain = pca.fit_transform(testData)
print(dfTrain)

kmeans = KMeans(n_clusters=6)
#predict the labels of clusters.
label = kmeans.fit_predict(dfTrain)

#plotting the results:
x, y = unpairLists(dfTrain)

plt.scatter(x, y, label = label)
plt.legend()
plt.show()