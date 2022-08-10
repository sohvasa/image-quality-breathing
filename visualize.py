#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:58:37 2022

@author: soham
"""
import read

#Importing required modules

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

#Load Data

trainReader = read.Read(test=0, preprocessing='none')
testReader = read.ReadLabeledData(preprocessing='none')

trainData, null = trainReader.getFormattedData()
testData, testLabels = testReader.getLabeledData()


trainReaderACF = read.Read(test=0)
testReaderACF = read.ReadLabeledData()

trainDataACF, null = trainReader.getFormattedData()
testDataACF, testLabelsACF = testReader.getLabeledData()


data = trainDataACF
pca = PCA(2)

 
# #Transform the data
df = pca.fit_transform(data)
# print(df)
# print(testLabelsACF)


#plt.scatter(df[])
 
#Import KMeans module

for k in range(2, 15):
#Initialize the class object
    kmeans = KMeans(n_clusters=k)
     
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
     
    #Getting unique labels
    u_labels = np.unique(label)
     
    #plotting the results:
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    plt.legend()
    plt.show()