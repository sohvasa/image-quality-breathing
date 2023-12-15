#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 11:03:20 2022

@author: soham
"""
import read

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
import random
import os

def convertToBinary(l):
    new = []
    for element in l:
        if element >= 4:
            new.append(1)
        else:
            new.append(0)
    return new

def pairLists(l1, l2):
    """
    Combine two lists into one list of tuple pairs.

    Parameters
    ----------
    l1 : list
        1D list.
    l2 : list
        1D list.

    Returns
    -------
    combined : list
        2D list.

    """
    combined = []
    for i in range(len(l1)):
        combined.append((l1[i], l2[i]))
    return combined

def unpairLists(l):
    l1 = []
    l2 = []
    for pair in l:
        l1.append(pair[0])
        l2.append(pair[1])
    return (l1, l2)

def shuffle(l):
    x = l
    random.shuffle(x)
    return x


def iteration(pcaDim, c, g, graph=False):

    testReader = read.ReadLabeledData(preprocessing='none')
    testData, testLabels = testReader.getLabeledData()
    
    pca = PCA(pcaDim)
    testData = pca.fit_transform(testData)
    
    X, y = unpairLists(shuffle(pairLists(testData, convertToBinary(testLabels))))

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
    
    unpaired = unpairLists(X_train)
    svc_model = SVC(kernel='linear')
    # svc_model = SVC(kernel='rbf', C=c, gamma=g)
    svc_model.fit(X_train, y_train)
    
    if graph:
        plt.figure(figsize=(10, 8), dpi=300)
        # Plotting our two-features-space
        sns.scatterplot(x=unpaired[0], 
                        y=unpaired[1], 
                        hue=y_train, 
                        s=50);
        plt.scatter()
        # Constructing a hyperplane using a formula.
        w = svc_model.coef_[0]           # w consists of 2 elements
        b = svc_model.intercept_[0]      # b consists of 1 element
        x_points = np.linspace(-1000, 1000)    # generating x-points from -n to n
        y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
        
        # Plotting a red hyperplane
        plt.plot(x_points, y_points, c='r')
        
        # Encircle support vectors
        plt.scatter(svc_model.support_vectors_[:, 0],
                    svc_model.support_vectors_[:, 1], 
                    s=100, 
                    facecolors='none', 
                    edgecolors='k', 
                    alpha=.5)
        # Step 2 (unit-vector):
        w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))
        # Step 3 (margin):
        margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))
        # Step 4 (calculate points of the margin lines):
        decision_boundary_points = np.array(list(zip(x_points, y_points)))
        points_of_line_above = decision_boundary_points + w_hat * margin
        points_of_line_below = decision_boundary_points - w_hat * margin
        # Plot margin lines
        # Blue margin line above
        plt.plot(points_of_line_above[:, 0], 
                 points_of_line_above[:, 1], 
                 'r--', 
                 linewidth=2)
        # Green margin line below
        plt.plot(points_of_line_below[:, 0], 
                 points_of_line_below[:, 1], 
                 'r--',
                 linewidth=2)
    
    
    y_pred = svc_model.predict(X_test)
    return [accuracy_score(y_test,y_pred), precision_score(y_test,y_pred), recall_score(y_test,y_pred), f1_score(y_test,y_pred)]

iteration(2, 1, 1, graph=True)


gammas = [10**-9, 10**-8, 10**-7, 10**-6, 10**-5]
c_values = [1, 10, 100, 1000, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10]
# for pcaDim in [2]:
    
    # accuracies = []
    # f1s = []
    # x = []
    # y = []
    
    # for g in gammas:
    #     for c in c_values:
    #         x.append(g)
    #         y.append(c)
    #         report = []
    #         for i in range(25):
    #             report.append(iteration(pcaDim, c, g))
      
    #         accuracies.append(np.average(report[0]))
            
    #         # print(report)
    #         # try:
    #         #     f1s.append(round(np.average(report[3]), 1))
    #         # except:
    #         #     f1s.append(0)
    
        
    # plt.figure(dpi=300)
    # plt.scatter(x, y, c=accuracies, cmap='BuGn')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("gamma")
    # plt.ylabel("C")
    # plt.title('Accuracy')
    # plt.colorbar()
    # plt.show()
    
    # print(f1s)
    # plt.figure(dpi=300)
    # plt.scatter(x, y, c=f1s, cmap='Greys')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("gamma")
    # plt.ylabel("C")
    # plt.title('F1, PCA Dimension: ' + str(pcaDim))
    # plt.colorbar()
    # plt.show()
    
    
    # c = 10**5
    # g = 10**-9
    # report = []
    # for i in range(2000):
    #     report.append(iteration(pcaDim, c, g))
    
    # # accuracy, precision, recall, f1
    # report = np.transpose(report)
    
    # fig = plt.figure(dpi=300)
    # #accuracy
    # plt.scatter(report[1], report[2], alpha=0.1)
    # plt.xlabel('Precision')
    # plt.ylabel('Recall')
    # plt.title('PCA Dimension: '+str(pcaDim) + ' Gamma: ' + str(g) + ' C: '+str(c))
    # plt.show()

    
            
# play sound when done
durationsound = 1 #seconds
freqsound = 400 # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(durationsound, freqsound))

# print(report)