#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:12:24 2022

@author: soham
"""
# import libraries
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np
from scipy.stats import sem


# import module
import read

# read data 
def readData(length, pcaDim, trainSize, rawOnly=False):
    data = []
    
    # instantiate data-reading classes
    trainReader = read.Read(dataPointSize=length, test=0, preprocessing='none', trainSize=trainSize)
    testReader = read.ReadLabeledData(dataPointSize=length, preprocessing='none')
    
    # get train data and the blank test data as it's unlabeled
    trainData, null = trainReader.getFormattedData()
    testData, testLabels = testReader.getLabeledData()
    
    # add first data tuple to returned list
    data.append((trainData, testData, testLabels))
    
    if not rawOnly:
        # calculate pca
        pca = PCA(pcaDim)
        trainDataPCA = pca.fit_transform(trainData)
        pca = PCA(pcaDim)
        testDataPCA = pca.fit_transform(testData)
        # add second data tuple of PCA data to returned list
        data.append((trainDataPCA, testDataPCA))
        
        # get preprocessed acf data and repeat the above
        trainReaderACF = read.Read(dataPointSize=length, test=0)
        testReaderACF = read.ReadLabeledData(dataPointSize=length)
            
        trainDataACF, null = trainReaderACF.getFormattedData()
        testDataACF, testLabelsACF = testReaderACF.getLabeledData()
        
        data.append((trainDataACF, testDataACF, testLabelsACF))
        
        pca = PCA(pcaDim)
        trainDataACFPCA = pca.fit_transform(trainDataACF)
        pca = PCA(pcaDim)
        testDataACFPCA = pca.fit_transform(testDataACF)
        
        data.append((trainDataACFPCA, testDataACFPCA))
    
    return data


def graphResults(cluster, values, colors, index):
    plt.scatter(cluster, values, c=colors, cmap='Reds')
    plt.title('Prediction Graph, # of Clusters: ' + str(index))
    plt.xlabel('cluster')
    plt.ylabel('image quality index')
    plt.show()


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



class Predictor:
    def __init__(self, predictions0, clusters0, predictions, clusters):
        self.predictions = predictions
        self.clusters = clusters
        
        clusterNames = list(set(clusters))
        clusterPoints = {}
        for name in clusterNames:
            clusterPoints[name] = []
        
        for i in range(len(predictions)):
            clusterPoints[clusters[i]].append(predictions[i])
        for key in clusterPoints.keys():
            clusterPoints[key] = self.binarifyList(clusterPoints[key])
            
        self.clusterPoints = clusterPoints 
        
        # kludgey solution
        
        clusterNames0 = list(set(clusters0))
        clusterPoints0 = {}
        for name in clusterNames0:
            clusterPoints0[name] = []
        
        for i in range(len(predictions0)):
            clusterPoints0[clusters0[i]].append(predictions0[i])
        for key in clusterPoints0.keys():
            clusterPoints0[key] = self.binarifyList(clusterPoints0[key])
            
        self.clusterPoints0 = clusterPoints0 
            
    def binarifyList(self, l):
        good = 0
        trash = 0
        for e in range(1, 6):
            if e >= 4:
                good += l.count(e)
            else:
                trash += l.count(e)
        return [good, trash]
        
    def singleTest(self):
        actual = list(self.clusterPoints.values())[0]
        isHighQual = actual[0] > actual[1]
        
        prediction = None
        
        try:
            prediction = self.clusterPoints0[list(self.clusterPoints.keys())[0]]
        except KeyError:
            if isHighQual:
                prediction = [0, 1]
            else:
                prediction = [1, 0]
            
        predHighQual = prediction[0] > prediction[1]
        
        
        # correctly classified as high quality
        if isHighQual and predHighQual:
            return 1
        # correctly classified as low quality
        elif not isHighQual and not predHighQual:
            return 2
        # incorrectly classified as high quality
        elif not isHighQual and predHighQual:
            return 3
        # incorrectly classified as low quality
        else:
            return 4

        
    def getPerformance(self):
        trueHighQual = 0
        trueLowQual = 0
        highCorrect = 0
        lowCorrect = 0
        
        clusterPoints = pairLists(list(self.clusterPoints.keys()), list(self.clusterPoints.values()))
        for cluster, pair in clusterPoints:
            trueHighQual += pair[0]
            trueLowQual += pair[1]
            
            clusterValues = None
            try:
                clusterValues = self.clusterPoints0[cluster]
            except KeyError:
                pass
            
            if clusterValues == None:
                continue
            # high
            elif clusterValues[0] > clusterValues[1]:
                highCorrect += pair[0]
            # low
            else:
                lowCorrect += pair[1]
            
        specificity = highCorrect / trueHighQual
        sensitivity = lowCorrect / trueLowQual
        accuracy = (highCorrect + lowCorrect) / (trueHighQual + trueLowQual)
        return (specificity, sensitivity, accuracy)


    def predict(self, clusterName):
        data = self.clusterPoints[clusterName]
        if data[0] > data[1]:
            return 'high quality'
        else:
            return 'low quality'
        

def iterate(start, end, trainData, testData, testLabels, graph=False):
    labeledTraining = len(testData) - 1
    
    sensitivities = []
    specificities = []
    accuracies = []

    # print('Fitting and evaluating K-means clustering models.')
    for index in range(start, end + 1):
        
        model = KMeans(index, random_state=read.RANDOM_SEED)
        model.fit(trainData)
        
        testCombinedInitial = pairLists(testData, testLabels)
        random.shuffle(testCombinedInitial)
        
        trueHighQual = 0
        trueLowQual = 0
        highCorrect = 0
        lowCorrect = 0
        
        for i in range(len(testCombinedInitial)):
            testCombined = testCombinedInitial.copy()
            
            test1, test2 = sklearn.model_selection.train_test_split(testCombined, 
                                                                    train_size=labeledTraining, 
                                                                    test_size=(len(testData)-labeledTraining))
            
            test1, test1Labels = unpairLists(test1)
            test2, test2Labels = unpairLists(test2)
    
            
            clusterPredictions = model.predict(test1)
            assigningPredictions = []
            for c in clusterPredictions:
                assigningPredictions.append(chr(ord('@')+(c+1)))
            
            numPredictions = model.predict(test2)
            predictions = []
            for num in numPredictions:
                predictions.append(chr(ord('@')+(num+1)))
            
            
            comparison = []
            for i in range(len(test2Labels)):
                comparison.append((predictions[i], test2Labels[i]))
            
            predictor = Predictor(test1Labels, assigningPredictions, test2Labels, predictions)
            result = predictor.singleTest()
            if result == 1:
                trueHighQual += 1
                highCorrect += 1
            elif result == 2:
                trueLowQual += 1
                lowCorrect += 1
            elif result == 3:
                trueLowQual += 1
            else:
                trueHighQual += 1
            
            #performance = predictor.getPerformance()
        specificities.append(highCorrect / trueHighQual)
        sensitivities.append(lowCorrect / trueLowQual)
        accuracies.append((highCorrect + lowCorrect) / (trueHighQual + trueLowQual))  
           
        # specificities.append(performance[0])
        # sensitivities.append(performance[1])
        # accuracies.append(performance[2])
            
        predictions = []
        values = []
        cluster = []
        colors = []
        
        for pair in comparison:
            if pair not in predictions:
                cluster.append(pair[0])
                values.append(pair[1])
                colors.append(comparison.count(pair))
                
        if graph:
            graphResults(cluster, values, colors, index)
    return (specificities, sensitivities, accuracies)
    
def avgLists(l, mode='sem'):
    avgList = []
    err = []
    for i in range(len(l[0])):
        transposedVec = []
        total = 0
        for subL in l:
            total += subL[i]
            transposedVec.append(subL[i])
        avgList.append(total / len(l))
        if mode == 'sem':
            err.append(sem(transposedVec))
        elif mode == 'std':
            err.append(np.std(transposedVec))

    return (avgList, err)
        
def getCol(arr, col):
    column = []
    for row in arr:
        column.append(row[col])
    return column 

def run(k0, kf, staLen, stoLen, steLen, pcaDim=2, trials=100, acf=False, trainSize=300):
    
    for dataPointLen in range(staLen, stoLen + steLen, steLen):
        
        accuracies = []
        
        for i in tqdm(range(trials)):
            
            data = readData(dataPointLen, pcaDim, trainSize, rawOnly=(not acf))
            trainData, testData, testLabels = data[0]
            # trainDataPCA, testDataPCA = data[1]
            if acf:
                trainDataACF, testDataACF, testLabelsACF = data[2]
            # trainDataACFPCA, testDataACFPCA = data[3]
            
            #print("Length of training data set:  ", len(trainData))
            #print("Length of test data set:  ", len(testData)) # 26 good 35 bad as of Aug 9
            
            spec, sens, accu = iterate(k0, kf, trainData, testData, testLabels)
            if acf:
                specACF, sensACF, accuACF = iterate(k0, kf, trainDataACF, testDataACF, testLabelsACF)
            
            accuracies.append(accu)
            
        
        
        accuracies1, errors = avgLists(accuracies, mode='sem')    
        print('standard error of the mean: ')
        print(errors)
        
        accuracies2, dev = avgLists(accuracies, mode='std')
        print('standard deviation: ')
        print(dev)
        
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=300)
        ax1.hist(getCol(accuracies, 0))
        ax1.set_title('2 Clusters') 
        ax2.hist(getCol(accuracies, 1))
        ax2.set_title('3 Clusters')
        ax3.hist(getCol(accuracies, 2))
        ax3.set_title('4 Clusters')
        ax4.hist(getCol(accuracies, 3))
        ax4.set_title('5 Clusters')
        
        fig.tight_layout()
        plt.show()
        
        if acf:
            # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(5, 4), dpi=300)
            
            
            
            ax1.plot(range(k0, kf + 1), accu, label='Accuracy')
            ax1.plot(range(k0, kf + 1), spec, label='Specificity')
            ax1.plot(range(k0, kf + 1), sens, label='Sensitivity')
            ax1.set_title('No preprocessing')
            ax1.set(xlabel='# of Clusters', ylabel='Performance - Best: '+str(round(max(accu), 3)))
            ax1.set_ylim([0, 1])
           # plt.errorbar(x, y, yerr = y_error,fmt='o',ecolor = 'cyan',color='black')
    
            ax1.legend(fontsize=7)
            
            ax2.plot(range(k0, kf + 1), accuACF, label='Accuracy')
            ax2.plot(range(k0, kf + 1), specACF, label='Specificity')
            ax2.plot(range(k0, kf + 1), sensACF, label='Sensitivity')
            ax2.set_title('ACF preprocessing')
            ax2.set(xlabel='# of Clusters', ylabel='Performance - Best: '+str(round(max(accuACF), 3)))
            ax2.set_ylim([0, 1])
            
            # spec, sens, accu = iterate(k0, kf, trainDataPCA, testDataPCA, testLabels)
            # specACF, sensACF, accuACF = iterate(k0, kf, trainDataACFPCA, testDataACFPCA, testLabelsACF)
            
            fig.suptitle('PCA Dim: '+ str(pcaDim) + ' , Data Points of Length ' + str(dataPointLen))
            
            # ax3.plot(range(k0, kf + 1), accu, label='Accuracy')
            # ax3.plot(range(k0, kf + 1), spec, label='Specificity')
            # ax3.plot(range(k0, kf + 1), sens, label='Sensitivity')
            # ax3.set_title('PCA preprocessing')
            # ax3.set(xlabel='# of Clusters', ylabel='Performance - Best: '+str(round(max(accu), 3)))
            # ax3.legend(fontsize=5)
            
            # ax4.plot(range(k0, kf + 1), accuACF, label='Accuracy')
            # ax4.plot(range(k0, kf + 1), specACF, label='Specificity')
            # ax4.plot(range(k0, kf + 1), sensACF, label='Sensitivity')
            # ax4.set_title('PCA + ACF preprocessing')
            # ax4.set(xlabel='# of Clusters', ylabel='Performance - Best: '+str(round(max(accuACF), 3)))
            
            fig.tight_layout()
            plt.show()
        
        
        fig = plt.figure(dpi=300)
        x = range(k0, kf + 1)
        plt.plot(x, accuracies1, label='Accuracy (Standard Error)')
        plt.errorbar(x, accuracies1, yerr=errors,fmt='.k',ecolor = 'cyan',color='black')
        plt.title('Accuracy vs Clusters')
        plt.xlabel('# of Clusters')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.show()
        
        fig = plt.figure(dpi=300)
        x = range(k0, kf + 1)
        plt.plot(x, accuracies2, label='Accuracy (Standard Deviation)')
        plt.errorbar(x, accuracies2, yerr=dev,fmt='.k',ecolor = 'cyan',color='black')
        plt.title('Accuracy vs Clusters')
        plt.xlabel('# of Clusters')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.show()

    
if __name__ == '__main__':
    for pcaDim in range(2, 3):
        run(2, 16, 1000, 1500, 500, pcaDim=pcaDim)
        
    # best performances (raw data)
    bp = [0.82, 0.787, 0.869, 0.902, 0.885, 0.902, 0.852, 0.883]
    for i in range(len(bp)):
        bp[i] *= 100
        
    # segment lengths
    sl = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    
    fig = plt.figure(dpi=300)
    
    
    plt.plot(sl, bp)
    plt.title('Accuracy vs Segment Length')
    plt.xlabel('Segment Length')
    plt.ylabel('Best Accuracy (%)')
    plt.ylim([0, 100])
    plt.show()
    
    # best performances (raw data)
    bp = [0.71, 0.823, 0.903, 0.71, 0.806, 0.871, 0.823, 0.839, 0.887, 0.82]
    for i in range(len(bp)):
        bp[i] *= 100
        
    # segment lengths
    sl = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    
    fig = plt.figure(dpi=300)
    
    
    plt.plot(sl, bp)
    plt.title('Accuracy vs Segment Length')
    plt.xlabel('Segment Length')
    plt.ylabel('Best Accuracy (%)')
    plt.ylim([0, 100])
    plt.show()
