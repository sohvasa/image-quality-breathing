#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:43:13 2022

@author: soham
"""

# Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
from sklearn.decomposition import PCA

# Set defaults

DATA_POINT_SIZE_DEFAULT = 1000 # the length of each data point
NUM_EXTRACTIONS_DEFAULT = 10 # how many random data points should be obtained from each file
TEST_DEFAULT = 21 # split the unlabeled
LAG_DEFAULT = DATA_POINT_SIZE_DEFAULT # lag for the ACF
DIVISIONS_DEFAULT = 1 # how many pieces should each data point be evenly cut into
VERBOSE_DEFAULT = False
ROOT_DIR_DEFAULT = '/Users/soham/Documents/Internships/Summer 2022/pauly/research/project/unsupervised'
RANDOM_SEED = 0 # for reproducibility
PREPROCESSING_DEFAULT = 'ACF' # whether to use the autocorrelation function
PCA_DEFAULT = 2 # final dimensions of the PCA
UNLABELED_DATASET_DEFAULT = 'respdata3' # respdata2 used prior to August 9
TRAIN_SIZE_DEFAULT = 675

class Read:
    def __init__(self, 
                 dataPointSize=DATA_POINT_SIZE_DEFAULT, 
                 numExtractions=NUM_EXTRACTIONS_DEFAULT, 
                 test=TEST_DEFAULT, 
                 preprocessing=PREPROCESSING_DEFAULT,
                 lag=LAG_DEFAULT, 
                 divisions=DIVISIONS_DEFAULT, 
                 verbose=VERBOSE_DEFAULT,
                 pcaDim=PCA_DEFAULT,
                 trainSize=TRAIN_SIZE_DEFAULT):
        """
        Tool for reading and organizing the unlabeled data.

        Parameters
        ----------
        dataPointSize : TYPE, optional
            DESCRIPTION. The default is DATA_POINT_SIZE_DEFAULT.
        numExtractions : TYPE, optional
            DESCRIPTION. The default is NUM_EXTRACTIONS_DEFAULT.
        test : TYPE, optional
            DESCRIPTION. The default is TEST_DEFAULT.
        preprocessing : TYPE, optional
            DESCRIPTION. The default is PREPROCESSING_DEFAULT.
        lag : TYPE, optional
            DESCRIPTION. The default is LAG_DEFAULT.
        divisions : TYPE, optional
            DESCRIPTION. The default is DIVISIONS_DEFAULT.
        verbose : TYPE, optional
            DESCRIPTION. The default is VERBOSE_DEFAULT.
        pcaDim : TYPE, optional
            DESCRIPTION. The default is PCA_DEFAULT.

        Returns
        -------
        None.

        """
        
        self.rootdir = ROOT_DIR_DEFAULT
        self.fileset = []
        self.filesetTrain = []
        self.filesetTest = []
        self.divisions = divisions
        self.dataPointSize = dataPointSize
        self.extractions = numExtractions
        self.verbose = verbose
        self.test = test
        self.lag = lag
        self.preprocessing = preprocessing
        self.pcaDim = pcaDim
        self.trainSize = trainSize
        

    def updateFileSet(self):
        """
        Adds all the file paths to the fileset attribute.

        Returns
        -------
        None.

        """
        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                if 'RESPData_cones' in path and UNLABELED_DATASET_DEFAULT in path:
                    self.fileset.append(path)
        self.splitFileSet()


    def dataFromFileName(self, fi):
        """
        Converts a file name into a data frame.

        Parameters
        ----------
        fi : str
            The file name.

        Returns
        -------
        df : pd.DataFrame
            The file data.

        """
        data = []
        with open(fi) as f:
            lines = f.readlines()
            for line in lines:
                strVal = line[1:-2]
                if strVal != '':
                    data.append(int(strVal))
                else:
                    data.append(0)
        df = pd.DataFrame(data).squeeze()
        return df
    
    
    def split(self, lst, n):
        """
        Splits a list into n pieces.

        Parameters
        ----------
        lst : list
            Original list.
        n : int
            Cuts.

        Returns
        -------
        list
            The list of n pieces.

        """
        return [lst[i::n] for i in range(n)]
        
    
    def splitFileSet(self):
        indicies = random.sample(range(100), self.test)
        self.filesetTest = [self.fileset[i] for i in indicies]
        self.filesetTrain = random.sample(list(set(self.fileset) - set(self.filesetTest)), self.trainSize)


    def getDataFromRawFile(self, file, extracts):
        useACF = 'ACF' in self.preprocessing
        
        x = []
        fileDat = self.dataFromFileName(file)
            
        if len(fileDat) <= self.dataPointSize:
            return x
            
        try:
            # Randomly splice list into n "extractions"
            extractions = []
            for i in range(extracts):
                start = random.randint(750, len(fileDat) - self.dataPointSize - 1)
                stop = start + self.dataPointSize
                extractions.append(fileDat[start:stop])
        
            for e in extractions:  
                # Split the extraction into n divisions                 
                splitDat = self.split(e, self.divisions)
                
                for i in range(self.divisions):
                    if useACF:
                        if 0 in splitDat[i]:
                            print('alert! Zero value in split Data!!')
                        dataPoint = sm.tsa.stattools.acf(splitDat[i], nlags=self.lag)
                        if dataPoint[0] == 1:
                            x.append(dataPoint)
                        else:
                            break
                    else:
                        x.append(splitDat[i])
        except:
            pass
        
        # Return list of autocor. functions
        return x
    
    
    def getFormattedData(self):
        random.seed(RANDOM_SEED)
        usePCA = 'PCA' in self.preprocessing
        
        train = []
        test = []
        
        # update the file set if empty
        if len(self.fileset) == 0:
            self.updateFileSet()
        
        if self.verbose:
            print("Total data points: ", len(self.filesetTrain))

        # add data from each file designated for training
        # print('Reading unlabeled training data.')
        for file in self.filesetTrain:
            x = self.getDataFromRawFile(file, self.extractions)
            if len(x) != 0:
                train.extend(x)
                
        # add data from each file designated for testing
        for file in self.filesetTest:
            x = self.getDataFromRawFile(file, 1)
            if len(x) != 0:
                test.extend(x)
    
        # return tuple of train and test data
        train = np.array(train, dtype=float)
        test = np.array(test, dtype=float)
        
        if usePCA:
            pca = PCA(self.pcaDim)
            pcaTest = PCA(self.pcaDim)
            return (pca.fit_transform(train), pcaTest.fit_transform(test))
        
        return (train, test)



class ReadLabeledData:
    def __init__(self, dataPointSize=DATA_POINT_SIZE_DEFAULT, preprocessing=PREPROCESSING_DEFAULT, lag=LAG_DEFAULT, 
                 divisions=1, numExtractions=1, verbose=VERBOSE_DEFAULT, pcaDim=PCA_DEFAULT):
        self.rootdir = ROOT_DIR_DEFAULT
        self.fileset = []
        self.divisions = divisions
        self.dataPointSize = dataPointSize
        self.extractions = numExtractions
        self.excel_data = pd.read_excel('list-dat.xlsx')
        self.data = pd.DataFrame(self.excel_data, columns=['Date', 'Exam', 'Series', 'IQ Score'])
        self.verbose = verbose
        self.lag = lag
        self.preprocessing = preprocessing
        self.pcaDim = pcaDim

    def updateFileSet(self):
        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                if 'RESPData_cones' in path and ('respdata1' in path or 'respdata4' in path):
                    self.fileset.append(path)

    def dataFromFileName(self, fi):
        data = []
        with open(fi) as f:
            lines = f.readlines()
            for line in lines:
                strVal = line[1:-2]
                if strVal != '':
                    data.append(int(strVal))
                else:
                    data.append(0)
        return pd.DataFrame(data).squeeze()
    
    def split(self, lst, n):
        return [lst[i::n] for i in range(n)]

    def booleanize(self, num):
        if num > 3:
            return [1, 0]
        else:
            return [0, 1]
            
    def getDataFromRawFile(self, file, ind):
        useACF = 'ACF' in self.preprocessing
        
        
        x = []
        y = []
        
        if 'Ser' + str(int(self.data['Series'][ind])) in file and 'Ex' + str(self.data['Exam'][ind]) in file:
            if self.verbose:
                print("exam  " +str(self.data['Exam'][ind]))
                print("series  " +str(int(self.data['Series'][ind])))
                print("file name   " +file)
                
            fileDat = np.nan_to_num(self.dataFromFileName(file))
            
                
            if len(fileDat) <= self.dataPointSize + 750:
                return (x, y)
            
            imgQualScore = self.data['IQ Score'][ind]
            
            extractions = []
            for i in range(self.extractions):
                start = random.randint(750, len(fileDat) - self.dataPointSize - 1)
                stop = start + self.dataPointSize
                extractions.append(fileDat[start:stop])
    
            for e in extractions:                   
                splitDat = self.split(e, self.divisions)
                
                for i in range(self.divisions):
                    if useACF:
                        x.append(sm.tsa.stattools.acf(splitDat[i], nlags=self.lag))
                    else:
                        x.append(splitDat[i])
                        
                    y.append(imgQualScore)
                    
                    if self.verbose:
                        plt.plot(splitDat[i])
                        plt.title(str(ind) + ' Raw - Image Quality: ' + str(imgQualScore))
                        plt.show()        
                        
                        plt.plot(x)
                        plt.title(str(ind) + ' autocor. func. - Image Quality: ' + str(imgQualScore))
                        plt.show()    
                    
        return (x, y)
            
    
    def getLabeledData(self):
        usePCA = 'PCA' in self.preprocessing
        
        data = []
        labels = []
        
        if len(self.fileset) == 0:
            self.updateFileSet()
        
        if self.verbose:
            print(self.fileset)
            print("Total training data points: ", len(self.filesetTrain))
            print("Total testing data points: ", len(self.filesetTest))
        
        # print('Reading labeled data.')
        for ind in self.data.index:
            if not pd.isna(self.data['Series'][ind]):
                for file in self.fileset:
                    x, y = self.getDataFromRawFile(file, ind)
                    if len(x) != 0 and len(y) != 0:
                        data.extend(x)
                        labels.extend(y)
    
        data = np.array(data, dtype=float)
        labels = np.array(labels, dtype=float)
                
        if usePCA:
            pca = PCA(self.pcaDim)
            return (pca.fit_transform(data), labels)
        
        return (data, labels)



def readingTest1(debug, graph):
    reader = Read()
    train, test = reader.getFormattedData()
    
    print("Train data set size: ", len(train))
    print("Test data set size: ", len(test))
    if not debug and not graph:
        return 'done'
    
    t = 1
    for dataPoint in test:
        if debug:
            print("Point " + str(t) + "  - Length: ", len(dataPoint))
            print(dataPoint[:20])
            print("\n")
            print("\n")
        if graph:
            plt.plot(dataPoint)
            plt.title("Test Point #" + str(t))
            plt.show()
            t += 1
        
# readingTest1(False, True)

def readingTest2(debug, graph):
    reader = ReadLabeledData(preprocessing='none')
    data, labels = reader.getLabeledData()
    
    print("Number of data points: ", len(data))
    print("Number of labels: ", len(labels))
    
    if not debug and not graph:
        return 'done'
    
    t = 0
    while t < len(data):
        dataPoint = data[t]
        label = labels[t]
        
        if debug:
            print("Point " + str(t) + "  - Length: ", len(dataPoint))
            print("Label: ", label)
            print(dataPoint[:20])
            print("\n")
            print("\n")
            
        if graph:
            plt.plot(dataPoint)
            plt.title("Test Point #" + str(t + 1) + " - Img Qual. = " + str(label))
            plt.show()
            t += 1

#readingTest2(False, True)
        