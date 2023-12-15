#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:44:45 2022

@author: soham
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm


class Read:
    def __init__(self, dataPointSize, numExtractions, test, lag, divisions=1, verbose=False):
        self.rootdir = '/Users/soham/Documents/Internships/Summer 2022/pauly/research/project/supervised'
        self.fileset = []
        self.filesetTrain = []
        self.filesetTest = []
        self.divisions = divisions
        self.dataPointSize = dataPointSize
        self.extractions = numExtractions
        self.excel_data = pd.read_excel('list-dat.xlsx')
        self.data = pd.DataFrame(self.excel_data, columns=['Date', 'Exam', 'Series', 'IQ Score'])
        self.verbose = verbose
        self.test = test
        self.lag = lag

    def updateFileSet(self):
        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                if 'RESPData_cones' in path and 'backup' not in path:
                    self.fileset.append(path)
        self.splitFileSet()

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
        df = pd.DataFrame(data).squeeze()
        return df
    
    def split(self, lst, n):
        return [lst[i::n] for i in range(n)]

    def vectorize(self, num):
        vec = []
        for i in range(1,6):
            if num == i:
                vec.append(1)
            else:
                vec.append(0)
        return vec

    def booleanize(self, num):
        if num > 4:
            return [1, 0]
        else:
            return [0, 1]
        
    def splitFileSet(self):
        indicies = random.sample(range(0, 30), self.test)
        self.filesetTest = [self.fileset[i] for i in indicies]
        self.filesetTrain = list(set(self.fileset) - set(self.filesetTest))
        
    
    def rejectOutliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]
            
    def getDataFromRawFile(self, file, ind):
        x = []
        y = []
        
        if 'Ser' + str(int(self.data['Series'][ind])) in file and 'Ex' + str(self.data['Exam'][ind]) in file:
            # if self.verbose:
            #     print("exam  " +str(self.data['Exam'][ind]))
            #     print("series  " +str(int(self.data['Series'][ind])))
            #     print("file name   " +file)
                
            fileDat = self.dataFromFileName(file)
            imgQualScore = self.data['IQ Score'][ind]
            #sm.graphics.tsa.plot_acf(fileDat, lags=self.lag, title=("Image #" + str(ind) + " - Quality: " + str(imgQualScore)))
            
            extractions = []
            for i in range(self.extractions):
                start = random.randint(750, len(fileDat) - self.dataPointSize - 1)
                stop = start + self.dataPointSize
                extractions.append(fileDat[start:stop])
    
            for e in extractions:                   
                splitDat = self.split(e, self.divisions)
                
                
                for i in range(self.divisions):
                    # avg = sum(splitDat[i]) / len(splitDat[i])
                    # shifted = [(point - avg) for point in splitDat[i]]
                    # freq = np.fft.fft(shifted)
                    # x = np.abs(freq)
                    x.append(sm.tsa.stattools.acf(splitDat[i], nlags=self.lag))
                    # x = shifted
                    y.append(self.booleanize(imgQualScore))
                    
                    if self.verbose:
                        plt.plot(splitDat[i])
                        plt.title(str(ind) + ' Raw - Image Quality: ' + str(imgQualScore))
                        plt.show()        
                        
                        plt.plot(x)
                        plt.title(str(ind) + ' autocor. func. - Image Quality: ' + str(imgQualScore))
                        plt.show()    
                        
                        # plt.plot(shifted)
                        # plt.title(str(ind) + ' Raw - Image Quality: ' + str(imgQualScore))
                        # plt.show()
        
                        # phase = np.angle(freq)
                        # plt.plot(phase)
                        # plt.title(str(ind) + ' Phase - Image Quality: ' + str(imgQualScore))
                        # plt.show()
        
                        # plt.title(str(ind) +' Magnitude - Image Quality: ' + str(imgQualScore))
                        # plt.plot(np.abs(freq))
                        # plt.show()
                    
        return (x, y)
            
    
    def getFormattedData(self):
        XTrain = []
        yTrain = []
        XTest = []
        yTest = []
        
        if len(self.fileset) == 0:
            self.updateFileSet()
        
        if self.verbose:
            print(self.fileset)
            print("Total training data points: ", len(self.filesetTrain))
            print("Total testing data points: ", len(self.filesetTest))

        
        for ind in self.data.index:
            if not pd.isna(self.data['Series'][ind]):
                for file in self.filesetTrain:
                    x, y = self.getDataFromRawFile(file, ind)
                    if len(x) != 0 and len(y) != 0:
                        XTrain.extend(x)
                        yTrain.extend(y)
                for file in self.filesetTest:
                    x, y = self.getDataFromRawFile(file, ind)
                    if len(x) != 0 and len(y) != 0:
                        XTest.extend(x)
                        yTest.extend(y)
        
        XTrain = np.array(XTrain, dtype=float)
        yTrain = np.array(yTrain, dtype=float)
        XTest = np.array(XTest, dtype=float)
        yTest = np.array(yTest, dtype=float)
            
        return (XTrain, XTest, yTrain, yTest)


def readingTest():
    reader = Read()
    dat = reader.getFormattedData()
    print(dat[0])
    print(dat[1])

        