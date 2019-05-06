# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:52:36 2019

@author: Dougl
"""
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np

class StatisticModel():
    def __init__(self, result_classes, test_classes, path):
        self.result_classes = result_classes
        self.test_classes = test_classes
        self.acurracy = 0
        self.precision = 0
        self.recall = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.mcc = 0
        self.path = path
        
        
    def getStatistic(self):
        self.acurracy = accuracy_score(self.test_classes, self.result_classes)
        self.mcc = matthews_corrcoef(self.test_classes, self.result_classes)
        self.matrix = confusion_matrix(self.test_classes, self.result_classes)
        self.tp = self.matrix[0,0]
        self.tn = self.matrix[0,1]
        self.fp = self.matrix[1,1]
        self.fn = self.matrix[1,0]
        #self.acurracy = (self.tp + self.fp) / (self.tp + self.fp + self.tn+ self.fn )
        self.precision = self.tp / (self.tp + self.tn)
        self.recall = self.tp / (self.tp + self.fn)
        data = {'Acurracy': [self.acurracy], 
        'MCC': [self.mcc], 
        'TP': [self.tp], 
        'TN': [self.tn],
        'FP': [self.fp],
        'FN': [self.fn],
        'Precision':[self.precision],
        'Recall':[self.recall]}
        print(data)
        statisticdf = pd.DataFrame(data, columns = ['Acurracy', 'MCC', 'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall'])
        statisticdf.to_csv(str(self.path+"statistic.csv"), encoding='utf-8', index=False)
    