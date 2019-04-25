# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:20:15 2019

@author: Douglas Giordano
"""

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

#from .PreProcessing import PreProcessing
import pandas as pd
import numpy as np

from sklearn.metrics import matthews_corrcoef

#cross-validation
from numpy import array
from sklearn.model_selection import KFold

#oversampling
#from imblearn.over_sampling import BorderlineSMOTE, ADASYN
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler

class LearningAlgorithm:
    def __init__(self, forecasts, classes, path):
        self.forecasts = forecasts
        self.classes = classes
        self.path = path

    def getModel(self, train_forecasts, train_classes):
        gaussianNB = GaussianNB()
        gaussianNB.fit(train_forecasts, train_classes)
        return gaussianNB

    def forecast(self, train_forecasts, train_classes, test_forecasts, test_classes):
        model = self.getModel(train_forecasts, train_classes)
        filename = '/input/'+path+'/model.sav'
        pickle.dump(model, open(filename, 'wb'))
        return model.predict(test_forecasts)

    def reportResult(self, result, test_forecasts, test_classes):
        acurracy = accuracy_score(test_classes, result)
        mcc = matthews_corrcoef(test_classes, result)
        matrix = self.cnf_matrix(result, test_classes)
        print("Acurracy: %s" % acurracy)
        print("MCC: %s" % mcc)
        print("Matrix Confusion: %s" % matrix)

    def cnf_matrix(self, result, test_classes):
        cnf_matrix = confusion_matrix(test_classes, result)
        cnf_matrix_norm = cnf_matrix
        print(cnf_matrix_norm)
        return cnf_matrix_norm

    def kFold(self):
        # prepare cross validation
        kfold = KFold(10, True, 1)
        # enumerate splits
        for train, test in kfold.split(self.forecasts):
            print('train: %s, test: %s' % (self.forecasts[train].size, self.forecasts[test].size))
            print('class train: %s, test: %s' % (self.classes[train].size, self.classes[test].size))
            print('index train: %s, index test: %s' % (train, test))
            train_forecasts = self.forecasts[train]
            test_forecasts = self.forecasts[test]
            train_classes = self.classes[train]
            test_classes = self.classes[test]
            self.execute(train_forecasts, train_classes, test_forecasts, test_classes)
            
            

    def execute(self, train_forecasts, train_classes, test_forecasts, test_classes):
        #train_forecasts, train_classes = BorderlineSMOTE().fit_resample(train_forecasts, train_classes)
        #ros = RandomOverSampler(random_state=0)
        #self.forecasts_train, self.class_train = ros.fit_resample(self.forecasts_train, self.class_train)
        #self.forecasts_train, self.class_train = ADASYN().fit_resample(self.forecasts_train, self.class_train)
        #rus = RandomUnderSampler()
        #self.forecasts_train, self.class_train = rus.fit_resample(self.forecasts_train, self.class_train)
        result = self.forecast(train_forecasts, train_classes, test_forecasts, test_classes)
        self.reportResult(result, test_forecasts, test_classes)