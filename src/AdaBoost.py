# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:20:15 2019

@author: Douglas Giordano
"""

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

#from .PreProcessing import PreProcessing

#oversampling
#from imblearn.over_sampling import BorderlineSMOTE, ADASYN
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler
class AdaBoost(LearningAlgorithm):
    
    def __init__(self, forecasts, classes, path2, base_estimator, estimators=25, rateL=0.8):
        super().__init__(forecasts, classes, path2)
        self.base_estimator = base_estimator
        self.estimators = estimators
        self.rateL = rateL

    def getModel(self, train_forecasts, train_classes):
        if self.base_estimator == "svc":
            svc = SVC(probability=True, kernel='linear')
            abc = AdaBoostClassifier(n_estimators=self.estimators, base_estimator=svc, learning_rate=self.rateL)
        else:
            abc = AdaBoostClassifier(n_estimators=self.estimators, learning_rate=self.rateL)
        model = abc.fit(train_forecasts, train_classes)        
        return model