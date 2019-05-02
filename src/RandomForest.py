# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 01:45:14 2019

@author: Dougl
"""


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

from src.LearningAlgorithm import LearningAlgorithm

from src.PreProcessing import PreProcessing
#oversampling
#from imblearn.over_sampling import BorderlineSMOTE, ADASYN
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler

class RandomForest(LearningAlgorithm):

    def __init__(self,data,  forecasts, classes, path2):
        super().__init__(data, forecasts, classes, path2)
        
    def getModel(self, train_forecasts, train_classes):
        clf = RandomForestClassifier()
        clf.fit(train_forecasts, train_classes)
        return clf