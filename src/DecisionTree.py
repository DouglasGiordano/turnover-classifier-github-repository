# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:02:10 2019

@author: Dougl
"""

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

#from .PreProcessing import PreProcessing
from src.LearningAlgorithm import LearningAlgorithm
#oversampling
#from imblearn.over_sampling import BorderlineSMOTE, ADASYN
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler

class DecisionTree(LearningAlgorithm):
    
    def getModel(self, train_forecasts, train_classes):
        decisionTree = DecisionTreeClassifier()
        decisionTree.fit(train_forecasts, train_classes)
        return decisionTree
    
    def __init__(self,data, forecasts, classes, path2):
        super().__init__(data, forecasts, classes, path2)
