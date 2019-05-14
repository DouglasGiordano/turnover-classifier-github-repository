 # -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:01:02 2019

@author: Dougl
"""

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

from src.LearningAlgorithm import LearningAlgorithm
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
class SVM(LearningAlgorithm):

    def __init__(self,data,  forecasts, classes, path, kernel="linear"):
       LearningAlgorithm.__init__(self,data, forecasts, classes, path)
       self.kernel = kernel

    def getModel(self, train_forecasts, train_classes):
        svclassifier = SVC(kernel=self.kernel)
        svclassifier.fit(train_forecasts, train_classes)
        return svclassifier
