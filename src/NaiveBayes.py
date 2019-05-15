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

from sklearn.metrics import matthews_corrcoef
from src.LearningAlgorithm import LearningAlgorithm
#oversampling
#from imblearn.over_sampling import BorderlineSMOTE, ADASYN
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler

class NaiveBayes(LearningAlgorithm):
    
    def __init__(self, data, forecasts, classes, path2):
        LearningAlgorithm.__init__(self,data, forecasts, classes, path2)

    
