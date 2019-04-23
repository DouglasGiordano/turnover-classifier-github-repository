# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:20:15 2019

@author: Douglas Giordano
"""

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

from .PreProcessing import PreProcessing

from sklearn.metrics import matthews_corrcoef

#oversampling
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class ForecastNaveBayes:
    def __init__(self, path):
        self.path = path

    def forecast(self):
        self.gaussianNB = GaussianNB()
        self.gaussianNB.fit(self.forecasts_train, self.class_train)
        self.forecasts = self.gaussianNB.predict(self.forecasts_test)

    def acurracy(self):
        self.acurracy = accuracy_score(self.class_test, self.forecasts)
        print("result nave bayes " + self.path)
        print(self.acurracy)
        self.matthews_corrcoef = matthews_corrcoef(self.class_test, self.forecasts)  
        return self.acurracy

    def cnf_matrix(self):
        cnf_matrix = confusion_matrix(self.class_test, self.forecasts)
        self.cnf_matrix_norm = cnf_matrix
        print(self.cnf_matrix_norm)
        return self.cnf_matrix_norm

    def execute(self):
        pAll = PreProcessing(self.path)
        forecasts_all, classes_all = pAll.process()
        self.forecasts_train, self.forecasts_test, self.class_train, self.class_test = train_test_split(forecasts_all,
                                                                                                        classes_all,
                                                                                                        test_size=0.10)
        self.forecasts_train, self.class_train = BorderlineSMOTE().fit_resample(self.forecasts_train, self.class_train)
        #ros = RandomOverSampler(random_state=0)
        #self.forecasts_train, self.class_train = ros.fit_resample(self.forecasts_train, self.class_train)
        #self.forecasts_train, self.class_train = ADASYN().fit_resample(self.forecasts_train, self.class_train)
        #rus = RandomUnderSampler()
        #self.forecasts_train, self.class_train = rus.fit_resample(self.forecasts_train, self.class_train)
        self.forecast()
        self.acurracy()
        self.cnf_matrix()
        filename = 'finalized_model_nave' + self.path.replace("/", "") + '.sav'
        pickle.dump(self.gaussianNB, open(filename, 'wb'))
