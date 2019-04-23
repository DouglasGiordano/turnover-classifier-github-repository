# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:01:02 2019

@author: Dougl
"""

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

from .PreProcessing import PreProcessing

#oversampling
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
class SVM:

    def __init__(self, path, kernel):
        self.path = path
        self.kernel = kernel

    def forecast(self):
        self.svclassifier = SVC(kernel=self.kernel)
        self.svclassifier.fit(self.forecasts_train, self.class_train)
        self.forecasts = self.svclassifier.predict(self.forecasts_test)

    def acurracy(self):
        self.acurracy = accuracy_score(self.class_test, self.forecasts)
        print("result svm " + self.kernel + "-" + self.path)
        print(self.acurracy)
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
                                                                                                        test_size=0.10,
                                                                                                        random_state=0)
        self.forecasts_train, self.class_train = BorderlineSMOTE().fit_resample(self.forecasts_train, self.class_train)
        self.forecast()
        self.acurracy()
        self.cnf_matrix()
        filename = 'finalized_model_svm' + self.path.replace("/", "") + '-' + self.kernel + '.sav'
        pickle.dump(self.svclassifier, open(filename, 'wb'))
