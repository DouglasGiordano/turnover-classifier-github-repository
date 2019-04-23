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

from .PreProcessing import PreProcessing

#oversampling
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
class AdaBoost:
    def __init__(self, path, base_estimator):
        self.path = path
        self.base_estimator = base_estimator

    def forecast(self):
        if self.base_estimator == "svc":
            svc = SVC(probability=True, kernel='linear')
            self.abc = AdaBoostClassifier(n_estimators=25, base_estimator=svc, learning_rate=0.8)
        else:
            self.abc = AdaBoostClassifier(n_estimators=25, learning_rate=0.8)
        self.model = self.abc.fit(self.forecasts_train, self.class_train)
        self.forecasts = self.model.predict(self.forecasts_test)

    def acurracy(self):
        self.acurracy = accuracy_score(self.class_test, self.forecasts)
        print("result decision adaboost "+self.base_estimator+" " + self.path)
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
                                                                                                        test_size=0.10)
        self.forecasts_train, self.class_train = BorderlineSMOTE().fit_resample(self.forecasts_train, self.class_train)
        #self.forecasts_train, self.class_train = ADASYN().fit_resample(self.forecasts_train, self.class_train)
        self.forecast()
        self.acurracy()
        self.cnf_matrix()
        filename = 'finalized_model_adaboost' + self.path.replace("/", "") + self.base_estimator+'-svc.sav'
        pickle.dump(self.model, open(filename, 'wb'))
