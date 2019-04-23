# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:20:15 2019

@author: Douglas Giordano
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class PreProcessing:

    def __init__(self, pathData):
        self.pathData = pathData

    def process(self):
        base = pd.read_csv(self.pathData)
        statistics = base.describe()
        print(statistics)
        # replace nan with zero
        base = base.fillna(0)

        # get significance and number interactions
        classe = base.turnover

        # define classe and predictors
        # forecasters = base.iloc[:, [3,4,5,6,7,8,9,11,12,13,14,15,16,17]].
        forecasters = base.loc[:,
                      ['betweenness', 'degree_in', 'degree_out', 'degree_total', 'closeness', 'eigenvector', 'coreness',
                       'mean_positive', 'mean_negative', 'received_negative', 'received_positive', 'num_interaction',
                       'mean_interval', 'num_active_days']].values
        # forecasters = base.loc[:, ['mean_positive','mean_negative','received_negative','received_positive']].values
        # forecasters = base.loc[:, ['betweenness','degree_in','degree_out','degree_total','closeness','eigenvector','coreness','mean_positive','mean_negative','received_negative','received_positive']].values
        #classe = self.getTurnoverValues(classe)
        # scaler forecasts
        scaler = StandardScaler()
        forecasters = scaler.fit_transform(forecasters)

        return forecasters, classe

    def getTurnoverValues(self, classe):
        # get true or false value for turnover based in significance e number interactions
        significance = pd.read_csv('input/significance5.csv')
        turnover = []
        for value in classe:
            interaction = value[1]
            user_sig = value[0]
            if user_sig <= 0:
                days_no_interaction = value[2]
                if days_no_interaction > 180:
                    turnover.append(True)
                    pass
                else:
                    turnover.append(False)
                    pass
            else:
                number = min(significance.iloc[:, 0].values, key=lambda x: abs(x - interaction))
                sig = significance.loc[significance['obs'] == number]
                sig = sig.iloc[:, 1].values
                if sig <= user_sig:
                    turnover.append(True)
                else:
                    turnover.append(False)
        return turnover
