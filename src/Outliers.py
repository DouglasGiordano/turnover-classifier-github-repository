# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:30:58 2019

@author: Dougl
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

def create_data(location, name):
    data = pd.read_csv(location)
    #o = Outliers(data, name)
    #o.detectarOutliers()
    data = data.fillna(0)
    return data

def create(): 
    static = create_data("input/dataset-usuario.csv", "static")
    static.drop(static.columns[[0]], axis=1, inplace=True)
    
    staticAll = create_data("input/dataset-usuario_all.csv", "static-all")
    staticAll.drop(staticAll.columns[[0]], axis=1, inplace=True)
    
    last = create_data("input/dataset-usuario_last.csv","last")
    last.drop(last.columns[[0]], axis=1, inplace=True)
    
    lastAll = create_data("input/dataset-usuario_last_all.csv","last-all")
    lastAll.drop(lastAll.columns[[0]], axis=1, inplace=True)
    return static, staticAll, last, lastAll

def createPlot(data, name):
    t = Outliers(data, name)
    t.detectarOutliers()
    

def removeOutlierSentiment(data):
    outlier_sentiment_positive = data[data.mean_positive<=0]
    print("positive = "+str(len(outlier_sentiment_positive.index)))
    data.loc[data.mean_positive<=0].mean_positive = 1
    
    outlier_sentiment_negative = data[data.mean_negative>=0]
    print("negative = "+str(len(outlier_sentiment_negative.index)))
    data.loc[data.mean_negative>=0 ].mean_negative = -1
    
    outlier_sentiment_received_negative = data[data.received_negative>=0]
    print("received negative = "+str(len(outlier_sentiment_received_negative.index)))
    data.loc[data.received_negative>=0].received_negative = -1
    
    outlier_sentiment_received_positive = data[data.received_positive<=0]
    print("received positive = "+str(len(outlier_sentiment_received_positive.index)))
    data.loc[data.received_positive<=0].received_positive = 1
    
    return data
    
def removeOutlier(data, column, limitMax):
    outlier = data[data[column] > limitMax]
    print(column+" = "+str(len(outlier.index)))
    data = data.drop(data[data[column] > limitMax].index)
    return data

def removeOutlierMetricStatic(data):
    data = removeOutlier(data, "betweenness", 15000000)
    data = removeOutlier(data, "closeness", 0.7)
    #degree in não foi removido nenhum
    data = removeOutlier(data, "degree_out", 8000)
    data = removeOutlier(data, "degree_total", 12500)    
    #eigenvetor não foi removido nenhum
    data = removeOutlier(data, "coreness", 8000)
    
    #simples
    data = removeOutlier(data, "num_interaction", 12000)   
    data = removeOutlier(data, "mean_interval", 1000) 
    data = removeOutlier(data, "num_active_days", 7500) 
    return data

def removeOutlierMetricStaticAll(data):
    data = removeOutlier(data, "betweenness", 100000000)
    #data = removeOutlier(data, "closeness", 0.75)
    data = removeOutlier(data, "degree_in", 50000)
    data = removeOutlier(data, "degree_out", 40000)
    data = removeOutlier(data, "degree_total", 100000)    
    #eigenvetor não foi removido nenhum
    data = removeOutlier(data, "coreness", 30000)   
    
    #simples
    data = removeOutlier(data, "num_interaction", 30000)   
    data = removeOutlier(data, "mean_interval", 2500) 
    data = removeOutlier(data, "num_active_days", 7500) 
    return data

def removeOutlierMetricLast(data):
    data = removeOutlier(data, "betweenness", 200000)
    #closeness não foi removido nenhum
    data = removeOutlier(data, "degree_in", 2500)
    data = removeOutlier(data, "degree_out", 1000)
    data = removeOutlier(data, "degree_total", 2500)    
    #eigenvetor não foi removido nenhum
    data = removeOutlier(data, "coreness", 500)    
    
    #simples
    data = removeOutlier(data, "num_interaction", 3000)   
    #data = removeOutlier(data, "mean_interval", 2500) 
    #data = removeOutlier(data, "num_active_days", 7500) 
    return data

def removeOutlierMetricLastAll(data):
    data = removeOutlier(data, "betweenness", 200000)
    #closeness não foi removido nenhum
    data = removeOutlier(data, "degree_in", 2500)
    data = removeOutlier(data, "degree_out", 1000)
    data = removeOutlier(data, "degree_total", 2500)    
    #eigenvetor não foi removido nenhum
    data = removeOutlier(data, "coreness", 500)    
    
    #simples
    data = removeOutlier(data, "num_interaction", 4000)   
    #data = removeOutlier(data, "mean_interval", 2500) 
    #data = removeOutlier(data, "num_active_days", 7500) 
    return data

static, staticAll, last, lastAll = create();

#createPlot(static, "static")
#createPlot(staticAll, "static-all")
#createPlot(last, "last")
#createPlot(lastAll, "last-all")
#sentimentos
print("******************* Detect and remove outliers static")
static = removeOutlierSentiment(static)
static = removeOutlierMetricStatic(static)
static.to_csv("output/dataset-usuario-nooutlier.csv", sep=',',index=False)

print("******************* Detect and remove outliers static all")
staticAll = removeOutlierSentiment(staticAll)
staticAll = removeOutlierMetricStaticAll(staticAll)
staticAll.to_csv("output/dataset-usuario-all-nooutlier.csv", sep=',',index=False)

print("******************* Detect and remove outliers last")
last = removeOutlierSentiment(last)
last = removeOutlierMetricLast(last)
last.to_csv("output/dataset-last-nooutlier.csv", sep=',',index=False)

print("******************* Detect and remove outliers last all")
lastAll = removeOutlierSentiment(lastAll)
lastAll = removeOutlierMetricLastAll(lastAll)
lastAll.to_csv("output/dataset-last-all-nooutlier.csv", sep=',',index=False)




