import sys

from src.LearningAlgorithm import LearningAlgorithm
from src.AdaBoost import AdaBoost
from src.SVM import SVM
from src.DecisionTree import DecisionTree
from src.NaiveBayes import NaiveBayes
from src.PreProcessing import PreProcessing
from src.RandomForest import RandomForest

def classificar(n):
    processingNoCasualAll = PreProcessing("input/dataset-usuario-nooutlier.csv")
    processingAll = PreProcessing("input/dataset-usuario-all-nooutlier.csv")
    processingNoCasualLast = PreProcessing("input/dataset-last-nooutlier.csv")
    processingAllLast = PreProcessing("input/dataset-last-all-nooutlier.csv")
    #tree decision 
    if n == 1:
        forecasts, classes, base = processingNoCasualAll.process()
        dT = AdaBoost(base, forecasts, classes, "1-decisionTree-no-casual", base_estimator="tree")
        dT.kFold()
    if n == 2:
        forecasts, classes, base = processingAll.process()
        dT = AdaBoost(base, forecasts, classes, "2-decisionTree-all", base_estimator="tree")
        dT.kFold()
    if n == 3:
        forecasts, classes, base = processingNoCasualLast.process()
        dT = AdaBoost(base, forecasts, classes, "3-decisionTree-no-casual-last", base_estimator="tree")
        dT.kFold()
    if n == 4:
        forecasts, classes, base = processingAllLast.process()
        dT = AdaBoost(base, forecasts, classes, "4-decisionTree-all-last", base_estimator="tree")
        dT.kFold()
    #tree decision 
    
    #SVM
    if n == 5:
        forecasts, classes, base = processingNoCasualAll.process()
        dT = SVM(base, forecasts, classes, "5-Linear-SVM-no-casual", "linear")
        dT.kFold()
    if n == 6:
        forecasts, classes, base = processingNoCasualLast.process()
        dT = SVM(base, forecasts, classes, "6-Linear-SVM-no-casual-last", "linear")
        dT.kFold()
    if n == 7:
        forecasts, classes, base = processingAll.process()
        dT = SVM(base, forecasts, classes, "7-SVM-no-casual", "rbf")
        dT.kFold()
    if n == 8:
        forecasts, classes, base = processingAll.process()
        dT = SVM(base, forecasts, classes, "8-SVM-no-casual-last", "rbf")
        dT.kFold()
        
    #SVM
    
    #Naive Bayes
    if n == 9:
        forecasts, classes, base = processingAll.process()
        dT = NaiveBayes(base, forecasts, classes, "9-naive-bayes-no-casual")
        dT.kFold()    
    if n == 10:
        forecasts, classes, base = processingAll.process()
        dT = NaiveBayes(base, forecasts, classes, "10-naive-bayese-all")
        dT.kFold() 
    if n == 11:
        forecasts, classes, base = processingAll.process()
        dT = NaiveBayes(base, forecasts, classes, "11-naive-bayes-no-casual-last")
        dT.kFold() 
    if n == 12:
        forecasts, classes, base = processingAll.process()
        dT = NaiveBayes(base, forecasts, classes, "12-naive-bayes-all-last")
        dT.kFold() 
    #Naive Bayes
    
    #Tree Decision
    if n == 13:
        forecasts, classes, base = processingAll.process()
        dT = DecisionTree(base, forecasts, classes, "13-decisionTree-no-casual")
        dT.kFold() 
    if n == 14:
        forecasts, classes, base = processingAll.process()
        dT = DecisionTree(base, forecasts, classes, "14-decisionTree-all")
        dT.kFold() 
    if n == 15:
        forecasts, classes, base = processingAll.process()
        dT = DecisionTree(base, forecasts, classes, "15-decisionTree-no-casual-last")
        dT.kFold()
    if n == 16:
        forecasts, classes, base = processingAll.process()
        dT = DecisionTree(base, forecasts, classes, "16-decisionTree-all-last")
        dT.kFold() 
    #Tree Decision
#n = sys.argv[1]
#print("classifier "+n)
#classificar(int(n))
classificar(1)