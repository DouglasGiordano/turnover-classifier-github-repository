import sys

from src.DecisionTree import DecisionTree
from src.PreProcessing import PreProcessing
from src.RandomForest import RandomForest


def classificar(n):
    processingCasualAll = PreProcessing("input/dataset-usuario-nooutlier.csv")
    processingAll = PreProcessing("input/dataset-usuario-all-nooutlier.csv")
    processingCasualLast = PreProcessing("input/dataset-last-nooutlier.csv")
    processingAllLast = PreProcessing("input/dataset-last-all-nooutlier.csv")
    if n == 1:
        forecasts, classes, base = processingCasualAll.process()
        dT = DecisionTree(base, forecasts, classes, "1-decisionTree-casual-all")
        dT.kFold()
    if n == 17:
        forecasts, classes, base = processingAll.process()
        dT = RandomForest(base, forecasts, classes, "1-decisionTree-casual-all")
        dT.kFold()


#n = sys.argv[1]
#print("classifier "+n)
#classificar(int(n))
classificar(1)