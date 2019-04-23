from algo import ForecastNaveBayes
from algo import DecisionTree
from algo import SVM
from algo import AdaBoost
from algo import PreProcessing
import sys





def classificar(n):
    staticUser = "input/dataset-usuario-nooutlier.csv"
    staticUserAll = "input/dataset-usuario-all-nooutlier.csv"
    lastUser = "input/dataset-last-nooutlier.csv"
    lastUserAll = "input/dataset-last-all-nooutlier.csv"
    if n == 21:
        f = ForecastNaveBayes.ForecastNaveBayes(staticUser)
        f.execute()
    if n == 22:
        f = ForecastNaveBayes.ForecastNaveBayes(staticUserAll)
        f.execute()
    if n == 23:
        f = ForecastNaveBayes.ForecastNaveBayes(lastUser)
        f.execute()
    if n == 24:
        f = ForecastNaveBayes.ForecastNaveBayes(lastUserAll)
        f.execute()
    if n == 25:
        d = DecisionTree.DecisionTree(staticUser)
        d.execute()
    if n == 26:
        d = DecisionTree.DecisionTree(staticUserAll)
        d.execute()
    if n == 27:
        d = DecisionTree.DecisionTree(lastUser)
        d.execute()
    if n == 28:
        d = DecisionTree.DecisionTree(lastUserAll)
        d.execute()
    if n == 9:
        d = SVM.SVM(staticUser, "linear")
        d.execute()
    if n == 10:
        d = SVM.SVM(staticUserAll, "linear")
        d.execute()
    if n == 11:
        d = SVM.SVM(lastUser, "linear")
        d.execute()
    if n == 12:
        d = SVM.SVM(lastUserAll, "linear")
        d.execute()
    if n == 13:
        d = SVM.SVM(staticUser, "poly")
        d.execute()
    if n == 14:
        d = SVM.SVM(staticUserAll, "poly")
        d.execute()
    if n == 15:
        d = SVM.SVM(lastUser, "poly")
        d.execute()
    if n == 16:
        d = SVM.SVM(lastUserAll, "poly")
        d.execute()
    if n == 17:
        d = SVM.SVM(staticUser, "rbf")
        d.execute()
    if n == 18:
        d = SVM.SVM(staticUserAll, "rbf")
        d.execute()
    if n == 19:
        d = SVM.SVM(lastUser, "rbf")
        d.execute()
    if n == 20:
        d = SVM.SVM(lastUserAll, "rbf")
        d.execute()
    if n == 5:
        d = AdaBoost.AdaBoost(staticUser, "svc")
        d.execute()
    if n == 6:
        d = AdaBoost.AdaBoost(staticUserAll, "svc")
        d.execute()
    if n == 7:
        d = AdaBoost.AdaBoost(lastUser, "svc")
        d.execute()
    if n == 8:
        d = AdaBoost.AdaBoost(lastUserAll, "svc")
        d.execute()
    if n == 1:
        d = AdaBoost.AdaBoost(staticUser, "tree")
        d.execute()
    if n == 2:
        d = AdaBoost.AdaBoost(staticUserAll, "tree")
        d.execute()
    if n == 3:
        d = AdaBoost.AdaBoost(lastUser, "tree")
        d.execute()
    if n == 4:
        d = AdaBoost.AdaBoost(lastUserAll, "tree")
        d.execute()


n = sys.argv[1]
print("classifier "+n)
classificar(int(n))
