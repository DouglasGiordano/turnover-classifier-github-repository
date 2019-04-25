
import os
n = 28
while n > 0:
    os.system("gnome-terminal -x 'python ExecuteParam.py "+str(n)+"'")
    print(n)
    n = n - 1


#os.system("python ExecuteParam.py 1")
