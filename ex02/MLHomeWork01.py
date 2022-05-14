import pandas as pd
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

train = pd.read_table(r"C:\Users\lwx\Dropbox\SWOT_simulator\LearnAndTest\ex02\train.txt", 
                      sep = " ",
                      header = None)

test = pd.read_table(r"C:\Users\lwx\Dropbox\SWOT_simulator\LearnAndTest\ex02\test.txt", 
                      sep = " ",
                      header = None)

def cal_accuracy(pre, tru, name):
    pre = np.asarray(pre)
    tru = np.asarray(tru)
    acc = (len(pre) - (np.abs(pre - tru)).sum()) * 100 / len(pre)
    info = name + " accuracy: " + str(acc) + "%"
    return acc, info


train, test = train_test_split(train, test_size = 0.3)
X_train = train[[1,2,3,4]]
y_train = train[[0]]
X_test = test[[1,2,3,4]]
y_test = test[[0]]
def training(clf,name):
    clf.fit(X_train,y_train)
    pre = clf.predict(X_test)
    tru = y_test[0]
    acc, info = cal_accuracy(pre,tru, name)
    return info
size = [50, 100, 150]
layers = [2, 4, 6]
infos = []
for i in size:
    for j in layers:
        hidden_layer_sizes = (i,) * j
        clf = MLPClassifier(solver='adam', alpha=1e-5,
                     hidden_layer_sizes=hidden_layer_sizes, random_state=1)
        name = str(hidden_layer_sizes)
        info = training(clf, name)
        infos.append(info)
        
print(infos)