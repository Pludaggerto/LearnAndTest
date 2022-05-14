import pandas as pd
import numpy  as np

from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier 
from sklearn.svm             import SVC

def cal_accuracy(pre, tru, name):

    pre = np.asarray(pre)
    tru = np.asarray(tru)
    acc = (len(pre) - (np.abs(pre - tru)).sum()) * 100 / len(pre)
    info = name + " accuracy: " + str(acc) + "%"

    return acc, info

def BPtraining(clf, name, X_train, y_train, X_test, y_test):

    clf.fit(X_train, y_train)
    pre = clf.predict(X_test)
    tru = y_test[0]
    acc, info = cal_accuracy(pre, tru, name)
    return info

def RFtraining(X_train, y_train, X_test, y_test):

    rf = RandomForestClassifier()           
    rf.fit(X_train, y_train)
    pre = rf.predict(X_test)
    tru = y_test[0]
    acc, info = cal_accuracy(pre, tru, "RF")
    return info

def BPtrainingu(X_train, y_train, X_test, y_test):

    clf = MLPClassifier()           
    clf.fit(X_train, y_train)
    pre = clf.predict(X_test)
    tru = y_test[0]
    acc, info = cal_accuracy(pre, tru, "BP")
    return info

def SVMtraining(X_train, y_train, X_test, y_test):
    svc = SVC()           
    svc.fit(X_train, y_train)
    pre = svc.predict(X_test)
    tru = y_test[0]
    acc, info = cal_accuracy(pre, tru, "SVC")
    return info

def main():

    train = pd.read_table(r"C:\Users\lwx\Dropbox\SWOT_simulator\LearnAndTest\ex02\train.txt", 
                          sep = " ",
                          header = None)

    test = pd.read_table(r"C:\Users\lwx\Dropbox\SWOT_simulator\LearnAndTest\ex02\test.txt", 
                          sep = " ",
                          header = None)

    train, test = train_test_split(train, test_size = 0.3)

    X_train = train[[1,2,3,4]]
    y_train = train[[0]]
    X_test = test[[1,2,3,4]]
    y_test = test[[0]]

    size = [50, 100, 150]
    layers = [2, 4, 6]
    infos = []

    for i in size:
        for j in layers:
            hidden_layer_sizes = (i,) * j
            clf = MLPClassifier(solver='adam', alpha=1e-5,
                            hidden_layer_sizes=hidden_layer_sizes, random_state=1)
            name = str(hidden_layer_sizes)
            info = BPtraining(clf, name, X_train, y_train, X_test, y_test)
            infos.append(info)
        
    infos.append(SVMtraining(X_train, y_train, X_test, y_test))
    infos.append(BPtrainingu(X_train, y_train, X_test, y_test))
    infos.append(RFtraining(X_train, y_train, X_test, y_test))
    print(infos)

if __name__ == '__main__':
    main()