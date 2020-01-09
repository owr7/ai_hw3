import csv

import pandas as pd
import sklearn
from dask.dot import graphviz
import graphviz
from sklearn import tree
from nltk import DecisionTreeClassifier

if __name__ == '__main__':
    with open("train.csv", "r") as f:
        reader = csv.reader(f)
        X_train = []
        Y_train = []
        for i, line in enumerate(reader):
            if i > 0:
                X_train.append([float(k) for i, k in enumerate(line) if i < 8])
                Y_train.append(float(line[8]))
        dt = tree.DecisionTreeClassifier(criterion='entropy')

        x_t_neg = [x for x, y in zip(X_train, Y_train) if y == 0]
        x_t_pos = [x for x, y in zip(X_train, Y_train) if y == 1]
        X_train = x_t_neg[:len(x_t_pos)] + x_t_pos
        Y_train = [0]*len(x_t_pos) + [1]*len(x_t_pos)

        dt = dt.fit(X_train, Y_train)
        with open("test.csv", "r") as f_2:
            reader_2 = csv.reader(f_2)
            X_test = []
            Y_true = []
            for i, line in enumerate(reader_2):
                if i > 0:
                    X_test.append([float(k) for i, k in enumerate(line) if i < 8])
                    Y_true.append(float(line[8]))
            y_pred = dt.predict(X_test)
            # print(sklearn.metrics.confusion_matrix(Y_true, y_pred))
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Y_true, y_pred).ravel()
            print('[', [tp, fp], '\n', [fn, tn], ']')
            print(sum(k for k in Y_true))