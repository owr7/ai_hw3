import csv

import pandas as pd
import random
import sklearn
from dask.dot import graphviz
import graphviz
from dask.tests.test_base import np
from sklearn import tree
from nltk import DecisionTreeClassifier

def flip_coin(p: float):
    return True if random.uniform(0, 1) <= p else False


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

            print(4*fn+fp)
            tn_2 = tn
            fn_2 = fn
            fp_2 = fp
            tp_2 = tp
            for i in range(tn):
                if flip_coin(0.05):
                    tn_2 -= 1
                    fp_2 += 1
            for i in range(fn):
                if flip_coin(0.05):
                    fn_2 -= 1
                    tp_2 += 1
            print(4*fn_2+fp_2)

            tn_2 = tn
            fn_2 = fn
            fp_2 = fp
            tp_2 = tp
            for i in range(tn):
                if flip_coin(0.1):
                    tn_2 -= 1
                    fp_2 += 1
            for i in range(fn):
                if flip_coin(0.1):
                    fn_2 -= 1
                    tp_2 += 1
            print(4*fn_2+fp_2)

            tn_2 = tn
            fn_2 = fn
            fp_2 = fp
            tp_2 = tp
            for i in range(tn):
                if flip_coin(0.2):
                    tn_2 -= 1
                    fp_2 += 1
            for i in range(fn):
                if flip_coin(0.2):
                    fn_2 -= 1
                    tp_2 += 1
            print(4 * fn_2 + fp_2)




            # 3
            dt_3 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=3)
            dt_3 = dt_3.fit(X_train, Y_train)
            y_pred = dt_3.predict(X_test)
            # print(sklearn.metrics.confusion_matrix(Y_true, y_pred))
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Y_true, y_pred).ravel()
            print('[', [tp, fp], '\n', [fn, tn], ']')

            # 9
            dt_9 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=9)
            dt_9 = dt_9.fit(X_train, Y_train)
            y_pred = dt_9.predict(X_test)
            # print(sklearn.metrics.confusion_matrix(Y_true, y_pred))
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Y_true, y_pred).ravel()
            print('[', [tp, fp], '\n', [fn, tn], ']')

            # 27
            dt_27 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=27)
            dt_27 = dt_27.fit(X_train, Y_train)
            y_pred = dt_27.predict(X_test)
            # print(sklearn.metrics.confusion_matrix(Y_true, y_pred))
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Y_true, y_pred).ravel()
            print('[', [tp, fp], '\n', [fn, tn], ']')
            dot_data = tree.export_graphviz(dt_27, out_file=None, feature_names=['Pregnancies', 'Glucose',
                                                                                 'BloodPressure',
                                                                                 'SkinThickness', 'Insulin', 'BMI',
                                                                                 'DiabetesPedigreeFunction', 'Age'],
                                            class_names=['0', '1'],
                                            filled=True, rounded=True,
                                            special_characters=True)

            graph = graphviz.Source(dot_data)
            graph.render('test')
            graph

            print(tree.plot_tree(dt_27))


