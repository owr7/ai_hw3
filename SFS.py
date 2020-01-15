import csv
import itertools
import math
from typing import List, Tuple

from dask.tests.test_base import np


def distance(train: List, test: List, f_in: List[int]):
    train_ = [t for i, t in enumerate(train) if i in f_in]
    test_ = [t for i, t in enumerate(test) if i in f_in]
    return math.sqrt(sum((a-b)*(a-b) for a, b in zip(train_, test_)))

def distance_2(train: List, test: List):
    return sum(abs(a-b) for a, b in zip(train, test))
#def cond

if __name__ == '__main__':
    with open("train.csv", "r") as f:
        reader = csv.reader(f)
        X_train = []
        Y_train = []
        for i, line in enumerate(reader):
            if i > 0:
                X_train.append([float(k) for i, k in enumerate(line) if i < 8])
                Y_train.append(float(line[8]))
        max_ = []
        min_ = []
        for i in range(8):
            max_.append(max(k[i] for k in X_train))
            min_.append(min(k[i] for k in X_train))
        for train in X_train:
            for i in range(8):
                train[i] = train[i]/(max_[i]-min_[i])

        with open("test.csv", "r") as f_2:
            reader_2 = csv.reader(f_2)
            X_test = []
            Y_true = []
            for i, line in enumerate(reader_2):
                if i > 0:
                    X_test.append([float(k) for i, k in enumerate(line) if i < 8])
                    Y_true.append(float(line[8]))

            for test in X_test:
                for i in range(8):
                    test[i] = test[i] / (max_[i] - min_[i])
            flg = False
            final_features = []
            curr_features = []
            max_accu_final = 0
            for place_i in range(8):
                if flg:
                    break
                max_accu = 0
                for feature in range(8):
                    if flg:
                        break
                    if feature in final_features:
                        continue
                    curr_features = final_features + [feature]
                    tp = tn = fp = fn = 0
                    for i, test in enumerate(X_test):
                        if flg:
                            break
                        distance_list = [(distance(train, test, curr_features), Y_train[i]) for i, train in enumerate(X_train)]
                        distance_list.sort(key=lambda x: x[0])
                        #print(sum(i[1] for i in distance_list))
                        result = 0
                        k = 9
                        if sum(s[1] for s in distance_list[:k]) > k/2:
                            result = 1
                        if result == Y_true[i]:
                            if result == 1:
                                tp += 1
                            elif result == 0:
                                tn += 1

                        if result != Y_true[i]:
                            if result == 1:
                                fp += 1
                            elif result == 0:
                                fn += 1
                    accu = (tn+tp)/X_test.__len__()
                    if accu > max_accu:
                        max_accu = accu
                        best_feature = feature

                if max_accu_final <= max_accu:
                    final_features.append(best_feature)
                    max_accu_final = max_accu
                else:
                    flg = True
                    break
            print(final_features)


