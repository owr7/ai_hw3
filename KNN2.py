import csv
import math
from typing import List


def distance(train: List, test: List):
    return math.sqrt(sum((a-b)*(a-b) for a, b in zip(train, test)))

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

            tp = tn = fp = fn = 0
            for i, test in enumerate(X_test):
                distance_list = [(distance(train, test), Y_train[i]) for i, train in enumerate(X_train)]
                distance_list.sort(key=lambda x: x[0])
                #print(sum(i[1] for i in distance_list))
                result = 0
                k = 27
                if sum(s[1] for s in distance_list[:k])*4 > k/2:
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
            print([tp, fp], '\n', [fn, tn])