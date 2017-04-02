import math
import numpy as np

from functools import reduce
from itertools import combinations
from random import randint
from random import random


def accuracy(pred_Y, test_Y):
    good = 0
    for i in range(len(pred_Y)):
        if pred_Y[i] == test_Y[i]:
            good += 1
    return good/len(pred_Y)


def train_test_split(Xy, test_size):
    test_inds = set()
    i = 0
    while i < test_size:
        k = randint(0, Xy.shape[0] - 1)
        if k not in test_inds:
            test_inds.add(k)
            i += 1
    train_inds = set(range(Xy.shape[0])).difference(test_inds)
    train_inds = list(train_inds)
    test_inds = list(test_inds)
    return Xy[train_inds, :], Xy[test_inds, :]


class Scaler:
    def __init__(self, file=None):
        if file is None:
            self.mean = []
            self.std = []
        else:
            with open(file) as reader:
                self.mean = list(map(float, reader.readline().split(",")))
                self.std = list(map(float, reader.readline().split(",")))

    def scale(self, arr):
        res = arr.copy()
        if len(self.mean) == 0:
            for j in range(res.shape[1]):
                self.mean.append(res[:, j].mean())
                if res[:, j].std() > 0:
                    self.std.append(res[:, j].std())
                else:
                    self.std.append(0.000001)
                res[:, j] = (res[:, j] - self.mean[j])/self.std[j]
            with open("mean_std", "w") as writer:
                writer.write(",".join([str(item) for item in self.mean]) + "\n")
                writer.write(",".join([str(item) for item in self.std]) + "\n")
        else:
            for j in range(res.shape[1]):
                res[:, j] = (res[:, j] - self.mean[j])/self.std[j]
        return res


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.n = None
        self.m = None

    def get_loss(self, x, y):
        return math.log(1 + math.exp(-self.w.dot(x) * y))

    def part_der(self, x, y, j):
        return -self.w[j]*y/(math.exp(y*self.w.dot(x)) + 1)

    def get_grad(self, x, y):
        return np.array([self.part_der(x, y, j) for j in range(self.m)])

    def target_function(self, X, Y):
        return sum(self.get_loss(X[i, :], Y[i]) for i in range(self.n))

    def instance_generator(self, X, Y):
        # выдаём примеры по возможности по очереди из разных классов, чтобы ускорить сходимость
        used_inds = set()
        next_label = 1
        attempts = 0
        yield
        while len(used_inds) < self.n:
            ind = randint(0, self.n - 1)
            if ind not in used_inds:
                if Y[ind] == next_label or attempts == 5:
                    yield X[ind, :], Y[ind], ind
                    used_inds.add(ind)
                    attempts = 0
                    next_label = - next_label
                else:
                    attempts += 1

    def fit(self, X, Y):
        for i in range(len(Y)):
            Y[i] = 1 if Y[i] == 1 else -1
        lam = 0.01
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.w = np.array([(random() * 2 - 1) * (1/(2*self.m)) for j in range(self.m)])
        old_Q = 0
        new_Q = self.target_function(X, Y)
        gen = self.instance_generator(X, Y)
        next(gen)
        num_step = 1
        while abs(old_Q - new_Q) > 0.0001:
            try:
                new_x, new_y, new_i = next(gen)
            except StopIteration:
                break
            try:
                new_loss = self.get_loss(new_x, new_y)
                new_grad = self.get_grad(new_x, new_y)
            except OverflowError:
                print("exception")
                continue
            old_Q = new_Q
            self.w -= 1/num_step * new_grad
            new_Q = (1 - lam) * old_Q - lam * new_loss
            num_step += 1

        #print(self.w)

    def predict_one(self, x):
        return 1 if self.w.dot(x) > 0 else 0

    def predict(self, X):
        pred_Y = []
        for i in range(X.shape[0]):
            pred_Y.append(self.predict_one(X[i, :]))
        return pred_Y


def extended_arr(arr):
    res = arr.copy()
    m = res.shape[1]
    # нулевой элемент ни с кем не перемножаем
    for i in range(1, m):
        res = np.hstack((res, np.array([(res[:,i] * res[:,j]).transpose() for j in range(i, m)]).transpose()))
    np.save("extended_arr", res)
    return res


def find_corr_features(arr, target, features, k):
    corr_features = features.copy()
    corr_features.sort(key=lambda item: abs(np.corrcoef(arr[:, item], target))[0][1], reverse=True)
    corr_features = corr_features[:k]
    with open("corr_features", "w") as writer:
        writer.write(",".join([str(item) for item in corr_features]))
    return corr_features


def hstack_ones_vector(arr):
    ones_vec = np.array([1 for i in range(arr.shape[0])])
    ones_vec.resize(arr.shape[0], 1)
    arr = np.hstack((ones_vec, arr))
    return arr


def harm_score(Y_pred, Y_test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] == 1 and Y_test[i] == 1:
            TP += 1
        elif Y_pred[i] == 1 and Y_test[i] == 0:
            FP += 1
        elif Y_pred[i] == 0 and Y_test[i] == 1:
            FN += 1
        elif Y_pred[i] == 0 and Y_test[i] == 0:
            TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2*precision*recall/(precision + recall)

# на основе множества испытаний находит оптимальные веса для заданных факторов
def find_bests():
    XY = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    XY_train, XY_test = train_test_split(XY, test_size=500)
    X_train, Y_train = XY_train[:, :-1], XY_train[:, -1]
    X_test, Y_test = XY_test[:, :-1], XY_test[:, -1]
    #X_train = extended_arr(X_train)
    X_train = X_train[:, [0, 1, 3, 4]]
    scaler = Scaler()
    X_train = scaler.scale(X_train)
    X_train = hstack_ones_vector(X_train)

    #X_test = extended_arr(X_test)
    X_test = X_test[:, [0, 1, 3, 4]]
    X_test = scaler.scale(X_test)
    X_test = hstack_ones_vector(X_test)

    bests = []
    for k in range(1000):
        est = LogisticRegression()
        est.fit(X_train, Y_train)
        pred_Y = est.predict(X_test)
        score = harm_score(Y_test, pred_Y)
        print(harm_score(Y_test, pred_Y))
        bests.append((est.w, score))
    bests.sort(key=lambda item: item[1], reverse=True)
    print(bests)


def forward_selection(max_num_features=5, k=100, t_s=500):
    XY = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    features = list(range(XY.shape[1] - 1))
    used_features = []
    best_score = 0
    for j in range(max_num_features):
        best_feature = None
        for feature in features:
            if feature in used_features:
                continue
            average_score = 0
            temp_features = used_features + [feature]
            for i in range(k):
                XY_train, XY_test = train_test_split(XY, test_size=t_s)
                X_train, Y_train = XY_train[:, :-1], XY_train[:, -1]
                X_test, Y_test = XY_test[:, :-1], XY_test[:, -1]
                scaler = Scaler()
                X_train = scaler.scale(X_train)
                X_test = scaler.scale(X_test)
                X_train = X_train[:, temp_features]
                X_test = X_test[:, temp_features]
                est = LogisticRegression()
                est.fit(X_train, Y_train)
                Y_pred = est.predict(X_test)
                score = harm_score(Y_test, Y_pred)
                average_score += score
            average_score /= k
            print("temp_features {}".format(temp_features))
            print("average_score {}".format(average_score))
            if average_score > best_score:
                best_score = average_score
                best_feature = feature
        if best_feature is None:
            break
        used_features.append(best_feature)
        print(used_features)
        print(best_score)
    return used_features

if __name__ == "__main__":
    # get answer
    XY = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    XY_train, XY_test = train_test_split(XY, test_size=500)
    X_train, Y_train = XY_train[:, :-1], XY_train[:, -1]
    X_test, Y_test = XY_test[:, :-1], XY_test[:, -1]
    X_train = extended_arr(X_train)
    scaler = Scaler()
    X_train = scaler.scale(X_train)

    Xid = np.loadtxt("test.csv", delimiter=",", skiprows=1)
    X, ids = Xid[:, 1:], Xid[:, 0]
    X = scaler.scale(X)
    # факторы выбраны на основе работы функции forward_selection и на основе корреляции этих факторов с label (find_corr_features)
    X = X[:, [0, 1, 3, 4]]
    X = hstack_ones_vector(X)
    est = LogisticRegression()
    # веса факторов были получены на основе работы функции find_best_weights
    est.w = np.array([0.20038421, -0.40711321,  0.31176292, -0.79409471, -0.19801522])
    pred_Y = est.predict(X)
    with open("answer", "w") as writer:
        writer.write("id,label\n")
        for i in range(len(pred_Y)):
            writer.write(str(int(ids[i])) + "," + str(pred_Y[i]) + "\n")

