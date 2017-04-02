import numpy as np
import pandas as pd
import math

from enum import Enum
from collections import namedtuple
from collections import Counter
from operator import itemgetter
from pprint import pformat
from random import randint
from functools import reduce
from scipy.linalg import fractional_matrix_power as power

#не правильно работающие скрипты
#долго работает
"""
class KDTree(object):

    def _build_tree(self, objects, k, axis=0):
        if not objects:
            return None
        objects.sort(key=lambda o: o[0][axis])
        median_idx = int(len(objects) / 2)
        median_point, median_label = objects[median_idx]
        median_id = self._id_gen
        self._id_gen += 1
        next_axis = (axis + 1) % k
        #print("point={}, label={}, id={}".format(median_point[:5], median_label, median_id))
        return Node(point=median_point,
                    left=self._build_tree(objects[:median_idx], k, next_axis),
                    right=self._build_tree(objects[median_idx + 1:], k, next_axis),
                    axis=axis, label=median_label, id=median_id)

    def __init__(self, k, objects=[]):
        self._id_gen = 0
        self.root = self._build_tree(list(objects), k)


    def nearest_neighbors(self, destination, num_neighbors):
        bests = [[None, None, float('inf'), None] for i in range(num_neighbors)]
        used_ids = set()
        for num_best in range(num_neighbors):
            stack = []
            stack.append(self.root)
            while stack:
                node = stack.pop(0)
                if node is not None:
                    point, left, right, axis, label, id = node
                    if id not in used_ids:
                        #print("id={}, used_ids={}, len(bests)={}".format(id, used_ids, [bests[i][3] for i in range(len(bests))]))
                        here_sd = square_distance(point, destination)
                        if here_sd < bests[num_best][2]:
                            if bests[num_best][3] is not None:
                                used_ids.remove(bests[num_best][3])
                            bests[num_best][:] = point, label, here_sd, id
                            used_ids.add(id)
                    diff = destination[axis] - point[axis]
                    close, away = (left, right) if diff <= 0 else (right, left)
                    stack.append(close)

                    if diff ** 2 < bests[num_best][2]:
                        stack.append(away)
            #self._recursive_search(self.root, num_best, bests, used_ids, destination)
        return bests
"""

def square_distance(a, b, D=None):
    if D is None:
        D = np.identity(a.shape[0])
    c = a - b
    s = c.dot(D).dot(c.transpose())
    return s


def cos_distance(a, b):
    return 1 - (a.dot(b)/(math.sqrt(a.dot(a)) * math.sqrt(b.dot(b))))


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
                    self.std.append(0.01)
                res[:, j] = (res[:, j] - self.mean[j])/self.std[j]
            with open("mean_std", "w") as writer:
                writer.write(",".join([str(item) for item in self.mean]) + "\n")
                writer.write(",".join([str(item) for item in self.std]) + "\n")
        else:
            for j in range(res.shape[1]):
                res[:, j] = (res[:, j] - self.mean[j])/self.std[j]
        return res


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


def accuracy(pred_y, test_y):
    good = 0
    for i in range(len(pred_y)):
        if pred_y[i] == test_y[i]:
            good += 1
    return good/len(pred_y)



def const(r):
    return 1


def epan(r):
    return 3/4 * (1 - r*r)


def quart(r):
    return 15/16 * (1 - r*r)*(1 - r*r)


def trian(r):
    return 1 - r


def gauss(r):
    return (2 * math.pi)**(-0.5) * math.e**(-0.5*r*r)


class KNN:

    def __init__(self):
        self.instances = []

    def fit(self, X, y):
        for i in range(X.shape[0]):
            self.instances.append((X[i, :], y[i]))

    def nearest_neighbors(self, dest, num_nrs, D):
        bests = [[None, float('inf'), None] for i in range(num_nrs)]
        for i in range(len(self.instances)):
            point, label = self.instances[i]
            #dist = square_distance(dest, point, D)
            dist = cos_distance(dest, point)
            if dist < bests[num_nrs - 1][1]:
                    bests.append([label, dist, point])
                    bests.sort(key=lambda item: item[1])
                    bests.pop()
        return bests

    def predict_one(self, instance, num_nrs, weight_func, D):
        res = self.nearest_neighbors(instance, num_nrs, D)
        h = res[-1][1]
        points = {}
        val_max = -1
        ind_max = None
        for i in range(len(res)):
            if res[i][0] not in points.keys():
                points[res[i][0]] = weight_func(res[i][1]/h)
            else:
                points[res[i][0]] += weight_func(res[i][1]/h)
            if points[res[i][0]] > val_max:
                val_max = points[res[i][0]]
                ind_max = res[i][0]
        #print(ind_max)
        #return ind_max, [(res[j][0], res[j][1]) for j in range(len(res))]
        return ind_max

    def predict(self, X, num_nrs, weight_func):
        y = []
        for i in range(X.shape[0]):
            #D = self.dann(X[i, :], num_nrs + 20)
            D = np.identity(X.shape[1])
            y.append(self.predict_one(X[i, :], num_nrs, weight_func, D))
        return y


def LOO(params, X, y, reps=100):
    D = np.identity(X.shape[1])
    val_max = 0
    param_max = None
    for param in params:
        s = 0
        used_inds = set()
        for i in range(reps):
            ind = randint(0, X.shape[0])
            if ind not in used_inds:
                used_inds.add(ind)
                X1 = np.vstack((X[:ind, :], X[ind + 1:, :]))
                y1 = np.hstack((y[:ind], y[ind + 1:]))
                x_test = X[ind, :]
                y_test = y[ind]
                est = KNN()
                est.fit(X1, y1)
                if est.predict_one(x_test, param[0], param[1], D) == y_test:
                    s += 1
        print("s", s)
        print("param", param)
        print("val_max", val_max)
        if s > val_max:
            val_max = s
            param_max = param
    return val_max, param_max


if __name__ == "__main__":
    """
    #testing
    #Xy = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 2:]
    Xy = np.load("Xy.npy")
    Xy_train, Xy_test = train_test_split(Xy, 10)
    X_train = Xy_train[:, :-1]
    y_train = Xy_train[:, -1]
    #scaler = Scaler()
    #X_train = scaler.scale(X_train)
    #corr_features = list(map(int, open("corr_features").readline().split(",")))
    #X_train = X_train[:, corr_features]
    est = KNN()
    est.fit(X_train, y_train)

    X_test = Xy_test[:, :-1]
    y_test = Xy_test[:, -1]
    #X_test = scaler.scale(X_test)
    #X_test = X_test[:, corr_features]
    pred_y = est.predict(X_test, 10, trian)
    for i in range(len(pred_y)):
        print(y_test[i], pred_y[i])
    print(accuracy(pred_y, y_test))
    """

    """
    # leave one out
    Xy = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    #Xy, _ = train_test_split(Xy, test_size=1000)
    print("Xy.shape", Xy.shape)
    X = Xy[:, :-1]
    scaler = Scaler()
    X = scaler.scale(X)
    #corr_features = list(map(int, open("corr_features").readline().split(",")))
    #X = X[:, corr_features]
    y = Xy[:, -1]
    print("X.shape", X.shape)
    params = []
    for num_nrs in [5, 15, 25, 50, 100, 150]:
        for func in [epan, const, gauss, quart, trian]:
            params.append((num_nrs, func))
    print(LOO(params, X, y))
    """


    #get answer
    Xy = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    X_train, y_train = Xy[:, :-1], Xy[:, -1]
    #with open("useful_columns") as reader:
    #    useful_cols = list(map(int, reader.readline().split(",")))
    #X_train = X_train[:, useful_cols[:-1]]
    scaler = Scaler()
    X_train = scaler.scale(X_train)
    est = KNN()
    est.fit(X_train, y_train)


    Xid = np.loadtxt("test.csv", delimiter=",", skiprows=1)
    #np.save("Xid.npy", Xid)
    #Xid = np.load("Xid.npy")
    X_test, ids = Xid[:, 1:], Xid[:, 0]

    #X_test = X_test[:, useful_cols[:-1]]

    X_test = scaler.scale(X_test)
    pred_y = est.predict(X_test, 25, quart)

    with open("answer", "w") as writer:
        writer.write("id,label\n")
        for i in range(len(pred_y)):
            writer.write(str(int(ids[i])) + "," + str(int(pred_y[i])) + "\n")
