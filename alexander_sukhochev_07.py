import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split



def find_optimal_test_set_size(X, Y, thresh=0.01):
    X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.5)
    clf = SVC(kernel="rbf", probability=True)
    clf.fit(X_train, Y_train)
    cur_thresh = 100
    cur_test_size = 0
    while cur_thresh > thresh and cur_test_size <= X_rest.shape[0]:
        cur_test_size += 10
        scores = []
        i = 0
        while i < 10:
            print("i:{}, cur_test_size:{}, cur_thresh:{}". format(i, cur_test_size, cur_thresh))
            X_test, X_notest, Y_test, Y_notest = train_test_split(X_rest, Y_rest, test_size=int(X_rest.shape[0] - cur_test_size))
            print(X_test.shape)
            if len(np.unique(Y_test)) == 1:
                continue
            i += 1
            Y_pred_proba = clf.predict_proba(X_test)[:, 1]
            scores.append(roc_auc_score(Y_test, Y_pred_proba))
        cur_thresh = abs(max(scores) - min(scores))
    return cur_test_size if cur_test_size <= X.shape[0]/2 else -1


def get_trainXY():
    df = pd.read_csv("learn.csv")
    Y = np.array(df["y"])
    df = df.drop("y", axis=1)
    X = df.as_matrix()[:, 1:].astype(np.float32)
    return X, Y


def get_testXid():
    df_test = pd.read_csv("test.csv")
    ids = np.array(df_test["id"])
    X_test = df_test.as_matrix()[:, 1:].astype(np.float32)
    return X_test, ids


def outliers_detection(X, Y):
    must_delete = set()
    for i in range(X.shape[0]):
        print(i)
        for j in range(X.shape[1]):
            if abs(X[i, j] - np.mean(X[Y == Y[i], j])) > 5*np.std(X[Y == Y[i], j]):
                must_delete.add(i)
                break
    must_stay = sorted(list(set(range(X.shape[0])) - must_delete))
    print(len(must_stay))
    return X[must_stay, :], Y[must_stay]


def feature_select(X, Y):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lsvc, prefit=True)
    X = model.transform(X)
    return X, Y


def test1():
    X, Y = get_trainXY()
    #X, Y = feature_select(X, Y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    clf = SVC(probability=True, max_iter=12000)
    param_grid = [{'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 1, 10]},
                  {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 1, 10], 'degree': [2, 3, 5], 'coef0': [-10.0, 0.0, 10.0]},
                  {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                  {'kernel': ['sigmoid'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 1, 10], 'coef0': [-10.0, 0.0, 10.0]}]

    cv = GridSearchCV(clf, cv=3, param_grid=param_grid, verbose=5, scoring="roc_auc")
    cv.fit(X, Y)
    print(cv.best_params_)
    print(cv.best_score_)


def get_answer():
    X_train, Y_train = get_trainXY()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = SVC(C=10, gamma=0.001, probability=True)
    clf.fit(X_train, Y_train)
    X_test, ids = get_testXid()
    X_test = scaler.transform(X_test)
    Y_pred = clf.predict_proba(X_test)[:, 1]
    with open("answer", "w") as writer:
        writer.write("id,label\n")
        for i in range(len(Y_pred)):
            writer.write(str(int(ids[i])) + "," + str(Y_pred[i]) + "\n")


if __name__ == "__main__":
   get_answer()
