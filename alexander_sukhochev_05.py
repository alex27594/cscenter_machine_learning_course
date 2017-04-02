import pandas
import numpy as np

from collections import Counter
from math import log


class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def uniquecounts(rows):
    results = Counter(rows[i][-1] for i in range(len(rows)))
    return results


def entropy(rows):
    results = uniquecounts(rows)
    # Now calculate the entropy
    ent = 0.0
    for r in results.keys():
        p = results[r]/len(rows)
        ent -= p * log(p, 2)
    return ent, results


class MyDecisionTree:
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
        self.cat_features = set()
        self.root = None

    def divideset(self, rows, column, value):
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value

        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]
        return set1, set2

    def build_tree(self, rows, scoref=entropy, depth=0):
        print("depth", depth)
        if len(rows) == 0:
            return decisionnode()
        if depth > self.max_depth:
            return decisionnode(results=uniquecounts(rows))
        if len(rows) < self.min_size:
            return decisionnode(results=uniquecounts(rows))
        current_score, _ = scoref(rows)

        best_gain = 0.0
        best_criteria = None
        best_sets = None

        for col in range(len(rows[0])-1):
            column_values = list([(row[col], row[-1]) for row in rows])
            column_values.sort(key=lambda item: item[0], reverse=True)

            cur_gini_true = None
            cur_gini_false = None
            i = 0
            cur_p = 0
            results_true = None
            results_false = None
            end = False
            for val, _ in column_values:
                prev_i = i
                while column_values[i][0] >= val:
                    if i < len(column_values) - 1:
                        i += 1
                    else:
                        end = True
                        break
                if end is True:
                    break
                if cur_gini_false is None and cur_gini_true is None:
                    set1, set2 = column_values[:i], column_values[i:]
                    cur_p += len(set1)/len(rows)
                    cur_gini_true, results_true = scoref(set1)
                    cur_gini_false, results_false = scoref(set2)
                    gain = current_score - cur_p * cur_gini_true - (1 - cur_p) * cur_gini_false
                    if gain > best_gain and len(set1) > self.min_size and len(set2) > self.min_size:
                        best_gain = gain
                        best_criteria = (col, val)
                        #print("best_gain:{}, set1, set2:{},{}".format(best_gain, len(set1), len(set2)))
                elif cur_gini_false is not None and cur_gini_true is not None:
                    dif_i = i - prev_i
                    cur_p += dif_i/len(rows)
                    dif_set = column_values[prev_i: i]
                    dif_results = uniquecounts(dif_set)

                    results_true += dif_results
                    cur_gini_true = 0
                    for r in results_true.keys():
                        p1 = results_true[r]/len(rows)
                        cur_gini_true -= p1*log(p1, 2)

                    results_false -= dif_results
                    cur_gini_false = 0
                    for r in results_false.keys():
                        p1 = results_false[r]/len(rows)
                        cur_gini_false -= p1*log(p1, 2)

                    gain = current_score- cur_p * cur_gini_true - (1 - cur_p) * cur_gini_false
                    if gain > best_gain and len(set1) > self.min_size and len(set2) > self.min_size:
                        best_gain = gain
                        best_criteria = (col, val)
                        #print("best_gain:{}, set1, set2:{},{}".format(best_gain, len(set1), len(set2)))

        if best_gain > 0:
            best_sets = self.divideset(rows, best_criteria[0], best_criteria[1])

            trueBranch = self.build_tree(best_sets[0], entropy, depth + 1)
            falseBranch = self.build_tree(best_sets[1], entropy, depth + 1)

            return decisionnode(col=best_criteria[0], value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
        else:
            return decisionnode(results=uniquecounts(rows))

    def fit(self, X, Y, scoref=entropy):
        rows = np.hstack((X, np.reshape(Y, (Y.shape[0], 1)))).tolist()

        self.root = self.build_tree(rows, entropy, 0)

    def classify(self, observation, node):
        if node.results is not None:
            #return max([(i, node.results[i]) for i in node.results.keys()], key=lambda item: item[1])[0])
            return node.results[1] / sum(node.results.values())
        else:
            v = observation[node.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= node.value:
                    branch = node.tb
                else:
                    branch = node.fb
            else:
                if v == node.value:
                    branch = node.tb
                else:
                    branch = node.fb
        return self.classify(observation, branch)

    def predict_one(self, observation):
        return self.classify(observation, self.root)

    def predict(self, X):
        observations = X.tolist()
        pred_Y = []
        for obs in observations:
            pred_Y.append(self.predict_one(obs))
        return pred_Y

def preprocess(df):
    categ_cols = ["f_16", "f_22", "f_30", "f_37"]
    df = pandas.get_dummies(df, columns=categ_cols)
    Y = np.array(df["label"])
    df = df.drop("label", axis=1)
    X = df.as_matrix()[:, 1:].astype(np.float32)
    return X, Y


def preprocess_answer(df):
    categ_cols = ["f_16", "f_22", "f_30", "f_37"]
    df = pandas.get_dummies(df, columns=categ_cols)
    ids = np.array(df["id"])
    df = df.drop("id", axis=1)
    X = df.as_matrix().astype(np.float32)
    return X, ids

if __name__ == "__main__":
    df = pandas.read_csv("learn.csv")
    X, Y = preprocess(df)
    tree = MyDecisionTree(max_depth=14, min_size=10)
    tree.fit(X, Y)
    df_test = pandas.read_csv("test.csv")
    X_test, ids = preprocess_answer(df_test)
    Y_pred = tree.predict(X_test)
    with open("answer", "w") as writer:
        writer.write("id,label\n")
        for i in range(len(Y_pred)):
            writer.write(str(int(ids[i])) + "," + str(Y_pred[i]) + "\n")