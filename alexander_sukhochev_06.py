import numpy as np
import pandas as pd

from math import exp
from random import random
from random import randint
from random import gauss
from random import sample
from scipy.stats import entropy


def mutual_info(a, b, bins):
    c_ab = np.histogram2d(a, b, bins=bins)[0].flatten()
    c_a = np.histogram(a, bins=bins)[0]
    c_b = np.histogram(b, bins=bins)[0]
    h_ab = entropy(c_ab)
    h_a = entropy(c_a)
    h_b = entropy(c_b)
    mi = h_a + h_b - h_ab
    return mi


class MyKBestSelector:
    def __init__(self, k):
        self.k = k
        self.scores = None

    def fit(self, X, Y):
        self.scores = []
        for i in range(X.shape[1]):
            self.scores.append((i, mutual_info(X[:, i], Y, bins=60)))
        self.scores.sort(key=lambda item: item[1], reverse=True)
        self.scores = self.scores

    def transform(self, X):
        X = X.copy()
        X = X[:, [item[0] for item in self.scores[:self.k]]]
        return X

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)


def my_train_test_split(X, Y, test_size):
    test_count = int(X.shape[0]*test_size)
    test_inds = sample(range(X.shape[0]), test_count)
    train_inds = list(set(list(range(X.shape[0]))) - set(test_inds))
    X_train = X[train_inds, :]
    X_test = X[test_inds, :]
    Y_train = Y[train_inds, :]
    Y_test = Y[test_inds, :]
    return X_train, X_test, Y_train, Y_test


class MyScaler:
    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, X):
        for j in range(X.shape[1]):
            self.means.append(np.mean(X[:, j]))
            self.stds.append(np.std(X[:, j]))

    def transform(self, X):
        X = X.copy()
        for j in range(X.shape[1]):
            X[:, j] = (X[:, j] - self.means[j])/self.stds[j]
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def sample_generator(X, Y):
    pos_labels = sorted(list(np.unique(Y)))
    cur_ind = 0
    yield
    while True:
        i = randint(0, Y.shape[0] - 1)
        if Y[i] == pos_labels[cur_ind]:
            yield X[i, :], Y[i]
            cur_ind = (cur_ind + 1) % len(pos_labels)


class Network:
    def __init__(self, n_inputs, n_hidden, n_outputs, l_rate, n_epoch):
        self.layers = list()
        hidden_layer = [{'weights': np.array([random() for i in range(n_inputs + 1)])} for i in range(n_hidden)]
        self.layers.append(hidden_layer)
        output_layer = [{'weights': np.array([random() for i in range(n_hidden + 1)])} for i in range(n_outputs)]
        self.layers.append(output_layer)
        self.n_outputs = n_outputs
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def activate(self, weights, inputs):
        return np.dot(weights[:-1], inputs) + weights[-1]

    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def forward_propagate(self, row):
        inputs = row
        for layer in self.layers:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = np.array(new_inputs)
        return inputs

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = list()
            if i != len(self.layers)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.layers[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, row):
        for i in range(len(self.layers)):
            inputs = row
            if i != 0:
                inputs = [neuron['output'] for neuron in self.layers[i - 1]]
            for neuron in self.layers[i]:
                for j in range(len(inputs)-1):
                    neuron['weights'][j] += self.l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.l_rate * neuron['delta']

    def fit(self, X, Y, autochange_start_weights=True):
        if autochange_start_weights:
            for i in range(len(self.layers[0])):
                for j in range(self.layers[0][i]["weights"].shape[0] - 1):
                    self.layers[0][i]["weights"][j] = np.dot(X[:, j], Y)/np.dot(X[:, j], X[:, j]) + gauss(0, 0.1)
        sg = sample_generator(X, Y)
        next(sg)
        prev_sum_error = 0
        cur_sum_error = 0
        epoch_num = 0
        t = 0
        init_l_rate = self.l_rate
        for epoch in range(self.n_epoch):
            sample, label = next(sg)
            outputs = self.forward_propagate(sample)
            expected = [0 for i in range(self.n_outputs)]
            expected[int(label)] = 1
            error = sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            self.backward_propagate_error(expected)
            self.update_weights(sample)
            print("epoch={}, lrate={}, error={}".format(epoch, self.l_rate, error))
            cur_sum_error += error
            epoch_num += 1
            if epoch_num == X.shape[0]:
                print("prev_sum_error={}, cur_sum_error={}".format(prev_sum_error, cur_sum_error))
                """
                if abs(cur_sum_error - prev_sum_error) <= 1 and epoch > 50000:
                    break
                """
                epoch_num = 0
                prev_sum_error = cur_sum_error
                cur_sum_error = 0
                t += 1
                self.l_rate = init_l_rate/pow(t, 0.5)
                print("new l_rate={}".format(self.l_rate))

    def predict_one(self, row):
        outputs = self.forward_propagate(row)
        return np.argmax(outputs)

    def predict_one_proba(self, row):
        outputs = self.forward_propagate(row)
        return outputs/np.sum(outputs)

    def predict(self, X):
        res = []
        for i in range(X.shape[0]):
            res.append(self.predict_one(X[i, :]))
        return np.array(res)

    def predict_proba(self, X):
        res = []
        for i in range(X.shape[0]):
            res.append(self.predict_one_proba(X[i, :]))
        return np.array(res)


class FeatureGenerator:
    def __init__(self):
        self.selector = MyKBestSelector(k=100)

    def fit_transform(self, X, Y):
        selX = self.selector.fit_transform(X, Y)
        new_features = []
        # product
        for j in range(selX.shape[1]):
            for j1 in range(j + 1, selX.shape[1]):
                new_features.append(selX[:, j] * selX[:, j1])
        new_features_arr = np.transpose(np.array(new_features))
        return np.hstack((X, new_features_arr))

    def transform(self, X, Y):
        selX = self.selector.transform(X)
        new_features = []
        # product
        for j in range(selX.shape[1]):
            for j1 in range(j + 1, selX.shape[1]):
                new_features.append(selX[:, j] * selX[:, j1])

        new_features_arr = np.transpose(np.array(new_features))
        return np.hstack((X, new_features_arr))


def get_answer():
    df = pd.read_csv("learn.csv")
    Y = np.array(df["y"])
    df = df.drop("y", axis=1)
    X = df.as_matrix()[:, 1:].astype(np.float32)
    feature_generator = FeatureGenerator()
    X = feature_generator.fit_transform(X, Y)
    scaler = MyScaler()
    X = scaler.fit_transform(X)
    k = 20
    k0 = int((X.shape[0]*k)**0.5)
    selector = MyKBestSelector(k=k)
    X = selector.fit_transform(X, Y)
    net = Network(n_inputs=X.shape[1], n_hidden=k0, n_outputs=2, l_rate=0.5, n_epoch=450000)
    net.fit(X, Y)

    df_test = pd.read_csv("test.csv")
    ids = np.array(df_test["id"])
    X_test = df_test.as_matrix()[:, 1:].astype(np.float32)
    X_test = feature_generator.transform(X_test, Y)
    X_test = scaler.transform(X_test)
    X_test = selector.transform(X_test)
    pred_Y = net.predict_proba(X_test)[:, 1]
    with open("answer", "w") as writer:
        writer.write("id,label\n")
        for i in range(len(pred_Y)):
            writer.write(str(int(ids[i])) + "," + str(pred_Y[i]) + "\n")
    print("selector.scores", selector.scores)


if __name__ == "__main__":
    get_answer()
