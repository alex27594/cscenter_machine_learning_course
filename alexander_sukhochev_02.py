import numpy as np
import pandas
from random import randint
from numpy.linalg import inv
from numpy.linalg import LinAlgError
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split


def two_parts_hor_split(arr, props=None, inds=None):
    if inds is None:
        assert sum(props) == 1
        used_inds = set()
        inds = []
        while len(inds) < int(arr.shape[0] * props[0]):
            ind = randint(0, arr.shape[0] - 1)
            if ind not in used_inds:
                used_inds.add(ind)
                inds.append(ind)
    else:
        used_inds = set(inds)
    other_inds = list(set(range(arr.shape[0])) - used_inds)
    inds.sort()
    other_inds.sort()
    return arr[inds, :].copy(), arr[other_inds, :].copy(), inds, other_inds


def two_parts_vert_split(arr, props=None, inds=None):
    if inds is None:
        assert sum(props) == 1
        used_inds = set()
        inds = []
        while len(inds) < int(arr.shape[1] * props[0]):
            ind = randint(0, arr.shape[1] - 1)
            if ind not in used_inds:
                used_inds.add(ind)
                inds.append(ind)
    else:
        used_inds = set(inds)
    other_inds = list(set(range(arr.shape[1])) - used_inds)
    inds.sort()
    other_inds.sort()
    return arr[:, inds].copy(), arr[:, other_inds].copy(), inds, other_inds


def hstack_ones_vector(arr):
    ones_vec = np.array([1 for i in range(arr.shape[0])])
    ones_vec.resize(arr.shape[0], 1)
    arr = np.hstack((ones_vec, arr))
    return arr


def hstack_features_product(arr, features, i, j):
    new_vec = arr[:, i] * arr[:, j]
    new_vec.resize(arr.shape[0], 1)
    features = features + [arr.shape[1]]
    return np.hstack((arr, new_vec)), features


class MyLinearRegression:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        self.beta = inv((X.transpose().dot(X))).dot(X.transpose()).dot(y.transpose())

    def predict_one(self, x):
        return self.beta.transpose().dot(x.transpose())

    def predict(self, X):
        pred_y = []
        for x in X:
            pred_y.append(self.predict_one(x))
        return np.array(pred_y)


def rmse(pred_y, test_y):
    assert pred_y.shape[0] == test_y.shape[0]
    n = pred_y.shape[0]
    return (sum((pred_y[i] - test_y[i])**2 for i in range(n))/n)**(1/2)


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
                self.std.append(res[:, j].std())
                res[:, j] = (res[:, j] - self.mean[j])/self.std[j]
            with open("mean_std", "w") as writer:
                writer.write(",".join([str(item) for item in self.mean]) + "\n")
                writer.write(",".join([str(item) for item in self.std]) + "\n")
        else:
            for j in range(res.shape[1]):
                res[:, j] = (res[:, j] - self.mean[j])/self.std[j]
        return res


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


# не доработано
def find_test_size(approx):
    Xy = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    # изначальный размер тестового множества
    test_size = 10
    # какой-то вектор beta с небольшим числом факторов, чтобы не было high variance
    beta = np.ones((2, 1))
    for i in range(100):
        Xy_test, _, _, _ = two_parts_hor_split(Xy, (test_size/Xy.shape[0], 1 - test_size/Xy.shape[0]))
        X_test, y_test = Xy_test[:, :2], Xy_test[-1]
        vals = []
        for j in range(X_test.shape[0]):
            line = X_test[j]
            line.resize(2, 1)
            vals.append(beta.transpose().dot(line) - y_test[j])
        #print("std", np.std(vals))
        if np.std(vals) * 3 < approx:
            return test_size
        else:
            test_size += 1
    return test_size


def super_fit(Xy, init_train_size, train_size_step, init_feature_size, feature_size_step,
              max_feature_size, test_size, max_iter_size, thresh1, thresh2):
    """
    В процессе свое работы функция, варьируя параметры
    train_size1 и feature_size, подбирает оптимальный beta,
    который бы способствовал компромиссному решению дилеммы high bias и high variance
    :param Xy: датасет
    :param init_train_size: стартовый размер обучающей выборки
    :param train_size_step: шаг, на который увеличивается размер обучающей выборки при подозрении на high variance
    :param init_feature_size: стартовое количество факторов
    :param feature_size_step: шаг, на который увеличивается размер списка используемых факторов
    :param max_feature_size: максимальное количество используемых факторов
    :param test_size: размер тестовой выборки
    :param max_iter_size: максимальное количество сделанных итераций
    :param thresh1: разница между train error и test error для определения high variance
    :param thresh2: желаемое значение rmse на тесте
    :return: beta оптимальный вектор линейной регресии
    """
    #print("super_fit is starting...")
    train_size1 = init_train_size
    feature_size = init_feature_size
    best_features, can_find = feature_selection(Xy, train_size1, feature_size, test_size)
    if can_find == "no":
        raise Exception("Can not find first version of features")
    for i in range(max_iter_size):
        test_error, train_error, beta = cv(Xy, 1000, train_size1, best_features, test_size)
        #print("Current situation: test_error={}, train_error={}".format(test_error, train_error))
        # high variance
        if test_error - train_error > thresh1:
            #print("high variance!!!")
            if train_size1 + train_size_step <= Xy.shape[0] - test_size:
                train_size1 += train_size_step
                #print("new train size={}".format(train_size1))
                continue
            else:
                raise Exception("Can not deal with high variance. All train object were used.")
        # high bias
        if test_error > thresh2:
            #print("high bias")
            if len(best_features) < max_feature_size:
                feature_size += feature_size_step
                best_features, can_find = feature_selection(Xy, train_size1, feature_size, test_size,
                                                            best_features)
                #print("best_features={}, can_find={}".format(best_features, can_find))
                if can_find == "yes":
                    continue
                elif can_find == "high variance":
                    if train_size1 + train_size_step <= Xy.shape[0] - test_size:
                        train_size1 += train_size_step
                        #print("new train size={}".format(train_size1))
                        continue
                    else:
                        raise Exception("Can not deal with high variance. All train object were used.")
                elif can_find == "no":
                    break

            else:
                raise Exception("Can not deal with high bias. Max feature size was used.")
        else:
            return test_error, train_error, beta
    #print("super_fit is ending")
    return test_error, train_error, beta, best_features, train_size1


def feature_selection(Xy, train_size1, feature_size, test_size=10, best_features=None, features=None):
    """
    :param Xy: датасет
    :param train_size1: размер обучающего множества
    :param feature_size: количество признаков, которые нужно выбрать
    :param test_size: размер обучающего множества
    :param features: отсортированный список факторов из которых нужно выбирать, если None,
    то список факторов берётся из файла "corr_features"
    :return: best_features: n наиболее результативных факторов при заданных условиях (размер обучения, количество факторов,
    которые нужно выбрать)
    """
    #print("-"*10)
    #print("feature_selection is starting...")
    assert train_size1 + test_size < Xy.shape[0]
    if features is None:
        #print("getting features from file")
        with open("corr_features") as reader:
            features = list(map(int, reader.readline().split(",")))
    #print("Size of accesible features={}".format(len(features)))
    if best_features is None:
        # нулевой фактор это 1, он всегда должен пристуствовать
        best_features = [0]
        # следующий фактор выбирается случайно из первых десяти факторов списка
        j1 = randint(0, 9)
        best_features.append(features[j1])
        feature_size -= 2
    else:
        feature_size -= len(best_features)
    test_error, _, _ = cv(Xy, 100, train_size1, best_features, test_size)
    i = 0
    j = 0
    can_find = "yes"
    num_attempts = 0
    while i < feature_size:
        if features[j] in best_features:
            j += 1
            continue
        #print("best_features={}".format(best_features))
        #print("try new feature={}".format(features[j]))
        if j == len(features) - 1:
            can_find = "no"
            return best_features, can_find
        #print(num_attempts)
        new_test_error, _, _ = cv(Xy, 100, train_size1, best_features + [features[j]], test_size)
        #print("test_error={}, new_test_error={}".format(test_error, new_test_error))
        if test_error - new_test_error >= 0.05 or (num_attempts >= 50 and abs(test_error - new_test_error) <= 0.05):
            best_features.append(features[j])
            test_error = new_test_error
            i += 1
        elif new_test_error > 100000:
            can_find = "high variance"
            return best_features, can_find
        else:
            num_attempts += 1
        j += 1
    #print("feature selection is ending")
    #print("best_features={}".format(best_features))
    #print("-"*10)
    return best_features, can_find


def cv(Xy, k, train_size1, features, test_size=10):
    """
    :param Xy: датасет
    :param k: количество испытаний
    :param train_size1: размер обучающего множества в единицах примеров
    :param features: используемые факторы
    :param test_size: размер тестового множества
    :return: s_test_error, s_train_error: Вычисляет среднее значение rmse на тренировочной
    и тестовой выборке на k тестах при заданном
    размере обучающей выборки и при заданном множестве факторов
    """
    #print("-"*5)
    #print("cv is starting...")
    s_test_error = 0
    s_train_error = 0
    beta = np.array([0.0 for i in range(len(features))])
    num_of_sings = 0
    i = 0
    while i < k:
        # делим на обучение и тест
        Xy_train, Xy_test, _, _ = two_parts_hor_split(Xy, (1 - test_size/Xy.shape[0], test_size/Xy.shape[0]))
        # разделение теста
        X_test, y_test = Xy_test[:, :-1], Xy_test[:, -1]
        X_test = X_test[:, features]
        # разделение на train и support(фактически то, что мы не используем)
        Xy_train1, Xy_supp, _, _ = two_parts_hor_split(Xy_train, (train_size1/Xy_train.shape[0], 1 - train_size1/Xy_train.shape[0]))
        X_train1, y_train1 = Xy_train1[:, :-1], Xy_train1[:, -1]
        X_train1 = X_train1[:, features]

        est = MyLinearRegression()
        try:
            est.fit(X_train1, y_train1)
        except np.linalg.linalg.LinAlgError:
            #print("Singular matrix")
            num_of_sings += 1
            if num_of_sings == 10:
                #print("singularity!!!!!")
                return 100, 100, beta
            else:
                continue
        beta += est.beta

        pred_y = est.predict(X_test)
        error = rmse(pred_y, y_test)
        # print("test_error={}".format(error))
        s_test_error += round(error, 5)

        pred_y = est.predict(X_train1)
        error = rmse(pred_y, y_train1)
        # print("train_error={}".format(error))
        s_train_error += round(error, 5)
        i += 1

    s_train_error /= k
    s_test_error /= k
    beta /= k
    #print("cv is ending")
    #print("-"*5)
    return s_test_error, s_train_error, beta

if __name__ == "__main__":
    # в этой закомменченной области представлен скрипт
    # для нахождения среди факторов и всевозможных их произведений таких факторов, которые бы
    # в наибольшей степени коррелировали бы с ответом
    """
    Xy = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    X, y = Xy[:, :-1], Xy[:, -1]
    X = hstack_ones_vector(X)
    # добавляем к признакам все их произведения
    X = extended_arr(X)
    # составляем список k фич, которые в наибольшей степени коррелируют с ответом
    features = list(range(X.shape[1]))
    k = 10000
    corr_features = find_corr_features(X, y, features, k)
    """

    # В этой закомменченной области представлен запуск функции super_fit (данный скрипт запускается при условии,
    # что уже был запущен скрипт, находящийся выше, так как использует файлы, созданные тем скриптом.
    # Автор закомментил вывод всей информации о работе super_fit(), который является достаточно информативным.
    """
    Xy = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    y = Xy[:, -1]
    X = np.load("extended_arr.npy")
    y = np.array(y)
    y.resize(y.shape[0], 1)
    Xy = np.hstack((X, y))
    test_error, train_error, beta, best_features, train_size1 = super_fit(Xy=Xy, init_train_size=30, train_size_step=10,
                                                                          init_feature_size=3, feature_size_step=1,
                                                                          max_feature_size=100, test_size=30,
                                                                          max_iter_size=200, thresh1=2, thresh2=4)
    """


    Xy = np.loadtxt("learn.csv", delimiter=",", skiprows=1)[:, 1:]
    X, y = Xy[:, :-1], Xy[:, -1]
    X = hstack_ones_vector(X)
    X = extended_arr(X)

    # Значения получены в результате работы функции super_fit.
    # Её работа занимает достаточно большое время, поэтому она была закоменчена в данном скрипте.
    # Я на саом деле так и не дождался конца её работы, взяв промежуточный результат за окончательный.
    best_features = [0, 4656, 4670, 17886, 24, 4695, 4721, 4644, 4710, 4660, 4569,
                   4612, 4716, 17272, 2358, 4659, 4690, 18761, 4604, 17948, 4718
                     ,132, 4692, 4581, 4615, 4559]
    train_size1 = 190

    y = np.array(y)
    y.resize(y.shape[0], 1)
    Xy = np.hstack((X, y))
    test_error, train_error, beta = cv(Xy=Xy, k=1000, train_size1=train_size1, features=best_features,
                                       test_size=Xy.shape[0] - train_size1 - 1)
    print("test_error={}, train_error={}, beta={}".format(test_error, train_error, beta))
    Xid = np.loadtxt("test.csv", delimiter=",", skiprows=1)
    X, ids = Xid[:, 1:], Xid[:, 0]
    X = hstack_ones_vector(X)
    X = extended_arr(X)
    X = X[:, best_features]

    est = MyLinearRegression()
    est.beta = beta
    pred_y = est.predict(X)
    with open("answer", "w") as writer:
        writer.write("id,target\n")
        for i in range(pred_y.shape[0]):
            writer.write(str(int(ids[i])) + "," + str(pred_y[i]) + "\n")