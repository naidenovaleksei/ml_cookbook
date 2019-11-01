#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import scipy.optimize as opt


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'naidenov.aleksei@yandex.ru'
# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 1}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def L(y, h):
    # L для y = 1
    p1 = sigmoid(h)
    # L для y = 0
    p0 = 1 - p1
    return - y * np.log(p1) - (1 - y) * (np.log(p0))

def deriative_L(y, h):
    # градиент dL/dh для y = 1
    grad1 = 1 / (1 + np.exp(h))
    # градиент dL/dh для y = 0
    grad0 = np.exp(h) / (1 + np.exp(h)) # 1 - grad1
    grad = - y * grad1 - (1 - y) * grad0
    return grad

def Loss(y, h):
    L_values = L(y, h)
    return np.mean(L_values)



class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau

    def fit(self, X_data, y_data):
        # self.base_algo = DecisionTreeClassifier(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        self.ks = []
        self.fxs = []
        self.losses = []
        # curr_pred = self.base_algo.predict(X_data)
        self.curr_pred = y_data.mean()
        curr_pred = np.ones_like(y_data) * y_data.mean()
        fx = np.log(curr_pred / (1 - curr_pred))
        self.fxs.append(fx)
        self.losses.append(Loss(y_data, fx))
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            grad = deriative_L(y_data, fx)
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            anti_grad = - grad
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, anti_grad)
            hx = algo.predict(X_data)
            #
            # func = lambda k: Loss(y_data, fx + k * hx)
            # k = opt.minimize(func, 0, bounds=((-1, 1),)).x[0]
            k = 1
            fx = fx + self.tau * k * hx
            self.losses.append(Loss(y_data, fx))
            self.ks.append(k)
            self.fxs.append(fx)

            self.estimators.append(algo)
            # # Обновите предсказания в каждой точке
            # curr_pred = sigmoid(fx)
        res = sigmoid(fx)
        self.thr = np.percentile(res, 100 * self.curr_pred)
        from sklearn.metrics import f1_score
        # func = lambda thr: ((res > thr).astype(int) != y_data).mean()
        # func = lambda thr: 1 - f1_score(y_data, (res > thr).astype(int))
        # thr = opt.minimize(func, 0, bounds=((0, 1),)).x[0]
        # print(thr)
        # self.thr = thr
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        curr_pred = np.ones(X_data.shape[0]) * self.curr_pred
        fx = np.log(curr_pred / (1 - curr_pred))
        for k,estimator in zip(self.ks, self.estimators):
            fx += self.tau * estimator.predict(X_data)
        # print(fx)
        res = sigmoid(fx)
        # print(res)
        limit = np.percentile(res, 100 * self.curr_pred)
        print(limit, self.thr)
        print(res > limit)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > limit
