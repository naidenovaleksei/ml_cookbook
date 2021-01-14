# coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np


# Пример параметров деревьев
# TREE_PARAMS_DICT = {
#     "max_depth": 4,
#     "min_samples_leaf": 9,
#     "min_samples_split": 8
# }
# Пример tau (learning_rate) для GB
# TAU = 0.01688


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logloss(y, h):
    # для y = 1
    p1 = sigmoid(h)
    # для y = 0
    p0 = 1 - p1
    return -y * np.log(p1) - (1 - y) * (np.log(p0))


def deriative_logloss(y, h):
    """градиент logloss по решaющей функции h(x)"""
    # градиент dL/dh для y = 1
    grad1 = 1 / (1 + np.exp(h))
    # градиент dL/dh для y = 0
    grad0 = np.exp(h) / (1 + np.exp(h))  # 1 - grad1
    grad = -y * grad1 - (1 - y) * grad0
    return grad


def Loss(y, h):
    """Общая функция потерь"""
    L_values = logloss(y, h)
    return np.mean(L_values)


class SimpleGB(BaseEstimator):
    """Реализация градиентного бустинга на деревьях решений для задачи классификации
    Параметры:
        tree_params_dict: dict - параметры дерева решений sklearn.tree.DecisionTreeRegressor
        iters: int - число деревьев в градиентном бустинге
        tau: float - learning rate бустинга (обновление решающей функции)
    """

    def __init__(self, tree_params_dict, iters, tau):
        # tree params
        self.tree_params_dict = tree_params_dict
        # n estimators
        self.iters = iters
        # learning rate
        self.tau = tau
        # trees
        self.estimators = []

    def fit(self, X_data, y_data):
        # среднее таргета
        self.curr_pred = y_data.mean()
        # априорная вероятность
        curr_pred = np.ones_like(y_data) * self.curr_pred
        # решающая функция
        fx = np.log(curr_pred / (1 - curr_pred))
        # создание деревьев в бустинге
        for iter_num in range(self.iters):
            # посчитать градиент функции потерь
            grad = deriative_logloss(y_data, fx)
            # обучить DecisionTreeRegressor предсказывать антиградиент
            anti_grad = -grad
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, anti_grad)
            hx = algo.predict(X_data)
            # обновление решающей функции
            fx = fx + self.tau * hx
            self.estimators.append(algo)
        return self

    def predict(self, X_data):
        # Предсказание на данных
        # априорная вероятность
        curr_pred = np.ones(X_data.shape[0]) * self.curr_pred
        # решающая фунция
        fx = np.log(curr_pred / (1 - curr_pred))
        # обновление решающей функции по деревьям бустинга
        for estimator in self.estimators:
            fx += self.tau * estimator.predict(X_data)
        # выбор порога
        res = sigmoid(fx)
        limit = np.percentile(res, 100 * self.curr_pred)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > limit
