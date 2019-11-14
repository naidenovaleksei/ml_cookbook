# coding=utf-8
import numpy as np
from sklearn.base import BaseEstimator


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'naidenov.aleksei@yandex.ru'

LR_PARAMS_DICT = {
    'C': 2545.7851287569724,
    'batch_size': 4704,
    'iters': 3319,
    'random_state': 777,
    'step': 0.7396818439186627
}


class MyLogisticRegression(BaseEstimator):
    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step

    # будем пользоваться этой функцией для подсчёта <w, x>
    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    # sklearn нужно, чтобы predict возвращал классы, поэтому оборачиваем наш __predict в это
    def predict(self, X):
        res = self.__predict(X)
        res[res > 0] = 1
        res[res < 0] = 0
        return res

    # производная регуляризатора
    def der_reg(self):
        return self.w / self.C

    # будем считать стохастический градиент не на одном элементе, а сразу на пачке (чтобы было эффективнее)
    def der_loss(self, x, y):
        # x.shape == (batch_size, features)
        # y.shape == (batch_size,)

        # confidence score - скалярное произведение значений признаков и весов
        z = self.__predict(x)
        # логистическая функция от confidence score
        h = 1 / (1 + np.exp(-z))
        # hint: это же частный случай Generalized Linear Model,
        # поэтому получаем корректировку весов как в линейной регрессии
        # p.19 Andrew Ng CS229 Lecture notes http://cs229.stanford.edu/notes/cs229-notes1.pdf
        ders_w = ((h - y).T * x.T).T
        der_w0 = (h - y)

        # для масштаба возвращаем средний градиент по пачке
        return ders_w.mean(axis=0), der_w0.mean(axis=0)

    def fit(self, X_train, y_train):
        # RandomState для воспроизводитмости
        random_gen = np.random.RandomState(self.random_state)
        
        # получаем размерности матрицы
        size, dim = X_train.shape
        
        # случайная начальная инициализация
        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):  
            # берём случайный набор элементов
            rand_indices = random_gen.choice(size, self.batch_size)
            # исходные метки классов это 0/1
            x = X_train[rand_indices]
            y = y_train[rand_indices]

            # считаем производные
            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg()

            # обновляемся по антиградиенту
            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step

        # метод fit для sklearn должен возвращать self
        return self