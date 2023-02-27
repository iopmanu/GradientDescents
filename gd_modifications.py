from gd import LearningRate, BaseDescent, LossFunction
from typing import Dict
from typing import Type

import numpy as np


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent, #if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticGradientDescent,
        'momentum': MomentumDescent,
        'adam': Adam
        #'adamax': Adamax
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))


class VanillaGradientDescent(BaseDescent):
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        weight_difference = -self.lr() * gradient
        self.w += weight_difference
        return weight_difference

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss is LossFunction.MSE:
            return -2 / y.shape[0] * (x.T @ y - x.T @ (self.predict(x)))
        elif self.loss is LossFunction.MAE:
            return 1 / y.shape[0] * x.T @ np.sign(self.predict(x) - y)
        elif self.loss is LossFunction.LogCosh:
            return 1 / y.shape[0] * x.T @ np.tanh(self.predict(x) - y)
        elif self.loss is LossFunction.Huber:
            difference = self.predict(x) - y
            mask = (np.abs(difference) <= 1)
            less_delta = x[mask].T @ difference[mask]
            more_delta = x[~mask].T @ np.sign(difference[~mask])
            return 1 / y.shape[0] * (less_delta - more_delta)


class StochasticGradientDescent(VanillaGradientDescent):
    """
    Mini-batch gradient descent
    """
    def __init__(self, batch_size: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        self._batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        idx = np.random.randint(0, x.shape[0], self._batch_size)
        x, y = x[idx], y[idx]

        return super().calc_gradient(x, y)


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum optimization for gradient descent
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._h = np.zeros(self.w.shape[0])

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self._h = self.momentum * self._h + self.lr() * gradient
        self.w -= self._h
        return -self._h


class Adam(StochasticGradientDescent):
    """
    Adaptive Moment Estimation.
    Using Adaptive Step and Momentum methods of optimizations
    for Stochastic Gradient Descent
    """
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self._beta1 = beta1
        self._beta2 = beta2
        self._m = np.zeros(self.w.shape[0])
        self._v = np.zeros(self.w.shape[0])
        self._eps = eps

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        k = self.iter

        self._m = self._beta1 * self._m + (1 - self._beta1) * gradient
        self._v = self._beta2 * self._v + (1 - self._beta2) * (gradient ** 2)
        m_hat = self._m / (1 - self._beta1 ** k)
        v_hat = self._v / (1 - self._beta2 ** k)

        w_prev = self.w.copy()
        self.w -= m_hat * self.lr() / (v_hat ** 0.5 + self._eps)

        return self.w - w_prev
