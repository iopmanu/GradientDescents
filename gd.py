from enum import Enum, auto
import numpy as np


class LearningRate:
    def __init__(self, lambda_: float = 1e-3, s0: float = 1, p: float = 0.5, eta0: float = np.nan) -> None:
        """
        :param lambda_: weight of the learning rate
        :param s0: param for a calculation of an optimal learning rate
        :param p: param for a calculation of an optimal learning rate
        :param eta0: param for class initialization with constant learning rate
        """
        self._lambda = lambda_
        self._s0 = s0
        self._p = p
        self._iteration = 0
        self._eta0 = eta0

    def __call__(self) -> float:
        """
        This method calculates learning rate for every iteration
        :return: value of the learning rate for current iteration
        """
        self._iteration += 1
        if self._eta0 is not np.nan:
            return self._eta0
        else:
            return self._lambda * (self._s0 / (self._s0 + self._iteration)) ** self._p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent(object):
    """
        iter:                   iteration number.
        max_iter:               maximum number of iterations.
        learning_rate:          the learning rate schedule:
                                    ‘constant’: eta = eta0;
                                    ‘optimal’: eta = lambda / (s0 / (s0 + k));
        momentum:               param for optimization of gradient descent.
        eta0:                   param which is using for calc 'constant' learning rate.
        lambda_:                 param which is using for calc 'optimal' learning rate.
        stop_grad:              checking mean loss absolute value for stop criteria.
        dimension:              quantity of features in feature matrix.
        loss:                   loss function.

    """
    allowed_kwargs = {'iter', 'max_iter', 'learning_rate', 'eta0', 'lambda_',
                      'stop_grad', 'dimension', 'momentum', 'loss'}

    __mse_dict = {'MSE': LossFunction.MSE, 'MAE': LossFunction.MAE,
                'Huber': LossFunction.Huber, 'LogCosh': LossFunction.LogCosh}

    def __init__(self, **kwargs):
        for param in kwargs:
            if param not in self.allowed_kwargs:
                raise TypeError('Unexpected keyword: ' + str(param))
        self.__dict__.update(kwargs)

        if not hasattr(self, 'dimension'):
            raise TypeError('Impossible to create without dimension')
        self.w = np.random.rand(kwargs.get('dimension'))

        if hasattr(self, 'learning_rate'):
            if kwargs.get('learning_rate') == 'constant':
                src_eta0 = kwargs.get('eta0') if kwargs.get('eta0') else 1.0
                self.lr = LearningRate(eta0=src_eta0)
            elif kwargs.get('learning_rate') == 'optimal':
                src_lambda = kwargs.get('lambda_') if kwargs.get('lambda_') else 1e-3
                self.lr = LearningRate(lambda_=src_lambda)
            else:
                raise TypeError('Invalid value for learning rate param')
        else:
            self.lr = LearningRate(eta0=1.0)

        if not hasattr(self, 'iter'):
            self.iter = 0

        if not hasattr(self, 'loss'):
            self.loss = LossFunction.MSE
        elif kwargs.get('loss') in self._mse_dict:
            self.loss = self._mse_dict[kwargs.get('loss')]
        else:
            raise ValueError("Unexpected loss function as argue")

        if not hasattr(self, 'max_iter'):
            self.max_iter = 1

        if not hasattr(self, 'momentum'):
            self.momentum = 0.0

        if not hasattr(self, 'stop_grad'):
            self.stop_grad = 1e-6

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This method for loss calculating between our predict and target variable.
        :param x: features matrix
        :param y: targets array
        :return: loss value
        """
        if self.loss is LossFunction.MSE:
            return 1 / y.shape[0] * (y - x @ self.w).T @ (y - x @ self.w)
        elif self.loss is LossFunction.MAE:
            return 1 / y.shape[0] * np.abs(self.predict(x) - y).sum()
        elif self.loss is LossFunction.LogCosh:
            return 1 / y.shape[0] * np.log(np.cosh(self.predict(x) - y)).sum()
        elif self.loss is LossFunction.Huber:
            difference = self.predict(x) - y
            mask = (np.abs(difference) <= 1)
            less_delta = (difference[mask] ** 2 / 2).sum()
            more_delta = (np.abs(difference[~mask]) - 0.5).sum()
            return 1 / y.shape[0] * (less_delta - more_delta)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: features matrix
        :return: predicted vector of target variable
        """
        return x @ self.w
