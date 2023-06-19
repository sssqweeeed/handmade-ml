import numpy as np


def ridge(weigts: np.ndarray, lmb: float):
    return weigts - lmb * np.sign(weigts)


def lasso(weigts: np.ndarray, lmb: float):
    return weigts - lmb * 1 / 2 * (weigts)


class LinearRegression:
    def __init__(self, num_iters=400, regularization='ridge', learning_rate=0.1, reg_lmb=0.001):
        self.num_iters = num_iters
        self.w = None
        self.b = None
        self.X = None

        self.regularization = ridge if regularization == 'ridge' else lasso
        self.learning_rate = learning_rate
        self.reg_lmb = reg_lmb

    def fit(self, X: np.ndarray, y: np.ndarray):
        m_obj, n_features = X.shape
        m_obj_y = y.shape[0]
        assert m_obj == m_obj_y
        self.X = np.hstack((np.ones((m_obj, 1)), X))
        self.y = y.reshape((-1, 1))
        self.w = np.zeros((n_features + 1, 1))
        # print(self.X.shape, self.w.shape)

        learning_rate = self.learning_rate
        lmb = self.reg_lmb
        for i in range(self.num_iters):
            self.w = self.w - (
                    learning_rate * (1 / (i + 1)) * np.sum((self.forward() - self.y)/m_obj * self.X, axis=0).reshape(-1, 1)
            )
            self.w[1:, 0] = self.regularization(self.w[1:, 0], lmb)

    def forward(self):
        return self.X @ self.w

    def predict(self, X):
        m_obj, n_features = X.shape
        X_bias = np.hstack((np.ones((m_obj, 1)), X))
        return X_bias @ self.w
