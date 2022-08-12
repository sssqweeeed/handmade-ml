import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self._mins = np.min(data, axis=0)
        self._maxs = np.max(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        result = np.zeros_like(data).astype(float)
        for j in range(data.shape[1]):
            result[:, j] = (data[:, j] - self._mins[j]) / (self._maxs[j] - self._mins[j])
        return result.copy()


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        # print(data)
        self._stds = np.std(data, axis=0)
        self._vars = np.mean(data, axis=0)
        # print(self._stds, self._vars)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        data = data.astype(float)
        result = np.zeros_like(data).astype(float)
        for j in range(data.shape[1]):
            result[:, j] = (data[:, j] - self._vars[j]) / self._stds[j]
        return result.copy()


class Preprocesser:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocesser):

    def __init__(self, dtype=np.float64):
        super(Preprocesser).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """

        self.uniques_lens = X.nunique().values
        self.uniques = np.concatenate(
            np.array([np.unique(X[f'{column}'].astype(object)) for column in X.columns], dtype=object))

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        str = lambda x: (np.repeat(np.array(x), self.uniques_lens, axis=0) == self.uniques).astype(int)
        return np.apply_along_axis(str, axis=1, arr=X.to_numpy())

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """

        self._data = dict()
        for column in X.columns:
            column_data_list = []
            for value in X[f'{column}'].unique():
                mean = Y[X[f'{column}'] == value].mean()
                proportion = Y[X[f'{column}'] == value].size / Y.size
                column_data_list.append((value, mean, proportion))

            self._data[column] = column_data_list

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """

        result_X = np.empty((X.shape[0], 0))
        for column in X.columns:
            df_for_value = np.zeros((X.shape[0], 3))
            for v, m, p in self._data[column]:
                df_for_value[(X[f'{column}'] == v).values] = [m, p, (m + a) / (p + b)]
            result_X = np.hstack((result_X, df_for_value))
        return result_X

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}
