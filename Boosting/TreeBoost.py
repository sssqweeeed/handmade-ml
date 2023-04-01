import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import copy, deepcopy
from Trees.Tree import TreeRegressor
from metrics import mse


class SimpleTreeBoostRegressor:
    def __init__(self, n_estimators, metric, derivative, max_depth: int, colsample_bytree, criterion,
                 subsample, minimize=False, debug=True, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.metric = metric
        self.derivative = derivative
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.criterion = criterion
        self.minimize = minimize
        self.debug = debug

        self.trees = []
        self.weights = []

    def fit(self, df: pd.DataFrame, target: str):
        features_names = list(df.columns.values)
        features_names.remove(target)

        tree = TreeRegressor(max_depth=self.max_depth, metric=mse, criterion=self.criterion, debug=self.debug,
                      minimize=True)

        tree.trivial_fit(df, target)
        self.trees.append(copy(tree))
        self.weights.append(1)

        for i in range(self.n_estimators):
            tree = TreeRegressor(max_depth=self.max_depth, metric=mse, criterion=self.criterion, debug=self.debug,
                                 minimize=True)

            train, _ = train_test_split(df, train_size=self.subsample, shuffle=True)

            if self.colsample_bytree != 1:
                train_features, _ = train_test_split(features_names, train_size=self.colsample_bytree)
            else:
                train_features = features_names

            if self.debug:
                print(train_features)

            predicts = self.predict(train)

            grad = self.derivative(predicts, train[target])
            # print('predict: ', predicts[0])
            # print('grad: ', grad[0])
            print(f'Iteration: {i}, Loss: {mse(predicts, train[target])}\n')

            train[target] = grad
            # print('----train-----')
            # print(train)
            # print('--------------\n\n')
            tree.fit(train[train_features + [target]], target)

            self.trees.append(deepcopy(tree))
            self.weights.append(self.lr)

    def predict(self, df: pd.DataFrame, predict_col=None, first_n_estimators=None):

        predicts = []
        for tree, weight in zip(self.trees, self.weights):
            pred = weight * tree.predict(df, predict_col=None)
            predicts.append(pred.copy())
        predicts = pd.DataFrame(predicts)

        if first_n_estimators:
            return predicts

        result = []
        for col in predicts.columns:
            result.append(predicts[col].sum())

        if predict_col is not None:
            df[predict_col] = result
            # return df
        else:
            return pd.Series(result)
