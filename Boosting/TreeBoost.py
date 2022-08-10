import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        # todo is it correct meaning?
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

        for _ in range(self.n_estimators):
            tree = TreeRegressor(max_depth=self.max_depth, metric=mse, criterion=self.criterion)

            train, _ = train_test_split(df, train_size=self.subsample)

            if self.colsample_bytree != 1:
                train_features, _ = train_test_split(features_names, train_size=self.colsample_bytree)
            else:
                train_features = features_names

            if self.debug:
                print(train_features)

            predicts = self.predict(df[train_features])
            grad = -self.derivative(predicts, df[target])
            print(grad)
            tree.fit(df[train_features + [target]], grad)
            self.trees.append(tree)
            # todo lr scheduler
            self.weights.append(self.lr)

    def predict(self, df: pd.DataFrame):
        if len(self.trees) == 0:
            return np.ones((len(df, )))

        predicts = []
        for tree, weight in zip(self.trees, self.weights):
            pred = weight * tree.predict(df, predict_col=None)
            predicts.append(pred)

        predicts = pd.DataFrame(predicts)
        result = []
        for col in predicts.columns:
            result.append(predicts[col].sum())
        return pd.Series(result)