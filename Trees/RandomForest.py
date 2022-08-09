import pandas as pd
from sklearn.model_selection import train_test_split

from Trees.Tree import TreeClassifier
from metrics import accuracy


class RandomForest:
    def __init__(self, n_estimators, criterion, max_depth, max_features, subsample, random_state, n_jobs, debug=False):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.subsample = subsample

        self.forest = []

        self.debug = debug

    def fit(self, df: pd.DataFrame, target: str):
        features_names = list(df.columns.values)
        features_names.remove(target)

        for _ in range(self.n_estimators):
            tree = TreeClassifier(max_depth=self.max_depth, metric=accuracy, criterion=self.criterion, debug=self.debug)
            train, _ = train_test_split(df, train_size=self.subsample)
            train_features, _ = train_test_split(features_names, train_size=self.max_features)
            if self.debug:
                print(train_features)
            tree.fit(df[train_features + [target]], target)
            self.forest.append(tree)

    def predict(self, df: pd.DataFrame):
        predicts = []
        for tree in self.forest:
            pred = tree.predict(df, predict_col=None)
            predicts.append(pred)

        predicts = pd.DataFrame(predicts)
        result = []
        for col in predicts.columns:
            result.append(predicts[col].mode().values[0])
        return pd.Series(result)
