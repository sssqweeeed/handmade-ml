import numpy as np
import pandas as pd


class Node:
    def __init__(self, data, feature, threshold, fit_score, left_value, right_value):
        self.data = data
        self.threshold = threshold
        self.feature = feature
        self.fit_score = fit_score
        self.left_value = left_value
        self.right_value = right_value

        self.left_leaf = None
        self.right_leaf = None


class Tree:
    def __init__(self, max_depth: int, metric, criterion, minimize=False, debug=True):
        self.metric = metric
        self.criterion = criterion
        self.minimize = minimize

        self.max_depth = max_depth

        self.tree = None

        self.debug = debug

    def split_predict(self, df: pd.DataFrame, feature: str, target: str, threshold: float, classes: dict):
        predict = df[feature].apply(
            func=lambda value: classes['left'] if value <= threshold else classes['right']
        )
        return self.metric(predict, df[target])

    def find_best_split(self, df: pd.DataFrame, target: str):
        if len(df) <= 2:
            self.print('Impossible to find split, len(df) <= 2\n------------------------------\n')
            return None
        if len(set(df[target])) == 1:
            self.print('Impossible to find split, len(set(df[target])) == 1\n------------------------------\n')
            return None

        best_feature = None
        best_threshold = None
        best_score = None
        best_classes = None

        if self.minimize:
            best_score = np.inf
        else:
            best_score = -np.inf

        self.print(df[target].value_counts())

        for col in df.columns:
            # todo remove hack to speed up
            if col != target:
                threshold = Tree.split_by_feature(df, feature=col, target=target, criterion=self.criterion)['threshold']
                values = self.find_values(data=df, feature=col, target=target, threshold=threshold)
                if values is None:
                    # todo bugfix: if all values are equal to None, then the result will contain only None values
                    continue
                self.print(f'feature:{col} values{values} '
                           f'split_predict{self.split_predict(df, col, target, threshold, values)},\n')

                score = self.split_predict(df, col, target, threshold, values)
                if self.minimize:
                    if score < best_score:
                        best_score = score
                        best_feature = col
                        best_threshold = threshold
                        best_classes = values.copy()
                else:
                    if score > best_score:
                        best_score = score
                        best_feature = col
                        best_threshold = threshold
                        best_classes = values.copy()

        result = {'best_score': best_score,
                  'best_feature': best_feature,
                  'best_threshold': best_threshold,
                  'left_value': best_classes['left'],
                  'right_value': best_classes['right']}
        self.print(f'{result} \n------------------------------\n')
        return result

    def build_step(self, target: str, nodes_to_build: [Node]):
        if len(nodes_to_build) == 0:
            self.print('There are no nodes to build')
        next_step_build = []
        for node in nodes_to_build:

            data_left = node.data[node.data[node.feature] <= node.threshold]
            data_right = node.data[node.data[node.feature] > node.threshold]

            left_split = self.find_best_split(data_left, target)
            right_split = self.find_best_split(data_right, target)

            if left_split is not None:
                node.left_leaf = Node(data=data_left,
                                      feature=left_split['best_feature'],
                                      threshold=left_split['best_threshold'],
                                      fit_score=left_split['best_score'],
                                      left_value=left_split['left_value'],
                                      right_value=left_split['right_value'])
                next_step_build.append(node.left_leaf)
            if right_split is not None:
                node.right_leaf = Node(data=data_right,
                                       feature=right_split['best_feature'],
                                       threshold=right_split['best_threshold'],
                                       fit_score=right_split['best_score'],
                                       left_value=right_split['left_value'],
                                       right_value=right_split['right_value'])

                next_step_build.append(node.right_leaf)
        return next_step_build

    # @dispatch(pd.DataFrame, str)
    def fit(self, df: pd.DataFrame, target: str):
        # todo X, y fit
        first_step_split = self.find_best_split(df, target)
        self.tree = Node(data=df,
                         feature=first_step_split['best_feature'],
                         threshold=first_step_split['best_threshold'],
                         fit_score=first_step_split['best_score'],
                         left_value=first_step_split['left_value'],
                         right_value=first_step_split['right_value'])
        nodes_to_build = [self.tree]

        for _ in range(self.max_depth - 1):
            nodes_to_build = self.build_step(target, nodes_to_build)
        # nodes_to_build = self.build_step(target, nodes_to_build)
        self.print('OK!')

    # @dispatch(pd.DataFrame, pd.DataFrame)
    # def fit(self, X: pd.DataFrame, y: pd.DataFrame):
    #     df = X.copy()
    #     if 'target' not in X.columns:
    #         df['target'] = y.copy()
    #     else:
    #         warnings.warn('name "target" in X.columns')
    #         return
    #     self.fit(df, target='target')

    def passing_tree(self, node: Node, row: pd.Series):
        if row[node.feature] <= node.threshold:
            if node.left_leaf is not None:
                return self.passing_tree(node.left_leaf, row)
            else:
                return node.left_value
        else:
            if node.right_leaf is not None:
                return self.passing_tree(node.right_leaf, row)
            else:
                return node.right_value

    def predict(self, df: pd.DataFrame, predict_col: str):
        # todo optimize predict
        predicts = []
        for i, row in df.iterrows():
            predicts.append(self.passing_tree(self.tree, row))
        if predict_col is not None:
            df[predict_col] = predicts
            # return df
        else:
            return np.array(predicts)

    def draw_tree(self):
        pass

    @staticmethod
    def H(data: pd.DataFrame, criterion='entropy'):
        # impurity
        props = np.array(data.value_counts(normalize=True))
        result = None
        if criterion == 'entropy':
            log_props = np.log2(props)
            result = (-1) * np.matmul(props, log_props)
        elif criterion == 'max':
            result = 1 - np.max(props)
        elif criterion == 'gini':
            result = 1 - np.sum(props ** 2)
        return result

    @staticmethod
    def Q(data: pd.DataFrame, feature, target, threshold: float, criterion='entropy'):
        R_left = data[data[feature] <= threshold][target]
        R_right = data[data[feature] > threshold][target]

        len_left = len(R_left)
        len_right = len(R_right)
        len_R = len(data)

        return Tree.H(data[target], criterion=criterion) - \
               (len_left / len_R) * Tree.H(R_left, criterion=criterion) - \
               (len_right / len_R) * Tree.H(R_right, criterion=criterion)

    @staticmethod
    def get_grid(data: pd.Series):
        values = np.array(data.sort_values())
        grid = (values[:-1] + values[1:]) / 2

        print('find threshold in:', grid[0], grid[-1])
        return grid

    @staticmethod
    def split_by_feature(data: pd.DataFrame, feature, target, criterion='entropy', mode='slow'):
        if mode == 'slow':
            Q_max = -np.inf
            threshold_max = None
            for threshold in Tree.get_grid(data[feature]):
                Q_value = Tree.Q(data, feature, target, threshold, criterion)
                # self.print(threshold, Q_value)
                if Q_value > 1:
                    # warnings.warn('Q_value > 1')
                    break
                if Q_max < Q_value:
                    Q_max = Q_value
                    threshold_max = threshold
            return {'threshold': threshold_max, 'feature': feature}

        elif mode == 'fast':
            # todo fast grid search
            pass

    def find_values(self, data: pd.DataFrame, feature, target, threshold) -> dict:
        pass

    def print(self, msg):
        if self.debug:
            print(msg)


class TreeRegressor(Tree):
    def find_values(self, data: pd.DataFrame, feature, target, threshold):
        left_data = data[data[feature] <= threshold]
        right_data = data[data[feature] > threshold]
        if len(left_data) == 0 or len(right_data) == 0:
            return None
        else:
            return {'left': np.mean(left_data[target]), 'right': np.mean(right_data[target])}


class TreeClassifier(Tree):
    def find_values(self, data: pd.DataFrame, feature, target, threshold):
        left_data = data[data[feature] <= threshold]
        right_data = data[data[feature] > threshold]
        if len(left_data) == 0 or len(right_data) == 0:
            return None
        else:
            return {'left': left_data[target].mode().values[0], 'right': right_data[target].mode().values[0]}
