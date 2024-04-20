import numpy as np
import pandas as pd


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def gini(p):
    return 2 * p * (1 - p)

def misclassification(p):
    return min(p, 1 - p)

def information_gain(left_data, right_data, criterion_function='entropy'):
    if criterion_function == 'entropy':
        criterion_function = entropy
    elif criterion_function == 'gini':
        criterion_function = gini
    elif criterion_function == 'misclassification':
        criterion_function = misclassification
    else:
        raise ValueError("Invalid criterion function.")
    parent_data = np.concatenate((left_data, right_data), axis=0)
    partent_impurity = criterion_function(parent_data.mean() if len(parent_data) > 0 else 0)
    left_impurity = criterion_function(left_data.mean() if len(left_data) > 0 else 0)
    right_impurity = criterion_function(right_data.mean() if len(right_data) > 0 else 0)
    return partent_impurity - (len(left_data) / len(parent_data)) * left_impurity - (len(right_data) / len(parent_data)) * right_impurity

class DecisionTree():

    def __init__(self, name='tree', max_features=np.inf, max_depth=np.inf, min_samples_split=2, search_depth=1, criterion_function='entropy'):
        self.name = name
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_function = criterion_function
        self.search_depth = search_depth

    def _terminal_node(self, y):
        return y.mean() >= 0.5
    
    def _get_best_split(self, X, y):
        # Draw random features depending on max_features
        if self.max_features == np.inf:
            features = X.columns
        else:
            features = np.random.choice(X.columns, self.max_features, replace=False)

        best_info_gain = -999
        best_feature = None
        for feature in features:
            # Find split for numerical features
            if X[feature].dtype != 'object':
                for split in X[feature].unique():
                    left_data = y[X[feature] <= split]
                    right_data = y[X[feature] > split]
                    info_gain = information_gain(left_data, right_data, self.criterion_function)
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_feature = feature
                        best_split = split
                        left_data_best = (X.loc[X[feature] <= split], y[X[feature] <= split])
                        right_data_best = (X.loc[X[feature] > split], y[X[feature] > split])
            # Find split for categorical features
            else:
                for category in X[feature].unique():
                    left_data = y[X[feature] == category]
                    right_data = y[X[feature] != category]
                    info_gain = information_gain(left_data, right_data, self.criterion_function)
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_feature = feature
                        best_split = category
                        left_data_best = (X.loc[X[feature] == category], y[X[feature] == category])
                        right_data_best = (X.loc[X[feature] != category], y[X[feature] != category])

        node = {
            'info_gain': best_info_gain,
            'feature': best_feature,
            'split': best_split,
            'left_split': left_data_best,
            'right_split': right_data_best
        }

        return node
    
    def _split_node(self, node, depth):
        left_X, left_y = node['left_split']
        right_X, right_y = node['right_split']
        del node['left_split']
        del node['right_split']
        
        if node['info_gain'] == 0:
            node['left_child'] = self._terminal_node(pd.concat([left_y, right_y]))
            node['right_child'] = self._terminal_node(pd.concat([left_y, right_y]))
            return
        
        if depth >= self.max_depth:
            node['left_child'] = self._terminal_node(left_y)
            node['right_child'] = self._terminal_node(right_y)
            return

        if len(left_y) < self.min_samples_split:
            node['left_child'] = self._terminal_node(left_y)
        else:
            node['left_child'] = self._get_best_split(left_X, left_y)
            self._split_node(node['left_child'], depth + 1)

        if len(right_y) < self.min_samples_split:
            node['right_child'] = self._terminal_node(right_y)
        else:
            node['right_child'] = self._get_best_split(right_X, right_y)
            self._split_node(node['right_child'], depth + 1)
        
    def _generate_tree(self, X, y):
        root = self._get_best_split(X, y)
        self._split_node(root, 1)

        return root

    def train(self, X, y, generated_X=None, generated_y=None):
        print(f"+-- Training trees {len(self.trees) + 1} to {len(self.trees) + self.n_trees}...")
        for _ in range(self.n_trees):
            X_train = pd.concat([X, generated_X])
            y_train = pd.concat([y, generated_y])
            tree = self._generate_tree(X_train, y_train)
            self.trees.append(tree)
            print(f"|   +-- Tree {len(self.trees)} trained.")            

    def _tree_row_prediction(self, X, node):
        if not isinstance(node, dict):
            return node
        
        feature = node['feature']
        split = node['split']

        if isinstance(X[feature], str):
            if X[feature] == split:
                return self._tree_row_prediction(X, node['left_child'])
            else:
                return self._tree_row_prediction(X, node['right_child'])
        else:
            if X[feature] <= split:
                return self._tree_row_prediction(X, node['left_child'])
            else:
                return self._tree_row_prediction(X, node['right_child'])
            
    def _tree_prediction(self, X, node):
        return X.apply(lambda row: self._tree_row_prediction(row, node), axis=1)
    
    def evaluate_accuracy(self, X_test, y_test):

        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
    
    def get_tree_depth(self, node):
        if not isinstance(node, dict):
            return 1
        return 1 + max(self.get_tree_depth(node['left_child']), self.get_tree_depth(node['right_child']))