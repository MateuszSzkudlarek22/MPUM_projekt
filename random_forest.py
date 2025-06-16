from collections import Counter

from decision_tree import DecisionTree
import numpy as np

class RandomForest:
    def __init__(self, n_trees, n_features, n_tree_size):
        self.n_trees = n_trees
        self.n_features = n_features
        self.n_trees = n_tree_size

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(r_features=self.n_features, max_depth=self.n_trees)
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)

            X_bootstrap = X[idxs]
            y_bootstrap = y[idxs]

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])