import numpy as np
import pandas as pd

class Node:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.feature_idx = None
        self.threshold = None
        self.value = None
        self.depth = depth
        self.is_leaf = False

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini', r_features = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_features = None
        self.r_features = r_features

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self.grow_tree(X, y)
        return self

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        node = Node(depth=depth)

        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_samples < self.min_samples_split or \
                n_classes == 1:
            node.is_leaf = True
            node.value = self._most_common_label(y)
            return node

        best_feature_idx, best_threshold = self._best_split(X, y)

        if best_feature_idx is None:
            node.is_leaf = True
            node.value = self._most_common_label(y)
            return node

        node.feature_idx = best_feature_idx
        node.threshold = best_threshold

        left_indices = X[:, best_feature_idx] <= best_threshold
        right_indices = ~left_indices

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            node.is_leaf = True
            node.value = self._most_common_label(y)
            return node

        node.left = self.grow_tree(X_left, y_left, depth + 1)
        node.right = self.grow_tree(X_right, y_right, depth + 1)

        return node

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        selected_indices = None

        if self.r_features is not None:
            selected_indices = np.random.choice(np.arange(0, n_features), size=self.r_features, replace=False)

        parent_impurity = self._calculate_impurity(y)

        best_info_gain = -np.inf
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(n_features):
            if selected_indices is not None:
                if feature_idx not in selected_indices:
                    continue
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])

                left_weight = np.sum(left_indices) / n_samples
                right_weight = np.sum(right_indices) / n_samples

                info_gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)

                # Aktualizacja najlepszego podziału
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _calculate_impurity(self, y):
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == 'gini':
            return 1 - np.sum(probabilities ** 2)
        else:
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy

    def _most_common_label(self, y):
        if len(y) == 0:
            return None

        unique_labels, counts = np.unique(y, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        node = self.root

        while not node.is_leaf:
            if sample[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def print_tree(self, feature_names=None, class_names=None, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth

        if node.is_leaf:
            class_label = node.value
            if class_names is not None:
                if class_label == -1:
                    class_label = 0
                class_label = class_names[int(class_label)]
            print(f"{indent}Liść: {class_label}")
        else:
            feature_name = f"X[{node.feature_idx}]"
            if feature_names is not None:
                feature_name = feature_names[node.feature_idx]
            print(f"{indent}Jeśli {feature_name} <= {node.threshold}:")
            self.print_tree(feature_names, class_names, node.left, depth + 1)
            print(f"{indent}W przeciwnym przypadku:")
            self.print_tree(feature_names, class_names, node.right, depth + 1)