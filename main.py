##
import numpy as np
import timeit
from sklearn.tree import DecisionTreeClassifier
from make_table import utworz_wykres_liniowy

from load_data import load_data
from decision_tree import DecisionTree
from precision import evaluate_classification_by_class, save_metrics_table_to_png, create_confusion_matrix

##
y_predicted = None

my_tree_fit_time = {}
sklearn_tree_fit_time = {}

my_tree_accuracy = {}
sklearn_tree_accuracy = {}

def decision_tree_fit(tree, X, y):
    global y_predicted
    tree.fit(X, y)

depths = list(range(5, 61, 5))

data = {}
for i in range(5):
    data[i] = load_data('raw-img')

for depth in depths:
    sklearn_tree_fit = np.array([])

    sklearn_tree_predict = np.array([])

    sklearn_tree_acc = np.array([])

    for i in range(5):
        print(i)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = data[i]

        decision_tree_sklearn = DecisionTreeClassifier(max_depth=depth)

        timer_sklearn_tree = timeit.Timer(lambda: decision_tree_fit(decision_tree_sklearn, X_train,
                                                                    y_train))
        sklearn_tree_fit = np.append(sklearn_tree_fit, timer_sklearn_tree.timeit(1))


        y_predicted = decision_tree_sklearn.predict(X_val)
        sklearn_tree_acc = np.append(sklearn_tree_acc, np.mean(y_predicted == y_val))
    sklearn_tree_fit_time[depth] = np.mean(sklearn_tree_fit)

    sklearn_tree_accuracy[depth] = np.mean(sklearn_tree_acc)

print(my_tree_accuracy)
print(sklearn_tree_accuracy)
print(my_tree_fit_time)
print(sklearn_tree_fit_time)

##
X_train, X_val, X_test, y_train, y_val, y_test, labels = load_data('raw-img')

#random_forest = RandomForest(n, 30, 10)
sklearn_forest = DecisionTreeClassifier(max_depth=5)
#random_forest.fit(X_train, y_train)
sklearn_forest.fit(X_train, y_train)

#y_predicted = random_forest.predict(X_val)

#accuracy = np.append(accuracy, np.mean(y_predicted == y_val))

y_predicted = sklearn_forest.predict(X_test)
##
print(np.mean(y_predicted == y_test))
save_metrics_table_to_png(evaluate_classification_by_class(y_test, y_predicted, labels))
##
create_confusion_matrix(y_test, y_predicted)