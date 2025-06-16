##
import numpy as np
import timeit
from sklearn.neighbors import KNeighborsClassifier
from make_table import utworz_wykres_liniowy
from load_data import load_data
from precision import evaluate_classification_by_class, save_metrics_table_to_png, create_confusion_matrix
##
y_predicted = None
knn_fit_time = {}
knn_accuracy = {}


def knn_fit(classifier, X, y):
    global y_predicted
    classifier.fit(X, y)


# Lista różnych wartości k (liczba sąsiadów) do przetestowania
neighbors_values = list(range(1, 21, 2))  # Nieparzyste wartości od 1 do 19
data = {}

# Ładowanie danych
for i in range(5):
    data[i] = load_data('raw-img')

# Testowanie dla różnych wartości k
for n_neighbors in neighbors_values:
    knn_fit_times = np.array([])
    knn_predict_times = np.array([])
    knn_accuracies = np.array([])

    for i in range(5):
        print(i)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = data[i]

        # Tworzenie klasyfikatora k-NN
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Pomiar czasu trenowania
        timer_knn = timeit.Timer(lambda: knn_fit(knn_classifier, X_train, y_train))
        knn_fit_times = np.append(knn_fit_times, timer_knn.timeit(1))

        # Predykcja i obliczenie dokładności
        y_predicted = knn_classifier.predict(X_val)
        knn_accuracies = np.append(knn_accuracies, np.mean(y_predicted == y_val))

    # Zapisanie średnich czasów i dokładności
    knn_fit_time[n_neighbors] = np.mean(knn_fit_times)
    knn_accuracy[n_neighbors] = np.mean(knn_accuracies)

# Wyświetlenie wyników
print("Dokładność dla różnych wartości k:")
print(knn_accuracy)
print("Czas trenowania dla różnych wartości k:")
print(knn_fit_time)

utworz_wykres_liniowy(knn_accuracy, "knn_all.png")

##
X_train, X_val, X_test, y_train, y_val, y_test, labels = load_data('raw-img')

#random_forest = RandomForest(n, 30, 10)
sklearn_forest = KNeighborsClassifier(1)
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