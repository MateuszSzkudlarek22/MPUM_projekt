##
import numpy as np
import timeit
from sklearn.ensemble import AdaBoostClassifier
from make_table import utworz_wykres_liniowy
from load_data import load_data
from precision import evaluate_classification_by_class, save_metrics_table_to_png, create_confusion_matrix

y_predicted = None
adaboost_fit_time = {}
adaboost_accuracy = {}

##
def adaboost_fit(classifier, X, y):
    global y_predicted
    classifier.fit(X, y)


# Lista różnych wartości liczby estymatorów do przetestowania
n_estimators_values = list(range(100, 501, 20))  # Od 10 do 200 co 20
data = {}

# Ładowanie danych
for i in range(5):
    data[i] = load_data('raw-img')

# Testowanie dla różnych wartości liczby estymatorów
for n_estimators in n_estimators_values:
    adaboost_fit_times = np.array([])
    adaboost_predict_times = np.array([])
    adaboost_accuracies = np.array([])

    for i in range(5):
        print(i)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = data[i]

        # Tworzenie klasyfikatora AdaBoost
        # Domyślnie używa drzew decyzyjnych o max_depth=1 jako słabych klasyfikatorów
        adaboost_classifier = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=1.0, random_state=42)

        # Pomiar czasu trenowania
        timer_adaboost = timeit.Timer(lambda: adaboost_fit(adaboost_classifier, X_train, y_train))
        adaboost_fit_times = np.append(adaboost_fit_times, timer_adaboost.timeit(1))

        # Predykcja i obliczenie dokładności
        y_predicted = adaboost_classifier.predict(X_val)
        adaboost_accuracies = np.append(adaboost_accuracies, np.mean(y_predicted == y_val))

    # Zapisanie średnich czasów i dokładności
    adaboost_fit_time[n_estimators] = np.mean(adaboost_fit_times)
    adaboost_accuracy[n_estimators] = np.mean(adaboost_accuracies)

# Wyświetlenie wyników
print("Dokładność dla różnych wartości liczby estymatorów:")
print(adaboost_accuracy)
print("Czas trenowania dla różnych wartości liczby estymatorów:")
print(adaboost_fit_time)

# Utworzenie wykresu dokładności
utworz_wykres_liniowy(
    adaboost_accuracy
)

##
X_train, X_val, X_test, y_train, y_val, y_test, labels = load_data('raw-img')

#random_forest = RandomForest(n, 30, 10)
adaboost_classifier = AdaBoostClassifier(n_estimators=500, learning_rate=1.0, random_state=42)
adaboost_classifier.fit(X_train, y_train)
#random_forest.fit(X_train, y_train)

#y_predicted = random_forest.predict(X_val)

#accuracy = np.append(accuracy, np.mean(y_predicted == y_val))

y_predicted = adaboost_classifier.predict(X_test)
##
print(np.mean(y_predicted == y_test))
save_metrics_table_to_png(evaluate_classification_by_class(y_test, y_predicted, labels))
##
create_confusion_matrix(y_test, y_predicted)