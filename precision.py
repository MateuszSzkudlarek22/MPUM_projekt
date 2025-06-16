import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification_by_class(y_true, y_pred, class_dict=None):
    """
    Ocenia skuteczność predykcji dla każdej klasy.

    Parametry:
    y_true : lista lub numpy array
        Prawdziwe etykiety klas.
    y_pred : lista lub numpy array
        Predykcje modelu.
    class_dict : słownik, opcjonalnie
        Słownik mapujący indeksy klas na ich nazwy, np. {0: 'Klasa A', 1: 'Klasa B', 2: 'Klasa C'}.
        Jeśli nie podano, zostaną użyte oryginalne wartości z y_true.

    Zwraca:
    DataFrame z metrykami dla każdej klasy.
    """
    # Konwersja do numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Znajdź unikalne klasy w danych
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))

    # Jeśli podano słownik z nazwami klas, przygotuj etykiety do wyświetlania
    if class_dict is not None:
        display_names = [class_dict.get(cls, str(cls)) for cls in unique_classes]
        # Sprawdź, czy wszystkie klasy mają swoje nazwy w słowniku
        missing_classes = [cls for cls in unique_classes if cls not in class_dict]
        if missing_classes:
            print(f"Uwaga: Brak nazw dla klas {missing_classes} w słowniku. Użyto wartości oryginalnych.")
    else:
        display_names = [str(cls) for cls in unique_classes]
        class_dict = {cls: str(cls) for cls in unique_classes}

    # Obliczenie metryk dla każdej klasy
    precision = precision_score(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)

    # Stworzenie macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Obliczenie dodatkowych metryk
    support = np.sum(cm, axis=1)
    accuracy_per_class = np.diag(cm) / support.astype(float)

    # Stworzenie DataFrame z wynikami
    metrics_df = pd.DataFrame({
        'Indeks klasy': unique_classes,
        'Nazwa klasy': display_names,
        'Precyzja': precision,
        'Czułość (Recall)': recall,
        'F1-Score': f1,
        'Dokładność': accuracy_per_class,
        'Liczba próbek': support
    })

    # Wyświetlenie raportu klasyfikacji
    print("Raport klasyfikacji:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=display_names, zero_division=0))

    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_names, yticklabels=display_names)
    plt.xlabel('Predykcja')
    plt.ylabel('Wartość rzeczywista')
    plt.title('Macierz pomyłek')
    plt.tight_layout()
    plt.show()

    return metrics_df


def save_metrics_table_to_png(metrics_df, filename='metrics_table_adaboost.png', title='Metryki klasyfikacji dla każdej klasy'):
    """
    Tworzy tabelę z metrykami klasyfikacji i zapisuje ją jako plik PNG.

    Parametry:
    metrics_df : DataFrame
        DataFrame zawierający metryki klasyfikacji.
    filename : str, opcjonalnie
        Nazwa pliku PNG do zapisania tabeli.
    title : str, opcjonalnie
        Tytuł tabeli.
    """
    # Formatowanie liczb zmiennoprzecinkowych do 4 miejsc po przecinku
    formatted_df = metrics_df.copy()
    float_cols = ['Precyzja', 'Czułość (Recall)', 'F1-Score', 'Dokładność']
    for col in float_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map(lambda x: f'{x:.4f}')

    # Ustawienia figury
    fig, ax = plt.subplots(figsize=(12, len(formatted_df) + 2))
    ax.axis('off')
    ax.axis('tight')

    # Tworzenie tabeli
    table = ax.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        loc='center',
        cellLoc='center'
    )

    # Stylizacja tabeli
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Zwiększenie wysokości wierszy

    # Stylizacja nagłówków
    for k, cell in table.get_celld().items():
        if k[0] == 0:  # Nagłówek
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:  # Co drugi wiersz z danymi ma inne tło
            if k[0] % 2 == 0:
                cell.set_facecolor('#D9E1F2')
            else:
                cell.set_facecolor('#E9EDF4')

    # Dostosowanie szerokości kolumn
    col_widths = [max(len(str(formatted_df.iloc[i, j])) for i in range(len(formatted_df)))
                  for j in range(len(formatted_df.columns))]
    col_widths = [max(len(col), width) for col, width in zip(formatted_df.columns, col_widths)]

    # Dodanie tytułu
    plt.title(title, fontsize=14, fontweight='bold', pad=20)

    # Dopasowanie układu
    plt.tight_layout()

    # Zapisanie do pliku
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Tabela z metrykami została zapisana do pliku '{filename}'")


def create_confusion_matrix(y_true, y_pred, class_dict=None, figsize=(10, 8),
                            save_fig=True, fig_name='confusion_matrix_adaboost.png', dpi=300,
                            show_percent=True, cmap='Blues'):
    """
    Tworzy i zapisuje do pliku macierz pomyłek.

    Parametry:
    y_true : lista lub numpy array
        Prawdziwe etykiety klas.
    y_pred : lista lub numpy array
        Predykcje modelu.
    class_dict : słownik, opcjonalnie
        Słownik mapujący indeksy klas na ich nazwy, np. {0: 'Klasa A', 1: 'Klasa B'}.
    figsize : tuple, opcjonalnie
        Rozmiar figury matplotlib.
    save_fig : bool, opcjonalnie
        Czy zapisać figurę do pliku. Domyślnie True.
    fig_name : str, opcjonalnie
        Nazwa pliku do zapisania macierzy pomyłek.
    dpi : int, opcjonalnie
        Rozdzielczość figury w DPI.
    show_percent : bool, opcjonalnie
        Czy wyświetlić również procentowy udział pomyłek. Domyślnie True.
    cmap : str, opcjonalnie
        Kolorystyka macierzy pomyłek. Domyślnie 'Blues'.

    Zwraca:
    DataFrame z macierzą pomyłek.
    """
    # Konwersja do numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Znajdź unikalne klasy w danych
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))

    # Mapowanie klas na nazwy
    if class_dict is None:
        class_dict = {cls: str(cls) for cls in unique_classes}

    class_names = [class_dict.get(cls, str(cls)) for cls in unique_classes]

    # Obliczenie macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Tworzenie DataFrame z macierzą pomyłek
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = 'Rzeczywista klasa'
    cm_df.columns.name = 'Predykcja'

    # Obliczenie procentu pomyłek jeśli potrzebne
    if show_percent:
        cm_percent = np.zeros_like(cm, dtype=float)
        for i in range(len(unique_classes)):
            row_sum = np.sum(cm[i, :])
            if row_sum > 0:
                cm_percent[i, :] = cm[i, :] / row_sum * 100

        cm_percent_df = pd.DataFrame(cm_percent, index=class_names, columns=class_names)

    # Tworzenie wykresu
    plt.figure(figsize=figsize)

    if show_percent:
        # Stworzenie subplotów - jeden dla liczb, drugi dla procentów
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        # Macierz liczb
        sns.heatmap(cm_df, annot=True, fmt='d', cmap=cmap, ax=ax1,
                    cbar_kws={'label': 'Liczba próbek'})
        ax1.set_title('Macierz pomyłek (liczby)', fontsize=14, fontweight='bold')

        # Macierz procentów
        sns.heatmap(cm_percent_df, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax2,
                    cbar_kws={'label': 'Procent [%]'})
        ax2.set_title('Macierz pomyłek (procenty)', fontsize=14, fontweight='bold')

        plt.tight_layout()
    else:
        # Tylko macierz liczb
        ax = plt.gca()
        sns.heatmap(cm_df, annot=True, fmt='d', cmap=cmap, ax=ax,
                    cbar_kws={'label': 'Liczba próbek'})
        ax.set_title('Macierz pomyłek', fontsize=14, fontweight='bold')

    # Zapisanie figury
    if save_fig:
        plt.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print(f"Macierz pomyłek została zapisana do pliku '{fig_name}'")