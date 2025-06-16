import matplotlib.pyplot as plt
import numpy as np

def utworz_wykres_liniowy(dane, zapisz_jako=None, figsize=(10, 6)):
    """
    Tworzy wykres liniowy na podstawie danych w słowniku.

    Parametry:
    -----------
    dane : dict
        Słownik zawierający dane do wykresu w formacie {nazwa_serii: [wartości_y]}
        lub {nazwa_serii: (wartości_x, wartości_y)}
    zapisz_jako : str, opcjonalny
        Nazwa pliku do zapisania wykresu (np. 'wykres.png')
    figsize : tuple, opcjonalny
        Rozmiar wykresu (szerokość, wysokość) w calach

    Zwraca:
    --------
    fig, ax : obiekty matplotlib
        Obiekty figure i axes dla utworzonego wykresu
    """
    # Zmienne wewnętrzne dla tytułu i etykiet osi
    etykieta_x = "Oś X"
    etykieta_y = "Oś Y"


    x = sorted(dane.keys())
    y = [dane[key] for key in x]

    # Tworzenie nowego wykresu
    fig, ax = plt.subplots(figsize=figsize)

    # Rysowanie linii
    ax.plot(x, y, marker='o', linestyle='-', color='blue')

    # Dodanie tytułu i etykiet
    ax.set_xlabel(etykieta_x)
    ax.set_ylabel(etykieta_y)

    # Dodanie siatki
    ax.grid(True, linestyle='--', alpha=0.7)

    # Ustawienie osi X, aby zawierała tylko wartości z kluczy słownika
    plt.xticks(x)

    # Poprawienie układu
    plt.tight_layout()

    # Zapisanie wykresu, jeśli podano nazwę pliku
    if zapisz_jako:
        plt.savefig(zapisz_jako, dpi=300, bbox_inches='tight')
        print(f"Wykres zapisany jako: {zapisz_jako}")