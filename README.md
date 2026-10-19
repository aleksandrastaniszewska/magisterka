# Klasyfikacja intencji wzrokowej na podstawie sygnału EEG z wykorzystaniem wybranych sieci neuronowych

Projekt poświęcony analizie sygnałów EEG i klasyfikacji intencji wzrokowej z wykorzystaniem sieci neuronowych. Dane pochodzą z eksperymentu SSVEP z publicznie dostępnego zestawu danych Figshare: https://doi.org/10.6084/m9.figshare.13560281

## Struktura repozytorium

Plik `main.py` zawiera główny skrypt analizy danych, a katalog `data/` przechowuje oryginalne pliki `.mat` z eksperymentu. Katalogi `filtered_data/` oraz `przyklady_test/` zawierają lokalnie utworzone dane po filtracji i wizualizacje. Są pomijane przez Git i nie są częścią repozytorium.

## Opis danych

Zbiór danych zawiera sygnały EEG z eksperymentu SSVEP. Pliki w katalogu `filtered_data/` mają format MATLAB `.mat` i zawierają tablicę `data` o wymiarach 8 kanałów, 500 próbek na próbę, 2 wariantów opaski elektrodowej, 10 bloków oraz 12 celów.

Wymiary tablicy w plikach przefiltrowanych to `(kanał, próbka, typ opaski, blok, target)`. Indeksy targetów odpowiadają kolejno znakom `1, 2, 3, 4, 5, 6, 7, 8, 9, 0, *, #`.

### Kanały EEG i mapowanie

Osiem kanałów rejestrujących EEG znajduje się nad obszarami ciemieniowym i potylicznym. Mapowanie indeksów Pythona, liczonych od zera, na pozycje elektrod jest następujące:
    indeks 0 to POz,
    indeks 1 to PO3,
    indeks 2 to PO4,
    indeks 3 to PO5,
    indeks 4 to PO6,
    indeks 5 to Oz,
    indeks 6 to O1,
    a indeks 7 to O2.
Jest to kolejność kanałów użyta w opublikowanym zestawie danych.

Opaska mokra i opaska sucha oznaczają dwa warianty rejestracji, a nie dodatkowe kanały EEG. W oryginalnym zestawie indeks wariantu 1 oznacza opaskę mokrą, a indeks 2 opaskę suchą. W Pythonie odpowiadają im indeksy 0 i 1.

## Krótki opis eksperymentu

Badani wykonywali zadanie wyboru celu sterowane wskazówką wzrokową w ramach interfejsu mózg komputer SSVEP. Na ekranie wyświetlano wirtualną klawiaturę przypominającą klawiaturę telefonu. Była to siatka 3 × 4 z 12 migającymi znakami: cyfry od 1 do 9, 0 oraz symbole `*` i `#`. Każdy znak miał przypisaną własną częstotliwość i fazę migotania.

W każdej próbie czerwony kwadrat, czyli wskazówka cue, oznaczał znak, na którym badany miał skupić wzrok. Podczas stymulacji wszystkie cele migotały. Uczestnik patrzył na wskazany cel przez 2 sekundy i starał się nie mrugać. Następnie wyświetlano krótką przerwę wraz z informacją zwrotną o rozpoznanym znaku. Każdy blok zawierał 12 prób, po jednej dla każdego celu. Rejestrację prowadzono osobno z opaską mokrą i suchą, po 10 bloków dla każdego wariantu.

## Pipeline przetwarzania

Funkcja `filter` w `main.py` realizuje usuwanie trendu, filtrację pasmową w zakresie od 6 do 50 Hz oraz filtrację notch 50 Hz, a następnie wycina okno próby. Jej wywołanie jest obecnie wyłączone. Domyślnie skrypt oczekuje gotowych plików w katalogu `filtered_data/`, oblicza dla każdej próby 12 wyników CCA i zwraca je w tablicy `x_features` razem z etykietami `y_labels` oraz identyfikatorami `subject_ids`. Wyniki pozostają w pamięci i nie są zapisywane do pliku. Diagnostyczne wykresy FFT także są obecnie wyłączone.

Katalog `filtered_data/` można odtworzyć z oryginalnych danych, uruchamiając funkcję `filter`. Uwaga: funkcja ta najpierw usuwa istniejący katalog `filtered_data/`, a następnie zapisuje w nim nowe wyniki.

## Badanie źródłowe

Dane zostały udostępnione w ramach badania [An Open Dataset for Wearable SSVEP Based Brain Computer Interfaces](https://doi.org/10.6084/m9.figshare.13560281). Szczegóły protokołu eksperymentu i rozmieszczenia elektrod opisano również w publikacji [Zhu i in. (2021)](https://doi.org/10.3390/s21041256).
