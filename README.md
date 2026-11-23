# Klasyfikacja intencji wzrokowej na podstawie sygnału EEG z wykorzystaniem wybranych sieci neuronowych

Projekt poświęcony analizie sygnałów EEG i klasyfikacji intencji wzrokowej z wykorzystaniem sieci neuronowych. Dane pochodzą z eksperymentu SSVEP z publicznie dostępnego zestawu danych Figshare: https://doi.org/10.6084/m9.figshare.13560281

## Struktura repozytorium

`main.py` i `README.md` znajdują się w głównym folderze projektu. Folder `src/` zawiera moduły `config.py` (konfiguracja), `preprocessing.py` (odczyt i filtracja danych), `cca.py` (metoda odniesienia CCA), `dataset.py` (przygotowanie zbiorów danych), `plots.py` (wykresy wyników i diagnostyczne FFT), `model.py` (architektura sieci), `train.py` (trenowanie) i `evaluate.py` (ewaluacja). Folder `online/` zawiera moduły `stream.py` (strumieniowe przetwarzanie sygnału), `decoder.py` (dekodowanie wyników) i `gui.py` (interfejs graficzny). Foldery `src/` i `online/` są pakietami Pythona. Katalog `data/` przechowuje oryginalne pliki `.mat` z eksperymentu. Katalogi `filtered_data/` oraz `przyklady_test/` zawierają lokalnie utworzone dane po filtracji i wizualizacje. Są pomijane przez Git i nie są częścią repozytorium.

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

Opaska mokra i opaska sucha oznaczają dwa warianty rejestracji, a nie dodatkowe kanały EEG. W oryginalnym zestawie indeks wariantu 1 oznacza opaskę mokrą, a indeks 2 opaskę suchą. W Pythonie indeks 0 oznacza opaskę suchą, a indeks 1 mokrą.

## Krótki opis eksperymentu

Badani wykonywali zadanie wyboru celu sterowane wskazówką wzrokową w ramach interfejsu mózg komputer SSVEP. Na ekranie wyświetlano wirtualną klawiaturę przypominającą klawiaturę telefonu. Była to siatka 3 × 4 z 12 migającymi znakami: cyfry od 1 do 9, 0 oraz symbole `*` i `#`. Każdy znak miał przypisaną własną częstotliwość i fazę migotania.

W każdej próbie czerwony kwadrat, czyli wskazówka cue, oznaczał znak, na którym badany miał skupić wzrok. Podczas stymulacji wszystkie cele migotały. Uczestnik patrzył na wskazany cel przez 2 sekundy i starał się nie mrugać. Następnie wyświetlano krótką przerwę wraz z informacją zwrotną o rozpoznanym znaku. Każdy blok zawierał 12 prób, po jednej dla każdego celu. Rejestrację prowadzono osobno z opaską mokrą i suchą, po 10 bloków dla każdego wariantu.

## Pipeline przetwarzania

Funkcja `filter` w `preprocessing.py` realizuje filtrację sygnałów, a gotowe próby są zapisywane do `filtered_data/`. Uruchomienie `main.py` pokazuje menu pozwalające wybrać metody, okna czasowe, same wykresy lub pełne przeliczenie. Wyniki trafiają do `results/baselines.csv`, a wykresy do `results/figures/`. Opaska 0 oznacza suchą, a opaska 1 mokrą. ITR obliczane jest ze wzoru Wolpawa przy czasie wyboru równym długości okna plus 0,5 s przerwy na przeniesienie wzroku.

### Uruchamianie ewaluacji

Uruchom `main.py` z katalogu projektu:

```powershell
.\.venv\Scripts\python.exe main.py
```

Program najpierw pyta o zakres osób: jedna wybrana osoba, osoby skrajne albo wszystkie osoby (`all`). Następnie wybierz metody (`1` = CCA, `2` = FBCCA, `3` = TRCA, `4` = eTRCA; można wpisać kilka, np. `1,3`, albo `5` / `all`) oraz okna (np. `0.25,1` albo `all`). Na końcu można zdecydować, czy generować wykresy. Dla wszystkich osób powstają wykresy zbiorcze (opcja `all` nigdy nie tworzy osobnego zestawu dla każdej osoby), dla jednej osoby wykresy tej osoby, a dla zakresu ekstremów trzy zbiorcze wykresy dla wybranych skrajnych osób. Przy częściowym wyborze wykresy ograniczają się do wybranych metod i okien. Diagnostyczne widma FFT kanału Oz dla celu 0 są zapisywane automatycznie dla mokrej i suchej opaski; korzystają z przefiltrowanych prób `dataset.npz`. Wyniki częściowych przeliczeń zastępują tylko odpowiadające im wiersze CSV, zachowując pozostałe wcześniej policzone dane.

### Kontrola i interaktywne wyniki odniesienia

W `main.py` można ustawić `RUN_CHECKS = True`, aby sprawdzić kompletność i powtarzalność wyników, zapisać tabele podsumowujące i statystyczne, zweryfikować wykresy oraz utworzyć raport `results/RAPORT_wyniki_odniesienia.md`. Kontrola spadku trafności przy dłuższym oknie na suchej opasce jest oznaczana jako ostrzeżenie (WARN), nie błąd. Ustawienie `RUN_INTERACTIVE = True` uruchamia samo generowanie samodzielnego pliku `results/figures/wyniki_interaktywne.html`. Wykresy korzystają z lokalnie osadzonych danych Plotly, więc HTML otwiera się w przeglądarce bez uruchomionego Pythona.

Do raportu statystycznego i wykresów interaktywnych wymagane są pakiety `statsmodels` i `plotly`. Jeśli nie są dostępne w używanym środowisku, zainstaluj je poleceniem `python -m pip install statsmodels plotly`.

Katalog `filtered_data/` można odtworzyć z oryginalnych danych, uruchamiając funkcję `filter`. Uwaga: funkcja ta najpierw usuwa istniejący katalog `filtered_data/`, a następnie zapisuje w nim nowe wyniki.

## Badanie źródłowe

Dane zostały udostępnione w ramach badania [An Open Dataset for Wearable SSVEP Based Brain Computer Interfaces](https://doi.org/10.6084/m9.figshare.13560281). Szczegóły protokołu eksperymentu i rozmieszczenia elektrod opisano również w publikacji [Zhu i in. (2021)](https://doi.org/10.3390/s21041256).
