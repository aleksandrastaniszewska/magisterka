import os
from pathlib import Path

import numpy as np

from src.config import mat_files_filtered, path_filtered
from src.dataset import build_dataset
from src.evaluate import evaluate_baselines
from src.interactive import build_interactive
from src.plots import (
    find_extremes,
    plot_all,
    plot_extremes,
    plot_summary,
    plot_subject,
)
from src.preprocessing import read_file
from src.results_report import run_results_report


METHODS = ("CCA", "FBCCA", "TRCA", "eTRCA")
WINDOWS = (0.25, 0.5, 1.0, 1.5, 2.0)
RUN_CHECKS = False
RUN_INTERACTIVE = False


def _ensure_dataset():
    dataset_path = Path(path_filtered) / "dataset.npz"
    if dataset_path.is_file():
        return dataset_path

    print("Nie znaleziono dataset.npz. Buduję zbiór z plików .mat...")
    read_file(path_filtered, mat_files_filtered)
    if not mat_files_filtered:
        raise FileNotFoundError(
            f"Nie znaleziono dataset.npz ani plików .mat w {path_filtered}"
        )
    build_dataset(mat_files_filtered)
    return dataset_path


def _choose_methods():
    print("\nCo przeliczyć?")
    print("  1. CCA")
    print("  2. FBCCA")
    print("  3. TRCA")
    print("  4. eTRCA")
    print("  5. Wszystkie metody (all)")
    print("  Możesz też podać kilka numerów, np. 1,3.")
    while True:
        choice = input("Wybór [5]: ").strip().lower() or "5"
        if choice in ("5", "all", "a", "wszystkie", "wszystko"):
            return METHODS
        try:
            selected = {
                int(value.strip())
                for value in choice.split(",")
            }
        except ValueError:
            print("Wpisz numery metod oddzielone przecinkami albo 5/all.")
            continue
        if selected and selected.issubset({1, 2, 3, 4}):
            method_by_number = dict(zip((1, 2, 3, 4), METHODS))
            return tuple(
                method for number, method in method_by_number.items()
                if number in selected
            )
        print("Wybierz numery 1, 2, 3, 4 albo 5/all.")


def _choose_windows():
    print("\nDla jakich okien liczyć?")
    print("  Dostępne: 0.25, 0.5, 1, 1.5, 2 s")
    print("  Wpisz np. 0.25,1 albo all, aby wybrać wszystkie.")
    while True:
        choice = input("Okna [all]: ").strip().lower() or "all"
        if choice in ("all", "a", "wszystkie", "wszystko"):
            return WINDOWS
        try:
            windows = tuple(
                sorted(
                    {
                        float(value.strip().replace(",", "."))
                        for value in choice.split(",")
                    }
                )
            )
        except ValueError:
            print("Nie rozpoznaję okien. Podaj je po przecinku albo wpisz all.")
            continue
        if windows and all(window in WINDOWS for window in windows):
            return windows
        print("Wybierz spośród: 0.25, 0.5, 1, 1.5, 2 albo wpisz all.")


def _yes_no(prompt, default=True):
    hint = "T/n" if default else "t/N"
    while True:
        choice = input(f"{prompt} [{hint}]: ").strip().lower()
        if not choice:
            return default
        if choice in ("t", "tak", "y", "yes"):
            return True
        if choice in ("n", "nie", "no"):
            return False
        print("Odpowiedz tak albo nie.")


def _choose_cohort():
    print("\nDla kogo liczyć?")
    print("  1. Jedna osoba (wybiorę ID)")
    print("  2. Osoby skrajne (3 najlepsze, 3 najgorsze, 3 z największą różnicą)")
    print("  3. Wszystkie osoby (all; wykresy zbiorcze)")
    while True:
        choice = input("Wybór [3]: ").strip().lower() or "3"
        if choice in ("3", "all", "a", "wszystkie", "wszystko"):
            return "all", None
        if choice in ("1", "osoba"):
            dataset_path = _ensure_dataset()
            with np.load(dataset_path) as dataset:
                subjects = sorted(np.unique(dataset["subject"]).tolist())
            print(f"Dostępne ID: {subjects[0]}–{subjects[-1]} ({len(subjects)} osób)")
            while True:
                subject_id = input("Podaj ID osoby, np. S001: ").strip().upper()
                if subject_id in subjects:
                    return "single", (subject_id,)
                print("Nie znaleziono takiego ID w dataset.npz.")
        if choice in ("2", "skrajne", "ekstrema"):
            extremes = find_extremes()
            subjects = tuple(
                sorted(
                    {
                        person["subject"]
                        for people in extremes.values()
                        for person in people
                    }
                )
            )
            return "extremes", subjects
        print("Wybierz 1, 2 albo 3/all.")


def _run_plots(scope, subjects, methods, windows, dataset_path):
    if scope == "all":
        if methods == METHODS and windows == WINDOWS:
            saved_plots = plot_all(dataset_path=os.fspath(dataset_path))
        else:
            saved_plots = plot_summary(
                methods,
                windows,
                dataset_path=os.fspath(dataset_path),
            )
    elif scope == "single":
        saved_plots = plot_subject(
            subjects[0],
            dataset_path=os.fspath(dataset_path),
            methods=methods,
            windows=windows,
        )
    else:
        extremes = find_extremes(print_results=False)
        saved_plots = plot_extremes(
            extremes,
            methods=methods,
            windows=windows,
            dataset_path=os.fspath(dataset_path),
        )
    print("\nZapisane wykresy:")
    for plot_path in saved_plots:
        print(f"  {plot_path}")


def main():
    if RUN_CHECKS:
        run_results_report()
        return
    if RUN_INTERACTIVE:
        print(f"Zapisano wykres interaktywny: {build_interactive()}")
        return

    scope, subjects = _choose_cohort()
    methods = _choose_methods()
    windows = _choose_windows()
    make_plots = _yes_no("Utworzyć wykresy dla wybranego zakresu?")
    dataset_path = _ensure_dataset()
    evaluate_baselines(
        os.fspath(dataset_path),
        windows=windows,
        subjects=subjects,
        methods=methods,
    )
    if make_plots:
        _run_plots(scope, subjects, methods, windows, dataset_path)


if __name__ == "__main__":
    main()
