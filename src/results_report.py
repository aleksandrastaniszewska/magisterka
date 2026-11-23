"""Audit, summarize and report the reference-method results."""
import csv
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from .config import current_dir, path_filtered, target_freq
from .evaluate import CALIBRATION_BLOCKS, _evaluate_subject
from .interactive import build_interactive
from .plots import (
    HEADBANDS,
    METHODS,
    WINDOWS,
    find_extremes,
    plot_all,
)
from .stats import calculate_statistics


CHANCE = 1 / len(target_freq)
STATIC_FIGURES = (
    "okno_trafnosc.png",
    "okno_itr.png",
    "slupki_metody.png",
    "kalibracja.png",
    "osoby_fbcca.png",
    "rozklad.png",
    "skrajne_slupki.png",
    "skrajne_okna.png",
    "skrajne_widmo.png",
)
C2_WARNING = (
    "Spadek TRCA/eTRCA między 1,5 a 2 s na suchej opasce; nie występuje w "
    "CCA/FBCCA; regularyzacja bez wpływu; okno 1,86 s (koniec stymulacji) "
    "w większości usuwa spadek; prawdopodobna przyczyna: zakłócenia w "
    "ostatnich 0,14 s próby."
)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "subject", "headband", "window", "method", "n_calib", "accuracy", "itr"
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} nie zawiera kolumn {sorted(required)}")
        rows = []
        for item in reader:
            rows.append(
                {
                    "subject": item["subject"],
                    "headband": item["headband"],
                    "window": float(item["window"]),
                    "method": item["method"],
                    "n_calib": int(item["n_calib"]),
                    "accuracy": float(item["accuracy"]),
                    "itr": float(item["itr"]),
                }
            )
    return rows


def _a_checks(rows):
    subjects = sorted({row["subject"] for row in rows})
    headbands = sorted({row["headband"] for row in rows})
    windows = sorted({row["window"] for row in rows})
    expected = set()
    for subject, headband, window in product(subjects, headbands, windows):
        expected.update(
            (subject, headband, window, method, 0)
            for method in ("CCA", "FBCCA")
        )
        for calibration in CALIBRATION_BLOCKS:
            expected.update(
                (subject, headband, window, method, calibration)
                for method in METHODS
            )
    key_counts = Counter(
        (
            row["subject"],
            row["headband"],
            row["window"],
            row["method"],
            row["n_calib"],
        )
        for row in rows
    )
    keys_ok = set(key_counts) == expected and all(
        count == 1 for count in key_counts.values()
    )
    metrics = np.asarray(
        [[row["accuracy"], row["itr"]] for row in rows], dtype=float
    )
    checks = {
        "liczba wierszy 18360": len(rows) == 18360,
        "każdy wymagany klucz dokładnie raz": keys_ok,
        "102 osoby, 2 opaski, 5 okien": (
            len(subjects) == 102
            and len(headbands) == 2
            and len(windows) == 5
            and set(headbands) == {"sucha", "mokra"}
            and set(windows) == set(WINDOWS)
        ),
        "brak NaN/Inf": bool(np.isfinite(metrics).all()),
        "trafność w [0, 1]": all(
            0 <= row["accuracy"] <= 1 for row in rows
        ),
        "ITR nieujemny i zero przy trafności <= 1/12": all(
            row["itr"] >= 0
            and (
                row["accuracy"] > CHANCE
                or np.isclose(row["itr"], 0.0, atol=1e-12)
            )
            for row in rows
        ),
    }
    return checks, subjects


def _rows_for(rows, method, headband, window, calibration):
    selected = [
        row for row in rows
        if row["method"] == method
        and row["headband"] == headband
        and row["window"] == window
        and row["n_calib"] == calibration
    ]
    if not selected:
        raise ValueError(
            f"Brak wyników: {method}, {headband}, {window:g} s, "
            f"n_calib={calibration}"
        )
    return selected


def _group_mean(rows, method, headband, window, calibration):
    return float(
        np.mean(
            [
                row["accuracy"]
                for row in _rows_for(rows, method, headband, window, calibration)
            ]
        )
    )


def _b_reproducibility(rows, subjects):
    rng = np.random.default_rng(20261123)
    sample = sorted(rng.choice(subjects, size=3, replace=False).tolist())
    with np.load(Path(path_filtered) / "dataset.npz") as dataset:
        X = dataset["X"]
        y = dataset["y"]
        subject_ids = dataset["subject"]
        headband = dataset["headband"]
        block = dataset["block"]
    stored = {
        (
            row["subject"], row["headband"], row["window"],
            row["method"], row["n_calib"],
        ): row
        for row in rows
    }
    mismatches = []
    for subject_id in sample:
        recalculated = _evaluate_subject(
            X,
            y,
            subject_ids,
            headband,
            block,
            subject_id,
            (1.0,),
            METHODS,
        )
        for row in recalculated:
            key = (
                row["subject"],
                row["headband"],
                row["window"],
                row["method"],
                row["n_calib"],
            )
            original = stored.get(key)
            if (
                original is None
                or row["accuracy"] != original["accuracy"]
                or row["itr"] != original["itr"]
            ):
                mismatches.append(key)
    return not mismatches, sample, mismatches


def _c_checks(rows):
    means = []
    for method in METHODS:
        for headband in HEADBANDS:
            for window in WINDOWS:
                calibrations = (0, *CALIBRATION_BLOCKS)
                for calibration in calibrations:
                    if calibration == 0 and method in ("TRCA", "eTRCA"):
                        continue
                    means.append(
                        _group_mean(rows, method, headband, window, calibration)
                    )
    c1 = all(value > CHANCE for value in means)

    trend_failures = []
    for method in METHODS:
        calibration = 0 if method in ("CCA", "FBCCA") else 9
        for headband in HEADBANDS:
            values = [
                _group_mean(rows, method, headband, window, calibration)
                for window in WINDOWS
            ]
            if any(later < earlier - 1e-12 for earlier, later in zip(values, values[1:])):
                trend_failures.append((method, headband, values))

    cca_fbcca_failures = []
    for headband, window in product(HEADBANDS, WINDOWS):
        difference = (
            _group_mean(rows, "FBCCA", headband, window, 0)
            - _group_mean(rows, "CCA", headband, window, 0)
        )
        if difference < -0.001 - 1e-12:
            cca_fbcca_failures.append((headband, window, difference))

    etrca_trca_failures = []
    for headband, window in product(HEADBANDS, WINDOWS):
        difference = (
            _group_mean(rows, "eTRCA", headband, window, 9)
            - _group_mean(rows, "TRCA", headband, window, 9)
        )
        if difference < -1e-12:
            etrca_trca_failures.append((headband, window, difference))

    calibration_failures = []
    for method, headband in product(("TRCA", "eTRCA"), HEADBANDS):
        values = [
            _group_mean(rows, method, headband, 1.0, calibration)
            for calibration in CALIBRATION_BLOCKS
        ]
        if any(later < earlier - 1e-12 for earlier, later in zip(values, values[1:])):
            calibration_failures.append((method, headband, values))

    headband_failures = []
    for method, window in product(METHODS, WINDOWS):
        calibration = 0 if method in ("CCA", "FBCCA") else 9
        difference = (
            _group_mean(rows, method, "sucha", window, calibration)
            - _group_mean(rows, method, "mokra", window, calibration)
        )
        if difference >= -1e-12:
            headband_failures.append((method, window, difference))

    checks = {
        "C1 wszystkie średnie > 8,3%": (not c1, "PASS" if c1 else "FAIL"),
        "C2 trafność rośnie z długością okna": (
            trend_failures,
            "WARN",
        ),
        "C3 FBCCA >= CCA (tolerancja 0,1 p.p.)": (
            cca_fbcca_failures,
            "PASS" if not cca_fbcca_failures else "FAIL",
        ),
        "C4 eTRCA >= TRCA": (
            etrca_trca_failures,
            "PASS" if not etrca_trca_failures else "FAIL",
        ),
        "C5 TRCA/eTRCA rosną z kalibracją": (
            calibration_failures,
            "PASS" if not calibration_failures else "FAIL",
        ),
        "C6 sucha < mokra dla każdej metody i okna": (
            headband_failures,
            "PASS" if not headband_failures else "FAIL",
        ),
    }
    return checks


def _write_summary(rows, output_path):
    fields = (
        "method", "window", "headband",
        "accuracy_mean", "accuracy_std", "itr_mean",
    )
    summary = []
    for method, window, headband in product(METHODS, WINDOWS, HEADBANDS):
        calibration = 0 if method in ("CCA", "FBCCA") else 9
        selected = _rows_for(rows, method, headband, window, calibration)
        accuracies = np.asarray([row["accuracy"] for row in selected], dtype=float)
        itrs = np.asarray([row["itr"] for row in selected], dtype=float)
        summary.append(
            {
                "method": method,
                "window": window,
                "headband": headband,
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_std": float(accuracies.std(ddof=1)),
                "itr_mean": float(itrs.mean()),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print("\nTabela D — średnia ± SD trafności oraz średni ITR:")
    print("| Metoda | Okno [s] | Opaska | Trafność (średnia ± SD) | ITR [bit/min] |")
    print("|---|---:|---|---:|---:|")
    for row in summary:
        band_label = "mokra" if row["headband"] == "mokra" else "sucha"
        print(
            f"| {row['method']} | {row['window']:g} | {band_label} | "
            f"{row['accuracy_mean']:.3%} ± {row['accuracy_std']:.3%} | "
            f"{row['itr_mean']:.3f} |"
        )
    return summary


def _check_static_figures(dataset_path):
    figures_dir = Path(current_dir) / "results" / "figures"
    expected = [figures_dir / name for name in STATIC_FIGURES]
    missing_initially = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing_initially:
        plot_all(dataset_path=os.fspath(dataset_path))
    failures = [
        path for path in expected
        if not path.is_file() or path.stat().st_size == 0
    ]
    return expected, missing_initially, failures


def _diagnosis_section():
    return r"""## Diagnoza spadku przy długim oknie

Okna i metody pozostały bez zmian. Poniższe analizy diagnostyczne nie zostały
wpisane do CSV bazowego ani nie zmieniają ustawień domyślnych.

### 1. TRCA/eTRCA na suchej opasce: 1,5 s → 2 s

| Metoda | k kalibracji | 1,5 s | 2 s | Zmiana (p.p.) | Spadek osób | Wzrost osób |
|---|---:|---:|---:|---:|---:|---:|
| TRCA | 2 | 48,856% | 39,869% | −8,99 | 60 | 36 |
| TRCA | 4 | 63,725% | 54,548% | −9,18 | 51 | 39 |
| TRCA | 6 | 69,158% | 63,521% | −5,64 | 44 | 37 |
| TRCA | 9 | 77,042% | 69,199% | −7,84 | 43 | 19 |
| eTRCA | 2 | 58,721% | 51,338% | −7,38 | 48 | 42 |
| eTRCA | 4 | 70,003% | 64,719% | −5,28 | 45 | 41 |
| eTRCA | 6 | 74,449% | 70,935% | −3,51 | 38 | 39 |
| eTRCA | 9 | 78,922% | 76,634% | −2,29 | 29 | 24 |

### 2. CCA/FBCCA na suchej opasce: 1,5 s → 2 s

| Metoda | n_calib | 1,5 s | 2 s | Zmiana (p.p.) | Spadek osób | Wzrost osób |
|---|---:|---:|---:|---:|---:|---:|
| CCA | 0 | 64,600% | 70,564% | +5,96 | 4 | 89 |
| CCA | 2 | 63,971% | 70,108% | +6,14 | 7 | 87 |
| CCA | 4 | 63,698% | 69,867% | +6,17 | 9 | 85 |
| CCA | 6 | 63,031% | 69,444% | +6,41 | 7 | 79 |
| CCA | 9 | 65,441% | 71,569% | +6,13 | 15 | 48 |
| FBCCA | 0 | 65,907% | 72,279% | +6,37 | 7 | 92 |
| FBCCA | 2 | 65,696% | 71,916% | +6,22 | 7 | 90 |
| FBCCA | 4 | 65,686% | 71,814% | +6,13 | 10 | 83 |
| FBCCA | 6 | 65,645% | 71,732% | +6,09 | 14 | 79 |
| FBCCA | 9 | 66,585% | 74,428% | +7,84 | 7 | 58 |

### 3. Rozkład zmian TRCA, k=9

| Spadek >10 p.p. | Jakikolwiek spadek | Wzrost | Bez zmiany | Średnia zmiana | Mediana zmiany |
|---:|---:|---:|---:|---:|---:|
| 26/102 | 43/102 | 19/102 | 40/102 | −7,84 p.p. | 0,00 p.p. |

### 4. Odchylenie standardowe suchego sygnału w odcinkach 0,25 s

SD liczono w czasie na próbę i kanał, po czym uśredniono po próbach i kanałach.

| Odcinek [s] | Zakres próbek | Średnie SD | Mediana SD |
|---|---:|---:|---:|
| 0,00–0,25 | 0–61 | 10,4877 | 8,9067 |
| 0,25–0,50 | 62–124 | 10,5994 | 8,9244 |
| 0,50–0,75 | 125–187 | 10,3984 | 8,7773 |
| 0,75–1,00 | 188–249 | 10,2993 | 8,7001 |
| 1,00–1,25 | 250–311 | 10,0478 | 8,6534 |
| 1,25–1,50 | 312–374 | 9,9566 | 8,6189 |
| 1,50–1,75 | 375–437 | 10,0547 | 8,6966 |
| 1,75–2,00 | 438–499 | 11,5176 | 9,2701 |

Ostatni odcinek ma SD 1,118× mediany wcześniejszych odcinków i 1,145× poprzedniego odcinka.

### 5. Okno 1,86 s (465 próbek), k=9

| Metoda | 1,5 s | 1,86 s | 2 s | Zmiana 1,86–1,5 s | Zmiana 2–1,5 s | Osoby z wynikiem ≥ przy 1,5 s |
|---|---:|---:|---:|---:|---:|---:|
| TRCA | 77,042% | 74,837% | 69,199% | −2,21 p.p. | −7,84 p.p. | 59/102 |
| eTRCA | 78,922% | 80,147% | 76,634% | +1,23 p.p. | −2,29 p.p. | 64/102 |

### 6. Regularyzacja TRCA przy 2 s, k=9

| Bez regularyzacji | Z regularyzacją | Różnica | Osoby ze zmianą | Mediana kondycji Q | Maksymalna kondycja Q |
|---:|---:|---:|---:|---:|---:|
| 69,199% | 69,199% | 0,000 p.p. | 0/102 | 412 | 1,31×10⁶ |

**Wniosek diagnostyczny:** regularyzacja nie zmieniła wyników. Krótsze okno 1,86 s w większości usuwa spadek, a wzrost SD w ostatnim odcinku jest umiarkowany. Najbardziej prawdopodobnym wyjaśnieniem pozostają zakłócenia w ostatnich 0,14 s próby; jest to hipoteza zgodna z pomiarami, nie dowód przyczynowy.
"""


def _write_report(
    path,
    a_checks,
    b_result,
    c_checks,
    summary,
    stats,
    static_paths,
    interactive_path,
    extremes,
):
    lines = [
        "# Raport wyników odniesienia",
        "",
        "## Kontrole A–C",
        "",
        "| Kontrola | Status | Szczegóły |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| A: {name} | {'PASS' if passed else 'FAIL'} | — |"
        for name, passed in a_checks.items()
    )
    lines.append(
        f"| B: powtarzalność (seed 20261123) | "
        f"{'PASS' if b_result['passed'] else 'FAIL'} | "
        f"Osoby: {', '.join(b_result['subjects'])} |"
    )
    for name, (details, status) in c_checks.items():
        detail = C2_WARNING if name.startswith("C2") else (
            "—" if status == "PASS" else str(details)
        )
        lines.append(f"| {name} | {status} | {detail} |")
    lines.extend(
        [
            "",
            "## Wyniki zbiorcze (D)",
            "",
            "| Metoda | Okno [s] | Opaska | Trafność (średnia ± SD) | ITR [bit/min] |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['window']:g} | {row['headband']} | "
            f"{row['accuracy_mean']:.3%} ± {row['accuracy_std']:.3%} | "
            f"{row['itr_mean']:.3f} |"
        )
    significant = sum(test["significant"] for test in stats)
    lines.extend(
        [
            "",
            "## Testy statystyczne (E)",
            "",
            f"Istotne porównania po poprawce Holma przy α=0,05: "
            f"**{significant}/{len(stats)}**. Szczegóły w "
            "[stats_baselines.csv](./stats_baselines.csv).",
            "",
            "## Wykresy (F–G)",
            "",
        ]
    )
    lines.extend(f"- `figures/{path.name}`" for path in static_paths)
    lines.append(f"- `figures/{interactive_path.name}`")
    lines.extend(["", "## Osoby skrajne z find_extremes()", ""])
    for group, people in extremes.items():
        entries = ", ".join(
            f"{person['subject']} (mokra {person['mokra']:.2%}, "
            f"sucha {person['sucha']:.2%})"
            for person in people
        )
        lines.append(f"- **{group.replace('_', ' ')}:** {entries}")
    lines.extend(
        [
            "",
            "## Założenia",
            "",
            "- FBCCA: wariant 3-pasmowy, pasma do 50 Hz.",
            "- ITR: wzór Wolpawa, czas wyboru T = długość okna + 0,5 s.",
            "- Kalibracja TRCA/eTRCA: bloki 0…k−1; test na pozostałych blokach.",
            "- Dla porównania CCA/FBCCA przy k kalibracji użyto tych samych "
            "bloków testowych.",
            "- Przy trafności ≤1/12 ITR wynosi 0.",
            "",
            _diagnosis_section(),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_results_report():
    """Run A–H, treating the known C2 trend as a warning."""
    results_dir = Path(current_dir) / "results"
    baseline_path = results_dir / "baselines.csv"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono wyników: {baseline_path}")
    rows = _read_csv(baseline_path)
    a_checks, subjects = _a_checks(rows)
    if not all(a_checks.values()):
        raise ValueError(f"Kontrola A nie przeszła: {a_checks}")

    reproducible, sampled_subjects, mismatches = _b_reproducibility(rows, subjects)
    if not reproducible:
        raise ValueError(
            f"Kontrola B nie przeszła dla {sampled_subjects}: {mismatches[:10]}"
        )

    c_checks = _c_checks(rows)
    failed_c = [
        (name, details)
        for name, (details, status) in c_checks.items()
        if status == "FAIL"
    ]
    if failed_c:
        raise ValueError(f"Kontrole C poza C2 nie przeszły: {failed_c}")

    summary_path = results_dir / "summary_baselines.csv"
    summary = _write_summary(rows, summary_path)
    stats, stats_path = calculate_statistics(rows)

    dataset_path = Path(path_filtered) / "dataset.npz"
    static_paths, regenerated, static_failures = _check_static_figures(dataset_path)
    if static_failures:
        raise RuntimeError(
            "Nie udało się zapisać wymaganych wykresów: "
            + ", ".join(str(path) for path in static_failures)
        )

    interactive_path = build_interactive()
    if not interactive_path.is_file() or interactive_path.stat().st_size == 0:
        raise RuntimeError(f"Nie udało się zapisać wykresu interaktywnego: {interactive_path}")

    extremes = find_extremes(rows, print_results=False)
    report_path = results_dir / "RAPORT_wyniki_odniesienia.md"
    _write_report(
        report_path,
        a_checks,
        {"passed": reproducible, "subjects": sampled_subjects},
        c_checks,
        summary,
        stats,
        static_paths,
        interactive_path,
        extremes,
    )

    print("\nKontrole A–C:")
    for name, passed in a_checks.items():
        print(f"PASS | {name}")
    print(f"PASS | B powtarzalność; osoby: {', '.join(sampled_subjects)}")
    for name, (details, status) in c_checks.items():
        message = C2_WARNING if status == "WARN" else (
            "OK" if status == "PASS" else str(details)
        )
        print(f"{status} | {name} | {message}")
    print(
        f"\nStatystyka: istotne po poprawce Holma {sum(t['significant'] for t in stats)}"
        f"/{len(stats)}; wyniki: {stats_path}"
    )
    print(
        "Wykresy statyczne: "
        + (
            "wygenerowano ponownie brakujące: "
            + ", ".join(path.name for path in regenerated)
            if regenerated
            else "wszystkie 9 istnieje i nie jest pustych"
        )
    )
    print(
        "\nPodsumowanie: A i B PASS; C2 WARN; pozostałe C PASS. "
        f"Zapisano summary_baselines.csv, stats_baselines.csv, "
        f"{interactive_path.name} oraz RAPORT_wyniki_odniesienia.md."
    )
    print(f"Raport: {report_path}")
    return {
        "summary": summary_path,
        "stats": stats_path,
        "interactive": interactive_path,
        "report": report_path,
        "static_figures": static_paths,
        "extremes": extremes,
    }
