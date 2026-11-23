"""Plots for the precomputed reference-method results."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy.signal import detrend

from .config import current_dir, fs, path_filtered, target_freq


METHODS = ("CCA", "FBCCA", "TRCA", "eTRCA")
METHOD_LABELS = {
    "CCA": "CCA",
    "FBCCA": "FBCCA (3 pasma)",
    "TRCA": "TRCA",
    "eTRCA": "eTRCA",
}
METHOD_COLORS = {
    "CCA": "#80bde9",
    "FBCCA": "#be90dd",
    "TRCA": "#4f9954",
    "eTRCA": "#d178b6",
}
HEADBANDS = ("mokra", "sucha")
HEADBAND_LABELS = {"mokra": "Opaska mokra", "sucha": "Opaska sucha"}
WINDOWS = (0.25, 0.5, 1.0, 1.5, 2.0)
CHANCE = 1 / len(target_freq)
CSV_FIELDS = {
    "subject",
    "headband",
    "window",
    "method",
    "n_calib",
    "accuracy",
    "itr",
}


def _read_results(csv_path=None):
    path = Path(csv_path) if csv_path is not None else Path(current_dir) / "results" / "baselines.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku wyników: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not CSV_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Plik {path} nie zawiera wymaganych kolumn {sorted(CSV_FIELDS)}")
        rows = []
        for row in reader:
            parsed = {
                "subject": row["subject"],
                "headband": row["headband"],
                "window": float(row["window"]),
                "method": row["method"],
                "n_calib": int(row["n_calib"]),
                "accuracy": float(row["accuracy"]),
                "itr": float(row["itr"]),
            }
            if parsed["headband"] not in HEADBANDS or parsed["method"] not in METHODS:
                raise ValueError(f"Nieznana opaska lub metoda w {path}: {row}")
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Plik wyników jest pusty: {path}")
    return rows


def _selected(rows, method, window, headband, n_calib):
    return [
        row
        for row in rows
        if row["method"] == method
        and row["window"] == window
        and row["headband"] == headband
        and row["n_calib"] == n_calib
    ]


def _subject_values(rows, metric):
    return {row["subject"]: row[metric] for row in rows}


def _mean_sem(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Brak wyników do obliczenia średniej")
    sem = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
    return float(values.mean()), float(sem)


def _save(fig, output_dir, filename):
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _prepare_output(output_dir=None):
    path = Path(output_dir) if output_dir is not None else Path(current_dir) / "results" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_fft(signal, title, target=None, output_dir=None, filename="diagnostyczne_fft.png"):
    """Save an FFT magnitude plot for one signal or the mean spectrum of trials."""
    signals = np.asarray(signal, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    if signals.ndim != 2 or signals.shape[-1] < 2:
        raise ValueError("Sygnał FFT musi mieć kształt (próbki,) lub (próby, próbki)")

    window = np.hanning(signals.shape[-1])
    centered = detrend(signals, axis=-1)
    magnitudes = np.abs(np.fft.rfft(centered * window, axis=-1))
    magnitudes *= 2 / window.sum()
    frequencies = np.fft.rfftfreq(signals.shape[-1], d=1 / fs)

    fig, ax = plt.subplots(figsize=(12, 5))
    mean_spectrum = magnitudes.mean(axis=0)
    ax.plot(frequencies, mean_spectrum, label="Średnie widmo")
    if target is not None:
        expected = float(target_freq[target])
        harmonics = tuple(expected * harmonic for harmonic in range(1, 4))
        for harmonic in range(1, 4):
            frequency = harmonics[harmonic - 1]
            if frequency <= frequencies[-1]:
                ax.axvline(
                    frequency,
                    linestyle="--" if harmonic == 1 else ":",
                    label=f"{harmonic}. harmoniczna: {frequency:g} Hz",
                )
        _label_spectrum_harmonics(
            ax, frequencies, mean_spectrum, harmonics=harmonics
        )
    ax.set_xlim(0, min(60, frequencies[-1]))
    ax.set_xticks(np.arange(0, min(61, frequencies[-1] + 1), 5))
    ax.set_xlabel("Częstotliwość [Hz]")
    ax.set_ylabel("Amplituda widma FFT [jedn. względne]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save(fig, _prepare_output(output_dir), filename)


def plot_diagnostic_fft(subject, dataset_path=None, output_dir=None):
    """Save wet and dry Oz spectra for target 0 from the filtered dataset."""
    path = (
        Path(dataset_path)
        if dataset_path is not None
        else Path(path_filtered) / "dataset.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono zbioru danych do diagnostycznego FFT: {path}")
    with np.load(path) as dataset:
        X = dataset["X"]
        y = dataset["y"]
        subjects = dataset["subject"]
        headbands = dataset["headband"]
    if X.ndim != 3 or X.shape[1:] != (8, 500):
        raise ValueError(f"Oczekiwano X o wymiarach (próby, 8, 500), otrzymano {X.shape}")

    target_dir = _prepare_output(output_dir)
    saved = []
    for headband, label in ((1, "mokra"), (0, "sucha")):
        selected = (subjects == subject) & (headbands == headband) & (y == 0)
        trials = X[selected, 5, :]
        if len(trials) == 0:
            raise ValueError(f"Brak prób celu 0 dla osoby {subject} i opaski {label}")
        saved.append(
            plot_fft(
                trials,
                f"Diagnostyczne widmo FFT kanału Oz — {subject}, opaska {label}, cel 0",
                target=0,
                output_dir=target_dir,
                filename=f"diagnostyczne_fft_{subject}_{label}.png",
            )
        )
    return saved


def _headband_panels(fig, axes, ylabel, title):
    for ax, headband in zip(axes, HEADBANDS):
        ax.set_title(HEADBAND_LABELS[headband])
        ax.set_xlabel("Długość okna [s]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WINDOWS, [str(w).replace(".", ",") for w in WINDOWS])
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)


def _accuracy_line(ax):
    ax.axhline(CHANCE, color="black", linestyle="--", linewidth=1, label="Poziom zgadywania (8,3%)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))


def _format_value(value, metric="accuracy"):
    if metric == "accuracy":
        return f"{value:.1%}"
    return f"{value:.1f}"


def _label_bars(ax, containers, metric="accuracy", fontsize=7, rotation=0):
    for container in containers:
        ax.bar_label(
            container,
            labels=[_format_value(bar.get_height(), metric) for bar in container],
            padding=2,
            fontsize=fontsize,
            rotation=rotation,
        )
    ax.margins(y=0.14)


def _label_line_points(ax, x_values, y_values, metric="accuracy", offset=7, fontsize=7):
    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            _format_value(y_value, metric),
            (x_value, y_value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset >= 0 else "top",
            fontsize=fontsize,
        )


def _label_spectrum_harmonics(
    ax, frequencies, spectrum, color="black", harmonics=(9.25, 18.5, 27.75)
):
    for harmonic in harmonics:
        index = int(np.abs(frequencies - harmonic).argmin())
        amplitude = float(spectrum[index])
        ax.annotate(
            f"{harmonic:g} Hz: {amplitude:.3f}",
            (frequencies[index], amplitude),
            xytext=(3, 5),
            textcoords="offset points",
            color=color,
            fontsize=7,
            rotation=35,
        )


def _plot_window_metric(
    rows,
    output_dir,
    metric,
    filename,
    ylabel,
    title,
    methods=METHODS,
    windows=WINDOWS,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, headband in zip(axes, HEADBANDS):
        for method_index, method in enumerate(methods):
            means, errors = [], []
            n_calib = 0 if method in ("CCA", "FBCCA") else 9
            for window in windows:
                selected = _selected(rows, method, window, headband, n_calib)
                mean, sem = _mean_sem([row[metric] for row in selected])
                means.append(mean)
                errors.append(sem)
            ax.errorbar(
                windows,
                means,
                yerr=errors,
                color=METHOD_COLORS[method],
                marker="o",
                capsize=3,
                label=METHOD_LABELS[method],
            )
            _label_line_points(
                ax,
                windows,
                means,
                metric=metric,
                offset=(8, 16, -10, -18)[method_index % 4],
            )
        if metric == "accuracy":
            _accuracy_line(ax)
        ax.legend()
    _headband_panels(fig, axes, ylabel, title)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_window_accuracy(rows, output_dir, methods=METHODS, windows=WINDOWS):
    return _plot_window_metric(
        rows,
        output_dir,
        "accuracy",
        "okno_trafnosc.png",
        "Średnia trafność",
        "Trafność metod odniesienia w zależności od długości okna",
        methods,
        windows,
    )


def plot_window_itr(rows, output_dir, methods=METHODS, windows=WINDOWS):
    return _plot_window_metric(
        rows,
        output_dir,
        "itr",
        "okno_itr.png",
        "Średnie ITR [bit/min]",
        "Szybkość transferu informacji w zależności od długości okna",
        methods,
        windows,
    )


def plot_method_bars(rows, output_dir, methods=METHODS, windows=WINDOWS):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    group_positions = np.arange(len(windows))
    bar_width = 0.8 / len(methods)
    for ax, headband in zip(axes, HEADBANDS):
        for method_index, method in enumerate(methods):
            means, errors = [], []
            n_calib = 0 if method in ("CCA", "FBCCA") else 9
            for window in windows:
                selected = _selected(rows, method, window, headband, n_calib)
                mean, sem = _mean_sem([row["accuracy"] for row in selected])
                means.append(mean)
                errors.append(sem)
            offset = (method_index - (len(methods) - 1) / 2) * bar_width
            bars = ax.bar(
                group_positions + offset,
                means,
                bar_width,
                yerr=errors,
                capsize=2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            _label_bars(ax, (bars,), rotation=90)
        _accuracy_line(ax)
        ax.set_title(HEADBAND_LABELS[headband])
        ax.set_xlabel("Długość okna [s]")
        ax.set_xticks(group_positions, [str(w).replace(".", ",") for w in windows])
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Średnia trafność")
    fig.suptitle("Porównanie metod odniesienia dla różnych długości okna")
    fig.tight_layout()
    return _save(fig, output_dir, "slupki_metody.png")


def plot_calibration(rows, output_dir, methods=METHODS):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    calibrations = (2, 4, 6, 9)
    for ax, headband in zip(axes, HEADBANDS):
        for method_index, method in enumerate(methods):
            means, errors = [], []
            used_all_blocks = False
            for n_calib in calibrations:
                selected = _selected(rows, method, 1.0, headband, n_calib)
                if not selected and method in ("CCA", "FBCCA"):
                    selected = _selected(rows, method, 1.0, headband, 0)
                    used_all_blocks = True
                mean, sem = _mean_sem([row["accuracy"] for row in selected])
                means.append(mean)
                errors.append(sem)
            ax.errorbar(
                calibrations,
                means,
                yerr=errors,
                color=METHOD_COLORS[method],
                marker="o",
                capsize=3,
                label=(
                    f"{METHOD_LABELS[method]} (wszystkie bloki)"
                    if used_all_blocks
                    else METHOD_LABELS[method]
                ),
            )
            _label_line_points(
                ax,
                calibrations,
                means,
                offset=(8, 16, -10, -18)[method_index % 4],
            )
        _accuracy_line(ax)
        ax.set_title(HEADBAND_LABELS[headband])
        ax.set_xlabel("Liczba bloków kalibracyjnych")
        ax.set_xticks(calibrations)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")
    axes[0].set_ylabel("Średnia trafność (okno 1 s)")
    fig.suptitle("Wpływ liczby bloków kalibracyjnych na trafność")
    fig.tight_layout()
    return _save(fig, output_dir, "kalibracja.png")


def plot_subject_fbcca(rows, output_dir):
    subjects = sorted({row["subject"] for row in rows})
    wet = _subject_values(_selected(rows, "FBCCA", 1.0, "mokra", 0), "accuracy")
    dry = _subject_values(_selected(rows, "FBCCA", 1.0, "sucha", 0), "accuracy")
    missing = (set(subjects) - wet.keys()) | (set(subjects) - dry.keys())
    if missing:
        raise ValueError(f"Brak wyników FBCCA dla osób: {sorted(missing)}")
    subjects.sort(key=lambda subject: (-(wet[subject] + dry[subject]) / 2, subject))
    positions = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(19, 7))
    width = 0.42
    wet_bars = ax.bar(
        positions - width / 2,
        [wet[subject] for subject in subjects],
        width,
        color=METHOD_COLORS["FBCCA"],
        label="Opaska mokra",
    )
    dry_bars = ax.bar(
        positions + width / 2,
        [dry[subject] for subject in subjects],
        width,
        color=METHOD_COLORS["FBCCA"],
        alpha=0.45,
        hatch="//",
        label="Opaska sucha",
    )
    _label_bars(ax, (wet_bars, dry_bars), fontsize=5, rotation=90)
    _accuracy_line(ax)
    ax.set_xticks(positions, subjects, rotation=90, fontsize=5)
    ax.set_xlabel("Osoba (posortowana według średniej z obu opasek)")
    ax.set_ylabel("Trafność FBCCA (okno 1 s)")
    ax.set_title("Trafność FBCCA dla poszczególnych osób")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, output_dir, "osoby_fbcca.png")


def plot_distribution(rows, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = {}
    data = []
    colors = []
    for method_index, method in enumerate(METHODS):
        for headband_offset, headband in ((-0.18, "sucha"), (0.18, "mokra")):
            selected = _selected(
                rows,
                method,
                1.0,
                headband,
                0 if method in ("CCA", "FBCCA") else 9,
            )
            values = _subject_values(selected, "accuracy")
            group_position = method_index * 1.2 + headband_offset
            positions[(method, headband)] = group_position
            data.append(list(values.values()))
            colors.append(METHOD_COLORS[method])
    artists = ax.boxplot(data, positions=list(positions.values()), widths=0.28, patch_artist=True)
    rng = np.random.default_rng(42)
    for index, (box, color) in enumerate(zip(artists["boxes"], colors)):
        box.set_facecolor(color)
        box.set_alpha(0.45 if index % 2 == 0 else 0.95)
        box.set_edgecolor(color)
        method, headband = list(positions.keys())[index]
        selected = _selected(
            rows,
            method,
            1.0,
            headband,
            0 if method in ("CCA", "FBCCA") else 9,
        )
        values = [row["accuracy"] for row in selected]
        jitter = rng.uniform(-0.09, 0.09, len(values))
        ax.scatter(
            positions[(method, headband)] + jitter,
            values,
            s=9,
            color=color,
            alpha=0.5,
            marker="o" if headband == "mokra" else "^",
        )
        median = float(np.median(values))
        ax.annotate(
            _format_value(median),
            (group_position, median),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            fontweight="bold",
        )
    _accuracy_line(ax)
    centers = [index * 1.2 for index in range(len(METHODS))]
    ax.set_xticks(centers, [METHOD_LABELS[method] for method in METHODS])
    ax.set_ylabel("Trafność (okno 1 s)")
    ax.set_title("Rozkład trafności między osobami")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor="#666666", alpha=0.95, label="Opaska mokra"),
            Patch(facecolor="#666666", alpha=0.45, label="Opaska sucha"),
            Line2D([], [], color="black", linestyle="--", label="Poziom zgadywania (8,3%)"),
        ],
        loc="best",
    )
    fig.tight_layout()
    return _save(fig, output_dir, "rozklad.png")


def find_extremes(rows=None, csv_path=None, print_results=True):
    """Select best, worst and most headband-sensitive subjects by 1 s FBCCA."""
    if rows is None:
        rows = _read_results(csv_path)
    wet = _subject_values(_selected(rows, "FBCCA", 1.0, "mokra", 0), "accuracy")
    dry = _subject_values(_selected(rows, "FBCCA", 1.0, "sucha", 0), "accuracy")
    subjects = sorted(wet.keys() & dry.keys())
    if len(subjects) < 3:
        raise ValueError("Potrzeba co najmniej trzech osób z wynikami FBCCA dla obu opasek")
    records = {
        subject: {
            "subject": subject,
            "mokra": wet[subject],
            "sucha": dry[subject],
            "mean": (wet[subject] + dry[subject]) / 2,
            "difference": wet[subject] - dry[subject],
        }
        for subject in subjects
    }
    extremes = {
        "najlepsze": sorted(subjects, key=lambda s: (-records[s]["mean"], s))[:3],
        "najgorsze": sorted(subjects, key=lambda s: (records[s]["mean"], s))[:3],
        "największa_różnica": sorted(
            subjects, key=lambda s: (-records[s]["difference"], s)
        )[:3],
    }
    result = {group: [records[subject] for subject in ids] for group, ids in extremes.items()}
    if print_results:
        for group, people in result.items():
            print(f"{group.replace('_', ' ').capitalize()} (FBCCA, 1 s):")
            for person in people:
                print(
                    f"  {person['subject']}: mokra={person['mokra']:.2%}, "
                    f"sucha={person['sucha']:.2%}, "
                    f"średnia={person['mean']:.2%}, "
                    f"mokra-sucha={person['difference']:+.2%}"
                )
    return result


def available_subjects(rows=None, csv_path=None):
    """Return subject IDs available in the results CSV."""
    if rows is None:
        rows = _read_results(csv_path)
    return sorted({row["subject"] for row in rows})


def _subject_plot_rows(rows, subject):
    selected = [row for row in rows if row["subject"] == subject]
    if not selected:
        raise ValueError(f"Brak wyników w CSV dla osoby {subject}")
    return selected


def plot_subject_bars(rows, subject, output_dir, methods=METHODS, window=1.0):
    """Plot selected methods for one subject at a chosen window."""
    subject_rows = _subject_plot_rows(rows, subject)
    positions = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, headband in ((-width / 2, "sucha"), (width / 2, "mokra")):
        values = []
        for method in methods:
            n_calib = 0 if method in ("CCA", "FBCCA") else 9
            selected = _selected(subject_rows, method, window, headband, n_calib)
            if not selected:
                raise ValueError(
                    f"Brak wyniku {method}, okno {window:g} s, "
                    f"opaska {headband} dla {subject}"
                )
            values.append(selected[0]["accuracy"])
        bars = ax.bar(
            positions + offset,
            values,
            width,
            color=[METHOD_COLORS[method] for method in methods],
            alpha=0.45 if headband == "sucha" else 0.95,
            hatch="//" if headband == "sucha" else None,
            label=HEADBAND_LABELS[headband],
        )
        _label_bars(ax, (bars,), fontsize=7, rotation=90)
    _accuracy_line(ax)
    ax.set_xticks(positions, [METHOD_LABELS[method] for method in methods])
    ax.set_ylabel(f"Trafność (okno {window:g} s)")
    ax.set_title(f"Metody odniesienia dla osoby {subject}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_dir, f"osoba_{subject}_slupki.png")


def plot_subject_windows(rows, subject, output_dir, methods=METHODS, windows=WINDOWS):
    """Plot selected methods across selected windows for one subject."""
    subject_rows = _subject_plot_rows(rows, subject)
    fig, ax = plt.subplots(figsize=(9, 5))
    for method_index, method in enumerate(methods):
        n_calib = 0 if method in ("CCA", "FBCCA") else 9
        for headband in HEADBANDS:
            values = []
            for window in windows:
                selected = _selected(
                    subject_rows, method, window, headband, n_calib
                )
                if not selected:
                    raise ValueError(
                        f"Brak wyniku {method}, okno {window:g} s, "
                        f"opaska {headband} dla {subject}"
                    )
                values.append(selected[0]["accuracy"])
            ax.plot(
                windows,
                values,
                color=METHOD_COLORS[method],
                linestyle="-" if headband == "mokra" else "--",
                marker="o" if headband == "mokra" else "s",
                label=f"{METHOD_LABELS[method]}, {HEADBAND_LABELS[headband].lower()}",
            )
            _label_line_points(
                ax,
                windows,
                values,
                offset=(8, -10, 16, -18)[(method_index * 2 + HEADBANDS.index(headband)) % 4],
            )
    _accuracy_line(ax)
    ax.set_xticks(windows, [str(w).replace(".", ",") for w in windows])
    ax.set_xlabel("Długość okna [s]")
    ax.set_ylabel("Trafność")
    ax.set_title(f"Trafność metod w zależności od okna — {subject}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    return _save(fig, output_dir, f"osoba_{subject}_okna.png")


def plot_subject_spectrum(subject, output_dir, dataset_path=None):
    """Plot the target-0 Oz spectrum for one subject, split by headband."""
    path = (
        Path(dataset_path)
        if dataset_path is not None
        else Path(path_filtered) / "dataset.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono zbioru danych do widma FFT: {path}")
    with np.load(path) as dataset:
        X = dataset["X"]
        y = dataset["y"]
        subjects = dataset["subject"]
        headbands = dataset["headband"]
    if X.ndim != 3 or X.shape[1:] != (8, 500):
        raise ValueError(f"Oczekiwano X o wymiarach (próby, 8, 500), otrzymano {X.shape}")

    time_window = np.hanning(X.shape[-1])
    frequencies = np.fft.rfftfreq(X.shape[-1], d=1 / fs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, headband in zip(axes, (1, 0)):
        label = "mokra" if headband == 1 else "sucha"
        selected = (subjects == subject) & (headbands == headband) & (y == 0)
        trials = X[selected, 5, :]
        if len(trials) == 0:
            raise ValueError(f"Brak prób celu 0 dla osoby {subject} i opaski {label}")
        centered = trials - trials.mean(axis=-1, keepdims=True)
        magnitudes = np.abs(np.fft.rfft(centered * time_window, axis=-1))
        magnitudes *= 2 / time_window.sum()
        ax.plot(frequencies, magnitudes.mean(axis=0), color=METHOD_COLORS["FBCCA"])
        _label_spectrum_harmonics(ax, frequencies, magnitudes.mean(axis=0))
        for harmonic in (9.25, 18.5, 27.75):
            ax.axvline(harmonic, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(5, 50)
        ax.set_title(f"{HEADBAND_LABELS[label]} ({len(trials)} prób)")
        ax.set_xlabel("Częstotliwość [Hz]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Średnia amplituda widma FFT [jedn. względne]")
    fig.suptitle(f"Widmo kanału Oz, cel 0 (9,25 Hz) — {subject}")
    fig.tight_layout()
    return _save(fig, output_dir, f"osoba_{subject}_widmo.png")


def plot_subject(
    subject,
    csv_path=None,
    dataset_path=None,
    output_dir=None,
    methods=METHODS,
    windows=WINDOWS,
):
    """Create selected-method/window bars, window trends and spectrum for one subject."""
    rows = _read_results(csv_path)
    _subject_plot_rows(rows, subject)
    target_dir = _prepare_output(output_dir)
    bar_window = 1.0 if 1.0 in windows else windows[0]
    return [
        plot_subject_bars(rows, subject, target_dir, methods, bar_window),
        plot_subject_windows(rows, subject, target_dir, methods, windows),
        plot_subject_spectrum(subject, target_dir, dataset_path),
        *plot_diagnostic_fft(subject, dataset_path, target_dir),
    ]


def plot_extreme_bars(rows, extremes, output_dir, methods=METHODS, window=1.0):
    groups = ("najlepsze", "najgorsze", "największa_różnica")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)
    for row_index, group in enumerate(groups):
        for col_index, person in enumerate(extremes[group]):
            ax = axes[row_index, col_index]
            subject = person["subject"]
            positions = np.arange(len(methods))
            width = 0.36
            bar_containers = []
            for headband_offset, headband in ((-width / 2, "sucha"), (width / 2, "mokra")):
                values = []
                for method in methods:
                    n_calib = 0 if method in ("CCA", "FBCCA") else 9
                    selected = _selected(rows, method, window, headband, n_calib)
                    subject_row = next(
                        (item for item in selected if item["subject"] == subject),
                        None,
                    )
                    if subject_row is None:
                        raise ValueError(f"Brak wyniku {method} dla osoby {subject}")
                    values.append(subject_row["accuracy"])
                bars = ax.bar(
                    positions + headband_offset,
                    values,
                    width,
                    color=[METHOD_COLORS[method] for method in methods],
                    alpha=0.45 if headband == "sucha" else 0.95,
                    hatch="//" if headband == "sucha" else None,
                )
                bar_containers.append(bars)
            _label_bars(ax, bar_containers, fontsize=6, rotation=90)
            _accuracy_line(ax)
            ax.set_title(
                f"{subject}\nFBCCA mokra/sucha: "
                f"{person['mokra']:.0%}/{person['sucha']:.0%}"
            )
            ax.set_xticks(
                positions,
                [METHOD_LABELS[method] for method in methods],
                rotation=30,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.2)
            if col_index == 0:
                ax.set_ylabel(f"{group.replace('_', ' ').capitalize()}\nTrafność")
    fig.legend(
        handles=[
            Patch(facecolor="#666666", alpha=0.95, label="Opaska mokra"),
            Patch(facecolor="#666666", alpha=0.45, hatch="//", label="Opaska sucha"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
    )
    fig.suptitle(
        f"Metody odniesienia dla osób skrajnych (okno {window:g} s)", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, output_dir, "skrajne_slupki.png")


def plot_extreme_windows(
    rows, extremes, output_dir, methods=("FBCCA", "eTRCA"), windows=WINDOWS
):
    groups = ("najlepsze", "najgorsze", "największa_różnica")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)
    for row_index, group in enumerate(groups):
        for col_index, person in enumerate(extremes[group]):
            ax = axes[row_index, col_index]
            subject = person["subject"]
            for headband_index, headband in enumerate(("mokra", "sucha")):
                for method_index, method in enumerate(methods):
                    n_calib = 0 if method in ("CCA", "FBCCA") else 9
                    means = []
                    for window in windows:
                        selected = _selected(rows, method, window, headband, n_calib)
                        subject_row = next(
                            (item for item in selected if item["subject"] == subject),
                            None,
                        )
                        if subject_row is None:
                            raise ValueError(f"Brak wyniku {method} dla osoby {subject}")
                        means.append(subject_row["accuracy"])
                    linestyle = "-" if headband == "mokra" else "--"
                    marker = "o" if headband == "mokra" else "s"
                    ax.plot(
                        windows,
                        means,
                        color=METHOD_COLORS[method],
                        linestyle=linestyle,
                        marker=marker,
                        label=f"{METHOD_LABELS[method]}, {HEADBAND_LABELS[headband].lower()}",
                    )
                    offset = (8, -10, 16, -18)[
                        (headband_index * len(methods) + method_index) % 4
                    ]
                    _label_line_points(ax, windows, means, offset=offset, fontsize=6)
            _accuracy_line(ax)
            ax.set_title(subject)
            ax.set_xticks(windows, [str(w).replace(".", ",") for w in windows])
            ax.grid(True, alpha=0.2)
            if col_index == 0:
                ax.set_ylabel(f"{group.replace('_', ' ').capitalize()}\nTrafność")
            if row_index == 2:
                ax.set_xlabel("Długość okna [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize="small")
    fig.suptitle("FBCCA i eTRCA dla osób skrajnych", y=1.02)
    fig.tight_layout()
    return _save(fig, output_dir, "skrajne_okna.png")


def plot_extremes(
    extremes,
    methods=METHODS,
    windows=WINDOWS,
    csv_path=None,
    dataset_path=None,
    output_dir=None,
):
    """Create the three aggregate extreme-subject plots, never one per person."""
    rows = _read_results(csv_path)
    target_dir = _prepare_output(output_dir)
    bar_window = 1.0 if 1.0 in windows else windows[0]
    return [
        plot_extreme_bars(rows, extremes, target_dir, methods, bar_window),
        plot_extreme_windows(rows, extremes, target_dir, methods, windows),
        plot_extreme_spectra(extremes, target_dir, dataset_path),
        *plot_diagnostic_fft(
            extremes["najlepsze"][0]["subject"], dataset_path, target_dir
        ),
    ]


def plot_extreme_spectra(extremes, output_dir, dataset_path=None):
    path = (
        Path(dataset_path)
        if dataset_path is not None
        else Path(path_filtered) / "dataset.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono zbioru danych do widma FFT: {path}")
    with np.load(path) as dataset:
        X = dataset["X"]
        y = dataset["y"]
        subjects = dataset["subject"]
        headbands = dataset["headband"]
    if X.ndim != 3 or X.shape[1:] != (8, 500):
        raise ValueError(f"Oczekiwano X o wymiarach (próby, 8, 500), otrzymano {X.shape}")

    selected_people = (
        extremes["najlepsze"][0]["subject"],
        extremes["najgorsze"][0]["subject"],
    )
    time_window = np.hanning(X.shape[-1])
    frequencies = np.fft.rfftfreq(X.shape[-1], d=1 / fs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    subject_colors = ("#1f77b4", "#d62728")
    for ax, headband in zip(axes, (1, 0)):
        label = "mokra" if headband == 1 else "sucha"
        for subject_index, subject in enumerate(selected_people):
            selected = (subjects == subject) & (headbands == headband) & (y == 0)
            trials = X[selected, 5, :]
            if len(trials) == 0:
                raise ValueError(
                    f"Brak prób celu 0 dla osoby {subject} i opaski {label}"
                )
            centered = trials - trials.mean(axis=-1, keepdims=True)
            magnitudes = np.abs(np.fft.rfft(centered * time_window, axis=-1))
            magnitudes *= 2 / time_window.sum()
            spectrum = magnitudes.mean(axis=0)
            ax.plot(
                frequencies,
                spectrum,
                color=subject_colors[subject_index],
                label=(
                    f"{'Najlepsza' if subject_index == 0 else 'Najgorsza'} "
                    f"({subject}, {len(trials)} prób)"
                ),
            )
            _label_spectrum_harmonics(ax, frequencies, spectrum, subject_colors[subject_index])
        for harmonic in (9.25, 18.5, 27.75):
            ax.axvline(harmonic, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(5, 50)
        ax.set_title(HEADBAND_LABELS[label])
        ax.set_xlabel("Częstotliwość [Hz]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")
    axes[0].set_ylabel("Średnia amplituda widma FFT [jedn. względne]")
    fig.suptitle("Widmo kanału Oz dla celu 0 (9,25 Hz)")
    fig.tight_layout()
    return _save(fig, output_dir, "skrajne_widmo.png")


def plot_all(csv_path=None, dataset_path=None, output_dir=None):
    """Create cohort figures from the saved CSV and diagnostic FFT figures."""
    rows = _read_results(csv_path)
    target_dir = _prepare_output(output_dir)
    extremes = find_extremes(rows, print_results=False)
    return [
        plot_window_accuracy(rows, target_dir),
        plot_window_itr(rows, target_dir),
        plot_method_bars(rows, target_dir),
        plot_calibration(rows, target_dir),
        plot_subject_fbcca(rows, target_dir),
        plot_distribution(rows, target_dir),
        plot_extreme_bars(rows, extremes, target_dir),
        plot_extreme_windows(rows, extremes, target_dir),
        plot_extreme_spectra(extremes, target_dir, dataset_path),
        *plot_diagnostic_fft(
            sorted({row["subject"] for row in rows})[0],
            dataset_path,
            target_dir,
        ),
    ]


def plot_summary(
    methods,
    windows,
    csv_path=None,
    output_dir=None,
    dataset_path=None,
    subjects=None,
):
    """Create cohort-level plots for selected subjects, methods and windows."""
    rows = _read_results(csv_path)
    if subjects is not None:
        selected_subjects = set(subjects)
        rows = [row for row in rows if row["subject"] in selected_subjects]
        if not rows:
            raise ValueError("Brak wyników dla wybranych osób")
    target_dir = _prepare_output(output_dir)
    figures = [
        _plot_window_metric(
            rows,
            target_dir,
            "accuracy",
            "okno_trafnosc.png",
            "Średnia trafność",
            "Trafność metod odniesienia w zależności od długości okna",
            methods,
            windows,
        ),
        _plot_window_metric(
            rows,
            target_dir,
            "itr",
            "okno_itr.png",
            "Średnie ITR [bit/min]",
            "Szybkość transferu informacji w zależności od długości okna",
            methods,
            windows,
        ),
        plot_method_bars(rows, target_dir, methods, windows),
    ]
    if 1.0 in windows:
        figures.append(plot_calibration(rows, target_dir, methods))
    diagnostic_subject = (
        sorted({row["subject"] for row in rows})[0]
        if subjects is None
        else sorted(set(subjects))[0]
    )
    figures.extend(plot_diagnostic_fft(diagnostic_subject, dataset_path, target_dir))
    return figures
