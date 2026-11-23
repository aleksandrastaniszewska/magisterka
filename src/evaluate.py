# Odpowiada za ewaluację wytrenowanego modelu.
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cca import cca_scores
from .config import current_dir, fs, n_harmonics, target_freq
from .fbcca import (
    DEFAULT_FBCCA_BANDS,
    fbcca_scores_from_bands,
    filter_bank,
)
from .trca import etrca_predict, make_templates, trca_filters, trca_predict


CALIBRATION_BLOCKS = (2, 4, 6, 9)


def _itr(accuracy, window):
    n_targets = len(target_freq)
    if accuracy <= 1 / n_targets:
        # Under chance-level accuracy, report zero information transfer.
        return 0.0
    if accuracy <= 0 or accuracy >= 1:
        bits_per_selection = 0.0 if accuracy <= 0 else np.log2(n_targets)
    else:
        bits_per_selection = (
            np.log2(n_targets)
            + accuracy * np.log2(accuracy)
            + (1 - accuracy) * np.log2((1 - accuracy) / (n_targets - 1))
        )
    # T includes a 0.5 s assumed gaze-transfer pause after each selection.
    return float(60 / (window + 0.5) * bits_per_selection)


def _append_result(rows, subject, headband, window, method, n_calib, truth, prediction):
    accuracy = float(np.mean(truth == prediction))
    rows.append(
        {
            "subject": subject,
            "headband": "sucha" if headband == 0 else "mokra",
            "window": window,
            "method": method,
            "n_calib": n_calib,
            "accuracy": accuracy,
            "itr": _itr(accuracy, window),
        }
    )


def _evaluate_subject(X, y, subject, headband, block, subject_id, windows, methods):
    rows = []
    for headband_id in (0, 1):
        selected = (subject == subject_id) & (headband == headband_id)
        subject_X = X[selected]
        subject_y = y[selected]
        subject_block = block[selected]
        if len(subject_X) != 120 or not np.array_equal(
            np.unique(subject_block), np.arange(10)
        ):
            raise ValueError(
                f"{subject_id}, headband {headband_id}: expected 120 trials in 10 blocks"
            )

        fbcca_bands = (
            filter_bank(subject_X, fs, n_bands=DEFAULT_FBCCA_BANDS)
            if "FBCCA" in methods
            else None
        )
        for window in windows:
            n_samples = int(window * fs)
            if n_samples < 28 or n_samples > subject_X.shape[-1]:
                raise ValueError(f"Unsupported window length: {window} seconds")
            window_X = subject_X[..., :n_samples]

            fbcca_predictions = None
            if "CCA" in methods:
                cca_predictions = np.argmax(
                    cca_scores(window_X, target_freq, fs, n_harmonics), axis=1
                )
                _append_result(
                    rows, subject_id, headband_id, window, "CCA", 0,
                    subject_y, cca_predictions
                )
            if "FBCCA" in methods:
                fbcca_predictions = np.argmax(
                    fbcca_scores_from_bands(
                        fbcca_bands, target_freq, fs, n_harmonics, n_samples=n_samples
                    ),
                    axis=1,
                )
                _append_result(
                    rows, subject_id, headband_id, window, "FBCCA", 0,
                    subject_y, fbcca_predictions
                )

            for n_calib in CALIBRATION_BLOCKS:
                calibration = subject_block < n_calib
                testing = subject_block >= n_calib
                y_test = subject_y[testing]
                if "CCA" in methods:
                    cca_prediction = np.argmax(
                        cca_scores(
                            subject_X[testing, ..., :n_samples],
                            target_freq,
                            fs,
                            n_harmonics,
                        ),
                        axis=1,
                    )
                    _append_result(
                        rows, subject_id, headband_id, window, "CCA", n_calib,
                        y_test, cca_prediction
                    )
                if "FBCCA" in methods:
                    _append_result(
                        rows, subject_id, headband_id, window, "FBCCA", n_calib,
                        y_test, fbcca_predictions[testing]
                    )

                if "TRCA" in methods or "eTRCA" in methods:
                    X_cal = subject_X[calibration, ..., :n_samples]
                    X_test = subject_X[testing, ..., :n_samples]
                    templates = make_templates(X_cal, subject_y[calibration])
                    filters = trca_filters(X_cal, subject_y[calibration])
                    if "TRCA" in methods:
                        _append_result(
                            rows, subject_id, headband_id, window, "TRCA", n_calib,
                            y_test,
                            np.asarray(
                                [
                                    trca_predict(trial, templates, filters)
                                    for trial in X_test
                                ]
                            ),
                        )
                    if "eTRCA" in methods:
                        _append_result(
                            rows, subject_id, headband_id, window, "eTRCA", n_calib,
                            y_test,
                            np.asarray(
                                [
                                    etrca_predict(trial, templates, filters)
                                    for trial in X_test
                                ]
                            ),
                        )
    return rows


def _print_subject_table(rows):
    print(
        "subject headband window method n_calib accuracy itr",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['subject']:>4} {row['headband']:<5} {row['window']:>5.2f} "
            f"{row['method']:<5} {row['n_calib']:>7} "
            f"{row['accuracy']:>8.3%} {row['itr']:>8.2f}",
            flush=True,
        )


def _report_checks(rows):
    chance = 1 / len(target_freq)
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (row["method"], row["headband"], row["window"], row["n_calib"])
        ].append(row["accuracy"])
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    below_chance = [(key, value) for key, value in means.items() if value <= chance]
    print(
        "Średnie trafności > 8,3%: "
        + ("TAK" if not below_chance else f"NIE ({below_chance})")
    )

    overall_by_window = {
        window: float(np.mean([r["accuracy"] for r in rows if r["window"] == window]))
        for window in sorted({r["window"] for r in rows})
    }
    trend_ok = all(
        later > earlier
        for earlier, later in zip(
            overall_by_window.values(), list(overall_by_window.values())[1:]
        )
    )
    print(f"Średnia trafność rośnie z długością okna: {trend_ok} {overall_by_window}")

    paired = defaultdict(lambda: defaultdict(list))
    for row in rows:
        paired[
            (row["subject"], row["headband"], row["window"], row["n_calib"])
        ][row["method"]].append(row["accuracy"])
    cca_fbcca_differences = []
    trca_etr_differences = []
    trca_by_calibration = defaultdict(list)
    for key, method_values in paired.items():
        if "CCA" in method_values and "FBCCA" in method_values:
            cca_fbcca_differences.append(
                np.mean(method_values["FBCCA"]) - np.mean(method_values["CCA"])
            )
        if "TRCA" in method_values and "eTRCA" in method_values:
            trca_etr_differences.append(
                np.mean(method_values["eTRCA"]) - np.mean(method_values["TRCA"])
            )
        if "TRCA" in method_values:
            trca_by_calibration[key[3]].extend(method_values["TRCA"])

    if cca_fbcca_differences:
        difference = float(np.mean(cca_fbcca_differences))
        print(
            "Średnio CCA <= FBCCA: "
            f"{difference >= 0} (różnica FBCCA-CCA={difference:.3%})"
        )
    if trca_etr_differences:
        difference = float(np.mean(trca_etr_differences))
        print(
            "Średnio TRCA <= eTRCA: "
            f"{difference >= 0} (różnica eTRCA-TRCA={difference:.3%})"
        )
    available_calibrations = [
        k for k in CALIBRATION_BLOCKS if trca_by_calibration[k]
    ]
    if available_calibrations:
        calibration_means = {
            k: float(np.mean(trca_by_calibration[k])) for k in available_calibrations
        }
        calibration_ok = all(
            later > earlier
            for earlier, later in zip(
                calibration_means.values(), list(calibration_means.values())[1:]
            )
        )
        print(
            "TRCA rośnie z liczbą bloków kalibracji: "
            f"{calibration_ok} {calibration_means}"
        )

    headband_means = {
        label: float(np.mean([r["accuracy"] for r in rows if r["headband"] == label]))
        for label in ("sucha", "mokra")
    }
    print(
        f"Sucha opaska gorsza od mokrej: "
        f"{headband_means['sucha'] < headband_means['mokra']} {headband_means}"
    )


def evaluate_baselines(
    dataset_path,
    windows=(0.25, 0.5, 1.0, 1.5, 2.0),
    subjects=None,
    methods=("CCA", "FBCCA", "TRCA", "eTRCA"),
):
    """Evaluate selected reference methods and windows and merge them into the CSV."""
    supported_methods = ("CCA", "FBCCA", "TRCA", "eTRCA")
    requested_methods = set(methods)
    unknown_methods = requested_methods - set(supported_methods)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    methods = tuple(method for method in supported_methods if method in requested_methods)
    if not methods:
        raise ValueError(f"methods must include one or more of {supported_methods}")
    supported_windows = (0.25, 0.5, 1.0, 1.5, 2.0)
    windows = tuple(sorted(set(float(window) for window in windows)))
    if not windows or any(window not in supported_windows for window in windows):
        raise ValueError(f"windows must be selected from {supported_windows}")

    with np.load(dataset_path) as dataset:
        X = dataset["X"]
        y = dataset["y"]
        subject = dataset["subject"]
        headband = dataset["headband"]
        block = dataset["block"]

    if X.ndim != 3 or not (len(X) == len(y) == len(subject) == len(headband) == len(block)):
        raise ValueError("Dataset arrays have inconsistent shapes")
    selected_subjects = sorted(np.unique(subject).tolist())
    if subjects is not None:
        missing = set(subjects) - set(selected_subjects)
        if missing:
            raise ValueError(f"Unknown subject IDs: {sorted(missing)}")
        selected_subjects = [s for s in selected_subjects if s in set(subjects)]
    if not selected_subjects:
        raise ValueError("No subjects selected for baseline evaluation")

    rows = []
    for index, subject_id in enumerate(selected_subjects, start=1):
        print(
            f"Postęp: [{index}/{len(selected_subjects)}] {subject_id}",
            flush=True,
        )
        subject_rows = _evaluate_subject(
            X, y, subject, headband, block, subject_id, windows, methods
        )
        rows.extend(subject_rows)
        if index == 1:
            print(f"Wyniki pilotażowe dla {subject_id}:")
            _print_subject_table(subject_rows)

    results_path = Path(current_dir) / "results" / "baselines.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "subject", "headband", "window", "method", "n_calib",
        "accuracy", "itr",
    )
    merged_rows = {}
    if results_path.is_file():
        with results_path.open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                key = (
                    row["subject"],
                    row["headband"],
                    float(row["window"]),
                    row["method"],
                    int(row["n_calib"]),
                )
                merged_rows[key] = {
                    **row,
                    "window": key[2],
                    "n_calib": key[4],
                    "accuracy": float(row["accuracy"]),
                    "itr": float(row["itr"]),
                }
    for row in rows:
        key = (
            row["subject"],
            row["headband"],
            row["window"],
            row["method"],
            row["n_calib"],
        )
        merged_rows[key] = row
    rows = list(merged_rows.values())
    rows.sort(
        key=lambda row: (
            row["subject"],
            row["headband"],
            row["window"],
            row["method"],
            row["n_calib"],
        )
    )
    with results_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Zapisano wyniki: {results_path}")
    if (
        subjects is None
        and methods == supported_methods
        and windows == supported_windows
    ):
        _report_checks(rows)
    return rows
