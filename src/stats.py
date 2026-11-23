"""Paired statistical tests for baseline-method comparisons."""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from .config import current_dir
from .plots import HEADBANDS, METHODS, WINDOWS, _read_results


COMPARISONS = (
    ("FBCCA vs CCA", "FBCCA", "CCA", 0, 0),
    ("eTRCA vs TRCA", "eTRCA", "TRCA", 9, 9),
)


def _paired_values(rows, method_a, method_b, headband, window, calib_a, calib_b):
    values_a = {
        row["subject"]: row["accuracy"]
        for row in rows
        if row["method"] == method_a
        and row["headband"] == headband
        and row["window"] == window
        and row["n_calib"] == calib_a
    }
    values_b = {
        row["subject"]: row["accuracy"]
        for row in rows
        if row["method"] == method_b
        and row["headband"] == headband
        and row["window"] == window
        and row["n_calib"] == calib_b
    }
    subjects = sorted(values_a.keys() & values_b.keys())
    if len(subjects) != 102:
        raise ValueError(
            f"Oczekiwano 102 sparowanych osób dla {method_a}/{method_b}, "
            f"{headband}, {window:g} s; znaleziono {len(subjects)}"
        )
    return (
        np.asarray([values_a[subject] for subject in subjects]),
        np.asarray([values_b[subject] for subject in subjects]),
    )


def calculate_statistics(rows=None, csv_path=None, output_path=None):
    """Run paired Wilcoxon tests and one Holm correction across all 20 tests."""
    if rows is None:
        rows = _read_results(csv_path)

    tests = []
    for label, method_a, method_b, calib_a, calib_b in COMPARISONS:
        for window in WINDOWS:
            for headband in HEADBANDS:
                first, second = _paired_values(
                    rows,
                    method_a,
                    method_b,
                    headband,
                    window,
                    calib_a,
                    calib_b,
                )
                differences = first - second
                if np.all(differences == 0):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(
                            first,
                            second,
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                tests.append(
                    {
                        "comparison": label,
                        "window": window,
                        "headband": headband,
                        "mean_difference": float(np.mean(differences)),
                        "p": p_value,
                    }
                )

    _, adjusted, _, _ = multipletests(
        [test["p"] for test in tests],
        method="holm",
    )
    for test, adjusted_p in zip(tests, adjusted):
        test["p_adjusted"] = float(adjusted_p)
        test["significant"] = bool(adjusted_p < 0.05)

    path = (
        Path(output_path)
        if output_path is not None
        else Path(current_dir) / "results" / "stats_baselines.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "comparison",
        "window",
        "headband",
        "mean_difference",
        "p",
        "p_adjusted",
        "significant",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tests)
    return tests, path
