"""Self-contained interactive Plotly views for baseline results."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .config import current_dir, target_freq
from .plots import (
    HEADBANDS,
    HEADBAND_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    WINDOWS,
    _read_results,
)


CHANCE = 1 / len(target_freq)
CALIBRATIONS = (2, 4, 6, 9)


def _selected(rows, method, window, headband, n_calib):
    return [
        row for row in rows
        if row["method"] == method
        and row["window"] == window
        and row["headband"] == headband
        and row["n_calib"] == n_calib
    ]


def _mean_metric(rows, method, window, headband, n_calib, metric):
    selected = _selected(rows, method, window, headband, n_calib)
    if not selected:
        raise ValueError(
            f"Brak wyników {method}, {headband}, okno {window:g} s, "
            f"n_calib={n_calib}"
        )
    return float(np.mean([row[metric] for row in selected]))


def _accuracy_reference_shapes():
    return [
        {
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": CHANCE,
            "y1": CHANCE,
            "line": {"color": "#333333", "dash": "dash", "width": 1},
        }
    ]


def _window_metric_figure(rows):
    fig = go.Figure()
    trace_pairs = {}
    metric_values = {"accuracy": [], "itr": []}
    for method in METHODS:
        for headband in HEADBANDS:
            values = {
                metric: [
                    _mean_metric(
                        rows,
                        method,
                        window,
                        headband,
                        0 if method in ("CCA", "FBCCA") else 9,
                        metric,
                    )
                    for window in WINDOWS
                ]
                for metric in ("accuracy", "itr")
            }
            trace_index = len(fig.data)
            trace_pairs[(method, headband)] = trace_index
            for metric in metric_values:
                metric_values[metric].append(values[metric])
            fig.add_trace(
                go.Scatter(
                    x=list(WINDOWS),
                    y=values["accuracy"],
                    mode="lines+markers",
                    name=METHOD_LABELS[method],
                    legendgroup=method,
                    line={"color": METHOD_COLORS[method], "width": 2},
                    visible=headband == "mokra",
                    customdata=np.asarray(values["itr"]),
                    hovertemplate=(
                        "Okno: %{x:g} s<br>Trafność: %{y:.1%}"
                        "<br>ITR: %{customdata:.2f} bit/min<extra>%{fullData.name}</extra>"
                    ),
                )
            )

    band_buttons = []
    for headband in HEADBANDS:
        visible = [
            key[1] == headband
            for key, _ in trace_pairs.items()
        ]
        band_buttons.append(
            {
                "label": HEADBAND_LABELS[headband],
                "method": "update",
                "args": [{"visible": visible}],
            }
        )
    metric_buttons = []
    for metric, label, y_title in (
        ("accuracy", "Trafność", "Średnia trafność"),
        ("itr", "ITR", "Średni ITR [bit/min]"),
    ):
        metric_buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"y": metric_values[metric]},
                    {
                        "yaxis.title.text": y_title,
                        "shapes[0].visible": metric == "accuracy",
                    },
                ],
            }
        )
    fig.update_layout(
        title="Wyniki metod w funkcji długości okna",
        xaxis_title="Długość okna [s]",
        yaxis_title="Średnia trafność",
        xaxis={"tickmode": "array", "tickvals": list(WINDOWS)},
        yaxis={"tickformat": ".0%"},
        shapes=_accuracy_reference_shapes(),
        legend_title="Metoda",
        updatemenus=[
            {
                "buttons": band_buttons,
                "direction": "down",
                "x": 0,
                "y": 1.2,
                "xanchor": "left",
                "yanchor": "top",
            },
            {
                "buttons": metric_buttons,
                "direction": "down",
                "x": 0.38,
                "y": 1.2,
                "xanchor": "left",
                "yanchor": "top",
            },
        ],
        margin={"t": 125},
    )
    return fig


def _subject_bars_figure(rows):
    fig = go.Figure()
    choices = []
    for method in METHODS:
        for window in WINDOWS:
            wet = {
                row["subject"]: row["accuracy"]
                for row in _selected(
                    rows,
                    method,
                    window,
                    "mokra",
                    0 if method in ("CCA", "FBCCA") else 9,
                )
            }
            dry = {
                row["subject"]: row["accuracy"]
                for row in _selected(
                    rows,
                    method,
                    window,
                    "sucha",
                    0 if method in ("CCA", "FBCCA") else 9,
                )
            }
            subjects = sorted(wet.keys() & dry.keys(), key=lambda s: (-(wet[s] + dry[s]) / 2, s))
            if len(subjects) != 102:
                raise ValueError(
                    f"Oczekiwano 102 osób dla {method}, {window:g} s; "
                    f"znaleziono {len(subjects)}"
                )
            choice_start = len(fig.data)
            for headband, values in (("mokra", wet), ("sucha", dry)):
                fig.add_trace(
                    go.Bar(
                        x=subjects,
                        y=[values[subject] for subject in subjects],
                        name=HEADBAND_LABELS[headband],
                        marker={
                            "color": METHOD_COLORS[method],
                            "opacity": 1 if headband == "mokra" else 0.48,
                            "pattern": {"shape": "" if headband == "mokra" else "/"},
                        },
                        visible=False,
                        customdata=subjects,
                        hovertemplate=(
                            "Osoba: %{customdata}<br>Trafność: %{y:.1%}<extra>"
                            + HEADBAND_LABELS[headband]
                            + "</extra>"
                        ),
                    )
                )
            choices.append(
                {
                    "label": f"{METHOD_LABELS[method]}, {window:g} s",
                    "start": choice_start,
                    "subjects": subjects,
                    "wet": [wet[subject] for subject in subjects],
                    "dry": [dry[subject] for subject in subjects],
                    "method": method,
                    "window": window,
                }
            )

    buttons = []
    for choice in choices:
        visible = [False] * len(fig.data)
        visible[choice["start"]:choice["start"] + 2] = [True, True]
        buttons.append(
            {
                "label": choice["label"],
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title.text": (
                            f"Trafność osób — {METHOD_LABELS[choice['method']]}, "
                            f"okno {choice['window']:g} s"
                        )
                    },
                ],
            }
        )
    first = choices[0]
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.update_layout(
        title=f"Trafność osób — {METHOD_LABELS[first['method']]}, okno {first['window']:g} s",
        xaxis_title="Osoba (posortowana według średniej z obu opasek)",
        yaxis_title="Trafność",
        yaxis={"tickformat": ".0%"},
        barmode="group",
        shapes=_accuracy_reference_shapes(),
        legend_title="Opaska",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.2,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"t": 115, "b": 140},
        xaxis={"tickangle": -90, "tickfont": {"size": 8}},
    )
    return fig


def _calibration_figure(rows):
    fig = go.Figure()
    choices = []
    for window in WINDOWS:
        for headband in HEADBANDS:
            start = len(fig.data)
            for method in METHODS:
                values = []
                for calibration in CALIBRATIONS:
                    values.append(
                        _mean_metric(
                            rows,
                            method,
                            window,
                            headband,
                            calibration,
                            "accuracy",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=list(CALIBRATIONS),
                        y=values,
                        mode="lines+markers",
                        name=METHOD_LABELS[method],
                        line={"color": METHOD_COLORS[method], "width": 2},
                        visible=False,
                        hovertemplate=(
                            "Bloki kalibracji: %{x}<br>Trafność: %{y:.1%}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
            choices.append(
                {
                    "label": f"{window:g} s, {HEADBAND_LABELS[headband]}",
                    "start": start,
                    "window": window,
                    "headband": headband,
                }
            )
    buttons = []
    for choice in choices:
        visible = [False] * len(fig.data)
        visible[choice["start"]:choice["start"] + len(METHODS)] = [True] * len(METHODS)
        buttons.append(
            {
                "label": choice["label"],
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title.text": (
                            f"Krzywa kalibracji — okno {choice['window']:g} s, "
                            f"{HEADBAND_LABELS[choice['headband']].lower()}"
                        )
                    },
                ],
            }
        )
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True
    fig.data[3].visible = True
    fig.update_layout(
        title="Krzywa kalibracji — okno 0,25 s, opaska mokra",
        xaxis={
            "title": "Liczba bloków kalibracyjnych",
            "tickmode": "array",
            "tickvals": list(CALIBRATIONS),
        },
        yaxis={"title": "Średnia trafność", "tickformat": ".0%"},
        shapes=_accuracy_reference_shapes(),
        legend_title="Metoda",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.2,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"t": 115},
    )
    return fig


def _paired_scatter_figure(rows):
    fig = go.Figure()
    choices = []
    for method in METHODS:
        calibration = 0 if method in ("CCA", "FBCCA") else 9
        for window in WINDOWS:
            wet = {
                row["subject"]: row["accuracy"]
                for row in _selected(rows, method, window, "mokra", calibration)
            }
            dry = {
                row["subject"]: row["accuracy"]
                for row in _selected(rows, method, window, "sucha", calibration)
            }
            subjects = sorted(wet.keys() & dry.keys())
            if len(subjects) != 102:
                raise ValueError(
                    f"Oczekiwano 102 sparowanych osób dla {method}, {window:g} s"
                )
            index = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=[wet[subject] for subject in subjects],
                    y=[dry[subject] for subject in subjects],
                    mode="markers",
                    name=METHOD_LABELS[method],
                    marker={"color": METHOD_COLORS[method], "size": 9, "opacity": 0.75},
                    customdata=subjects,
                    visible=False,
                    hovertemplate=(
                        "Osoba: %{customdata}<br>Mokra: %{x:.1%}"
                        "<br>Sucha: %{y:.1%}<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            choices.append(
                {
                    "label": f"{METHOD_LABELS[method]}, {window:g} s",
                    "index": index,
                    "wet": [wet[subject] for subject in subjects],
                    "dry": [dry[subject] for subject in subjects],
                    "subjects": subjects,
                    "method": method,
                    "window": window,
                }
            )
    buttons = []
    for choice in choices:
        visible = [False] * len(fig.data)
        visible[choice["index"]] = True
        buttons.append(
            {
                "label": choice["label"],
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title.text": (
                            f"Opaska mokra vs sucha — {METHOD_LABELS[choice['method']]}, "
                            f"okno {choice['window']:g} s"
                        )
                    },
                ],
            }
        )
    first = choices[0]
    fig.data[0].visible = True
    fig.update_layout(
        title=f"Opaska mokra vs sucha — {METHOD_LABELS[first['method']]}, okno {first['window']:g} s",
        xaxis={"title": "Trafność — opaska mokra", "range": [0, 1], "tickformat": ".0%"},
        yaxis={"title": "Trafność — opaska sucha", "range": [0, 1], "tickformat": ".0%"},
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "line": {"color": "#555555", "dash": "dash"},
            },
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y",
                "y0": CHANCE,
                "y1": CHANCE,
                "line": {"color": "#333333", "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "x0": CHANCE,
                "x1": CHANCE,
                "yref": "paper",
                "y0": 0,
                "y1": 1,
                "line": {"color": "#333333", "dash": "dot"},
            },
        ],
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.2,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"t": 115},
        showlegend=False,
    )
    return fig


def build_interactive(csv_path=None, output_path=None):
    """Write all four interactive views into one standalone HTML document."""
    rows = _read_results(csv_path)
    output = (
        Path(output_path)
        if output_path is not None
        else Path(current_dir) / "results" / "figures" / "wyniki_interaktywne.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = (
        ("trafnosc-okno", _window_metric_figure(rows)),
        ("trafnosc-osoby", _subject_bars_figure(rows)),
        ("kalibracja", _calibration_figure(rows)),
        ("opaski-sparowane", _paired_scatter_figure(rows)),
    )
    sections = []
    for div_id, figure in figures:
        sections.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=False,
                div_id=div_id,
                config={"responsive": True, "displaylogo": False},
            )
        )
    html = (
        "<!doctype html><html lang=\"pl\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Interaktywne wyniki metod odniesienia</title>"
        "<style>body{font-family:Arial,sans-serif;margin:0 auto;max-width:1500px;"
        "padding:20px;color:#222}h1{text-align:center}.chart{margin:28px 0 52px;"
        "padding:12px;border:1px solid #ddd;border-radius:8px}</style>"
        f"<script>{get_plotlyjs()}</script></head><body>"
        "<h1>Wyniki metod odniesienia — widoki interaktywne</h1>"
        + "".join(f"<section class=\"chart\">{section}</section>" for section in sections)
        + "</body></html>"
    )
    output.write_text(html, encoding="utf-8")
    return output
