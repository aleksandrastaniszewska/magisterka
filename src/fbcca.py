# Metoda odniesienia FBCCA, bez kalibracji. Służy tylko do porównania z siecią.
import numpy as np
from scipy.signal import cheby1, filtfilt

from .cca import cca_scores


# Three bands win on all 2 s results and on overall mean accuracy; five lose to CCA.
DEFAULT_FBCCA_BANDS = 3


def filter_bank(X, fs, n_bands=DEFAULT_FBCCA_BANDS):
    """Filter (..., channels, samples) into three or five sub-bands."""
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError("X must have shape (..., channels, samples)")
    if fs <= 100:
        raise ValueError("fs must be greater than twice the 50 Hz upper cutoff")

    if n_bands not in (3, 5):
        raise ValueError("n_bands must be 3 or 5")
    lower_edges = (6, 14, 22, 30, 38)[:n_bands]
    filtered_bands = []
    for lower_edge in lower_edges:
        numerator, denominator = cheby1(
            4, 1.0, (lower_edge, 50), btype="bandpass", fs=fs
        )
        filtered_bands.append(
            filtfilt(numerator, denominator, X, axis=-1)
        )
    return np.stack(filtered_bands, axis=0)


def fbcca_scores_from_bands(bands, freqs, fs, n_harmonics=3, n_samples=None):
    """Score prefiltered trials, optionally using a prefix after filtering."""
    bands = np.asarray(bands)
    if bands.ndim != 4 or bands.shape[0] not in (3, 5):
        raise ValueError("bands must have shape (3 or 5, trials, channels, samples)")
    if n_samples is not None:
        if not 1 <= n_samples <= bands.shape[-1]:
            raise ValueError("n_samples must be within the filtered trial length")
        bands = bands[..., :n_samples]

    n_bands = bands.shape[0]
    weights = np.arange(1, n_bands + 1, dtype=float) ** -1.25 + 0.25
    frequencies = len(freqs)
    scores = np.zeros((bands.shape[1], frequencies), dtype=float)
    for weight, band in zip(weights, bands):
        scores += weight * cca_scores(band, freqs, fs, n_harmonics) ** 2
    return scores


def fbcca_scores(
    X, freqs, fs, n_harmonics=3, n_bands=DEFAULT_FBCCA_BANDS, n_samples=None
):
    """Filter full trials, then score their optional prefix with weighted CCA."""
    bands = filter_bank(X, fs, n_bands=n_bands)
    return fbcca_scores_from_bands(
        bands, freqs, fs, n_harmonics=n_harmonics, n_samples=n_samples
    )


def fbcca_predict(X, freqs, fs, n_harmonics=3):
    """Return the highest-scoring FBCCA target index for each trial."""
    return np.argmax(fbcca_scores(X, freqs, fs, n_harmonics), axis=1)
