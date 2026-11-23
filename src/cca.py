# Metoda odniesienia służąca wyłącznie do porównania wyników z siecią; nie jest wejściem do sieci.
import os
from functools import lru_cache

import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import CCA

from .config import fs, n_harmonics, target_freq


@lru_cache(maxsize=256)
def _reference_basis(freq, n_samples, sampling_rate, harmonics):
    t = np.arange(n_samples) / sampling_rate
    columns = []
    for harmonic in range(1, harmonics + 1):
        columns.extend(
            (
                np.sin(2 * np.pi * harmonic * freq * t),
                np.cos(2 * np.pi * harmonic * freq * t),
            )
        )
    reference = np.column_stack(columns)
    reference -= reference.mean(axis=0, keepdims=True)
    return np.linalg.qr(reference, mode="reduced")[0]


def _frequency_values(freqs):
    if hasattr(freqs, "values"):
        return [freqs[key] for key in sorted(freqs)]
    return list(freqs)


def cca_scores(X, freqs, fs, n_harmonics=3):
    """Return the largest canonical correlation per trial and target."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must have shape (trials, channels, samples)")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    frequencies = _frequency_values(freqs)
    if not frequencies:
        raise ValueError("freqs must contain at least one target frequency")

    signals = X.transpose(0, 2, 1)
    signals = signals - signals.mean(axis=1, keepdims=True)
    signal_basis = np.linalg.qr(signals, mode="reduced")[0]
    scores = np.empty((X.shape[0], len(frequencies)), dtype=float)

    for target, freq in enumerate(frequencies):
        reference_basis = _reference_basis(
            float(freq), X.shape[-1], float(fs), int(n_harmonics)
        )
        cross_basis = np.einsum(
            "ntc,tk->nck", signal_basis, reference_basis, optimize=True
        )
        scores[:, target] = np.linalg.svd(
            cross_basis, compute_uv=False
        )[:, 0]
    return np.clip(scores, 0.0, 1.0)


def cca_predict(X, freqs, fs, n_harmonics=3):
    """Return the highest-scoring target index for each trial."""
    return np.argmax(cca_scores(X, freqs, fs, n_harmonics), axis=1)


# Tworzy sinusoidalne i cosinusoidalne sygnały referencyjne dla częstotliwości i harmonicznych.
def create_reference_signals(freq, n_samples, fs, n_harmonics):
    t = np.arange(n_samples) / fs # tablica 500 probek /250 kazda wartość to 0.004s
    ref = []
    for h in range(1, n_harmonics + 1):
        ref.append(np.sin(2 * np.pi * h * freq * t))
        ref.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(ref).T

# Oblicza dla jednej próby EEG wektor korelacji CCA, po jednej wartości na każdy target.
def cca_features_for_trial(trial, target_freq, fs, n_harmonics):
    n_samples = 500
    X = trial.T #transpozycja macierzy w wierszach sygnal w kolumnie kanaly eeg
    features = np.zeros(len(target_freq))# tu tworzymy macierz na cechy wypelniona zerami

    for target, freq in target_freq.items():
        Y = create_reference_signals(freq, n_samples, fs, n_harmonics)
        cca = CCA(n_components=1)
        cca.fit(X,Y)
        X_c, Y_c = cca.transform(X,Y)
        corr = np.corrcoef(X_c[:,0], Y_c[:,0])[0,1]
        features[target] = corr
    return features

# Wypisuje wyniki CCA dla jednej próby i sprawdza, czy najwyższy wynik wskazuje jej target.
def check_features(trial, true_target, target_freq, fs, n_harmonics=3):
    """Diagnostyczny print - sprawdza czy cechy CCA sensownie wskazuja prawdziwy target."""
    feats = cca_features_for_trial(trial, target_freq, fs, n_harmonics)

    print(f"Prawdziwy target: {true_target} ({target_freq[true_target]} Hz)")
    print("-" * 50)
    for i, freq in target_freq.items():
        marker = " <-- PRAWDZIWY" if i == true_target else ""
        print(f"target {i:2d} ({freq:6.2f} Hz): korelacja = {feats[i]:.3f}{marker}")

    predicted = np.argmax(feats)
    print("-" * 50)
    print(f"Przewidziany target (argmax): {predicted} ({target_freq[predicted]} Hz)")
    print(f"Trafienie: {'TAK' if predicted == true_target else 'NIE'}")
    print(f"Roznica miedzy najlepszym a drugim najlepszym: "
          f"{np.sort(feats)[-1] - np.sort(feats)[-2]:.3f}")

# Wczytuje każdy plik raz, oblicza cechy CCA dla każdej próby i zwraca cechy, etykiety oraz identyfikatory osób.
def process(mat_files):
    all_features = []
    all_labels = []
    all_subjects = []

    for file in mat_files:
        mat = sio.loadmat(file)
        if 'data' not in mat:
            continue

        EEG = mat['data']
        n_channels, n_time, n_etype, n_block, n_target = EEG.shape
        subject_id = os.path.basename(file)[:-4]

        for electrode_type in range(n_etype):
            for block in range(n_block):
                for target in range(n_target):
                    trial = EEG[:, :, electrode_type, block, target]
                    features = cca_features_for_trial(
                        trial, target_freq, fs, n_harmonics
                    )
                    all_features.append(features)
                    all_labels.append(target)
                    all_subjects.append(subject_id)
                    # Diagnostyczne wykresy FFT
                    # if subject_id == "S035" and electrode_type == 0 and target == 2:
                    #     for channel in range(n_channels):
                    #         signal = trial[channel]
                    #         plot_fft(
                    #             signal,
                    #             f"FFT {subject_id}, blok {block}, kanał {channel}, target {target}",
                    #             target
                    #         )

    return (np.asarray(all_features), np.asarray(all_labels), np.asarray(all_subjects))
