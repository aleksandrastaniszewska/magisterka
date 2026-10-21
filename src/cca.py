# Metoda odniesienia służąca wyłącznie do porównania wyników z siecią; nie jest wejściem do sieci.
import os

import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import CCA

from .config import fs, n_harmonics, target_freq


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
