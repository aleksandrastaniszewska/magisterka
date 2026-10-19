import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    detrend
)
import shutil
from sklearn.cross_decomposition import CCA


current_dir = os.getcwd()
path_raw = f"{current_dir}\\data\\13560281"
path_filtered = f"{current_dir}\\filtered_data"
mat_files_raw = []
mat_files_filtered = []
n_harmonics = 3 #ile harmonicznych uwzględniamy w referencji
fs = 250 #w pracy napisali, że zdownsamplowali do 250Hz

target_freq = {
    0:9.25,
    1:11.25,
    2:13.25,
    3:9.75,
    4:11.75,
    5:13.75,
    6:10.25,
    7:12.25,
    8:14.25,
    9:10.75,
    10:12.75,
    11:14.75
}

# Zbiera ścieżki do plików MAT z podanego katalogu i dopisuje je do listy.
def read_file(path, mat_files):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mat"):
                mat_files.append(os.path.join(root, file))

# Filtruje sygnał filtrem Butterwortha o podanym zakresie częstotliwości i typie.
def signal_filter(signal, min, max, type):
    b_band, a_band = butter(4, [min/(fs/2), max/(fs/2)], btype=type)
    filtered_signal = filtfilt(b_band, a_band, signal)
    return filtered_signal

# Tłumi zakłócenia o zadanej częstotliwości, domyślnie 50 Hz.
def notch_filter(signal, freq=50, Q=30):
    b_notch, a_notch = iirnotch(freq, Q, fs=fs)
    return filtfilt(b_notch, a_notch, signal)

# Rysuje przebieg sygnału w czasie.
def plot_signal(signal, title):
    time = np.arange(len(signal)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Rysuje widmo FFT sygnału i opcjonalnie zaznacza częstotliwość targetu oraz harmoniczne.
def plot_fft(signal, title, target=None):

    #signal = signal - np.mean(signal)
    signal = detrend(signal)
    window = np.hanning(len(signal))
    fft_vals = np.fft.rfft(signal * window)
    power = np.abs(fft_vals)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1/fs
    )

    plt.figure(figsize=(12, 5))
    plt.plot(frequencies, power)

    if target is not None:
        expected = target_freq[target]
        plt.axvline(
            expected,
            linestyle='--',
            label=f'Expected: {expected} Hz'
        )

        plt.axvline(
            2 * expected,
            linestyle=':',
            label=f'2F={2*expected:.2f} Hz'
        )

        plt.axvline(
            3 * expected,
            linestyle=':',
            label=f'3F={3*expected:.2f} Hz'
        )

    plt.xlim(0, 60)
    plt.xticks(np.arange(0, 61, 5))
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# Wypisuje porównanie widm przed filtracją, po filtrze pasmowym i po filtrze notch.
def check_freq(signal_ssvep, filtered_signal, final_signal):
    freqs = np.fft.rfftfreq(len(signal_ssvep), 1/fs)

    window = np.hanning(len(signal_ssvep))

    fft_before = np.abs(np.fft.rfft(detrend(signal_ssvep) * window))
    fft_filtered = np.abs(np.fft.rfft(detrend(filtered_signal) * window))
    fft_after = np.abs(np.fft.rfft(detrend(final_signal) * window))

    for i in range(0, 255, 5):
        print(
        f"{freqs[i]:.2f} Hz | "
        f"before={fft_before[i]:.1f} | "
        f"filtered={fft_filtered[i]:.1f} | "
        f"after={fft_after[i]:.1f}"
        )

# Zapisuje przefiltrowane dane EEG w pliku MAT o tej samej nazwie.
def save_filtered_file(file, EEG):
    output_dir = os.path.join(current_dir, "filtered_data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, os.path.basename(file))

    sio.savemat(output_path, {"data": EEG})

    print(f"Saved: {output_path}")
                
# Filtruje wszystkie próby z podanych plików MAT, wycina okno SSVEP i zapisuje wyniki.
def filter(mat_files):
    if os.path.exists(path_filtered):
        shutil.rmtree(path_filtered)
    for file in mat_files:
        mat = sio.loadmat(file)
        if 'data' in mat:# and file[-8:-4] in ('S001','S004','S035'):#('S004', 'S018', 'S023', 'S025', 'S035'):
            EEG = mat['data']

            n_channels, n_time, n_etype, n_block, n_target = EEG.shape
            start, end = 160, 660
            EEG_filtered = np.zeros((n_channels, end - start, n_etype, n_block, n_target))

            for electrode_type in range(2):
                for block in range(10):
                    for target in range(12):
                        trial = EEG[:, :, electrode_type, block, target]
                        for channel in range(8):
                            if True:#electrode_type == 0 and block == 1 and channel == 5 and target == 2:
                                signal = trial[channel]  

                                signal_full   = detrend(signal) # usunięcie artefaktów funkcji filtrowania??
                                filtered_full = signal_filter(signal_full, 6, 50, 'bandpass')
                                #final_full    = signal_filter(filtered_full, 49, 50, 'bandstop')
                                final_full = notch_filter(filtered_full, 50)

                                #signal_ssvep    = signal_full[start:end]
                                #filtered_signal = filtered_full[start:end]
                                #final_signal    = final_full[start:end]

                                #plot_fft(signal_ssvep, f"FFT - Oz, raw, {file[-8:-4]}, target - {target}", target)
                                #plot_fft(final_signal, f"FFT - Oz, {file[-8:-4]}, target - {target}", target)
                                #check_freq(signal_ssvep, filtered_signal, final_signal)

                                #plot_fft(signal, "RAW - przed filtrem", target)
                                #plot_fft(final_full, "PO filtrze - w tej samej pętli", target)
                                #plot_fft(final_full[160:660], "PO filtrze - okno ssvep", target)
                                EEG_filtered[channel, :, electrode_type, block, target] = final_full[start:end]
            save_filtered_file(file, EEG_filtered)

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

#read_file(path_raw, mat_files_raw)
#filter(mat_files_raw)
read_file(path_filtered, mat_files_filtered)
x_features, y_labels, subject_ids = process(mat_files_filtered)
print(x_features.shape)
print(y_labels.shape)
print(subject_ids.shape)

i = 0
print("Osoba:", subject_ids[i])
print("Prawdziwy target:", y_labels[i])
print("Wyniki CCA:", x_features[i])
print("Przewidziany target:", np.argmax(x_features[i]))
