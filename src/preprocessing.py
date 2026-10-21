import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import butter, detrend, filtfilt, iirnotch

from .config import current_dir, fs, path_filtered, target_freq


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
