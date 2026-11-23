import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import butter, detrend, filtfilt, iirnotch

from .config import current_dir, fs, path_filtered


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

                                EEG_filtered[channel, :, electrode_type, block, target] = final_full[start:end]
            save_filtered_file(file, EEG_filtered)
