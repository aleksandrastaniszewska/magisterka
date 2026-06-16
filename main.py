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
import requests
import zipfile


current_dir = os.getcwd()
path = f"{current_dir}\\data\\13560281"
print(path)
mat_files = []
fs = 250 #w paperze napisali, że zdownsamplowali do 250Hz

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

def readfile():

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mat"):
                mat_files.append(os.path.join(root, file))


def signal_filter(signal, min, max, type):
    b_band, a_band = butter(4, [min/(fs/2), max/(fs/2)], btype=type)
    filtered_signal = filtfilt(b_band, a_band, signal)
    return filtered_signal

def notch_filter(signal, freq=50, Q=30):
    b_notch, a_notch = iirnotch(freq, Q, fs=fs)
    return filtfilt(b_notch, a_notch, signal)

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

def check_freq(signal_ssvep, filtered_signal, final_signal):
    freqs = np.fft.rfftfreq(len(signal_ssvep), 1/fs)

    window = np.hanning(len(signal_ssvep))

    fft_before   = np.abs(np.fft.rfft(detrend(signal_ssvep)   * window))
    fft_filtered = np.abs(np.fft.rfft(detrend(filtered_signal) * window))
    fft_after    = np.abs(np.fft.rfft(detrend(final_signal)    * window))

    for i in range(0, 255, 5):
        print(
        f"{freqs[i]:.2f} Hz | "
        f"before={fft_before[i]:.1f} | "
        f"filtered={fft_filtered[i]:.1f} | "
        f"after={fft_after[i]:.1f}"
        )


                
def process(mat_files):
    for file in mat_files:
        mat = sio.loadmat(file)
        if 'data' in mat and file[-8:-4] in ('S004','S035'):#('S004', 'S018', 'S023', 'S025', 'S035'):
            EEG = mat['data']
            for electrode_type in range(2):
                for block in range(10):
                    for target in range(12):
                        trial = EEG[:, :, electrode_type, block, target]
                        for channel in range(8):
                            if electrode_type == 0 and block == 1 and channel == 5 and target == 2:
                                signal = trial[channel]  
                                start = 160
                                end   = 660

                                signal_full   = detrend(signal) # usunięcie artefaktów funkcji filtrowania??
                                filtered_full = signal_filter(signal_full, 6, 50, 'bandpass')
                                #final_full    = signal_filter(filtered_full, 49, 50, 'bandstop')
                                final_full = notch_filter(filtered_full, 50)

                                signal_ssvep    = signal_full[start:end]
                                filtered_signal = filtered_full[start:end]
                                final_signal    = final_full[start:end]

                                plot_fft(signal_ssvep, f"FFT - Oz, raw, {file[-8:-4]}, target - {target}", target)
                                plot_fft(final_signal, f"FFT - Oz, {file[-8:-4]}, target - {target}", target)
                                check_freq(signal_ssvep, filtered_signal, final_signal)



            
readfile()
process(mat_files)
