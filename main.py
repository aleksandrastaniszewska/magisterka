import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import (
    butter,
    filtfilt,
    iirnotch
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

    signal = signal - np.mean(signal)
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
                
def process(mat_files):
    for file in mat_files:
        mat = sio.loadmat(file)
        if 'data' in mat and file[-8:-4] in ('S035'):
            EEG = mat['data']
            for electrode_type in range(2):
                for block in range(10):
                    for target in range(12):
                        trial = EEG[:, :, electrode_type, block, target]
                        for channel in range(8):
                            if electrode_type == 0 and block == 1 and channel == 5 and target == 2:
                                signal = trial[channel]
                                start = 50
                                end   = 660
                                signal_ssvep = signal[start:end]
                                    #0.00 - 0.50 s    przed stymulacją
                                    #0.50 - 0.64 s    opóźnienie wzrokowe
                                    #0.64 - 2.64 s    właściwa stymulacja SSVEP (2 s)
                                    #2.64 - 2.84 s    po stymulacji
                                plot_fft(signal_ssvep, f"FFT - Oz, raw, {file[-8:-4]}, target - {target}", target)
                                filtered_signal = signal_filter(signal_ssvep, 6, 50, 'bandpass')
                                final_signal = signal_filter(filtered_signal, 48, 52, 'bandstop')
                                plot_fft(final_signal, f"FFT - Oz, {file[-8:-4]}, target - {target}", target)

            
readfile()
process(mat_files)
