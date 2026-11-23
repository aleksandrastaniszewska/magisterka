# Odpowiada za przygotowanie i ładowanie zbiorów danych.
import os

import numpy as np
import scipy.io as sio

from .config import path_filtered


def build_dataset(mat_files):
    all_signals = []
    all_labels = []
    all_subjects = []
    all_headbands = []
    all_blocks = []

    for file in mat_files:
        mat = sio.loadmat(file)
        if "data" not in mat:
            continue

        data = mat["data"]
        _, _, n_headbands, n_blocks, n_targets = data.shape
        subject_id = os.path.basename(file)[:-4]

        for headband in range(n_headbands):
            for block in range(n_blocks):
                for target in range(n_targets):
                    all_signals.append(data[:, :, headband, block, target])
                    all_labels.append(target)
                    all_subjects.append(subject_id)
                    all_headbands.append(headband)
                    all_blocks.append(block)

    X = (
        np.stack(all_signals)
        if all_signals
        else np.empty((0, 8, 500))
    )
    y = np.asarray(all_labels)
    subject = np.asarray(all_subjects)
    headband = np.asarray(all_headbands)
    block = np.asarray(all_blocks)

    os.makedirs(path_filtered, exist_ok=True)
    np.savez(
        os.path.join(path_filtered, "dataset.npz"),
        X=X,
        y=y,
        subject=subject,
        headband=headband,
        block=block,
    )

    return X, y, subject, headband, block
