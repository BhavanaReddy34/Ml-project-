import numpy as np


# =========================
# SEQUENCE CREATION (IMPROVED)
# =========================
def create_seq(data, seq_len=24):
    X, Y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len])

    return np.array(X), np.array(Y)


# =========================
# PER-NODE NORMALIZATION (VERY IMPORTANT)
# =========================
def normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8

    norm = (data - mean) / std

    return norm, mean, std


def denormalize(data, mean, std):
    return data * std + mean