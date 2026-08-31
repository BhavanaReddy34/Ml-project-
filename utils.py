import numpy as np
from typing import Tuple


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_seq(
    data: np.ndarray,
    seq_len: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for time-series prediction.

    Args:
        data: 2D array of shape [time, nodes]
        seq_len: Length of input sequence

    Returns:
        X: Array of shape [samples, seq_len, nodes]
        Y: Array of shape [samples, nodes]
    """

    X, Y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len])

    return np.array(X), np.array(Y)


# ============================================================
# NORMALIZATION (PER NODE)
# ============================================================

def normalize(
    data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data per node (column-wise z-score normalization).

    Args:
        data: 2D array [time, nodes]

    Returns:
        norm: Normalized data
        mean: Mean per node
        std: Standard deviation per node
    """

    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8

    norm = (data - mean) / std

    return norm, mean, std


# ============================================================
# DENORMALIZATION
# ============================================================

def denormalize(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Reverse normalization.

    Args:
        data: Normalized data
        mean: Mean per node
        std: Standard deviation per node

    Returns:
        Original scale data
    """

    return data * std + mean