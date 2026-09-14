import numpy as np
from utils import create_seq, normalize, denormalize

def test_sequence():
    data = np.random.rand(100, 5)
    X, Y = create_seq(data, seq_len=10)
    assert X.shape[0] == 90
    assert Y.shape[0] == 90

def test_normalization():
    data = np.random.rand(50, 3)
    norm, mean, std = normalize(data)
    restored = denormalize(norm, mean, std)
    assert np.allclose(data, restored, atol=1e-5)