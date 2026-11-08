"""Naive benchmarks and the ridge regression baseline."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter


# Gets the gain for the training data to make the training data louder for the naive benchmark.
def gain_from_train(x_train, y_train):
    wet = np.sum(np.asarray(y_train, dtype=np.float64) ** 2)
    dry = np.sum(np.asarray(x_train, dtype=np.float64) ** 2)
    return float(np.sqrt(wet / dry))


# Finding X^T X and X^T for solving the ridge regression to find the optimal weight.
def accumulate_normal_equations(x, y, max_past_samples):
    xtx = np.zeros((max_past_samples, max_past_samples), dtype=np.float64)
    xty = np.zeros(max_past_samples, dtype=np.float64)
    count = 0
    padding = np.zeros(max_past_samples - 1, dtype=np.float64)
    for x_slice, y_slice in zip(x, y):
        padded = np.concatenate([padding, x_slice.astype(np.float64)])
        input_windows = sliding_window_view(padded, max_past_samples)[:, ::-1]  # column k = k samples in the past
        xtx += input_windows.T @ input_windows
        xty += input_windows.T @ y_slice.astype(np.float64)
        count += len(y_slice)
    return xtx, xty, count

#Solve to find the optimal weights for our model
def solve_ridge(xtx, xty, count, num_past_samples, alpha):
    """Ridge solution for the first num_past_samples past samples. alpha is relative to mean input power."""
    matrix_a = xtx[:num_past_samples, :num_past_samples] / count + alpha * np.eye(num_past_samples)
    vector_b = xty[:num_past_samples] / count
    return np.linalg.solve(matrix_a, vector_b)


def linear_predict(x, weights):
    """y_hat[n] = sum_k weights[k] * x[n-k], applied independently to each slice (zero start)."""
    return lfilter(weights, [1.0], x.astype(np.float64), axis=1)


fir_predict = linear_predict  # alias for backward compatibility
