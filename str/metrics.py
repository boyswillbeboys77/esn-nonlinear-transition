import numpy as np

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def nrmse_range(a, b, eps=1e-12):
    return rmse(a, b) / (np.max(a) - np.min(a) + eps)

def r2(a, b):
    return 1 - np.sum((b - a) ** 2) / (np.sum((a - a.mean()) ** 2) + 1e-12)