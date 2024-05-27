import pandas as pd
import numpy as np
import time
import asyncio


def calculate_a(X):
    shape = X.shape
    a = (3 * (shape[1] + 1)) / shape[0]
    return a

def calculate_doa_matrix(X):
    x_T = X.transpose()
    x_out = x_T.dot(X)
    x_out_inv = pd.DataFrame(np.linalg.pinv(x_out.values), x_out.columns, x_out.index)
    return x_out_inv
