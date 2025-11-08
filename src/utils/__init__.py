import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

def gumbel_weighted_permutation(weights: NDArray[np.integer]) -> NDArray[np.integer]:
    with np.errstate(divide='ignore'):
        log_weights = np.log(weights)
    log_weights[np.isneginf(log_weights)] = np.finfo(log_weights.dtype).min

    gumbel_noise = np.random.gumbel(size=log_weights.shape)
    perturbed_weights = log_weights + gumbel_noise
    _row_ind, col_ind = linear_sum_assignment(perturbed_weights)
    return col_ind
