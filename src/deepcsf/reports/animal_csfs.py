"""
Collection of animal CSFs.
"""

import numpy as np


def generic_model(f, k1, k2, alpha, beta):
    the_csf = (
            (k1 * np.exp(-2 * np.pi * alpha * f)) -
            (k2 * np.exp(-2 * np.pi * beta * f))
    )
    the_csf = np.maximum(the_csf, 0)
    return the_csf


def csf(f, method='uhlrich'):
    if method == 'uhlrich':
        return generic_model(
            f, k1=295.42, k2=295.92, alpha=0.03902, beta=0.0395
        )
    else:
        return 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)
