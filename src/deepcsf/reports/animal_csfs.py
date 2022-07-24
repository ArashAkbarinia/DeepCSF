"""
Collection of animal CSFs.
"""

import numpy as np


def generic_model(f, k1, k2, alpha, beta):
    the_csf = ((k1 * np.exp(-2 * np.pi * alpha * f)) - (k2 * np.exp(-2 * np.pi * beta * f)))
    the_csf = np.maximum(the_csf, 0)
    return the_csf


def model_fest(fs):
    frequency = np.array([0.5, 1.12, 2, 2.83, 4, 5.66, 8, 11.3, 16, 22.6, 30, 56])
    # for sf=1.12 1.82095312?
    sensitivity = np.array(
        [0.631945311, 1.053242185, 1.9603125, 2.06315625, 2.10648437, 1.99190625,
         1.84360937, 1.62089063, 1.29775, 0.95945312, 0.56746875, 0.1]
    )
    return frequency, sensitivity


def rg_csf():
    frequency = [0.5, 1, 2, 3, 5, 7, 10, 30, 56]
    sensitivity = np.array([1, 0.9, 0.7, 0.5, 0.27, 0.14, 0.02, 1e-2, 1e-2]) * 0.7
    return frequency, sensitivity


def yb_csf():
    frequency = [0.5, 1, 2, 3, 5, 7, 10, 30, 56]
    sensitivity = np.array([1, 0.85, 0.6, 0.4, 0.2, 0.02, 0.01, 1e-2, 1e-2]) * 0.525
    return frequency, sensitivity


def chromatic_csf(chn):
    return rg_csf() if 'rg' in chn else yb_csf()


def get_csf(frequency, method='model_fest', chn='lum'):
    if 'lum' in chn:
        if method == 'model_fest':
            frequency, sensitivity = model_fest(frequency)
        else:
            sensitivity = [csf(f, method=method) for f in frequency]
        sensitivity = np.array(sensitivity)
        sensitivity /= sensitivity.max()
    else:
        frequency, sensitivity = chromatic_csf(chn)
    return np.array(frequency), sensitivity


def csf(f, method='uhlrich'):
    if method == 'uhlrich':
        org_f = f
        if org_f < 1:
            f = 1
        sensitivity = generic_model(f, k1=295.42, k2=295.92, alpha=0.03902, beta=0.0395)
        if org_f < 1:
            sensitivity = sensitivity * org_f
        return sensitivity
    elif method == 'falcon':
        f = f * 2
        org_f = f
        if org_f < 1:
            f = 1
        sensitivity = generic_model(f, k1=424.83, k2=424.87, alpha=0.00953, beta=0.00961)
        if org_f < 1:
            sensitivity = sensitivity * org_f
        return sensitivity
    else:
        return 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)
