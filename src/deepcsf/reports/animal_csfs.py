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


def csf(f, method='human'):
    uhlrich_pars = {
        'human': [295.42, 295.92, 0.03902, 0.03958],
        'cat': [1.6847, 2.8457, 0.2278, 2.0526],
        'falcon': [424.83, 424.87, 0.00953, 0.00961],
        'goldfish': [2.1411, 2.4682, 0.4840, 0.9219],
        'macaque': [246.14, 246.36, 0.03908, 0.03945],
        'pigeon': [30.921, 30.934, 0.04500, 0.04536],
        'owl': [2.4823, 2.5085, 0.2569, 0.2611]
    }
    if method in uhlrich_pars.keys():
        param = uhlrich_pars[method]
        if method == 'human' and f < 1:
            sensitivity = generic_model(1, k1=param[0], k2=param[1], alpha=param[2], beta=param[3])
            sensitivity = sensitivity * f
        else:
            sensitivity = generic_model(f, k1=param[0], k2=param[1], alpha=param[2], beta=param[3])
        return sensitivity
    else:
        return 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)
