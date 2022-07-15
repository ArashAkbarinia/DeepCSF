"""
Different colour spaces.
"""

import numpy as np

rgb_from_dkl = np.array(
    [[+0.49995000, +0.50001495, +0.49999914],
     [+0.99998394, -0.29898596, +0.01714922],
     [-0.17577361, +0.15319546, -0.99994349]]
)

dkl_from_rgb = np.array(
    [[0.4251999971, +0.8273000025, +0.2267999991],
     [1.4303999955, -0.5912000011, +0.7050999939],
     [0.1444000069, -0.2360000005, -0.9318999983]]
)

yog_from_rgb = np.array(
    [[+0.25, +0.50, +0.25],
     [+0.50, +0.00, -0.50],
     [-0.25, +0.50, -0.25]]
).T

rgb_from_yog = np.array(
    [[+1.0, +1.0, -1.0],
     [+1.0, +0.0, +1.0],
     [+1.0, -1.0, -1.0]]
).T


def rgb012dkl(x):
    return np.dot(x, dkl_from_rgb)


def rgb2dkl(x):
    return rgb012dkl(rgb2double(x))


def rgb2dkl01(x):
    x = rgb2dkl(x)
    x /= 2
    x[:, :, 1] += 0.5
    x[:, :, 2] += 0.5
    return x


def dkl2rgb(x):
    return uint8im(dkl2rgb01(x))


def dkl2rgb01(x):
    x = np.dot(x, rgb_from_dkl)
    return clip01(x)


def dkl012rgb01(x):
    x = x.copy()
    x[:, :, 1] -= 0.5
    x[:, :, 2] -= 0.5
    x *= 2
    return dkl2rgb01(x)


def dkl012rgb(x):
    return uint8im(dkl012rgb01(x))


def rgb012yog(x):
    return np.dot(x, yog_from_rgb)


def rgb2yog(x):
    return rgb012yog(rgb2double(x))


def rgb2yog01(x):
    x = rgb2yog(x)
    x[:, :, 1] += 0.5
    x[:, :, 2] += 0.5
    return x


def yog2rgb(x):
    return uint8im(yog2rgb01(x))


def yog2rgb01(x):
    x = np.dot(x, rgb_from_yog)
    return clip01(x)


def yog012rgb(x):
    return uint8im(yog012rgb01(x))


def yog012rgb01(x):
    x = x.copy()
    x[:, :, 1] -= 0.5
    x[:, :, 2] -= 0.5
    return yog2rgb01(x)


def rgb2double(x):
    if x.dtype == 'uint8':
        x = np.float32(x) / 255
    else:
        assert x.max() <= 1, 'rgb must be either uint8 or in the range of [0 1]'
    return x


def clip01(x):
    x = np.maximum(x, 0)
    x = np.minimum(x, 1)
    return x


def uint8im(image):
    image = clip01(image)
    image *= 255
    return np.uint8(image)
