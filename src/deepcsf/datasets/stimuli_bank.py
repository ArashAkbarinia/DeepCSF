"""
A collection of functions to generate stimuli.
"""

import numpy as np
import math
import cv2


def sinusoid_grating(img_size, amp, omega, rho, lambda_wave):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(img_size[0] / 2.0), int(img_size[1] / 2.0))
    [x, y] = np.meshgrid(
        range(-radius[0], radius[0] + 1),
        range(-radius[1], radius[1] + 1)
    )

    stimuli = amp * np.cos((omega[0] * x + omega[1] * y) / lambda_wave + rho)
    return stimuli


def circular_gratings(contrast, rad_length, sf_cpi=None, target_size=None, theta=0, rho=0):
    if target_size is None:
        target_size = [256, 256]
    rows, cols = target_size
    smaller_side = np.minimum(rows, cols)
    if sf_cpi is None:
        sf_cpi = round(smaller_side / 2)
    omega = [np.cos(theta), np.sin(theta)]
    sf_base = ((target_size[0] / 2) / np.pi)
    lambda_wave = sf_base / sf_cpi
    sinusoid_param = {
        'amp': contrast, 'omega': omega, 'rho': rho,
        'img_size': target_size, 'lambda_wave': lambda_wave
    }
    img = sinusoid_grating(**sinusoid_param)

    # if target size is even, the generated stimuli is 1 pixel larger.
    if np.mod(target_size[0], 2) == 0:
        img = img[:-1]
    if np.mod(target_size[1], 2) == 0:
        img = img[:, :-1]

    centre = (int(math.floor(cols / 2)), int(math.floor(rows / 2)))

    mask_img = np.zeros(img.shape, np.uint8)
    mask_img = cv2.circle(mask_img, centre, rad_length, (1, 1, 1), -1)

    img = np.multiply(img, mask_img)
    return img
