"""
Image manipulation functions.
"""

import numpy as np
import math
import cv2

from . import colour_spaces


def adjust_contrast(image, amount):
    amount = np.array(amount)

    assert np.all(amount >= 0.0), 'contrast_level too low.'
    assert np.all(amount <= 1.0), 'contrast_level too high.'

    is_uint8 = image.dtype == 'uint8'
    if is_uint8:
        image = np.float32(image) / 255
    image_contrast = (1 - amount) / 2.0 + np.multiply(image, amount)
    if is_uint8:
        image_contrast *= 255
        image_contrast = np.uint8(image_contrast)

    return image_contrast


def filter_img_sf(img, **kwargs):
    if len(img.shape) > 2:
        # assuming RGB
        dkl = colour_spaces.rgb012dkl01(img)
        img_freq = np.fft.fft2(dkl[:, :, 0])
    else:
        img_freq = np.fft.fft2(img)

    img_freq_cent = np.fft.fftshift(img_freq)
    img_sf_filtered = filter_chn_sf(img_freq_cent, **kwargs)

    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(img_sf_filtered)))
    img_back[img_back < 0] = 0
    img_back[img_back > 1] = 1

    if len(img.shape) > 2:
        dkl[:, :, 0] = img_back
        return colour_spaces.dkl012rgb01(dkl)
    else:
        return img_back


def filter_chn_sf(img, hsf_cut, lsf_cut):
    rows = img.shape[0]
    cols = img.shape[1]
    smaller_side = np.minimum(rows, cols)
    centre = (int(math.floor(cols / 2)), int(math.floor(rows / 2)))

    if hsf_cut == 0:
        mask_hsf = np.ones(img.shape, np.uint8)
    else:
        hsf_cut = 1 - hsf_cut
        hsf_length = int(math.floor(hsf_cut * smaller_side * 0.5))
        mask_hsf = np.zeros(img.shape, np.uint8)
        mask_hsf = cv2.circle(mask_hsf, centre, hsf_length, (1, 1, 1), -1)

    if lsf_cut == 0:
        mask_lsf = np.ones(img.shape, np.uint8)
    else:
        lsf_length = int(math.floor(lsf_cut * smaller_side * 0.5))
        mask_lsf = np.zeros(img.shape, np.uint8)
        mask_lsf = 1 - cv2.circle(mask_lsf, centre, lsf_length, (1, 1, 1), -1)

    mask_img = np.logical_and(mask_lsf, mask_hsf).astype('uint8')
    img_sf_filtered = np.multiply(img, mask_img)
    return img_sf_filtered
