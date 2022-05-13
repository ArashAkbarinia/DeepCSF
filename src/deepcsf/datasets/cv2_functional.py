"""
cv2 implementation of PyTorch functional file.
"""

from __future__ import division

import numpy as np
import numbers
import sys
import collections
import warnings

import torch

import cv2

if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable

INTER_MODE = {
    'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR,
    'BICUBIC': cv2.INTER_CUBIC
}
PAD_MOD = {
    'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_DEFAULT, 'symmetric': cv2.BORDER_REFLECT
}


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a
    torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted to tensor,
         (H x W x C[RGB]).

    Returns:
        Tensor: Converted image.
    """

    if _is_numpy_image(pic):
        if len(pic.shape) == 2:
            pic = np.expand_dims(pic, axis=2)
        img = torch.from_numpy(pic.transpose((2, 0, 1))).type(torch.FloatTensor)
        # backward compatibility
        if isinstance(img, torch.ByteTensor) or img.max() > 1:
            return img.float().div(255)
        else:
            return img
    elif _is_tensor_image(pic):
        return pic
    else:
        try:
            return to_tensor(np.array(pic))
        except Exception:
            raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))


def to_tensor_classes(pic):
    return torch.as_tensor(np.asarray(pic), dtype=torch.int64)


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if type(mean) not in [tuple, list]:
        mean = tuple([mean for _ in range(tensor.shape[0])])
    if type(std) not in [tuple, list]:
        std = tuple([std for _ in range(tensor.shape[0])])

    if _is_tensor_image(tensor):
        if not inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        if not inplace:
            tensor = tensor.copy()
        return (tensor - np.array(mean)) / np.array(std)
    else:
        raise RuntimeError('Undefined type')


def resize(img, size, interpolation='BILINEAR'):
    """Resize the input CV Image to the given size.

    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. If size is a sequence like
         (h, w), the output size will be matched to this. If size is an int,
         the smaller edge of the image will be matched to this number  maintaing
         the aspect ratio. i.e, if height > width, then image will be rescaled
         to (size * height / width, size)
        interpolation (str, optional): Desired interpolation. Default is
         ``BILINEAR``

    Returns:
        cv Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(
                img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation]
            )
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(
                img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation]
            )
    else:
        oh, ow = size
        return cv2.resize(
            img, dsize=(int(ow), int(oh)),
            interpolation=INTER_MODE[interpolation]
        )


def crop(img, top, left, height, width):
    """Crop the given CV Image.

    Args:
        img (np.ndarray): Image to be cropped.
        top: Upper pixel coordinate.
        left: Left pixel coordinate.
        height: Height of the cropped image.
        width: Width of the cropped image.

    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image. Got {}'.format(
        type(img))
    assert (
            height > 0 and width > 0
    ), 'h={} and w={} should greater than 0'.format(height, width)

    x1, y1, x2, y2 = round(top), round(left), round(top + height), round(
        left + width)

    try:
        _ = img[x1, y1, ...]
        _ = img[x2 - 1, y2 - 1, ...]
    except IndexError:
        warnings.warn(
            'crop region is {} but image size is {}'.format(
                (x1, y1, x2, y2), img.shape
            )
        )
        img = cv2.copyMakeBorder(
            img, - min(0, x1), max(x2 - img.shape[0], 0),
            -min(0, y1), max(y2 - img.shape[1], 0),
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)

    finally:
        return img[x1:x2, y1:y2, ...].copy()


def pad(img, padding, fill=(0, 0, 0), padding_mode='constant'):
    """Pad the given CV Image on all sides with speficified padding mode.
    Args:
        img (np.ndarray): Image to be padded.
        padding (int or tuple): Padding on each border.
            If a single int is provided this is used to pad all borders.
            If tuple of length 2 is provided this is the padding on left/right
            and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int, tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding: constant, edge, reflect or symmetric.
            constant: pads with a constant value that is specified with fill
            edge: pads with the last value on the edge of the image
            reflect: pads with reflection of image (without repeating the last
                value on the edge) padding [1, 2, 3, 4] with 2 elements on both
                sides in reflect mode will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value
                on the edge) padding [1, 2, 3, 4] with 2 elements on both sides
                in symmetric mode will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        CV Image: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert (
            padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
    ), 'Padding mode should be either constant, edge, reflect or symmetric'

    pad_left = pad_right = pad_top = pad_bottom = 0
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left, pad_top, pad_right, pad_bottom = padding

    if isinstance(fill, numbers.Number):
        fill = (fill,) * (2 * len(img.shape) - 3)

    if padding_mode == 'constant':
        assert (
            ((len(fill) == 3 and len(img.shape) == 3) or
             (len(fill) == 1 and len(img.shape) == 2))
        ), 'channel of image is {} but length of fill is {}'.format(
            img.shape[-1], len(fill)
        )

    img = cv2.copyMakeBorder(
        src=img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
        borderType=PAD_MOD[padding_mode], value=fill
    )
    return img


def center_crop(img, output_size: int):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[:2]
    th, tw = output_size
    i = int(round((h - th) * 0.5))
    j = int(round((w - tw) * 0.5))
    return crop(img, i, j, th, tw)


def resized_crop(img, top, left, height, width, size, interpolation='BILINEAR'):
    """Crop the given CV2 Image and resize it to desired size. Notably used in
    RandomResizedCrop.

    Args:
        img (np.ndarray): Image to be cropped.
        top: Upper pixel coordinate.
        left: Left pixel coordinate.
        height: Height of the cropped image.
        width: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantic as ``scale``.
        interpolation (str, optional): Desired interpolation. Default is
            ``BILINEAR``.
    Returns:
        np.ndarray: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image'
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def pad_crop(img, padding, pad_if_needed, size, **kwargs):
    if padding > 0:
        img = pad(img, padding)

    # pad the width if needed
    if pad_if_needed and img.shape[1] < size[1]:
        img = pad(img, (int((1 + size[1] - img.shape[1]) / 2), 0))
    # pad the height if needed
    if pad_if_needed and img.shape[0] < size[0]:
        img = pad(img, (0, int((1 + size[0] - img.shape[0]) / 2)))

    return crop(img, **kwargs)


def hflip(img):
    """Horizontally flip the given CV2 Image.

    Args:
        img (np.ndarray): Image to be flipped.

    Returns:
        np.ndarray:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV2 Image. Got {}'.format(type(img)))

    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given CV2 Image.

    Args:
        img (CV2 Image): Image to be flipped.

    Returns:
        CV2 Image:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV2 Image. Got {}'.format(type(img)))

    return cv2.flip(img, 0)
