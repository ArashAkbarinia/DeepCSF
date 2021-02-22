"""
Implementation of PyTorch transforms functions in open-cv to support more
flexible type of inputs.
"""

import sys
import math
import random
import warnings
import numbers

import numpy as np

import collections

if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable

from . import cv2_functional as tfunctional


class RandomCrop(object):
    """Crop the given CV Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size: int, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (CV Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        try:
            i = random.randint(0, h - th)
        except ValueError:
            i = random.randint(h - th, 0)
        try:
            j = random.randint(0, w - tw)
        except ValueError:
            j = random.randint(w - tw, 0)
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Args:
            img (np.ndarray): Image to be cropped.
        Returns:
            np.ndarray: Cropped image.
        """

        top, left, height, width = self.get_params(
            _find_first_image_recursive(imgs), self.size
        )
        fun = tfunctional.pad_crop
        kwargs = {
            'top': top, 'left': left, 'height': height, 'width': width,
            'size': self.size,
            'padding': self.padding, 'pad_if_needed': self.pad_if_needed
        }
        return _call_recursive(imgs, fun, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(
            self.size, self.padding)


class RandomResizedCrop(object):
    """Crop the given CV2 Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is
    made. This crop is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='BILINEAR'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (CV2 Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio
                           cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): List of images to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized images.
        """
        top, left, height, width = self.get_params(
            _find_first_image_recursive(imgs), self.scale, self.ratio
        )
        fun = tfunctional.resized_crop
        kwargs = {
            'top': top, 'left': left, 'height': height, 'width': width,
            'size': self.size, 'interpolation': self.interpolation
        }
        return _call_recursive(imgs, fun, **kwargs)

    def __repr__(self):
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(
            tuple(round(s, 4) for s in self.scale)
        )
        format_string += ', ratio={0}'.format(
            tuple(round(r, 4) for r in self.ratio)
        )
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomResizedCropSegmentation(RandomResizedCrop):
    """Crop the given CV2 Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is
    made. This crop is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: BILINEAR
    """

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): List of images to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized images.
        """
        assert len(imgs) == 2
        top, left, height, width = self.get_params(
            _find_first_image_recursive(imgs), self.scale, self.ratio
        )

        kwargs = {
            'top': top, 'left': left, 'height': height, 'width': width,
            'size': self.size, 'interpolation': self.interpolation
        }
        image = tfunctional.resized_crop(imgs[0], **kwargs)
        kwargs = {
            'top': top, 'left': left, 'height': height, 'width': width,
            'size': self.size, 'interpolation': 'NEAREST'
        }
        target = tfunctional.resized_crop(imgs[1], **kwargs)
        return image, target

    def __repr__(self):
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(
            tuple(round(s, 4) for s in self.scale)
        )
        format_string += ', ratio={0}'.format(
            tuple(round(r, 4) for r in self.ratio)
        )
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomHorizontalFlip(object):
    """Horizontally flip the given cv2 Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (cv2 Image): List of images to be flipped.

        Returns:
            cv2 Image: Randomly flipped images.
        """
        if random.random() < self.p:
            fun = tfunctional.hflip
            return _call_recursive(imgs, fun)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize(object):
    """Resize the input cv2 Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.Image.BILINEAR``
    """

    def __init__(self, size, interpolation='BILINEAR'):
        assert (isinstance(size, int) or
                (isinstance(size, Iterable) and len(size) == 2))
        self.size = size
        self.interpolation = interpolation
        self.kwargs = {'size': self.size, 'interpolation': self.interpolation}

    def __call__(self, imgs):
        """
        Args:
            imgs (cv2 Image): List of images to be scaled.

        Returns:
            cv2 Image: Rescaled image.
        """
        fun = tfunctional.resize
        kwargs = self.kwargs
        return _call_recursive(imgs, fun, **kwargs)

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str
        )


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensor): List of tensor images of size (C, H, W) to be
             normalised.

        Returns:
            Tensor: Normalized Tensor image.
        """
        fun = tfunctional.normalize
        kwargs = {'mean': self.mean, 'std': self.std}
        return _call_recursive(tensors, fun, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std
        )


class NormalizeSegmentation(object):
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.kwargs = {
            'mean': self.mean, 'std': self.std,  # FIXME'inplace': self.inplace
        }

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensor): List of tensor images of size (C, H, W) to be
             normalised.

        Returns:
            Tensor: Normalized Tensor image.
        """
        assert len(tensors) == 2
        kwargs = self.kwargs
        image = tfunctional.normalize(tensors[0], **kwargs)
        return image, tensors[1]

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std
        )


class ToTensor(object):
    """Convert a ``cv2 Image`` or ``numpy.ndarray`` to tensor.
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pics):
        """
        Args:
            pics (List of cv2 Image or numpy.ndarray): Image to be converted to
             tensor.

        Returns:
            Tensor: Converted images.
        """
        fun = tfunctional.to_tensor
        return _call_recursive(pics, fun)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensorSegmentation(object):
    """Convert a ``cv2 Image`` or ``numpy.ndarray`` to tensor.
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pics):
        """
        Args:
            pics (List of cv2 Image or numpy.ndarray): Image to be converted to
             tensor.

        Returns:
            Tensor: Converted images.
        """
        assert len(pics) == 2
        image = tfunctional.to_tensor(pics[0])
        target = tfunctional.to_tensor_classes(pics[1])
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterCrop(object):
    """Crops the given cv2 Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.kwargs = {'output_size': self.size}

    def __call__(self, imgs):
        """
        Args:
            imgs (cv2 Image): List of images to be cropped.

        Returns:
            cv2 Image: Cropped images.
        """
        fun = tfunctional.center_crop
        kwargs = self.kwargs
        return _call_recursive(imgs, fun, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def _call_recursive(imgs, fun, **kwargs):
    if type(imgs) is list:
        inner_list = []
        for img in imgs:
            inner_list.append(_call_recursive(img, fun, **kwargs))
        return inner_list
    elif type(imgs) is dict:
        inner_dict = dict()
        for key, val in imgs.items():
            inner_dict[key] = _call_recursive(val, fun, **kwargs)
        return inner_dict
    else:
        return fun(imgs, **kwargs)


def _find_first_image_recursive(imgs):
    if type(imgs) is list:
        return _find_first_image_recursive(imgs[0])
    elif type(imgs) is dict:
        key = list(imgs.keys())[0]
        return _find_first_image_recursive(imgs[key])
    else:
        return imgs


def inverse_mean_std(mean, std):
    mean = np.array(mean)
    std = np.array(std)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    return mean_inv, std_inv


def normalize_inverse(imgs, mean, std):
    mean_inv, std_inv = inverse_mean_std(mean, std)
    return tfunctional.normalize(imgs, mean_inv, std_inv)


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input
    domain.
    """

    def __init__(self, mean, std):
        mean_inv, std_inv = inverse_mean_std(mean, std)
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor)
