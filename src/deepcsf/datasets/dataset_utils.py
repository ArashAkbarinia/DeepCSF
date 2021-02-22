"""
Utility functions of all CSF related datasets.
"""

import sys
import os
import numpy as np
import random

import cv2
from torchvision import datasets as tdatasets
from torch.utils import data as torch_data

from ..utils import colour_spaces, system_utils


def _two_pairs_stimuli(img0, img1, con0, con1, p=0.5, contrast_target=None):
    imgs_cat = [img0, img1]
    max_contrast = np.argmax([con0, con1])
    if contrast_target is None:
        contrast_target = 0 if random.random() < p else 1
    if max_contrast != contrast_target:
        imgs_cat = imgs_cat[::-1]

    return (imgs_cat[0], imgs_cat[1]), contrast_target


def _prepare_vision_types(img, colour_space, vision_type):
    if 'grey' not in colour_space and vision_type != 'trichromat':
        opp_img = colour_spaces.rgb2dkl(img)
        if vision_type == 'dichromat_rg':
            opp_img[:, :, 1] = 0
        elif vision_type == 'dichromat_yb':
            opp_img[:, :, 2] = 0
        elif vision_type == 'monochromat':
            opp_img[:, :, [1, 2]] = 0
        else:
            sys.exit('Vision type %s not supported' % vision_type)
        img = colour_spaces.dkl2rgb(opp_img)
    return img


def _adjust_contrast(image, amount):
    amount = np.array(amount)

    assert np.all(amount >= 0.0), 'contrast_level too low.'
    assert np.all(amount <= 1.0), 'contrast_level too high.'

    image = np.float32(image) / 255
    image_contrast = (1 - amount) / 2.0 + np.multiply(image, amount)
    image_contrast *= 255

    return np.uint8(image_contrast)


def _prepare_stimuli(img0, colour_space, vision_type, contrasts, mask_image,
                     pre_transform, post_transform, same_transforms, p,
                     avg_illuminant=0, current_param=None):
    img0 = _prepare_vision_types(img0, colour_space, vision_type)
    img1 = img0.copy()

    # converting to range 0 to 1
    img0 = np.float32(img0) / 255
    img1 = np.float32(img1) / 255

    contrast_target = None
    # if current_param is passed no randomness is generated on the fly
    if current_param is not None:
        # cropping
        srow0, scol0, srow1, scol1 = current_param['crops']
        img0 = img0[srow0:, scol0:, :]
        img1 = img1[srow1:, scol1:, :]
        # flipping
        hflip0, hflip1 = current_param['hflips']
        if hflip0 > 0.5:
            img0 = img0[:, ::-1, :]
        if hflip1 > 0.5:
            img1 = img1[:, ::-1, :]
        # contrast
        contrasts = current_param['contrasts']
        # side of high contrast
        contrast_target = 0 if current_param['ps'] < 0.5 else 1

    if pre_transform is not None:
        if same_transforms:
            img0, img1 = pre_transform([img0, img1])
        else:
            [img0] = pre_transform([img0])
            [img1] = pre_transform([img1])

    if contrasts is None:
        contrast0 = random.uniform(0, 1)
        contrast1 = random.uniform(0, 1)
    else:
        contrast0, contrast1 = contrasts

    if 'grey' in colour_space:
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if colour_space == 'grey':
            img0 = np.expand_dims(img0, axis=2)
            img1 = np.expand_dims(img1, axis=2)
        elif colour_space == 'grey3':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)

    # manipulating the contrast
    img0 = _adjust_contrast(img0, contrast0)
    img1 = _adjust_contrast(img1, contrast1)

    if mask_image == 'gaussian':
        img0 -= 0.5
        img0 = img0 * _gauss_img(img0.shape)
        img0 += 0.5
        img1 -= 0.5
        img1 = img1 * _gauss_img(img1.shape)
        img1 += 0.5

    # adding the avgerage illuminant
    if avg_illuminant is None:
        half_max_contrast = max(contrast0, contrast1) / 2
        ill_gamut = (0.5 - half_max_contrast)
        avg_illuminant = np.random.uniform(low=-ill_gamut, high=ill_gamut)
    img0 += avg_illuminant
    img1 += avg_illuminant

    if post_transform is not None:
        img0, img1 = post_transform([img0, img1])

    img_out, contrast_target = _two_pairs_stimuli(
        img0, img1, contrast0, contrast1, p, contrast_target=contrast_target
    )
    return img_out, contrast_target


def _gauss_img(img_size):
    midx = np.floor(img_size[1] / 2) + 1
    midy = np.floor(img_size[0] / 2) + 1
    y = np.linspace(img_size[0], 0, img_size[0]) - midy
    x = np.linspace(0, img_size[1], img_size[1]) - midx
    [x, y] = np.meshgrid(x, y)
    sigma = min(img_size[0], img_size[1]) / 6
    gauss_img = np.exp(
        -(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))
    )

    gauss_img = gauss_img / np.max(gauss_img)
    if len(img_size) > 2:
        gauss_img = np.repeat(gauss_img[:, :, np.newaxis], img_size[2], axis=2)
    return gauss_img


def _cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class AfcDataset(object):
    def __init__(self, post_transform=None, pre_transform=None, p=0.5,
                 contrasts=None, same_transforms=False, colour_space='grey',
                 vision_type='trichromat', mask_image=None,
                 avg_illuminant=0, train_params=None):
        self.p = p
        self.contrasts = contrasts
        self.same_transforms = same_transforms
        self.colour_space = colour_space
        self.vision_type = vision_type
        self.mask_image = mask_image
        self.post_transform = post_transform
        self.pre_transform = pre_transform
        self.avg_illuminant = avg_illuminant
        if train_params is None:
            self.train_params = train_params
        else:
            self.train_params = system_utils.read_pickle(train_params)
        self.img_counter = 0


class CelebA(AfcDataset, tdatasets.CelebA):
    def __init__(self, afc_kwargs, celeba_kwargs):
        AfcDataset.__init__(self, **afc_kwargs)
        tdatasets.CelebA.__init__(self, **celeba_kwargs)
        self.loader = _cv2_loader

    def __getitem__(self, index):
        path = os.path.join(
            self.root, self.base_folder, "img_align_celeba",
            self.filename[index]
        )
        img0 = self.loader(path)

        img_out, contrast_target = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts,
            self.mask_image, self.pre_transform, self.post_transform,
            self.same_transforms, self.p, self.avg_illuminant
        )

        return img_out, contrast_target, path


class ImageFolder(AfcDataset, tdatasets.ImageFolder):
    def __init__(self, afc_kwargs, folder_kwargs):
        AfcDataset.__init__(self, **afc_kwargs)
        tdatasets.ImageFolder.__init__(self, **folder_kwargs)
        self.loader = _cv2_loader

    def __getitem__(self, index):
        current_param = None
        if self.train_params is not None:
            index = self.train_params['image_inds'][self.img_counter]
            current_param = {
                'ps': self.train_params['ps'][self.img_counter],
                'contrasts': self.train_params['contrasts'][self.img_counter],
                'hflips': self.train_params['hflips'][self.img_counter],
                'crops': self.train_params['crops'][self.img_counter]
            }
            self.img_counter += 1

        path, class_target = self.samples[index]
        img0 = self.loader(path)
        img_out, contrast_target = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts,
            self.mask_image, self.pre_transform, self.post_transform,
            self.same_transforms, self.p, self.avg_illuminant,
            current_param=current_param
        )

        return img_out[0], img_out[1], contrast_target, path


def _create_samples(samples):
    if 'avg_illuminant' in samples:
        avg_illuminant = samples['avg_illuminant']
        del samples['avg_illuminant']
    else:
        avg_illuminant = 0
    settings = samples
    settings['lenghts'] = (
        len(settings['amp']), len(settings['lambda_wave']),
        len(settings['theta']), len(settings['rho']), len(settings['side'])
    )
    num_samples = np.prod(np.array(settings['lenghts']))
    return num_samples, settings, avg_illuminant


class GratingImages(AfcDataset, torch_data.Dataset):
    def __init__(self, samples, afc_kwargs, target_size,
                 contrast_space=None, theta=None, rho=None, lambda_wave=None):
        AfcDataset.__init__(self, **afc_kwargs)
        torch_data.Dataset.__init__(self)
        if type(samples) is dict:
            # under this condition one contrast will be zero while the other
            # takes the arguments of samples.
            (
                self.samples, self.settings, self.avg_illuminant
            ) = _create_samples(samples)
        else:
            self.samples = samples
            self.settings = None
        if type(target_size) not in [list, tuple]:
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.contrast_space = contrast_space
        self.theta = theta
        self.rho = rho
        self.lambda_wave = lambda_wave

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        if self.settings is None:
            if self.contrasts is None:
                contrast0 = random.uniform(0, 1)
                contrast1 = random.uniform(0, 1)
            else:
                contrast0, contrast1 = self.contrasts

            # randomising the parameters
            if self.theta is None:
                theta = random.uniform(0, np.pi)
            else:
                theta = self.theta
            if self.rho is None:
                rho = random.uniform(0, np.pi)
            else:
                rho = self.rho
            if self.lambda_wave is None:
                lambda_wave = random.uniform(np.pi / 2, np.pi * 10)
            else:
                lambda_wave = self.lambda_wave
        else:
            inds = np.unravel_index(index, self.settings['lenghts'])
            contrast0 = self.settings['amp'][inds[0]]
            lambda_wave = self.settings['lambda_wave'][inds[1]]
            theta = self.settings['theta'][inds[2]]
            rho = self.settings['rho'][inds[3]]
            self.p = self.settings['side'][inds[4]]
            contrast1 = 0
        omega = [np.cos(theta), np.sin(theta)]

        # generating the gratings
        sinusoid_param = {
            'amp': contrast0, 'omega': omega, 'rho': rho,
            'img_size': self.target_size, 'lambda_wave': lambda_wave
        }
        img0 = _sinusoid(**sinusoid_param)
        sinusoid_param['amp'] = contrast1
        img1 = _sinusoid(**sinusoid_param)

        # multiply it by gaussian
        if self.mask_image == 'fixed_size':
            radius = (
                int(self.target_size[0] / 2.0),
                int(self.target_size[1] / 2.0)
            )
            [x, y] = np.meshgrid(
                range(-radius[0], radius[0] + 1),
                range(-radius[1], radius[1] + 1)
            )
            x1 = +x * np.cos(theta) + y * np.sin(theta)
            y1 = -x * np.sin(theta) + y * np.cos(theta)

            k = 2
            o1 = 8
            o2 = o1 / 2
            omg = (1 / 8) * (np.pi ** 2 / lambda_wave)
            gauss_img = omg ** 2 / (o2 * np.pi * k ** 2) * np.exp(
                -omg ** 2 / (o1 * k ** 2) * (1 * x1 ** 2 + y1 ** 2))

            gauss_img = gauss_img / np.max(gauss_img)
            img0 *= gauss_img
            img1 *= gauss_img
        elif self.mask_image == 'fixed_cycle':
            radius = (
                int(self.target_size[0] / 2.0),
                int(self.target_size[1] / 2.0)
            )
            [x, y] = np.meshgrid(
                range(-radius[0], radius[0] + 1),
                range(-radius[1], radius[1] + 1)
            )

            sigma = self.target_size[0] / 6
            gauss_img = np.exp(
                -(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))
            )

            gauss_img = gauss_img / np.max(gauss_img)
            img0 *= gauss_img
            img1 *= gauss_img

        img0 = (img0 + 1) / 2
        img1 = (img1 + 1) / 2
        # adding the avgerage illuminant
        img0 += self.avg_illuminant
        img1 += self.avg_illuminant

        # if target size is even, the generated stimuli is 1 pixel larger.
        if np.mod(self.target_size[0], 2) == 0:
            img0 = img0[:-1]
            img1 = img1[:-1]
        if np.mod(self.target_size[1], 2) == 0:
            img0 = img0[:, :-1]
            img1 = img1[:, :-1]

        if self.colour_space != 'grey':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
            if self.contrast_space == 'yb':
                img0[:, :, [0, 1]] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, [0, 1]] = 0.5
                img1 = colour_spaces.dkl012rgb01(img1)
            elif self.contrast_space == 'rg':
                img0[:, :, [0, 2]] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, [0, 2]] = 0.5
                img1 = colour_spaces.dkl012rgb01(img1)
            elif self.contrast_space != 'rgb':
                sys.exit('Contrast %s not supported' % self.contrast_space)

        if 'grey' not in self.colour_space and self.vision_type != 'trichromat':
            dkl0 = colour_spaces.rgb2dkl(img0)
            dkl1 = colour_spaces.rgb2dkl(img1)
            if self.vision_type == 'dichromat_rg':
                dkl0[:, :, 1] = 0
                dkl1[:, :, 1] = 0
            elif self.vision_type == 'dichromat_yb':
                dkl0[:, :, 2] = 0
                dkl1[:, :, 2] = 0
            elif self.vision_type == 'monochromat':
                dkl0[:, :, [1, 2]] = 0
                dkl1[:, :, [1, 2]] = 0
            else:
                sys.exit('Vision type %s not supported' % self.vision_type)
            img0 = colour_spaces.dkl2rgb01(dkl0)
            img1 = colour_spaces.dkl2rgb01(dkl1)

        if self.post_transform is not None:
            img0, img1 = self.post_transform([img0, img1])
        img_out, contrast_target = _two_pairs_stimuli(
            img0, img1, contrast0, contrast1, self.p
        )

        item_settings = np.array([contrast0, lambda_wave, theta, rho, self.p])

        return img_out[0], img_out[1], contrast_target, item_settings

    def __len__(self):
        return self.samples


def _sinusoid(img_size, amp, omega, rho, lambda_wave):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(img_size[0] / 2.0), int(img_size[1] / 2.0))
    [x, y] = np.meshgrid(
        range(-radius[0], radius[0] + 1),
        range(-radius[1], radius[1] + 1)
    )

    stimuli = amp * np.cos((omega[0] * x + omega[1] * y) / lambda_wave + rho)
    return stimuli
