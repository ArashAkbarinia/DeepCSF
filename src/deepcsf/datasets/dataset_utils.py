"""
Utility functions of all CSF related datasets.
"""

import sys
import os
import numpy as np
import random
import glob

import cv2
from skimage import io
from torchvision import datasets as tdatasets
from torch.utils import data as torch_data

from ..utils import colour_spaces, system_utils
from . import imutils
from . import stimuli_bank


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


def _prepare_grating_detector(img0, colour_space, vision_type, contrasts, mask_image,
                              pre_transform, post_transform, p, illuminant_range=1.0,
                              sf_filter=None, contrast_space='rgb'):
    img0 = _prepare_vision_types(img0, colour_space, vision_type)
    # converting to range 0 to 1
    img0 = np.float32(img0) / 255

    if pre_transform is not None:
        [img0] = pre_transform([img0])

    if contrasts is None:
        contrast0 = random.uniform(0, 1)
    else:
        contrast0 = random.uniform(contrasts[0], contrasts[1])

    if 'grey' in colour_space:
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        if colour_space == 'grey':
            img0 = np.expand_dims(img0, axis=2)
        elif colour_space == 'grey3':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)

    # applying SF filter
    if sf_filter is not None:
        hsf_cut, lsf_cut = sf_filter
        img0 = imutils.filter_img_sf(img0, hsf_cut=hsf_cut, lsf_cut=lsf_cut)

    # manipulating the contrast
    if contrast_space != 'rgb':
        img0 = colour_spaces.rgb2dkl01(img0)

    if random.random() < p:
        grating = _random_grating(img0.shape[0], contrast0)
        contrast_target = 1

        if contrast_space == 'lum':
            img0[:, :, 0] = (img0[:, :, 0] + grating) / 2
        elif contrast_space == 'rg':
            img0[:, :, 1] = (img0[:, :, 1] + grating) / 2
        elif contrast_space == 'yb':
            img0[:, :, 2] = (img0[:, :, 2] + grating) / 2
        elif contrast_space == 'rgb':
            grating = np.repeat(grating[:, :, np.newaxis], 3, axis=2)
            img0 = (img0 + grating) / 2
        elif contrast_space == 'dkl':
            chn = random.randint(0, 2)
            img0[:, :, chn] = (img0[:, :, chn] + grating) / 2
            for i in range(3):
                if i == chn:
                    continue
                if random.random() < 0.5:
                    img0[:, :, i] = (img0[:, :, i] + grating) / 2
    else:
        grating = 0.5
        contrast_target = 0
        img0 = (img0 + grating) / 2

    if contrast_space != 'rgb':
        img0 = colour_spaces.dkl012rgb01(img0)

    # multiplying by the illuminant
    if illuminant_range is None:
        illuminant_range = [1e-4, 1.0]
    if type(illuminant_range) in (list, tuple):
        if len(illuminant_range) == 1:
            ill_val = illuminant_range[0]
        else:
            ill_val = np.random.uniform(low=illuminant_range[0], high=illuminant_range[1])
    else:
        ill_val = illuminant_range
    # we simulate the illumination with multiplication
    # is ill_val is very small, the image becomes very dark
    img0 *= ill_val
    half_ill = ill_val / 2

    if mask_image == 'gaussian':
        img0 -= half_ill
        img0 = img0 * _gauss_img(img0.shape)
        img0 += half_ill

    if post_transform is not None:
        [img0] = post_transform([img0])

    return img0, contrast_target


def _prepare_stimuli(img0, colour_space, vision_type, contrasts, mask_image,
                     pre_transform, post_transform, same_transforms, p,
                     illuminant_range=1.0, current_param=None, sf_filter=None,
                     contrast_space='rgb', grating_detector=False):
    if grating_detector:
        if current_param:
            sys.exit('For grating_detector current_param cant be true')
        return _prepare_grating_detector(
            img0, colour_space, vision_type, contrasts, mask_image, pre_transform, post_transform,
            p, illuminant_range, sf_filter, contrast_space
        )

    img0 = _prepare_vision_types(img0, colour_space, vision_type)
    # converting to range 0 to 1
    img0 = np.float32(img0) / 255
    # copying to img1
    img1 = img0.copy()

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

    # applying SF filter
    if sf_filter is not None:
        hsf_cut, lsf_cut = sf_filter
        img0 = imutils.filter_img_sf(img0, hsf_cut=hsf_cut, lsf_cut=lsf_cut)
        img1 = imutils.filter_img_sf(img1, hsf_cut=hsf_cut, lsf_cut=lsf_cut)

    # manipulating the contrast
    if contrast_space == 'dkl':
        img0 = colour_spaces.rgb2dkl01(img0)
        img1 = colour_spaces.rgb2dkl01(img1)
    img0 = imutils.adjust_contrast(img0, contrast0)
    img1 = imutils.adjust_contrast(img1, contrast1)
    if contrast_space == 'dkl':
        img0 = colour_spaces.dkl012rgb01(img0)
        img1 = colour_spaces.dkl012rgb01(img1)

    # multiplying by the illuminant
    if illuminant_range is None:
        illuminant_range = [1e-4, 1.0]
    if type(illuminant_range) in (list, tuple):
        if len(illuminant_range) == 1:
            ill_val = illuminant_range[0]
        else:
            ill_val = np.random.uniform(low=illuminant_range[0], high=illuminant_range[1])
    else:
        ill_val = illuminant_range
    # we simulate the illumination with multiplication
    # is ill_val is very small, the image becomes very dark
    img0 *= ill_val
    img1 *= ill_val
    half_ill = ill_val / 2

    if mask_image == 'gaussian':
        img0 -= half_ill
        img0 = img0 * _gauss_img(img0.shape)
        img0 += half_ill
        img1 -= half_ill
        img1 = img1 * _gauss_img(img1.shape)
        img1 += half_ill

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
    gauss_img = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

    gauss_img = gauss_img / np.max(gauss_img)
    if len(img_size) > 2:
        gauss_img = np.repeat(gauss_img[:, :, np.newaxis], img_size[2], axis=2)
    return gauss_img


def _cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class AfcDataset(object):
    def __init__(self, post_transform=None, pre_transform=None, p=0.5, contrasts=None,
                 same_transforms=False, colour_space='grey', vision_type='trichromat',
                 mask_image=None, illuminant_range=1.0, train_params=None, sf_filter=None,
                 contrast_space='rgb', grating_detector=False):
        self.p = p
        self.contrasts = contrasts
        self.same_transforms = same_transforms
        self.colour_space = colour_space
        self.vision_type = vision_type
        self.mask_image = mask_image
        self.post_transform = post_transform
        self.pre_transform = pre_transform
        self.illuminant_range = illuminant_range
        if train_params is None:
            self.train_params = train_params
        else:
            self.train_params = system_utils.read_pickle(train_params)
        self.img_counter = 0
        self.sf_filter = sf_filter
        self.contrast_space = contrast_space
        self.grating_detector = grating_detector


class CelebA(AfcDataset, tdatasets.CelebA):
    def __init__(self, afc_kwargs, celeba_kwargs):
        AfcDataset.__init__(self, **afc_kwargs)
        tdatasets.CelebA.__init__(self, **celeba_kwargs)
        self.loader = _cv2_loader

    def __getitem__(self, index):
        path = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        img0 = self.loader(path)

        img_out, contrast_target = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts, self.mask_image,
            self.pre_transform, self.post_transform, self.same_transforms, self.p,
            self.illuminant_range, sf_filter=self.sf_filter, contrast_space=self.contrast_space,
            grating_detector=self.grating_detector
        )

        if self.grating_detector:
            return img_out, contrast_target, path
        else:
            return img_out[0], img_out[1], contrast_target, path


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
            img0, self.colour_space, self.vision_type, self.contrasts, self.mask_image,
            self.pre_transform, self.post_transform, self.same_transforms, self.p,
            self.illuminant_range, current_param=current_param, sf_filter=self.sf_filter,
            contrast_space=self.contrast_space, grating_detector=self.grating_detector
        )

        if self.grating_detector:
            return img_out, contrast_target, path
        else:
            return img_out[0], img_out[1], contrast_target, path


class ShapeDataset(torch_data.Dataset):
    def __init__(self, root, background=None, **kwargs):
        self.root = root
        self.target_size = (224, 224)
        self.mask_size = (128, 128)
        self.imgdir = '%s/shape2D/' % self.root
        self.bg = background

    def _prepare_out_imgs(self, bw_img, target_colour, place_fun):
        mask = cv2.resize(bw_img, self.mask_size, interpolation=cv2.INTER_NEAREST)
        current_colour = target_colour
        if os.path.exists(self.bg):
            bg_img = io.imread(self.bg)
            mask_img = cv2.resize(bg_img, self.mask_size, interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(bg_img, self.target_size, interpolation=cv2.INTER_NEAREST)
        elif self.bg == 'rnd':
            mask_img = np.random.randint(0, 256, (*self.mask_size, 3), dtype='uint8')
            img = np.random.randint(0, 256, (*self.target_size, 3), dtype='uint8')
        else:
            mask_img = np.zeros((*self.mask_size, 3), dtype='uint8') + int(self.bg)
            img = np.zeros((*self.target_size, 3), dtype='uint8') + int(self.bg)
        # converting images to float
        mask_img = mask_img.astype('float32') / 255
        img = img.astype('float32') / 255

        for chn_ind in range(3):
            current_chn = mask_img[:, :, chn_ind]
            current_chn[mask == 255] = current_colour[chn_ind]

        srow, scol = place_fun(self.mask_size, self.target_size)
        erow = srow + self.mask_size[0]
        ecol = scol + self.mask_size[1]
        img[srow:erow, scol:ecol] = mask_img

        return (img * 255).astype('uint8')


def _random_place(mask_size, target_size):
    srow = random.randint(0, target_size[0] - mask_size[0])
    scol = random.randint(0, target_size[1] - mask_size[1])

    return srow, scol


class ShapeTrain(ShapeDataset):

    def __init__(self, root, transform=None, colour_dist=None, **kwargs):
        ShapeDataset.__init__(self, root, transform=transform, **kwargs)
        if self.bg is None:
            self.bg = 'rnd'
        self.angles = (1, 11)
        self.img_paths = sorted(glob.glob(self.imgdir + '*.png'))
        self.colour_dist = colour_dist
        if self.colour_dist is not None:
            self.colour_dist = np.loadtxt(self.colour_dist, delimiter=',', dtype=int)

    def _prepare_train_imgs(self, mask_img, target_colour):
        target_colour = np.array(target_colour).astype('float32') / 255
        return self._prepare_out_imgs(mask_img, target_colour, _random_place)

    def _get_target_colour(self):
        if self.colour_dist is not None:
            rand_row = random.randint(0, len(self.colour_dist) - 1)
            target_colour = self.colour_dist[rand_row]
        else:
            target_colour = [random.randint(0, 255) for _ in range(3)]
        return target_colour

    def __len__(self):
        return len(self.img_paths)


class BinaryShapes(AfcDataset, ShapeTrain):
    def __init__(self, afc_kwargs, shape_kwargs):
        AfcDataset.__init__(self, **afc_kwargs)
        ShapeTrain.__init__(self, **shape_kwargs)
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

        path = self.img_paths[index]
        mask_img = io.imread(path)
        target_colour = self._get_target_colour()
        img0 = self._prepare_train_imgs(mask_img, target_colour)

        img_out, contrast_target = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts, self.mask_image,
            self.pre_transform, self.post_transform, self.same_transforms, self.p,
            self.illuminant_range, current_param=current_param, sf_filter=self.sf_filter,
            contrast_space=self.contrast_space, grating_detector=self.grating_detector
        )

        if self.grating_detector:
            return img_out, contrast_target, path
        else:
            return img_out[0], img_out[1], contrast_target, path


def _create_samples(samples):
    if 'illuminant_range' in samples:
        illuminant_range = samples['illuminant_range']
        del samples['illuminant_range']
    else:
        illuminant_range = 1.0
    settings = samples
    settings['lenghts'] = (
        len(settings['amp']), len(settings['lambda_wave']),
        len(settings['theta']), len(settings['rho']), len(settings['side'])
    )
    num_samples = np.prod(np.array(settings['lenghts']))
    return num_samples, settings, illuminant_range


def _random_grating(target_size, contrast0):
    rho = random.uniform(0, np.pi)
    sf = random.randint(1, target_size / 2)
    lambda_wave = ((target_size / 2) / np.pi) / sf
    theta = random.uniform(0, np.pi)
    omega = [np.cos(theta), np.sin(theta)]
    sinusoid_param = {
        'amp': contrast0, 'omega': omega, 'rho': rho,
        'img_size': [target_size, target_size], 'lambda_wave': lambda_wave
    }
    img0 = stimuli_bank.sinusoid_grating(**sinusoid_param)
    img0 = (img0 + 1) / 2

    # if target size is even, the generated stimuli is 1 pixel larger.
    if np.mod(target_size, 2) == 0:
        img0 = img0[:-1]
    if np.mod(target_size, 2) == 0:
        img0 = img0[:, :-1]
    return img0


class GratingImages(AfcDataset, torch_data.Dataset):
    def __init__(self, samples, afc_kwargs, target_size, theta=None, rho=None, lambda_wave=None):
        AfcDataset.__init__(self, **afc_kwargs)
        torch_data.Dataset.__init__(self)
        if type(samples) is dict:
            # under this condition one contrast will be zero while the other
            # takes the arguments of samples.
            self.samples, self.settings, self.illuminant_range = _create_samples(samples)
        else:
            self.samples = samples
            self.settings = None
        if type(target_size) not in [list, tuple]:
            target_size = (target_size, target_size)
        self.target_size = target_size
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
        img0 = stimuli_bank.sinusoid_grating(**sinusoid_param)
        sinusoid_param['amp'] = contrast1
        img1 = stimuli_bank.sinusoid_grating(**sinusoid_param)

        # multiply it by gaussian
        if self.mask_image == 'fixed_size':
            radius = (int(self.target_size[0] / 2.0), int(self.target_size[1] / 2.0))
            [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
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
            radius = (int(self.target_size[0] / 2.0), int(self.target_size[1] / 2.0))
            [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))

            sigma = self.target_size[0] / 6
            gauss_img = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

            gauss_img = gauss_img / np.max(gauss_img)
            img0 *= gauss_img
            img1 *= gauss_img

        img0 = (img0 + 1) / 2
        img1 = (img1 + 1) / 2
        # multiplying with the illuminant value
        img0 *= self.illuminant_range
        img1 *= self.illuminant_range

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
            elif self.contrast_space == 'lum':
                # this is really not necessary, but just for the sake of floating point
                img0[:, :, [1, 2]] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, [1, 2]] = 0.5
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
        img_out, contrast_target = _two_pairs_stimuli(img0, img1, contrast0, contrast1, self.p)

        item_settings = np.array([contrast0, lambda_wave, theta, rho, self.p])

        if self.grating_detector:
            return img_out[contrast_target], contrast_target, item_settings
        else:
            return img_out[0], img_out[1], contrast_target, item_settings

    def __len__(self):
        return self.samples
