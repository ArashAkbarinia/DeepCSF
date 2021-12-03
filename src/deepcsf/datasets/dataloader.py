"""
Data loader for contrast discrimination routine.
"""

import os
import sys

import torchvision.transforms as torch_transforms

from . import dataset_utils
from . import cv2_transforms

NATURAL_DATASETS = ['imagenet', 'celeba', 'land']


def train_set(db, target_size, preprocess, extra_transformation=None, **kwargs):
    mean, std = preprocess
    if extra_transformation is None:
        extra_transformation = []
    if kwargs['train_params'] is None:
        shared_pre_transforms = [
            *extra_transformation,
            cv2_transforms.RandomHorizontalFlip(),
        ]
    else:
        shared_pre_transforms = [*extra_transformation]
    shared_post_transforms = _get_shared_post_transforms(mean, std)
    if db in NATURAL_DATASETS:
        # if train params are passed don't use any random processes
        if kwargs['train_params'] is None:
            scale = (0.08, 1.0)
            size_transform = cv2_transforms.RandomResizedCrop(target_size, scale=scale)
            pre_transforms = [size_transform, *shared_pre_transforms]
        else:
            pre_transforms = [
                cv2_transforms.Resize(target_size),
                cv2_transforms.CenterCrop(target_size),
                *shared_pre_transforms
            ]
        post_transforms = [*shared_post_transforms]
        return _natural_dataset(db, 'train', pre_transforms, post_transforms, **kwargs)
    elif db in ['gratings']:
        return _get_grating_dataset(
            shared_pre_transforms, shared_post_transforms, target_size, **kwargs
        )
    return None


def validation_set(db, target_size, preprocess, extra_transformation=None, **kwargs):
    mean, std = preprocess
    if extra_transformation is None:
        extra_transformation = []
    shared_pre_transforms = [*extra_transformation]
    shared_post_transforms = _get_shared_post_transforms(mean, std)
    if db in NATURAL_DATASETS:
        pre_transforms = [
            cv2_transforms.Resize(target_size),
            cv2_transforms.CenterCrop(target_size),
            *shared_pre_transforms
        ]
        post_transforms = [*shared_post_transforms]
        return _natural_dataset(db, 'validation', pre_transforms, post_transforms, **kwargs)
    elif db in ['gratings']:
        return _get_grating_dataset(
            shared_pre_transforms, shared_post_transforms, target_size, **kwargs
        )
    return None


def _natural_dataset(db, which_set, pre_transforms, post_transforms, data_dir, **kwargs):
    torch_pre_transforms = torch_transforms.Compose(pre_transforms)
    torch_post_transforms = torch_transforms.Compose(post_transforms)
    afc_kwargs = {
        'pre_transform': torch_pre_transforms,
        'post_transform': torch_post_transforms,
        **kwargs
    }
    if db == 'imagenet':
        natural_kwargs = {'root': os.path.join(data_dir, which_set)}
        current_db = dataset_utils.ImageFolder(afc_kwargs, natural_kwargs)
    elif db == 'land':
        natural_kwargs = {'root': os.path.join(data_dir, 'Images')}
        current_db = dataset_utils.ImageFolder(afc_kwargs, natural_kwargs)
    elif db == 'celeba':
        split = 'test' if which_set == 'validation' else 'train'
        natural_kwargs = {'root': data_dir, 'split': split}
        current_db = dataset_utils.CelebA(afc_kwargs, natural_kwargs)
    else:
        sys.exit('Dataset %s is not supported!' % db)
    return current_db


def _get_grating_dataset(pre_transforms, post_transforms, target_size, data_dir, **kwargs):
    torch_pre_transforms = torch_transforms.Compose(pre_transforms)
    torch_post_transforms = torch_transforms.Compose(post_transforms)
    afc_kwargs = {
        'pre_transform': torch_pre_transforms,
        'post_transform': torch_post_transforms,
        **kwargs
    }
    return dataset_utils.GratingImages(
        samples=data_dir, afc_kwargs=afc_kwargs, target_size=target_size
    )


def _get_shared_post_transforms(mean, std):
    return [
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ]
