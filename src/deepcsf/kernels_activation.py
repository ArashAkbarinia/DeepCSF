"""
Evaluating a network with contrast-discrimination to generate its CSF.
"""

import numpy as np
import os
import sys

import torch

from skimage import io

import torchvision.transforms as torch_transforms

from .datasets import stimuli_bank, cv2_transforms, colour_spaces
from .models import pretrained_models, model_utils, lesion_utils
from .utils import report_utils, system_utils, argument_handler


def _get_activation(name, acts_dict):
    def hook(model, input_x, output_y):
        acts_dict[name] = output_y.detach()

    return hook


def _create_resnet_hooks(model):
    act_dict = dict()
    rfhs = dict()
    for attr_name in ['maxpool', 'contrast_pool']:
        if hasattr(model, attr_name):
            area0 = getattr(model, attr_name)
            rfhs['area0'] = area0.register_forward_hook(
                _get_activation('area0', act_dict)
            )
    for i in range(1, 5):
        attr_name = 'layer%d' % i
        act_name = 'area%d' % i
        area_i = getattr(model, attr_name)
        rfhs[act_name] = area_i.register_forward_hook(
            _get_activation(act_name, act_dict)
        )
        for j in range(len(area_i)):
            for k in range(1, 4):
                attr_name = 'bn%d' % k
                if hasattr(area_i[j], attr_name):
                    act_name = 'area%d.%d_%d' % (i, j, k)
                    area_ijk = getattr(area_i[j], attr_name)
                    rfhs[act_name] = area_ijk.register_forward_hook(
                        _get_activation(act_name, act_dict)
                    )
    return act_dict, rfhs


def _prepapre_colour_space(img, colour_space, contrast_space):
    if colour_space != 'grey':
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        if contrast_space == 'yb':
            img[:, :, [0, 1]] = 0.5
            img = colour_spaces.dkl012rgb01(img)
        elif contrast_space == 'rg':
            img[:, :, [0, 2]] = 0.5
            img = colour_spaces.dkl012rgb01(img)
        elif contrast_space == 'red':
            img[:, :, [1, 2]] = 0.5
        elif contrast_space == 'green':
            img[:, :, [0, 2]] = 0.5
        elif contrast_space == 'blue':
            img[:, :, [0, 1]] = 0.5
        elif contrast_space != 'rgb':
            sys.exit('Contrast %s not supported' % contrast_space)
    return img


def run_gratings_radius(model, out_file, args):
    act_dict, rfhs = _create_resnet_hooks(model)

    max_rad = round(round(args.target_size / 2))

    mean, std = args.preprocess
    transform = torch_transforms.Compose([
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

    all_activations = dict()
    with torch.no_grad():
        for i, (contrast) in enumerate(args.contrasts):
            acts_rads = []
            for grating_radius in range(1, max_rad, args.print_freq):
                img = stimuli_bank.circular_gratings(contrast, grating_radius)
                img = (img + 1) / 2
                img = _prepapre_colour_space(
                    img, args.colour_space, args.contrast_space
                )

                # making it pytorch friendly
                img = transform(img)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()

                _ = model(img)
                tmp_acts = dict()
                for key, val in act_dict.items():
                    current_acts = val.clone().cpu().numpy().squeeze()
                    if args.save_all:
                        tmp_acts[key] = current_acts
                    else:
                        tmp_acts[key] = [
                            np.mean(current_acts, axis=(1, 2)),
                            np.median(current_acts, axis=(1, 2)),
                            np.max(current_acts, axis=(1, 2)),
                        ]
                acts_rads.append(tmp_acts)

                if args.visualise:
                    img_inv = report_utils.inv_normalise_tensor(img, mean, std)
                    img_inv = img_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    img_inv = np.concatenate(img_inv, axis=1)
                    save_path = '%s%.5d.png' % (out_file, i)
                    img_inv = np.uint8((img_inv.squeeze() * 255))
                    io.imsave(save_path, img_inv)

                print('Contrast %.2f [%d/%d]' % (contrast, grating_radius, max_rad))
            all_activations['con%.3d' % (contrast * 100)] = acts_rads

    save_path = out_file + '.pickle'
    system_utils.write_pickle(save_path, all_activations)
    return all_activations


def main(argv):
    args = argument_handler.activation_arg_parser(argv)

    args.output_dir = '%s/activations/t%.3d/' % (args.output_dir, args.target_size)
    system_utils.create_dir(args.output_dir)
    out_file = '%s/%s' % (args.output_dir, args.experiment_name)
    if os.path.exists(out_file + '.pickle'):
        return

    args.preprocess = model_utils.get_mean_std(args.colour_space, args.vision_type)

    # creating the model, args.architecture should be a path
    model = pretrained_models.get_pretrained_model(
        args.architecture, args.transfer_weights
    )
    model = lesion_utils.lesion_kernels(
        model, args.lesion_kernels, args.lesion_planes, args.lesion_lines
    )
    model.eval()
    model.cuda()

    # TODO: support different types of experiments
    if args.stimuli == 'grating_radius':
        _ = run_gratings_radius(model, out_file, args)
