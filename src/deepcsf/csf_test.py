"""
Evaluating a network with contrast-discrimination to generate its CSF.
"""

import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .datasets import dataloader
from .models import model_csf, model_utils, lesion_utils
from .utils import system_utils, argument_handler
from .train_contrast_discrimination import _train_val


def sensitivity_sf(result_mat, sf, th=0.75, low=0, high=1):
    result_mat = result_mat[result_mat[:, 1] == sf, :]
    unique_contrast = np.unique(result_mat[:, 0])
    accs = []
    for contrast in unique_contrast:
        accs.append(result_mat[result_mat[:, 0] == contrast, -1].mean())

    max_ind = 0
    diff_acc = accs[max_ind] - th
    contrast_i = unique_contrast[max_ind]
    if abs(diff_acc) < 0.005:
        return None, 0, 1
    elif diff_acc > 0:
        return (low + contrast_i) / 2, low, contrast_i
    else:
        return (high + contrast_i) / 2, contrast_i, high


def _compute_mean(a, b):
    return (a + b) / 2


def _midpoint_sf(accuracy, low, mid, high, th, ep=1e-4):
    diff_acc = accuracy - th
    if abs(diff_acc) < ep:
        return None, None, None
    elif diff_acc > 0:
        new_mid = _compute_mean(low, mid)
        return low, new_mid, mid
    else:
        new_mid = _compute_mean(high, mid)
        return mid, new_mid, high


def main(argv):
    args = argument_handler.test_arg_parser(argv)
    args.batch_size = 16
    args.workers = 2

    colour_space = args.colour_space
    target_size = args.target_size

    # which illuminant to test
    illuminant = 0 if args.illuminant is None else args.illuminant[0]
    ill_suffix = '' if illuminant == 0 else '_%d' % int(illuminant * 100)

    res_out_dir = os.path.join(args.output_dir, 'evals%s' % ill_suffix)

    system_utils.create_dir(res_out_dir)
    out_file = '%s/%s_evolution.csv' % (res_out_dir, args.experiment_name)
    if os.path.exists(out_file):
        return

    tb_path = os.path.join(args.output_dir, 'test_%s%s' % (args.experiment_name, ill_suffix))
    args.tb_writers = {'test': SummaryWriter(tb_path)}

    preprocess = model_utils.get_mean_std(args.colour_space, args.vision_type)
    args.mean, args.std = preprocess

    # testing setting
    freqs = args.freqs
    # frequencies should be devisable to image row/column
    if freqs is None:
        sf_base = ((target_size / 2) / np.pi)
        readable_sfs = [i for i in range(1, int(target_size / 2) + 1) if target_size % i == 0]
        lambda_waves = [sf_base / e for e in readable_sfs]
    else:
        lambda_waves = freqs
    test_thetas = [0, 45, 90, 135]
    test_rhos = [0, 180]

    # creating the model, args.architecture should be a path
    if args.grating_detector:
        net_t = model_csf.load_grating_detector
        test_ps = [0.0]
    else:
        net_t = model_csf.load_contrast_discrimination
        test_ps = [0.0, 1.0]
    model = net_t(args.architecture, target_size, args.classifier)

    model = lesion_utils.lesion_kernels(
        model, args.lesion_kernels, args.lesion_planes, args.lesion_lines
    )
    model.eval()
    model.cuda()

    max_high = 1.0
    min_low = 0.0
    mid_start = (min_low + max_high) / 2

    csf_flags = [mid_start for _ in lambda_waves]

    db_params = {
        'colour_space': colour_space,
        'vision_type': args.vision_type,
        'mask_image': args.mask_image,
        'grating_detector': args.grating_detector
    }
    criterion = nn.CrossEntropyLoss().cuda()

    header = 'LambdaWave,SF,ACC,Contrast'
    all_results = []
    tb_writer = args.tb_writers['test']
    for i in range(len(csf_flags)):
        low = min_low
        high = max_high
        mid = mid_start
        j = 0
        psf_i = {'acc': [], 'contrast': []}
        while csf_flags[i] is not None:
            test_samples = {
                'amp': [csf_flags[i]], 'lambda_wave': [lambda_waves[i]], 'theta': test_thetas,
                'rho': test_rhos, 'side': test_ps, 'illuminant': illuminant
            }

            db = dataloader.validation_set(
                'gratings', target_size, preprocess, data_dir=test_samples, **db_params
            )
            db.contrast_space = args.contrast_space

            db_loader = torch.utils.data.DataLoader(
                db, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )

            epoch_out = _train_val(db_loader, model, criterion, None, -1 - j, args)
            accuracy = epoch_out[3] / 100
            contrast = int(csf_flags[i] * 1000)
            psf_i['acc'].append(accuracy)
            psf_i['contrast'].append(contrast)
            print(lambda_waves[i], csf_flags[i], accuracy, low, high)
            all_results.append(np.array([lambda_waves[i], readable_sfs[i], accuracy, mid]))
            # th=0.749 because test samples are 16, 12 correct equals 0.75 and test stops
            new_low, new_mid, new_high = _midpoint_sf(accuracy, low, mid, high, th=0.749)
            if abs(csf_flags[i] - max_high) < 1e-3 or new_mid == csf_flags[i] or j == 20:
                print('had to skip', csf_flags[i])
                csf_flags[i] = None
            else:
                low, mid, high = new_low, new_mid, new_high
                csf_flags[i] = new_mid

                min_diff = csf_flags[i] - 0
                max_diff = 1 - csf_flags[i]
                if illuminant < -min_diff or illuminant > max_diff:
                    print('Ill %.3f not possible for contrast %.3f' % (illuminant, csf_flags[i]))
                    csf_flags[i] = None
            j += 1
        np.savetxt(out_file, np.array(all_results), delimiter=',', fmt='%f', header=header)
        tb_writer.add_scalar("{}".format('csf'), 1 / all_results[-1][-1], readable_sfs[i])

        # making the psf
        psf_i['acc'] = np.array(psf_i['acc'])
        psf_i['contrast'] = np.array(psf_i['contrast'])
        for c in np.argsort(psf_i['contrast']):
            tb_writer.add_scalar(
                "{}_{:03d}".format('psf', readable_sfs[i]), psf_i['acc'][c], psf_i['contrast'][c]
            )
    tb_writer.close()
