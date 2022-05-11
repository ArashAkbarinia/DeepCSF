"""
Evaluating a network with contrast-discrimination to generate its CSF.
"""

import numpy as np
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from skimage import io

from .datasets import dataloader
from .models import model_csf, model_utils, lesion_utils
from .utils import report_utils, system_utils, argument_handler
from .train_contrast_discrimination import _train_val


def run_gratings_separate(db_loader, model, out_file, print_freq=0, preprocess=None,
                          old_results=None, grating_detector=False):
    with torch.no_grad():
        header = 'Contrast,SpatialFrequency,Theta,Rho,Side,Prediction'
        new_results = []
        num_batches = db_loader.__len__()
        for i, data in enumerate(db_loader):
            if grating_detector:
                timg0, targets, item_settings = data
                timg0 = timg0.cuda()
                out = model(timg0)
            else:
                timg0, timg1, targets, item_settings = data
                timg0 = timg0.cuda()
                timg1 = timg1.cuda()
                out = model(timg0, timg1)
            preds = out.cpu().numpy().argmax(axis=1)
            targets = targets.numpy()
            item_settings = item_settings.numpy()

            if preprocess is not None:
                mean, std = preprocess
                timgs = torch.cat([timg0, timg1], dim=2)
                img_inv = report_utils.inv_normalise_tensor(timgs, mean, std)
                img_inv = img_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
                img_inv = np.concatenate(img_inv, axis=1)
                save_path = '%s%.5d.png' % (out_file, i)
                img_inv = np.uint8((img_inv.squeeze() * 255))
                io.imsave(save_path, img_inv)

            for j in range(len(preds)):
                current_settings = item_settings[j]
                params = [*current_settings, preds[j] == targets[j]]
                new_results.append(params)
            num_tests = num_batches * timg0.shape[0]
            test_num = i * timg0.shape[0]
            percent = float(test_num) / float(num_tests)
            if print_freq > 0 and (i % print_freq) == 0:
                print('%.2f [%d/%d]' % (percent, test_num, num_tests))

    save_path = out_file + '.csv'
    if old_results is not None:
        all_results = [*old_results, *new_results]
    else:
        all_results = new_results
    all_results = np.array(all_results)
    np.savetxt(save_path, all_results, delimiter=',', header=header)
    return np.array(new_results), all_results


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
    c = (a + b) / 2
    return c


def _midpoint_sf(accuracy, low, mid, high, th=0.75):
    diff_acc = accuracy - th
    if abs(diff_acc) < 0.005:
        return None, None, None
    elif diff_acc > 0:
        new_mid = _compute_mean(low, mid)
        return low, new_mid, mid
    else:
        new_mid = _compute_mean(high, mid)
        return mid, new_mid, high


def main(argv):
    args = argument_handler.test_arg_parser(argv)
    # NOTE: a hack to handle taskonomy preprocessing
    if 'taskonomy' in args.architecture:
        args.colour_space = 'taskonomy_rgb'

    colour_space = args.colour_space
    target_size = args.target_size

    system_utils.create_dir(args.output_dir)
    out_file = '%s/%s' % (args.output_dir, args.experiment_name)
    if os.path.exists(out_file + '.csv'):
        return

    args.tb_writer = SummaryWriter(os.path.join(args.output_dir, 'test'))

    preprocess = model_utils.get_mean_std(args.colour_space, args.vision_type)
    args.mean, args.std = preprocess
    visualise_preprocess = preprocess if args.visualise else None

    # testing setting
    freqs = args.freqs
    if freqs is None:
        sf_base = ((target_size / 2) / np.pi)
        readable_sfs = [i for i in range(1, int(target_size / 2) + 1) if target_size % i == 0]
        lambda_waves = [sf_base / e for e in readable_sfs]
    else:
        if len(freqs) == 3:
            lambda_waves = np.linspace(freqs[0], freqs[1], int(freqs[2]))
        else:
            lambda_waves = freqs
    test_thetas = np.linspace(0, np.pi, 7)
    test_rhos = np.linspace(0, np.pi, 4)

    # creating the model, args.architecture should be a path
    if args.grating_detector:
        model = model_csf.GratingDetector(args.architecture, target_size)
        test_ps = [0.0]
    else:
        model = model_csf.ContrastDiscrimination(args.architecture, target_size)
        test_ps = [0.0, 1.0]
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

    out_file = out_file + '_evolution.csv'
    header = 'LambdaWave,SF,ACC,Contrast'
    all_results = []
    for i in range(len(csf_flags)):
        low = min_low
        high = max_high
        mid = mid_start
        j = 0
        while csf_flags[i] is not None:
            args.tb_writer.add_scalar("{}".format('low'), low, readable_sfs[i])
            args.tb_writer.add_scalar("{}".format('mid'), high, readable_sfs[i])
            args.tb_writer.add_scalar("{}".format('high'), mid, readable_sfs[i])
            test_samples = {
                'amp': [csf_flags[i]], 'lambda_wave': [lambda_waves[i]],
                'theta': test_thetas, 'rho': test_rhos, 'side': test_ps
            }

            db = dataloader.validation_set(
                'gratings', target_size, preprocess, data_dir=test_samples, **db_params
            )
            db.contrast_space = args.contrast_space

            db_loader = torch.utils.data.DataLoader(
                db, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
            )

            epoch_out = _train_val(db_loader, model, criterion, None, -1, args)
            accuracy = epoch_out[3] / 100
            print(lambda_waves[i], csf_flags[i], accuracy, low, high)
            all_results.append(np.array([lambda_waves[i], readable_sfs[i], accuracy, mid]))
            new_low, new_mid, new_high = _midpoint_sf(accuracy, low, mid, high, th=0.75)
            if abs(csf_flags[i] - max_high) < 1e-3 or new_mid == csf_flags[i] or j == 20:
                print('had to skip', csf_flags[i])
                csf_flags[i] = None
            else:
                low, mid, high = new_low, new_mid, new_high
                csf_flags[i] = new_mid
            j += 1
        np.savetxt(out_file, np.array(all_results), delimiter=',', fmt='%f', header=header)
