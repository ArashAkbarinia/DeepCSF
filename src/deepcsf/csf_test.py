"""
Evaluating a network with contrast-discrimination to generate its CSF.
"""

import numpy as np
import os

import torch

from skimage import io

from .datasets import dataloader
from .models import model_csf, model_utils, lesion_utils
from .utils import report_utils, system_utils, argument_handler


def run_gratings_separate(db_loader, model, out_file, print_freq=0,
                          preprocess=None, old_results=None):
    with torch.no_grad():
        header = 'Contrast,SpatialFrequency,Theta,Rho,Side,Prediction'
        new_results = []
        num_batches = db_loader.__len__()
        for i, (timg0, timg1, targets, item_settings) in enumerate(db_loader):
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


def main(argv):
    args = argument_handler.test_arg_parser(argv)
    colour_space = args.colour_space
    target_size = args.target_size

    args.output_dir = '%s/tests/t%.3d/' % (args.output_dir, args.target_size)
    system_utils.create_dir(args.output_dir)
    out_file = '%s/%s' % (args.output_dir, args.experiment_name)
    if os.path.exists(out_file + '.csv'):
        return

    preprocess = model_utils.get_mean_std(args.colour_space, args.vision_type)
    visualise_preprocess = preprocess if args.visualise else None

    # testing setting
    freqs = args.freqs
    if freqs is None:
        if target_size == 256:
            t4s = [
                target_size / 2, target_size / 2.5, target_size / 3,
                target_size / 3.5, target_size / 3.75, target_size / 4,
            ]
        else:
            # assuming 128
            t4s = [64]

        sf_base = ((target_size / 2) / np.pi)
        test_sfs = [
            sf_base / e for e in
            [*np.arange(1, 21), *np.arange(21, 61, 5),
             *np.arange(61, t4s[-1], 25), *t4s]
        ]
    else:
        if len(freqs) == 3:
            test_sfs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
        else:
            test_sfs = freqs
    # so the sfs gets sorted
    test_sfs = np.unique(test_sfs)
    test_thetas = np.linspace(0, np.pi, 7)
    test_rhos = np.linspace(0, np.pi, 4)
    test_ps = [0.0, 1.0]

    # creating the model, args.architecture should be a path
    model = model_csf.ContrastDiscrimination(args.architecture, target_size)
    model = lesion_utils.lesion_kernels(
        model, args.lesion_kernels, args.lesion_planes, args.lesion_lines
    )
    model.eval()
    model.cuda()

    max_high = 1.0
    mid_contrast = (0 + max_high) / 2

    all_results = None
    csf_flags = [mid_contrast for _ in test_sfs]

    for i in range(len(csf_flags)):
        low = 0
        high = max_high
        j = 0
        while csf_flags[i] is not None:
            print('%.2d %.3d Doing %f - %f %f %f' % (i, j, test_sfs[i], csf_flags[i], low, high))

            test_samples = {
                'amp': [csf_flags[i]], 'lambda_wave': [test_sfs[i]],
                'theta': test_thetas, 'rho': test_rhos, 'side': test_ps
            }
            db_params = {
                'colour_space': colour_space,
                'vision_type': args.vision_type,
                'mask_image': args.mask_image
            }

            db = dataloader.validation_set(
                'gratings', target_size, preprocess,
                data_dir=test_samples, **db_params
            )
            db.contrast_space = args.contrast_space

            db_loader = torch.utils.data.DataLoader(
                db, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
            )

            new_results, all_results = run_gratings_separate(
                db_loader, model, out_file, args.print_freq,
                preprocess=visualise_preprocess, old_results=all_results
            )
            new_contrast, low, high = sensitivity_sf(
                new_results, test_sfs[i], th=0.75, low=low, high=high
            )
            if abs(csf_flags[i] - max_high) < 1e-3 or new_contrast == csf_flags[i] or j == 20:
                print('had to skip', csf_flags[i])
                csf_flags[i] = None
            else:
                csf_flags[i] = new_contrast
            j += 1
