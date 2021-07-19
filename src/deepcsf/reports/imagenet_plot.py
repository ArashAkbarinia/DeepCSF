"""
Plotting different functionalities of ImageNet.
"""
import sys

import numpy as np
import glob

from matplotlib import pyplot as plt

from . import animal_csfs
from ..utils import report_utils


def _get_sf_ring_accuracies(data_dir, dataset):
    result_files = sorted(
        glob.glob(data_dir + '/*.csv'),
        key=report_utils.natural_keys
    )

    accs = []
    for file_path in result_files:
        if dataset == 'imagenet':
            tmp_res = np.loadtxt(file_path, delimiter=',')
            current_acc = tmp_res[:, 0].mean()
        elif dataset in ['voc_coco']:
            tmp_res = np.loadtxt(file_path, delimiter=';')
            current_acc = tmp_res[1] / 100
        else:
            sys.exit('sf_ring doesnt support dataset %s' % dataset)
        accs.append(current_acc)
    return accs


def imagenet_sf_ring_report(data_dir):
    result_files = sorted(
        glob.glob(data_dir + '/*.csv'),
        key=report_utils.natural_keys
    )

    accs = []
    for file_path in result_files:
        tmp_res = np.loadtxt(file_path, delimiter=',')
        current_acc = imagenet_result_summary(tmp_res)
        accs.append(current_acc)
    return accs


def imagenet_result_summary(predictions):
    summary_report = dict()
    summary_report['top1'] = predictions[:, 0].mean()
    summary_report['top5'] = predictions[:, 1].mean()

    category_inds = imagenet_category_inds()
    num_categories = category_inds.shape[0]
    cats_top1 = np.zeros(num_categories)
    cats_top5 = np.zeros(num_categories)
    for i in range(num_categories):
        si = int(category_inds[i, 0])
        ei = int(category_inds[i, 1])
        cats_top1[i] = predictions[si:ei, 0].mean()
        cats_top5[i] = predictions[si:ei, 1].mean()

    summary_report['cats_top1'] = cats_top1
    summary_report['cats_top5'] = cats_top5

    return summary_report


def imagenet_category_inds():
    num_samples = 50
    num_classes = 1000
    num_images = num_samples * num_classes
    class_inds = np.zeros((num_classes, 2))
    for j, i in enumerate(range(0, num_images, num_samples)):
        class_inds[j, :] = [i, i + num_samples]
    return class_inds


def plot_sf_ring_net(net_dir, dataset,
                     figsize=(8, 6), font_size=16, legend_loc='best',
                     log_axis=False, model_csf=None, normalise=False,
                     net_name=None, old_fig=None, plot_params=None):
    accs = _get_sf_ring_accuracies(net_dir, dataset)
    num_freqs = len(accs)

    xaxis = [e / (num_freqs / 60) for e in range(1, num_freqs + 1)]
    error_rate = 1 - np.array(accs)

    if normalise:
        error_rate /= error_rate.max()
        error_rate = (error_rate - np.min(error_rate)) / np.ptp(error_rate)

    if net_name is None:
        net_name = net_dir.split('/')[-2]

    if old_fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = old_fig
        ax = fig.axes[0]

    ax.set_title(net_name, **{'size': font_size})

    if model_csf == 'model_fest':
        model_fest_path = '/home/arash/Desktop/projects/deep_csf/model_fest/extracted_avg.csv'
        hcsf_data = np.loadtxt(model_fest_path, delimiter=',')
        xaxis = hcsf_data[0]
        hcsf = hcsf_data[1]
    elif model_csf is not None:
        org_freqs = [e / 2 for e in range(1, 120)]
        hcsf = np.array([animal_csfs.csf(f, model_csf) for f in org_freqs])
        hcsf /= hcsf.max()

    if model_csf is not None and old_fig is None:
        ax.plot(org_freqs, hcsf, '--', color='black', label='Human CSF')

    if plot_params is None:
        plot_params = {'color': 'green', 'marker': 'x', 'linestyle': '-'}
    if dataset == 'voc_coco':
        error_rate = np.interp(org_freqs, xaxis, error_rate)
        xaxis = org_freqs
    ax.plot(xaxis, error_rate, label=net_name, **plot_params)

    ax.set_xlabel('Spatial Frequency (Cycle/Image)', **{'size': font_size})
    ax.set_ylabel('Error Rate (%)', **{'size': font_size})

    if log_axis:
        ax.set_xscale('log')
        ax.set_yscale(
            'symlog',
            **{'linthreshy': 10e-2, 'linscaley': 0.25, 'subsy': [*range(2, 10)]}
        )
    ax.legend(loc=legend_loc)

    return fig


def plot_sf_ring_dir(data_dir, dataset, **kwargs):
    for net_dir in glob.glob(data_dir + '/*/'):
        plot_sf_ring_net(net_dir, dataset, **kwargs)
