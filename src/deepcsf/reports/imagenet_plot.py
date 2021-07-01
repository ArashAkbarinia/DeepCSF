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


def plot_sf_ring_net(net_dir, dataset,
                     figsize=(8, 4), font_size=16, legend_loc='best',
                     log_axis=False, model_csf=None, normalise=False):
    accs = _get_sf_ring_accuracies(net_dir, dataset)
    num_freqs = len(accs)

    xaxis = [e / (num_freqs / 60) for e in range(1, num_freqs + 1)]
    error_rate = 1 - np.array(accs)

    if normalise:
        error_rate /= error_rate.max()
        error_rate = (error_rate - np.min(error_rate)) / np.ptp(error_rate)

    net_name = net_dir.split('/')[-2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(net_name, **{'size': font_size})

    plot_params = {'color': 'gray', 'marker': 'x', 'linestyle': '-'}
    ax.plot(xaxis, error_rate, label=net_name, **plot_params)

    if model_csf == 'model_fest':
        model_fest_path = '/home/arash/Desktop/projects/deep_csf/model_fest/extracted_avg.csv'
        hcsf_data = np.loadtxt(model_fest_path, delimiter=',')
        xaxis = hcsf_data[0]
        hcsf = hcsf_data[1]
    elif model_csf is not None:
        hcsf = np.array([animal_csfs.csf(f, model_csf) for f in xaxis])
        hcsf /= hcsf.max()

    if model_csf is not None:
        ax.plot(xaxis, hcsf, '--', color='black', label='Human CSF')

    ax.set_xlabel('Spatial Frequency (Cycle/Image)', **{'size': font_size})
    ax.set_ylabel('Error Rate (%)', **{'size': font_size})

    if log_axis:
        ax.set_xscale('log')
    ax.legend(loc=legend_loc)

    return fig


def plot_sf_ring_dir(data_dir, dataset, **kwargs):
    for net_dir in glob.glob(data_dir + '/*/'):
        plot_sf_ring_net(net_dir, dataset, **kwargs)
