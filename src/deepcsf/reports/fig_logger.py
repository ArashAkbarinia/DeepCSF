"""
Generating figures for large data stored in presentable format.
"""

import os

from matplotlib import pyplot as plt

from . import resnet_plot
from ..utils import system_utils


def log_grating_radius(data_dir, fig_dir, net_name, chn, measure='avg'):
    experiment_name = 'grating_radius_%s' % chn

    file_path = '%s/%s/%s.pickle' % (data_dir, net_name, experiment_name)
    data = system_utils.read_pickle(file_path)

    for area_name in data['con100'][0].keys():

        out_dir = '%s/%s/%s/%s/' % (fig_dir, net_name, experiment_name, measure)
        out_file = os.path.join(out_dir, '%s.png' % area_name)
        if os.path.exists(out_file):
            continue

        tmp_fig = resnet_plot.plot_area_activation(data, area_name, measure)

        os.makedirs(out_dir, exist_ok=True)
        tmp_fig.tight_layout()
        tmp_fig.savefig(out_file)
        plt.close('all')
