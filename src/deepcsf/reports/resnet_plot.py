"""
Plotting the output of CSF tests.
"""
import sys

import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy import stats

from . import animal_csfs


def _area_name_from_path(file_name):
    for area_ind in range(5):
        area_name = 'area%d' % area_ind
        if area_name in file_name:
            return area_name
    return None


def _load_network_results(path, chns=None, area_suf=None):
    if area_suf is None:
        area_suf = ''
    net_results = dict()
    for chns_dir in sorted(glob.glob(path + '/*/')):
        chn_name = chns_dir.split('/')[-2]
        if chns is not None and chn_name not in chns:
            continue
        chn_res = []
        for file_path in sorted(glob.glob('%s/*%s*.csv' % (chns_dir, area_suf))):
            file_name = file_path.split('/')[-1]
            # finding the appropriate name
            area_name = _area_name_from_path(file_name)
            chn_res.append([np.loadtxt(file_path, delimiter=','), area_name])
        # if results were read add it to dictionary
        if len(chn_res) > 0:
            net_results[chn_name] = chn_res
    return net_results


def _load_lesion_results(path, chns=None, area_suf=None):
    if area_suf is None:
        area_suf = ''
    lesion_results = []
    for kernel_dir in sorted(glob.glob(path + '/*/')):
        lesion_results.append(
            _load_network_results(kernel_dir, chns=chns, area_suf=area_suf)
        )
    return lesion_results


def _extract_sensitivity(result_mat):
    unique_waves = np.unique(result_mat[:, 1])
    csf_inds = []
    for wave in unique_waves:
        lowest_contrast = result_mat[result_mat[:, 1] == wave][-1]
        csf_inds.append(1 / lowest_contrast[0])
    return csf_inds


def wave2sf(wave, target_size):
    base_sf = ((target_size / 2) / np.pi)
    return [((1 / e) * base_sf) for e in wave]


def uniform_sfs(xvals, yvals, target_size):
    max_xval = (target_size / 2)
    base_sf = max_xval / np.pi
    new_xs = [base_sf / e for e in np.arange(1, max_xval + 0.5, 0.5)]
    new_ys = np.interp(new_xs, xvals, yvals)
    return new_xs, new_ys


def _extract_data(result_mat, target_size):
    keys = ['contrast', 'wave', 'angle', 'phase', 'side']
    unique_params = dict()
    unique_params['wave'] = np.unique(result_mat[:, 1])
    # networks see the entire image, this assuming similar to fovea of 2 deg
    # to convert it to one degree, divided by 2
    unique_params['sf'] = np.array(
        wave2sf(unique_params['wave'], target_size)
    ) / 2
    accuracies = dict()
    contrasts_waves = dict()

    var_keys = ['angle', 'phase', 'side']

    sensitivities = dict()
    sensitivities['all'] = _extract_sensitivity(result_mat)
    data_summary = {
        'unique_params': unique_params, 'accuracies': accuracies,
        'contrasts_waves': contrasts_waves, 'sensitivities': sensitivities
    }
    # interpolating to all points
    int_xvals, int_yvals = uniform_sfs(
        unique_params['wave'], sensitivities['all'], target_size
    )
    unique_params['sf_int'] = np.array(wave2sf(int_xvals, target_size)) / 2
    sensitivities['all_int'] = int_yvals

    return data_summary


def _extract_network_summary(net_results, target_size):
    net_summary = dict()
    for chn_name, chn_data in net_results.items():
        net_summary[chn_name] = []
        num_tests = len(chn_data)
        for i in range(num_tests):
            test_summary = _extract_data(chn_data[i][0], target_size)
            net_summary[chn_name].append([test_summary, chn_data[i][1]])
    return net_summary


def _extract_lesion_summary(lesion_results, target_size):
    lesion_summary = []
    for res in lesion_results:
        lesion_summary.append(_extract_network_summary(res, target_size))
    return lesion_summary


def _chn_plot_params(chn_name):
    label = chn_name
    kwargs = {}
    if chn_name in ['lum']:
        colour = 'gray'
        kwargs = {'color': colour, 'marker': 'x', 'linestyle': '-'}
    elif chn_name == 'rg':
        colour = 'green'
        label = 'rg   '
        kwargs = {
            'color': colour, 'marker': '1', 'linestyle': '-',
            'markerfacecolor': 'white', 'markeredgecolor': 'r'
        }
    elif chn_name == 'yb':
        colour = 'blue'
        label = 'yb   '
        kwargs = {
            'color': colour, 'marker': '2', 'linestyle': '-',
            'markerfacecolor': 'white', 'markeredgecolor': 'y'
        }
    return label, kwargs


def _plot_chn_csf(chn_summary, chn_name, figsize=(22, 4), log_axis=False,
                  normalise=True, model_info=None, old_fig=None,
                  chn_info=None, legend_dis=False, legend=True, legend_loc='auto',
                  font_size=16):
    if old_fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = old_fig

    num_tests = len(chn_summary)
    for i in range(num_tests):
        # getting the x and y values
        org_yvals = np.array(chn_summary[i][0]['sensitivities']['all'])
        org_freqs = np.array(chn_summary[i][0]['unique_params']['sf'])

        if old_fig:
            ax = fig.axes[i]
        else:
            ax = fig.add_subplot(1, num_tests, i + 1)
        ax.set_title(chn_summary[i][1], **{'size': font_size})

        if chn_info is None:
            label, chn_params = _chn_plot_params(chn_name)
        else:
            label, chn_params = chn_info

        if normalise:
            org_yvals /= org_yvals.max()

        # first plot the human CSF
        if model_info is not None:
            model_name, plot_model = model_info
            if plot_model:
                hcsf = np.array([animal_csfs.csf(f, model_name) for f in org_freqs])
                hcsf /= hcsf.max()
                hcsf *= np.max(org_yvals)
                ax.plot(org_freqs, hcsf, '--', color='black', label='human')

            # use interpolation for corelation
            int_freqs = np.array(chn_summary[i][0]['unique_params']['sf_int'])
            hcsf = np.array([animal_csfs.csf(f, model_name) for f in int_freqs])
            hcsf /= hcsf.max()

            int_yvals = np.array(chn_summary[i][0]['sensitivities']['all_int'])
            int_yvals /= int_yvals.max()
            p_corr, r_corr = stats.pearsonr(int_yvals, hcsf)
            if not legend:
                suffix_label = ''
            elif legend_dis:
                euc_dis = np.linalg.norm(hcsf - int_yvals)
                suffix_label = ' [r=%.2f | d=%.2f]' % (p_corr, euc_dis)
            else:
                suffix_label = ' [r=%.2f]' % p_corr
        else:
            suffix_label = ''
        chn_label = '%s%s' % (label, suffix_label)
        ax.plot(org_freqs, org_yvals, label=chn_label, **chn_params)

        ax.set_xlabel('Spatial Frequency (Cycle/Image)', **{'size': font_size})
        ax.set_ylabel('Sensitivity (1/Contrast)', **{'size': font_size})
        if log_axis:
            ax.set_xscale('log')
        if legend:
            ax.legend(loc=legend_loc)
    return fig


def _plot_lesion_csf(chn_summary, chn_name, lesion_summary,
                     figsize=None, log_axis=False, normalise=True,
                     model_info=None, old_fig=None, chn_info=None):
    num_kernels = len(lesion_summary)
    if figsize is None:
        fig_height = int((num_kernels / 64) * 22)
        figsize = (22, fig_height)
        rows = int((num_kernels / 64) * 8)
        cols = 8
    if old_fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = old_fig

    # the original network without lesion
    f_org_yvals = np.array(chn_summary[0][0]['sensitivities']['all'])
    f_org_freqs = np.array(chn_summary[0][0]['unique_params']['sf'])
    if normalise:
        f_org_yvals /= f_org_yvals.max()
    # computing the human CSF
    if model_info is not None:
        model_name, plot_model = model_info
        if plot_model:
            f_hcsf = np.array([animal_csfs.csf(f, model_name) for f in f_org_freqs])
            f_hcsf /= f_hcsf.max()
            f_hcsf *= np.max(f_org_yvals)

        # use interpolation for corelation
        int_freqs = np.array(chn_summary[0][0]['unique_params']['sf_int'])
        hcsf = np.array([animal_csfs.csf(f, model_name) for f in int_freqs])
        hcsf /= hcsf.max()

        int_yvals = np.array(chn_summary[0][0]['sensitivities']['all_int'])
        int_yvals /= int_yvals.max()
        p_corr, r_corr = stats.pearsonr(int_yvals, hcsf)
        euc_dis = np.linalg.norm(hcsf - int_yvals)
        f_suffix_label = ' [r=%.2f | d=%.2f]' % (p_corr, euc_dis)
        f_suffix_label = ' [r=%.2f]' % (p_corr)
    else:
        f_suffix_label = ''

    for k in range(num_kernels):
        # getting the x and y values
        org_yvals = np.array(lesion_summary[k][chn_name][0][0]['sensitivities']['all'])
        org_freqs = np.array(lesion_summary[k][chn_name][0][0]['unique_params']['sf'])

        if old_fig:
            ax = fig.axes[k]
        else:
            ax = fig.add_subplot(rows, cols, k + 1)
        ax.set_title('k%.4d' % k)

        if normalise:
            org_yvals /= org_yvals.max()

        # first plot the human CSF
        if model_info is not None:
            model_name, plot_model = model_info
            if plot_model:
                ax.plot(org_freqs, f_hcsf, '--', color='black', label='human')

            int_yvals = np.array(lesion_summary[k][chn_name][0][0]['sensitivities']['all_int'])
            int_yvals /= int_yvals.max()
            p_corr, r_corr = stats.pearsonr(int_yvals, hcsf)
            euc_dis = np.linalg.norm(hcsf - int_yvals)
            suffix_label = ' [r=%.2f | d=%.2f]' % (p_corr, euc_dis)
            suffix_label = ' [r=%.2f]' % (p_corr)
        else:
            suffix_label = ''
        chn_label = '%s%s' % ('full', f_suffix_label)
        _, chn_params = _chn_plot_params(chn_name)
        ax.plot(f_org_freqs, f_org_yvals, label=chn_label, **chn_params)
        chn_label = '%.3d%s' % (k, suffix_label)
        _, chn_params = chn_info
        ax.scatter(org_freqs, org_yvals, label=chn_label, **chn_params)

        ax.set_xlabel('Spatial Frequency (Cycle/Image)')
        ax.set_ylabel('Sensitivity (1/Contrast)')
        if log_axis:
            ax.set_xscale('log')
            loc = 'lower center'
        else:
            loc = 'upper right'
        ax.legend(loc=loc)
    return fig


def plot_area_activation(activations, area_name, which_measure='avg'):
    if which_measure == 'avg':
        m_ind = 0
    elif which_measure == 'med':
        m_ind = 1
    elif which_measure == 'max':
        m_ind = 2
    else:
        sys.exit('Measure %s is not supported!' % which_measure)

    contrasts = activations.keys()
    num_rads = len(activations['con100'])
    num_kernels = activations['con100'][0][area_name][m_ind].shape[0]

    fig_height = int((num_kernels / 64) * 22)
    figsize = (28, fig_height)
    fig = plt.figure(figsize=figsize)
    rows = int((num_kernels / 64) * 8)
    cols = 8

    for i in range(num_kernels):
        ax = fig.add_subplot(rows, cols, i + 1)

        xaxis = np.arange(1, num_rads + 1)
        for contrast in contrasts:
            yaxis = []
            for rad_ind in range(num_rads):
                yaxis.append(activations[contrast][rad_ind][area_name][m_ind][i])
            con_f = 1 - (float(contrast[3:]) / 100)
            color = [con_f, con_f, con_f]
            ax.plot(xaxis, yaxis, color=color)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_xticks([])
    return fig


def plot_csf_lesion(lesion_path, org_path, area_suf,
                    target_size, chns=None, **kwargs):
    lesion_res = _load_lesion_results(lesion_path, chns=chns, area_suf=area_suf)
    lesion_summary = _extract_lesion_summary(lesion_res, target_size)

    net_results = _load_network_results(org_path, chns=chns, area_suf=area_suf)
    net_summary = _extract_network_summary(net_results, target_size)

    net_csf_fig = None
    for chn_key, chn_val in net_summary.items():
        if net_csf_fig is not None:
            kwargs['old_fig'] = net_csf_fig
            kwargs['model_info'] = None
        net_csf_fig = _plot_lesion_csf(chn_val, chn_key, lesion_summary, **kwargs)
    return net_csf_fig


def plot_csf_areas(path, target_size, chns=None, **kwargs):
    net_results = _load_network_results(path, chns=chns)
    net_summary = _extract_network_summary(net_results, target_size)
    net_csf_fig = None
    for chn_key, chn_val in net_summary.items():
        if net_csf_fig is not None:
            kwargs['old_fig'] = net_csf_fig
            kwargs['model_info'] = None
        net_csf_fig = _plot_chn_csf(chn_val, chn_key, **kwargs)
    return net_csf_fig
