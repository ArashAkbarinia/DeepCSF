"""
Plotting the output of CSF tests.
"""

import numpy as np
import glob
import ntpath
from matplotlib import pyplot as plt
from scipy import stats

from . import animal_csfs


def extract_csf(file_path):
    results = np.loadtxt(file_path, delimiter=',')
    frequency = np.unique(results[:, 1])
    sensitivity = [results[results[:, 1] == f][-1][-1] for f in frequency]
    return np.array(frequency) / 2, 1 / np.array(sensitivity)


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
            area_name = ntpath.basename(file_path)[:-4]
            frequency, sensitivity = extract_csf(file_path)
            chn_res.append({'freq': frequency, 'sens': sensitivity, 'name': area_name})
            # if results were read add it to dictionary
            if len(chn_res) > 0:
                net_results[chn_name] = chn_res
    return net_results


def _chn_plot_params(chn_name):
    label = chn_name
    kwargs = {}
    if chn_name in ['lum']:
        colour = 'gray'
        kwargs = {'color': colour, 'marker': 'o', 'linestyle': '-'}
    elif chn_name == 'rg':
        colour = 'green'
        label = 'rg   '
        kwargs = {
            'color': colour, 'marker': '^', 'linestyle': '-',
            'markerfacecolor': 'r', 'markeredgecolor': 'r'
        }
    elif chn_name == 'yb':
        colour = 'blue'
        label = 'yb   '
        kwargs = {
            'color': colour, 'marker': 's', 'linestyle': '-',
            'markerfacecolor': 'y', 'markeredgecolor': 'y'
        }
    return label, kwargs


def _instances_summary(instances, std=True):
    net_results = dict()
    for chn, areas in instances[0].items():
        chn_res = []
        for i in range(len(areas)):
            area_results = []
            for instance in instances:
                # normalising by the max to compare across instances
                csf = instance[chn][i]
                area_results.append(csf['sens'] / np.max(csf['sens']))
            areas_summary = {
                'freq': instance[chn][i]['freq'],
                'sens': np.mean(area_results, axis=0),
                'name': instance[chn][i]['name'],
            }
            if std:
                areas_summary['std'] = np.std(area_results, axis=0)
            chn_res.append(areas_summary)
        net_results[chn] = chn_res
    return net_results


def _plot_chn_csf(net_results, chn_name, figwidth=7, log_axis=False, normalise=True,
                  model_info=None, old_fig=None, chn_info=None, legend_dis=False, legend=True,
                  legend_loc='lower center', font_size=16):
    chn_summary = net_results[chn_name]
    num_tests = len(chn_summary)
    fig = plt.figure(figsize=(figwidth * num_tests, 5)) if old_fig is None else old_fig

    for i in range(num_tests):
        # getting the x and y values
        org_yvals = np.array(chn_summary[i]['sens'])
        org_freqs = np.array(chn_summary[i]['freq'])
        org_error = chn_summary[i]['std'] if 'std' in chn_summary[i] else None

        ax = fig.axes[i] if old_fig else fig.add_subplot(1, num_tests, i + 1)
        ax.set_title(chn_summary[i]['name'], **{'size': font_size})

        label, chn_params = _chn_plot_params(chn_name) if chn_info is None else chn_info

        if normalise:
            max_val = max([np.array(val[i]['sens']).max() for val in net_results.values()])
            org_yvals /= max_val

        # first plot the human CSF
        if model_info is not None:
            model_name, plot_model = model_info
            if plot_model:
                hcsf_freq, hcsf_sens = animal_csfs.get_csf(org_freqs, method=model_name)
                hcsf_sens /= hcsf_sens.max()
                hcsf_sens *= np.max(org_yvals)
                ax.plot(hcsf_freq, hcsf_sens, '-x', color='black', label='human')
                hcsf_inter = np.interp(org_freqs, hcsf_freq, hcsf_sens)
                ax.plot(org_freqs, hcsf_inter, '--', color='black')

            # use interpolation for correlation
            hcsf_sens = np.interp(org_freqs, hcsf_freq, hcsf_sens)
            # int_freqs = np.array(chn_summary[i][0]['unique_params']['sf_int'])
            # hcsf_sens = np.array([animal_csfs.csf(f, model_name) for f in int_freqs])
            # hcsf_sens /= hcsf_sens.max()
            #
            # int_yvals = np.array(chn_summary[i][0]['sensitivities']['all_int'])
            # int_yvals /= int_yvals.max()
            p_corr, r_corr = stats.pearsonr(org_yvals, hcsf_sens)
            if not legend:
                suffix_label = ''
            elif legend_dis:
                euc_dis = np.linalg.norm(hcsf_sens - org_yvals)
                suffix_label = ' [r=%.2f | d=%.2f]' % (p_corr, euc_dis)
            else:
                suffix_label = ' [r=%.2f]' % p_corr
        else:
            suffix_label = ''
        chn_label = '%s%s' % (label, suffix_label)
        # if standard error exists, plot it
        if org_error is None:
            ax.plot(org_freqs, org_yvals, label=chn_label, **chn_params)
        else:
            ax.errorbar(org_freqs, org_yvals, org_error, label=chn_label, **chn_params)

        ax.set_xlabel('Spatial Frequency (Cycle/Image)', **{'size': font_size})
        ax.set_ylabel('Sensitivity (1/Contrast)', **{'size': font_size})
        if log_axis:
            ax.set_xscale('log')
            ax.set_yscale(
                'symlog', **{'linthresh': 10e-2, 'linscale': 0.25, 'subs': [*range(2, 10)]}
            )
        if normalise:
            ax.set_ylim([0, 1])
        if legend:
            ax.legend(loc=legend_loc)
    return fig


def plot_csf_areas(path, chns=None, **kwargs):
    net_results = _load_network_results(path, chns=chns)
    net_csf_fig = None
    for chn_key in net_results.keys():
        if net_csf_fig is not None:
            kwargs['old_fig'] = net_csf_fig
            kwargs['model_info'] = None
        net_csf_fig = _plot_chn_csf(net_results, chn_key, **kwargs)
    return net_csf_fig


def plot_csf_instances(paths, std, area_suf=None, chns=None, **kwargs):
    instances = [_load_network_results(path, chns=chns, area_suf=area_suf) for path in paths]
    net_results = _instances_summary(instances, std)
    net_csf_fig = None
    for chn_key in net_results.keys():
        if net_csf_fig is not None:
            kwargs['old_fig'] = net_csf_fig
            kwargs['model_info'] = None
        net_csf_fig = _plot_chn_csf(net_results, chn_key, **kwargs)
    return net_csf_fig
