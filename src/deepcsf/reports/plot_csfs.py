"""
Plotting the output of CSF tests.
"""

import numpy as np
import glob
import ntpath
from matplotlib import pyplot as plt
from scipy import stats

from . import animal_csfs
from ..utils import report_utils


def extract_csf(file_path):
    results = np.loadtxt(file_path, delimiter=',')
    frequency = np.unique(results[:, 1])
    psfs = {'contrast': [], 'accs': []}
    sensitivity = []
    for f in frequency:
        f_inds = results[:, 1] == f
        f_results = results[f_inds]
        # there is a range of contrasts with accuracy=0.75, we take the mean
        # psfs['accs'].append(np.interp(psfs['contrast'], f_results[:, 1], f_results[:, 2]))
        psfs['contrast'].append(f_results[:, 3])
        psfs['accs'].append(f_results[:, 2])
        l_inds = f_results[:, 2] == 0.75
        high_sens = f_results[-1][-1]
        low_sens = f_results[l_inds][0][-1] if np.sum(l_inds) > 0 else high_sens
        sensitivity.append((low_sens + high_sens) / 2)
    return np.array(frequency) / 2, 1 / np.array(sensitivity), psfs


def _load_network_results(path, chns=None, area_suf=None, exclude=None):
    if area_suf is None:
        area_suf = ''
    net_results = dict()
    for chns_dir in sorted(glob.glob(path + '/*/')):
        chn_name = chns_dir.split('/')[-2]
        if chns is not None and chn_name not in chns:
            continue
        chn_res = []
        file_paths = sorted(
            glob.glob('%s/*%s*.csv' % (chns_dir, area_suf)), key=report_utils.natural_keys
        )
        if exclude is not None:
            for exclude_i in exclude:
                file_paths = [e for e in file_paths if exclude_i not in e]
        # move fc to the last element
        if '%sfc.csv' % chns_dir in file_paths:
            file_paths.remove('%sfc.csv' % chns_dir)
            file_paths.append('%sfc.csv' % chns_dir)
        if '%sstem.csv' % chns_dir in file_paths:
            file_paths.remove('%sstem.csv' % chns_dir)
            file_paths = ['%sstem.csv' % chns_dir, *file_paths]
        for file_path in file_paths:
            area_name = ntpath.basename(file_path)[:-4]
            frequency, sensitivity, psfs = extract_csf(file_path)
            chn_res.append({'freq': frequency, 'sens': sensitivity, 'name': area_name, 'psf': psfs})
            # if results were read add it to dictionary
            if len(chn_res) > 0:
                net_results[chn_name] = chn_res
    return net_results


def _chn_plot_params(chn_name):
    label = chn_name
    kwargs = {}
    if chn_name in ['lum', 'lum_yog']:
        colour = 'black'
        label = 'lum'
        kwargs = {'color': colour, 'marker': 'o', 'linestyle': '-'}
    elif chn_name in ['rg', 'rg_yog']:
        colour = 'green'
        label = 'rg   '
        kwargs = {
            'color': colour, 'marker': '^', 'linestyle': '-',
            'markerfacecolor': 'r', 'markeredgecolor': 'r'
        }
    elif chn_name in ['yb', 'yb_yog']:
        colour = 'blue'
        label = 'yb   '
        kwargs = {
            'color': colour, 'marker': 's', 'linestyle': '-',
            'markerfacecolor': 'y', 'markeredgecolor': 'y'
        }
    kwargs['linewidth'] = 6
    kwargs['markersize'] = kwargs['linewidth'] * 3
    return label, kwargs


def _minmax_instance_area(instance, area_ind):
    min_val = min([np.array(val[area_ind]['sens']).min() for val in instance.values()])
    max_val = max([np.array(val[area_ind]['sens']).max() for val in instance.values()])
    return min_val, max_val


def _instances_summary(instances, std=True):
    net_results = dict()
    for chn, areas in instances[0].items():
        chn_res = []
        for i in range(len(areas)):
            area_results = []
            for instance in instances:
                # normalising by the max to compare across instances
                csf = instance[chn][i]
                _, max_val = _minmax_instance_area(instance, i)
                area_results.append(csf['sens'] / max_val)
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


def _report_chn_csf(net_results, chn_name, model_info):
    chn_summary = net_results[chn_name]

    correlations = []
    distances = []
    for i in range(len(chn_summary)):
        # getting the x and y values
        org_yvals = np.array(chn_summary[i]['sens'])
        org_freqs = np.array(chn_summary[i]['freq'])

        # first compute the human CSF
        model_name, _ = model_info
        hcsf_freq, hcsf_sens = animal_csfs.get_csf(org_freqs, method=model_name, chn=chn_name)
        hcsf_sens *= np.max(org_yvals)

        # use interpolation for correlation
        hcsf_sens = np.interp(org_freqs, hcsf_freq, hcsf_sens)
        p_corr, r_corr = stats.pearsonr(org_yvals, hcsf_sens)
        euc_dis = np.linalg.norm(hcsf_sens - org_yvals)
        correlations.append(p_corr)
        distances.append(euc_dis)

    return correlations, distances


def _plot_chn_csf(net_results, chn_name, figwidth=7, log_axis=False, normalise='max',
                  model_info=None, old_fig=None, chn_info=None, legend_dis=False, legend=True,
                  legend_loc='upper right', font_size=16):
    chn_summary = net_results[chn_name]
    num_tests = len(chn_summary)
    fig = plt.figure(figsize=(figwidth * num_tests, figwidth * 0.8)) if old_fig is None else old_fig

    for i in range(num_tests):
        # getting the x and y values
        org_yvals = np.array(chn_summary[i]['sens'])
        org_freqs = np.array(chn_summary[i]['freq'])
        org_error = chn_summary[i]['std'] if 'std' in chn_summary[i] else None

        ax = fig.axes[i] if old_fig else fig.add_subplot(1, num_tests, i + 1)
        ax.set_title(chn_summary[i]['name'], **{'size': font_size})

        label, chn_params = _chn_plot_params(chn_name) if chn_info is None else chn_info

        if normalise is not None:
            min_val, max_val = _minmax_instance_area(net_results, i)
            if normalise == 'max':
                org_yvals /= max_val
            elif normalise == 'min_max':
                org_yvals = report_utils.min_max_normalise(org_yvals, 0, 1, min_val, max_val)

        # first plot the human CSF
        if model_info is not None:
            model_name, plot_model = model_info
            if plot_model:
                if 'lum' in chn_name:
                    ax.plot(
                        [], [], '--x', color='gray', label="Human",
                        linewidth=chn_params['linewidth'], markersize=chn_params['markersize']
                    )

                hcf_chn_params = chn_params.copy()
                hcf_chn_params['marker'] = 'x'
                hcf_chn_params['linestyle'] = '--'
                hcf_freq, hcf_sens = animal_csfs.get_csf(org_freqs, method=model_name, chn=chn_name)
                # plotting only if normalised
                # hcsf_sens *= np.max(org_yvals)
                ax.plot(hcf_freq, hcf_sens, **hcf_chn_params)
                # hcsf_inter = np.interp(org_freqs, hcsf_freq, hcsf_sens)
                # ax.plot(org_freqs, hcsf_inter, '--', color='black')

            # use interpolation for correlation
            hcf_sens = np.interp(org_freqs, hcf_freq, hcf_sens)
            p_corr, r_corr = stats.pearsonr(org_yvals, hcf_sens)
            if not legend:
                suffix_label = ''
            elif legend_dis:
                euc_dis = np.linalg.norm(hcf_sens - org_yvals)
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
            ax.errorbar(org_freqs, org_yvals, org_error, label=chn_label, capsize=6, **chn_params)

        if i == 0 and 'lum' in chn_name:
            ax.set_xlabel('Spatial Frequency (Cycle/Image)', **{'size': font_size * 0.9})
            ax.set_ylabel('Sensitivity (1/Contrast)', **{'size': font_size * 0.9})
        if log_axis:
            ax.set_xscale('log')
            ax.set_yscale(
                'symlog', **{'linthresh': 10e-2, 'linscale': 0.25, 'subs': [*range(2, 10)]}
            )
        if normalise is not None:
            ax.set_ylim([0, 1])
        if legend:
            ax.legend(loc=legend_loc, prop={'size': font_size * 0.55},
                      frameon=True, ncol=2, labelspacing=0.0, columnspacing=0.1,
                      bbox_to_anchor=(-0.03, 0.55, 0.5, 0.5)
                      )
        ax.tick_params(axis='both', which='major', labelsize=font_size * 0.65)
    return fig


def plot_csf_areas(path, chns=None, exclude=None, **kwargs):
    net_results = _load_network_results(path, chns=chns, exclude=exclude)
    net_csf_fig = None
    for chn_key in net_results.keys():
        if net_csf_fig is not None:
            kwargs['old_fig'] = net_csf_fig
            # kwargs['model_info'] = None
        net_csf_fig = _plot_chn_csf(net_results, chn_key, **kwargs)
    return net_csf_fig


def plot_csf_instances(paths, std, area_suf=None, chns=None, exclude=None, **kwargs):
    instances = [_load_network_results(path, chns, area_suf, exclude) for path in paths]
    net_results = _instances_summary(instances, std)
    net_csf_fig = None
    for chn_key in net_results.keys():
        if net_csf_fig is not None:
            kwargs['old_fig'] = net_csf_fig
            # kwargs['model_info'] = None
        net_csf_fig = _plot_chn_csf(net_results, chn_key, **kwargs)
    return net_csf_fig


def report_csf_areas(paths, area_suf=None, chns=None, **kwargs):
    net_results = _load_network_results(paths, chns=chns, area_suf=area_suf)
    report_summary = dict()
    for chn_key in net_results.keys():
        p_corr, euc_dis = _report_chn_csf(net_results, chn_key, **kwargs)
        report_summary[chn_key] = {'corr': p_corr, 'dist': euc_dis}
    return net_results, report_summary
