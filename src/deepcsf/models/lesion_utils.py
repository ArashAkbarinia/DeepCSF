"""
Routines to lesion and ablation.
"""

import sys


def lesion_lines(model, layer, kernel, kill_lines):
    for l_item in kill_lines:
        # pattern <P1>_<L1>_<P2>_<L2>
        current_line = l_item.split('_')
        if len(current_line) != 4:
            sys.exit(
                'The order of lines to be killed should follow '
                '<P1>_<L1>_<P2>_<L2>. Invalid item %d' %
                l_item
            )
        else:
            print(
                'Removing %d %s' % (kernel, l_item)
            )
            ax0 = int(current_line[0])
            ax1 = int(current_line[2])
            ln0 = int(current_line[1])
            ln1 = int(current_line[3])
            if ax0 == 0 and ax1 == 1:
                model.state_dict()[layer][kernel, ln0, ln1, :] = 0
            elif ax0 == 0 and ax1 == 2:
                model.state_dict()[layer][kernel, ln0, :, ln1] = 0
            elif ax0 == 1 and ax1 == 2:
                model.state_dict()[layer][kernel, :, ln0, ln1] = 0
    return model


def lesion_planes(model, layer, kernel, kill_planes):
    axis_num = None
    for p_item in kill_planes:
        if p_item.isdigit():
            plane_index = int(p_item)
            if axis_num is None:
                sys.exit(
                    'The order of planes to be killed should follow '
                    'ax_<NUMBER> and plane indices. Invalid axis %d' %
                    axis_num
                )
            else:
                print(
                    'Removing axis %d plane %d' % (axis_num, plane_index)
                )
                if axis_num == 0:
                    model.state_dict()[layer][kernel, plane_index, :, :] = 0
                elif axis_num == 1:
                    model.state_dict()[layer][kernel, :, plane_index, ] = 0
                elif axis_num == 2:
                    model.state_dict()[layer][kernel, :, :, plane_index, ] = 0
        else:
            # pattern ax_<NUMBER>
            axis_num = int(p_item.split('_')[-1])
    return model


def lesion_kernels(model, kernels=None, planes=None, lines=None):
    if kernels is not None:
        layer_name = ''
        for k_item in kernels:
            if k_item.isdigit():
                kernel_index = int(k_item)
                if layer_name == '':
                    sys.exit(
                        'The order of kernels to be killed should follow '
                        'layer name and kernel indices. Invalid layer name %s' %
                        layer_name
                    )
                else:
                    print(
                        'Removing layer %s kernel %d' %
                        (layer_name, kernel_index)
                    )
                    # check whether planes or lines are specified
                    # TODO: move this to TXT file to allow better combinations
                    if planes is not None:
                        model = lesion_planes(
                            model, layer_name, kernel_index, planes
                        )
                    elif lines is not None:
                        model = lesion_lines(
                            model, layer_name, kernel_index, lines
                        )
                    else:
                        model.state_dict()[layer_name][kernel_index] = 0
            else:
                layer_name = k_item
    return model
