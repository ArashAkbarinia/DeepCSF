"""
Supported arguments for train and evaluation.
"""

import argparse


def train_arg_parser(argvs, extra_args_fun=None):
    parser = _common_arg_parser(description='Contrast discrimination training')

    _add_optimisation_group(parser)

    misc_group = parser.add_argument_group('miscellaneous')
    misc_group.add_argument(
        '--random_seed',
        default=None,
        type=int,
        help='To make the results more reproducible (default: None)'
    )
    misc_group.add_argument(
        '--train_params',
        default=None,
        type=str,
        help='Path to a predefined set of parameters (default: None)'
    )
    misc_group.add_argument(
        '--illuminant_range',
        default=None,
        nargs='+',
        type=float,
        help='Images are multiplied to a value in this range (default: None)'
    )

    if extra_args_fun is not None:
        extra_args_fun(parser)

    args = parser.parse_args(argvs)
    return args


def test_arg_parser(argvs, extra_args_fun=None):
    parser = _common_arg_parser(description='Contrast discrimination testing')

    _add_optimisation_group(parser)

    misc_group = parser.add_argument_group('csf')
    misc_group.add_argument(
        '--freqs',
        default=None,
        type=float,
        help='The frequencies to be tested (default: None)'
    )
    misc_group.add_argument(
        '--contrast_space',
        default=None,
        type=str,
        help='The channel where contrast is manipulated (default: None)'
    )

    if extra_args_fun is not None:
        extra_args_fun(parser)

    args = parser.parse_args(argvs)
    return args


def _common_arg_parser(description='No description!'):
    parser = ArgumentParser(description=description)

    _add_dataset_group(parser)
    _add_network_group(parser)
    _add_logging_group(parser)
    _add_routine_group(parser)
    _add_input_group(parser)

    return parser


def _add_logging_group(parser):
    logging_group = parser.add_argument_group('logging')

    logging_group.add_argument(
        '--output_dir',
        type=str,
        default='../outputs/',
        help='The path to the output directory (default: ../outputs/)'
    )
    logging_group.add_argument(
        '--experiment_name',
        type=str,
        default='anonymous',
        help='The name of the experiment (default: anonymous)'
    )
    logging_group.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='Frequency of reporting (default: 100)'
    )
    logging_group.add_argument(
        '--save_all',
        action='store_true',
        default=False,
        help='Saving all check points (default: False)'
    )
    logging_group.add_argument(
        '--visualise',
        action='store_true',
        default=False,
        help='Visualising the input images to network (default: False)'
    )


def _add_routine_group(parser):
    routine_group = parser.add_argument_group('routine')

    routine_group.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='Which GPU to use (default: 0)'
    )
    routine_group.add_argument(
        '-j', '--workers',
        default=1,
        type=int,
        help='Number of workers for image generator (default: 1)'
    )
    routine_group.add_argument(
        '-b', '--batch_size',
        default=16,
        type=int,
        help='Batch size (default: 16)'
    )


def _add_network_group(parser):
    network_group = parser.add_argument_group('optimisation')

    network_group.add_argument(
        '-aname', '--architecture',
        required=True,
        type=str,
        help='Name of the architecture or network'
    )
    network_group.add_argument(
        '--resume',
        default=None,
        type=str,
        help='Path to the latest checkpoint (default: None)'
    )
    network_group.add_argument(
        '--transfer_weights',
        default=None,
        nargs='+',
        type=str,
        help='Whether transferring weights from a model (default: None)'
    )


def _add_optimisation_group(parser):
    optimisation_group = parser.add_argument_group('optimisation')

    optimisation_group.add_argument(
        '-lr', '--learning_rate',
        default=0.1,
        type=float,
        help='The learning rate parameter (default: 0.1)'
    )
    optimisation_group.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='The momentum for optimisation (default 0.9)'
    )
    optimisation_group.add_argument(
        '-wd', '--weight_decay',
        default=1e-4,
        type=float,
        help='The decay weight parameter (default: 1e-4)'
    )
    optimisation_group.add_argument(
        '-e', '--epochs',
        default=90,
        type=int,
        help='Number of epochs (default: 90)'
    )
    optimisation_group.add_argument(
        '--initial_epoch',
        default=0,
        type=int,
        help='The initial epoch number (default: 0)'
    )


def _add_input_group(parser):
    input_group = parser.add_argument_group('input')

    input_group.add_argument(
        '--colour_space',
        default='rgb',
        type=str,
        choices=[
            'rgb', 'imagenet_rgb',
            'lab',
            'grey', 'grey3'
        ],
        help='The colour space of network (default: rgb)'
    )
    input_group.add_argument(
        '--vision_type',
        default='trichromat',
        type=str,
        choices=[
            'trichromat',
            'monochromat',
            'dichromat_rg',
            'dichromat_yb'
        ],
        help='The vision type of the network (default: trichromat)'
    )
    input_group.add_argument(
        '--target_size',
        required=True,
        type=int,
        help='Target size'
    )


def _add_dataset_group(parser):
    dataset_group = parser.add_argument_group('dataset')

    dataset_group.add_argument(
        '-dname', '--dataset',
        type=str,
        help='Name of the dataset'
    )
    dataset_group.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='The path to the data directory (default: None)'
    )
    dataset_group.add_argument(
        '--train_dir',
        type=str,
        default=None,
        help='The path to the train directory (default: None)'
    )
    dataset_group.add_argument(
        '--validation_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    dataset_group.add_argument(
        '--train_samples',
        type=int,
        default=None,
        help='Number of training samples (default: All)'
    )
    dataset_group.add_argument(
        '--val_samples',
        type=int,
        default=None,
        help='Number of validation samples (default: All)'
    )


class ArgumentParser(argparse.ArgumentParser):
    """
    Overriding the add_argument_group function. If a group already exists, it
    returns it, otherwise creates a new group and returns it.
    """

    def add_argument_group(self, *args, **kwargs):
        ignore = ['positional arguments', 'optional arguments']
        if (
                args[0] in ignore or
                ('title' in kwargs.keys() and kwargs['title'] in ignore)
        ):
            return super().add_argument_group(*args, **kwargs)
        for group in self._action_groups:
            if (
                    group.title == args[0] or
                    ('title' in kwargs and group.title == kwargs['title'])
            ):
                return group
        return super().add_argument_group(*args, **kwargs)
