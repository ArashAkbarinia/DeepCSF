"""
A set of system utility functions.
"""

import os
import numpy as np
import random
import json
import shutil
import torch
import glob
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp'
]


def set_random_environment(random_seed):
    if random_seed is not None:
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def save_arguments(args):
    json_file_name = os.path.join(args.output_dir, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    if args.save_all:
        filename = 'e%.3d_%s' % (state['epoch'], filename)
    file_path = os.path.join(args.output_dir, filename)
    torch.save(state, file_path)
    if is_best:
        model_best_path = os.path.join(args.output_dir, 'model_best.pth.tar')
        shutil.copyfile(file_path, model_best_path)


def _read_extension(root, extension):
    img_paths = []
    img_paths.extend(
        sorted(glob.glob(root + '/*' + extension))
    )
    # with upper case
    img_paths.extend(
        sorted(glob.glob(root + '/*' + extension.upper()))
    )
    return img_paths


def image_in_folder(root, extensions=None):
    if extensions is None:
        extensions = IMG_EXTENSIONS

    img_paths = []
    # reading all extensions
    for extension in extensions:
        img_paths.extend(_read_extension(root, extension))

    return img_paths


def read_pickle(in_file):
    pickle_in = open(in_file, 'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data


def write_pickle(out_file, data):
    pickle_out = open(out_file, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()
