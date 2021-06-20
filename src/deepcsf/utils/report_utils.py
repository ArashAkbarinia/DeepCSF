"""
Utility routines for logging and reporting.
"""

import re
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_preds(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        corrects = []
        for k in topk:
            corrects.append(correct[:k])
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res, corrects


def accuracy(output, target, topk=(1,)):
    res, _ = accuracy_preds(output, target, topk=topk)
    return res


def inv_normalise_tensor(tensor, mean, std):
    tensor = tensor.clone()
    if type(mean) not in [tuple, list]:
        mean = tuple([mean for _ in range(tensor.shape[1])])
    if type(std) not in [tuple, list]:
        std = tuple([std for _ in range(tensor.shape[1])])
    # inverting the normalisation for each channel
    for i in range(tensor.shape[1]):
        tensor[:, i, ] = (tensor[:, i, ] * std[i]) + mean[i]
    tensor = tensor.clamp(0, 1)
    return tensor


def atof(value):
    try:
        return float(value)
    except ValueError:
        return value


def atoi(value):
    try:
        return int(value)
    except ValueError:
        return value


def natural_keys(text, delimiter=None, remove=None):
    """
    alist.sort(key=natural_keys) sorts in human order
    adapted from http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    if remove is not None:
        text = text.replace(remove, '')
    if delimiter is None:
        return [atoi(c) for c in re.split(r'(\d+)', text)]
    else:
        return [atof(c) for c in text.split(delimiter)]
