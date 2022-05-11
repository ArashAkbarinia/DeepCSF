"""
PyTorch contrast-discrimination training script for various datasets.
"""

import os
import numpy as np
import time
import sys
import ntpath

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from .datasets import dataloader
from .models import model_csf, model_utils
from .utils import report_utils, system_utils, argument_handler


def main(argv):
    args = argument_handler.train_arg_parser(argv)
    system_utils.set_random_environment(args.random_seed)

    # NOTE: a hack to handle taskonomy preprocessing
    if 'taskonomy' in args.architecture:
        args.colour_space = 'taskonomy_rgb'

    # it's a binary classification
    args.num_classes = 2

    # preparing the output folder
    args.output_dir = '%s/networks/%s/t%.3d/%s/%s/' % (
        args.output_dir, args.dataset, args.target_size, args.architecture, args.experiment_name
    )
    system_utils.create_dir(args.output_dir)

    # this is just a hack for when the training script has crashed
    filename = 'e%.3d_%s' % (8, 'checkpoint.pth.tar')
    file_path = os.path.join(args.output_dir, filename)
    if os.path.exists(file_path):
        return

    # dumping all passed arguments to a json file
    system_utils.save_arguments(args)

    _main_worker(args)


def _main_worker(args):
    mean, std = model_utils.get_mean_std(args.colour_space, args.vision_type)

    # create model
    if args.grating_detector:
        model = model_csf.GratingDetector(
            args.architecture, args.target_size, args.transfer_weights,
        )
    else:
        model = model_csf.ContrastDiscrimination(
            args.architecture, args.target_size, args.transfer_weights,
        )

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # if transfer_weights, only train the fc layer, otherwise all parameters
    if args.transfer_weights is None:
        params_to_optimize = [{'params': [p for p in model.parameters()]}]
    else:
        for p in model.features.parameters():
            p.requires_grad = False
        params_to_optimize = [{'params': [p for p in model.fc.parameters()]}]
    # optimiser
    optimizer = torch.optim.SGD(
        params_to_optimize, lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    args.tb_writers = {}
    for mode in ["train", "val"]:
        args.tb_writers[mode] = SummaryWriter(os.path.join(args.output_dir, mode))

    model_progress = []
    model_progress_path = os.path.join(args.output_dir, 'model_progress.csv')

    # optionally resume from a checkpoint
    best_acc1 = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            args.initial_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            best_acc1 = best_acc1.to(args.gpu)
            model = model.cuda(args.gpu)

            optimizer.load_state_dict(checkpoint['optimizer'])

            if os.path.exists(model_progress_path):
                model_progress = np.loadtxt(model_progress_path, delimiter=',')
                model_progress = model_progress.tolist()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_trans = []
    valid_trans = []
    both_trans = []

    # to have 100% deterministic behaviour train_params must be passed
    if args.train_params is not None:
        args.workers = 0
        shuffle = False
        args.illuminant_range = 1.0
    else:
        shuffle = True

    if args.sf_filter is not None and len(args.sf_filter) != 2:
        sys.exit('Length of the sf_filter must be two %s' % args.sf_filter)

    # loading the training set
    train_trans = [*both_trans, *train_trans]
    db_params = {
        'colour_space': args.colour_space,
        'vision_type': args.vision_type,
        'mask_image': args.mask_image,
        'contrasts': args.contrasts,
        'illuminant_range': args.illuminant_range,
        'train_params': args.train_params,
        'sf_filter': args.sf_filter,
        'contrast_space': args.contrast_space,
        'same_transforms': args.same_transforms,
        'grating_detector': args.grating_detector
    }
    if args.dataset in dataloader.NATURAL_DATASETS:
        path_or_sample = args.data_dir
    else:
        # this would be only for the grating dataset to generate
        path_or_sample = args.train_samples
    train_dataset = dataloader.train_set(
        args.dataset, args.target_size, preprocess=(mean, std),
        extra_transformation=train_trans, data_dir=path_or_sample, **db_params
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    # validation always is random
    db_params['train_params'] = None
    # loading validation set
    valid_trans = [*both_trans, *valid_trans]
    validation_dataset = dataloader.validation_set(
        args.dataset, args.target_size, preprocess=(mean, std),
        extra_transformation=valid_trans, data_dir=path_or_sample, **db_params
    )

    val_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        _adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_log = _train_val(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        validation_log = _train_val(val_loader, model, criterion, None, epoch, args)

        model_progress.append([*train_log, *validation_log[1:]])

        # remember best acc@1 and save checkpoint
        acc1 = validation_log[2]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save the checkpoints
        system_utils.save_checkpoint(
            {
                'epoch': epoch,
                'arch': args.architecture,
                'transfer_weights': args.transfer_weights,
                'preprocessing': {'mean': mean, 'std': std},
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'target_size': args.target_size,
            },
            is_best, args
        )
        header = 'epoch,t_time,t_loss,t_top1,v_time,v_loss,v_top1'
        np.savetxt(model_progress_path, np.array(model_progress), delimiter=',', header=header)


def _adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1 ** (epoch // (args.epochs / 3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _train_val(db_loader, model, criterion, optimizer, epoch, args):
    batch_time = report_utils.AverageMeter()
    data_time = report_utils.AverageMeter()
    losses = report_utils.AverageMeter()
    top1 = report_utils.AverageMeter()

    is_train = optimizer is not None

    if is_train:
        model.train()
        num_samples = args.train_samples
        tb_writer = args.tb_writers['train']
    else:
        model.eval()
        num_samples = args.val_samples
        tb_writer = args.tb_writers['val']

    end = time.time()
    with torch.set_grad_enabled(is_train):
        for i, data in enumerate(db_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.grating_detector:
                img0, target, _ = data
                img0 = img0.cuda(args.gpu, non_blocking=True)
                output = model(img0)
            else:
                img0, img1, target, img_path = data
                img0 = img0.cuda(args.gpu, non_blocking=True)
                img1 = img1.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(img0, img1)

                if i == 0:
                    for j in range(min(16, img0.shape[0])):
                        img_disp = torch.cat([img0[j], img1[j]], dim=2)
                        img_inv = report_utils.inv_normalise_tensor(img_disp, args.mean, args.std)
                        img_inv = img_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
                        tb_writer.add_image(
                            "{}_{}/{}".format(ntpath.basename(img_path), i, j), img_inv, epoch
                        )

            target = target.cuda(args.gpu, non_blocking=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = report_utils.accuracy(output, target)
            losses.update(loss.item(), img0.size(0))
            top1.update(acc1[0].cpu().numpy()[0], img0.size(0))

            if is_train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if i % args.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(db_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1
                    )
                )
            if num_samples is not None and i * len(img0) > num_samples:
                break
        if not is_train:
            # printing the accuracy of the epoch
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    # writing to tensorboard
    tb_writer.add_scalar("{}".format('loss'), losses.avg, epoch)
    tb_writer.add_scalar("{}".format('top1'), top1.avg, epoch)
    tb_writer.add_scalar("{}".format('time'), batch_time.avg, epoch)

    return [epoch, batch_time.avg, losses.avg, top1.avg]
