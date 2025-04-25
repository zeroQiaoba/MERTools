# -*- coding: utf-8 -*-
"""Imagenet validation set benchmark

The module evaluates the performance of a pytorch model on the ILSVRC 2012
validation set.

Based on PyTorch imagenet example:
    https://github.com/pytorch/examples/tree/master/imagenet
"""

from __future__ import division

import os
import time

from PIL import ImageFile
import torch
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from utils.benchmark_helpers import compose_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def imagenet_benchmark(model, data_dir, res_cache, refresh_cache, batch_size=256,
                       num_workers=20, remove_blacklist=False, center_crop=True,
                       override_meta_imsize=False):
    if not refresh_cache:  # load result from cache, if available
        if os.path.isfile(res_cache):
            res = torch.load(res_cache)
            prec1, prec5, speed = res['prec1'], res['prec5'], res['speed']
            print("=> loaded results from '{}'".format(res_cache))
            info = (100 - prec1, 100 - prec5, speed)
            msg = 'Top 1 err: {:.2f}, Top 5 err: {:.2f}, Speed: {:.1f}Hz'
            print(msg.format(*info))
            return

    meta = model.meta
    cudnn.benchmark = True

    if override_meta_imsize:  # NOTE REMOVE THIS LATER!
        import torch.nn as nn
        model.features_8 = nn.AdaptiveAvgPool2d(1)

    model = torch.nn.DataParallel(model).cuda()
    if remove_blacklist:
        subset = 'val_blacklisted'
    else:
        subset = 'val'
    valdir = os.path.join(data_dir, subset)
    preproc_transforms = compose_transforms(meta, resize=256, center_crop=center_crop,
                                            override_meta_imsize=override_meta_imsize)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, preproc_transforms), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True)
    prec1, prec5, speed = validate(val_loader, model)
    torch.save({'prec1': prec1, 'prec5': prec5, 'speed': speed}, res_cache)


def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    speed = WarmupAverageMeter()
    end = time.time()
    with torch.no_grad():
        for ii, (ims, target) in enumerate(val_loader):
            target = target.cuda()
            # ims_var = torch.autograd.Variable(ims, volatile=True)
            output = model(ims)  # compute output
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], ims.size(0))
            top5.update(prec5[0], ims.size(0))
            speed.update(time.time() - end, ims.size(0))
            end = time.time()
            if ii % 10 == 0:
                msg = ('Test: [{0}/{1}]\tSpeed {speed.current:.1f}Hz\t'
                       '({speed.avg:.1f})Hz\tPrec@1 {top1.avg:.3f} '
                       '{top5.avg:.3f}')
                print(msg.format(ii, len(val_loader), speed=speed, top1=top1,
                                 top5=top5))
    top1_err, top5_err = 100 - top1.avg, 100 - top5.avg
    print(' * Err@1 {0:.3f} Err@5 {1:.3f}'.format(top1_err, top5_err))

    return top1.avg, top5.avg, speed.avg


class WarmupAverageMeter(object):
    """Computes and stores the average and current value, after a fixed
    warmup period (useful for approximate benchmarking)

    Args:
        warmup (int) [3]: The number of updates to be ignored before the
        average starts to be computed.
    """

    def __init__(self, warmup=3):
        self.reset()
        self.warmup = warmup

    def reset(self):
        self.avg = 0
        self.current = 0
        self.delta_sum = 0
        self.count = 0
        self.warmup_count = 0

    def update(self, delta, n):
        self.warmup_count = self.warmup_count + 1
        if self.warmup_count >= self.warmup:
            self.current = n / delta
            self.delta_sum += delta
            self.count += n
            self.avg = self.count / self.delta_sum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
