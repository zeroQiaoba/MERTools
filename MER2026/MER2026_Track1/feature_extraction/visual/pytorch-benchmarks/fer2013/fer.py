# -*- coding: utf-8 -*-
"""Fer2013 benchmark

The module evaluates the performance of a pytorch model on the FER2013
benchmark.
"""

from __future__ import division

import os
import time

import torch
import numpy as np
import torch.utils.data
import torch.backends.cudnn as cudnn
from fer2013.fer_loader import Fer2013Dataset, Fer2013PlusDataset
from utils.benchmark_helpers import compose_transforms

def fer2013_benchmark(model, data_dir, res_cache, refresh_cache,
                       batch_size=256, num_workers=2, fer_plus=False):
    if not refresh_cache: # load result from cache, if available
        if os.path.isfile(res_cache):
            res = torch.load(res_cache)
            prec1_val, prec1_test = res['prec1_val'], res['prec1_test']
            print("=> loaded results from '{}'".format(res_cache))
            info = (prec1_val, prec1_test, res['speed'])
            msg = 'val acc: {:.2f}, test acc: {:.2f}, Speed: {:.1f}Hz'
            print(msg.format(*info))
            return

    meta = model.meta
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()
    preproc_transforms = compose_transforms(meta, center_crop=False)
    if fer_plus:
        dataset = Fer2013PlusDataset
    else:
        dataset = Fer2013Dataset
    speeds = []
    res = {}
    for mode in 'val', 'test':
        loader = torch.utils.data.DataLoader(
            dataset(data_dir, mode=mode, transform=preproc_transforms),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        prec1, speed = validate(loader, model, mode)
        res['prec1_{}'.format(mode)] = prec1
        speeds.append(speed)
    res['speed'] = np.mean(speed)
    torch.save(res, res_cache)

def validate(val_loader, model, mode):
    model.eval()
    top1 = AverageMeter()
    speed = WarmupAverageMeter()
    end = time.time()
    with torch.no_grad():
        for ii, (ims, target) in enumerate(val_loader):
            # target = target.cuda(async=True)
            target = target.cuda()
            output = model(ims) # compute output
            prec1, = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], ims.size(0))
            speed.update(time.time() - end, ims.size(0))
            end = time.time()
            if ii % 10 == 0:
                msg = ('{0}: [{1}/{2}]\tSpeed {speed.current:.1f}Hz\t'
                       '({speed.avg:.1f})Hz\tPrec@1 {top1.avg:.3f}')
                print(msg.format(mode, ii, len(val_loader),
                      speed=speed, top1=top1))
    print(' * Accuracy {0:.3f}'.format(top1.avg))
    return top1.avg, speed.avg

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.squeeze(-1).squeeze(-1)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
