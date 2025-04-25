# -*- coding: utf-8 -*-
"""This module evaluates imported PyTorch models on fer2013
"""

import os
import argparse
from os.path import join as pjoin
from fer2013.fer import fer2013_benchmark
from utils.benchmark_helpers import load_module_2or3

# MODEL_DIR = os.path.expanduser('~/data/models/pytorch/mcn_imports')
# FER_DIR = os.path.expanduser('~/data/datasets/fer2013+')
MODEL_DIR = './pretrained/'
FER_DIR = os.path.expanduser('~/Affective Computing/Dataset/FERPlus')

CACHE_DIR = 'res_cache/fer2013+'

def load_model(model_name):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = pjoin('model', model_name + '.py')
    weights_path = pjoin(MODEL_DIR, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net

def run_benchmarks(gpus, refresh, fer_plus):
    """Run bencmarks for imported models

    Args:
        gpus (str): comma separated gpu device identifiers
        refresh (bool): whether to overwrite the results of existing runs
        fer_plus (bool): whether to evaluate on the ferplus benchmark,
          rather than the standard fer benchmark.
    """

    # Select models (and their batch sizes) to include in the benchmark.
    if fer_plus:
        model_list = [
            ('resnet50_ferplus_dag', 32),
            ('senet50_ferplus_dag', 32),
        ]
    else:
        model_list = [
            ('alexnet_face_fer_bn_dag', 32),
            ('vgg_m_face_bn_fer_dag', 32),
            ('vgg_vd_face_fer_dag', 32),
        ]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

    opts = {'data_dir': FER_DIR, 'refresh_cache': refresh}

    for model_name, batch_size in model_list:
        cache_name = model_name
        if fer_plus:
            cache_name = cache_name + 'fer_plus'
        opts['res_cache'] = '{}/{}.pth'.format(CACHE_DIR, cache_name)
        opts['fer_plus'] = fer_plus
        model = load_model(model_name)
        print('benchmarking {}'.format(model_name))
        fer2013_benchmark(model, batch_size=batch_size, **opts)

parser = argparse.ArgumentParser(description='Run PyTorch benchmarks.')
parser.add_argument('--gpus', nargs='?', dest='gpus',
                    help='select gpu device id')
parser.add_argument('--refresh', dest='refresh', action='store_true',
                    help='refresh results cache')
parser.add_argument('--ferplus', dest='ferplus', action='store_true',
                    help='run ferplus (rather than fer) benchmarks')
parser.set_defaults(gpus=None)
parser.set_defaults(refresh=False)
parsed = parser.parse_args()

if __name__ == '__main__':
    run_benchmarks(parsed.gpus, parsed.refresh, parsed.ferplus)
