# -*- coding: utf-8 -*-
"""This script evaluates imported PyTorch models on the
ImageNet validation set

e.g.
python run_imagenet_benchmarks.py --model_subset pt_tpu --gpus 2
"""

import os
import argparse
from torchvision.models import densenet
from imagenet.evaluation import imagenet_benchmark
from pathlib import Path
from utils.benchmark_helpers import load_module_2or3

# directory containing imported pytorch models
MODEL_DIR = os.path.expanduser('~/data/models/pytorch/mcn_imports/')

# imagenet directory
ILSVRC_DIR = os.path.expanduser('~/data/shared-datasets/ILSVRC2012-pytorch-val')

# results cache directory
CACHE_DIR = 'res_cache/imagenet'


def load_torchvision_model(model_name):
    if 'densenet' in model_name:
        func = getattr(densenet, model_name)
        net = func(pretrained=True)
        net.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [224, 224]}
    return net


def load_tpu_converted_model(model_name, model_def_path, weights_path, **kwargs):
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    mean, std = getattr(mod, "MEAN_RGB"), getattr(mod, "STDDEV_RGB")
    net = func(pretrained=weights_path)
    net.meta = {
        'mean': mean,
        'std': std,
        'imageSize': [224, 224]}
    return net


def load_model(model_name):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    if 'tv_' in model_name:
        import ipdb
        ipdb.set_trace()
        net = load_torchvision_model(model_name)
    else:
        model_def_path = os.path.join(MODEL_DIR, model_name + ".py")
        weights_path = os.path.join(MODEL_DIR, model_name + ".pth")
        mod = load_module_2or3(model_name, model_def_path)
        func = getattr(mod, model_name)
        net = func(weights_path=weights_path)
    return net


def run_benchmarks(gpus, refresh, remove_blacklist, workers, no_center_crop,
                   override_meta_imsize, model_subset, tpu_model_dir, tpu_weights_dir):
    """Run bencmarks for imported models

    Args:
        gpus (str): comma separated gpu device identifiers
        refresh (bool): whether to overwrite the results of existing runs
        remove_blacklist (bool): whether to remove images from the 2014 ILSVRC
          blacklist from the validation images used in the benchmark
        workers (int): the number of workers
    """
    model_loader = load_model

    # Select models (and their batch sizes) to include in the benchmark.
    if model_subset == "all":
        raise NotImplementedError("TODO: update to use config dicts")
        model_list = [
            ('alexnet_pt_mcn', 256),
            ('squeezenet1_0_pt_mcn', 128),
            ('squeezenet1_1_pt_mcn', 128),
            ('vgg11_pt_mcn', 128),
            ('vgg13_pt_mcn', 92),
            ('vgg16_pt_mcn', 32),
            ('vgg19_pt_mcn', 24),
            ('resnet18_pt_mcn', 50),
            ('resnet34_pt_mcn', 50),
            ('resnet50_pt_mcn', 32),
            ('resnet101_pt_mcn', 24),
            ('resnet152_pt_mcn', 20),
            ('inception_v3_pt_mcn', 64),
            ("densenet121_pt_mcn", 50),
            ("densenet161_pt_mcn", 32),
            ("densenet169_pt_mcn", 32),
            ("densenet201_pt_mcn", 32),
            ('imagenet_matconvnet_alex', 256),
            ('imagenet_matconvnet_vgg_f_dag', 128),
            ('imagenet_matconvnet_vgg_m_dag', 128),
            ('imagenet_matconvnet_vgg_verydeep_16_dag', 32),
        ]
    elif model_subset == "pt_mcn":
        model_list = [('resnet50_pt_mcn', 32)]
    elif model_subset == "pt_tpu":
        model_loader = load_tpu_converted_model
        model_list = [{
            "model_name": 'resnet50',
            "cache_name": 'resnet50_from_tpu',
            "batch_size": 32,
            "model_def_path": Path(tpu_model_dir) / "resnet_models.py",
            "weights_path": Path(tpu_weights_dir) / "resnet50_ported.pth"},
        ]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

    opts = {
        'data_dir': ILSVRC_DIR,
        'refresh_cache': refresh,
        'remove_blacklist': remove_blacklist,
        'num_workers': workers,
        'center_crop': not no_center_crop,
        'override_meta_imsize': override_meta_imsize}

    for model_config in model_list:
        model_name = model_config["model_name"]
        cache_name = '{}/{}'.format(CACHE_DIR, model_config["cache_name"])
        if no_center_crop:
            cache_name = '{}-no-center-crop'.format(cache_name)
        opts['res_cache'] = '{}.pth'.format(cache_name)
        model = model_loader(**model_config)
        print('benchmarking {}'.format(model_name))
        imagenet_benchmark(model, batch_size=model_config["batch_size"], **opts)


parser = argparse.ArgumentParser(description='Run PyTorch benchmarks.')
parser.add_argument('--gpus', nargs='?', dest='gpus', help='select gpu device id')
parser.add_argument('--workers', type=int, default=4, dest='workers',
                    help='select number of workers')
parser.add_argument('--model_subset', type=str, default="all", help='eval subset')
parser.add_argument('--refresh', dest='refresh', action='store_true',
                    help='refresh results cache')
parser.add_argument('--no_center_crop', dest='no_center_crop', action='store_true',
                    help='prevent center cropping')
parser.add_argument('--override_meta_imsize', dest='override_meta_imsize',
                    action='store_true', help='allow arbitrary resizing of input image')
parser.add_argument(
    '--remove-blacklist', dest='remove_blacklist', action='store_true',
    help=('evaluate on 2012 validation subset without including'
          'the 2014 list of blacklisted images (only applies to'
          'imagenet models)'))
parser.add_argument(
    "--tpu_model_dir",
    default=Path.home() / "coding/libs/tf/tpu-fork/models/official/resnet/tf2pytorch",
)
parser.add_argument(
    "--tpu_weights_dir",
    default=Path.home() / "data/models/tensorflow/tpu/conversion_dir",
)
parser.set_defaults(gpus=None)
parser.set_defaults(refresh=False)
parser.set_defaults(remove_blacklist=False)
parser.set_defaults(no_center_crop=False)
parser.set_defaults(override_meta_imsize=False)
args = parser.parse_args()

if __name__ == '__main__':
    run_benchmarks(
        gpus=args.gpus,
        refresh=args.refresh,
        remove_blacklist=args.remove_blacklist,
        workers=args.workers,
        no_center_crop=args.no_center_crop,
        override_meta_imsize=args.override_meta_imsize,
        model_subset=args.model_subset,
        tpu_model_dir=args.tpu_model_dir,
        tpu_weights_dir=args.tpu_weights_dir,
    )
