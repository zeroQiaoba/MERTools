import os
import time
import random
import argparse
import numpy as np

import torch
from datetime import datetime
import torch.backends.cudnn as cudnn

import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank, init_distributed_mode
from my_affectgpt.common.logger import setup_logger
from my_affectgpt.common.registry import registry
from my_affectgpt.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *

def setup_seeds(config): 
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    # parser.add_argument("--job_name", required=True, help="which path to save model.")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="overwrite params in xxx.config (only for run and model). Example: --options 'ckpt=aaa' 'ckpt_2=bbb'")
    args = parser.parse_args()
    return args

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base")) # 'affectgpt.runners.runner_base.RunnerBase'
    return runner_cls

def main():

    args = parse_args()
    cfg = Config(args)

    # # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    # # max_epoch = cfg.run_cfg['max_epoch'] 
    # # job_id = f"{args.job_name}_affectgpt_epoch_{max_epoch}_{time.time()}"
    # # job_id = f"{args.job_name}_affectgpt_{time.time()}"
    # job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    # job_id = f"{job_name}_{str(int(time.time()))}" # 减少小数点后的存储，便于复制
    # # job_id = job_name # debug
    # # job_id = f"affectgpt_{time.time()}"

    # 用于分布式训练：防止 nccl barrier 卡死，job_id 统一确保进程能找到对应的进程组
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d%H%M')[:-1]}" # zhuofan

    print (job_id)

    # print logging files
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger() 
    cfg.pretty_print()

    # load task and start training
    task = tasks.setup_task(cfg) # video_text_pretrain
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id, 
        task=task, 
        model=model, 
        datasets=datasets
    )
    runner.train()

if __name__ == "__main__":
    main()
