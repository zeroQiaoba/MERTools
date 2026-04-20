"""
Adapted from salesforce@LAVIS. Below is the original copyright:
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import my_affectgpt.common.dist_utils as dist_utils
from my_affectgpt.common.dist_utils import download_cached_file
from my_affectgpt.common.utils import is_url
from my_affectgpt.common.logger import MetricLogger
from my_affectgpt.models.base_model import BaseModel
from my_affectgpt.models.Qformer import BertConfig, BertLMHeadModel
from my_affectgpt.models.eva_vit import create_eva_vit_g
from transformers import AutoTokenizer
import config


## 在 AffectGPT 中，每个 LLM 都需要自己的 'eos', 'pad', 'bos'；否则模型会报错
def load_tokenizer_from_LLM(model_name):
    if model_name in ['Baichuan2']:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False)
    if model_name in ['Qwen2', 'Qwen25']: tokenizer.bos_token='<|im_start|>'
    tokenizer.pad_token = tokenizer.eos_token # 看看如果全设置成这样子会有什么影响？ vicuna, llama2, llama3
    tokenizer.add_tokens([config.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FRAME_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FACE_PATCH_TOKEN],  special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_MULTI_PATCH_TOKEN], special_tokens=True)
    return tokenizer

