import os
import math
import time
import glob
import tqdm
import random
import torch
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

from utils.common import * 
import config

import warnings
warnings.filterwarnings('ignore') # 不显示 warning

def normal_batchcalling(model_cls, output_type='xxx', round=1):

    save_npz = f'{args.saveroot}/{args.model}-{args.input_type}-normal-{output_type}-round{round}.npz'
    if os.path.exists(save_npz): return

    gt_labels, whole_messages = [], []

    ## => (gt_labels, whole_messages)
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        name = row['name']
        a1 = row['a1'].replace('\n', ' ').replace('\t', ' ').strip()
        a2 = row['a2'].replace('\n', ' ').replace('\t', ' ').strip()
        preference = row['preference']
        gt_labels.append(preference)
        ###########################
        if args.debug and ii == 2: break # debug
        ###########################
        
        audio_path = os.path.join(config.PATH_TO_RAW_AUDIO[output_type], name+'.wav')
        video_path = os.path.join(config.PATH_TO_RAW_VIDEO[output_type], name+'.mp4')
        prompt = f"""We provide two descriptions for a given input: \
a1: \"{a1}\". \
a2: \"{a2}\". \
Please determinate which one is better aligned with the input content. \
If both of them equally align with the input content, please output 'same'. \
Therefore, the output should be a1, a2, or same. Please direct output the answer without additional reasoning process."""
        message = model_cls.generate_message(audio_path, video_path, prompt, args.input_type)
        whole_messages.append(message)

    ## 2. (whole_messages) => (pred_labels)
    whole_responses = model_cls.func_calling(whole_messages)
    pred_labels = func_postprocess_preference(whole_responses)
    np.savez_compressed(save_npz,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels)
    

## Solution: cot 形式，先给出 description, 然后计算 answer
def cot_step1_description_batchcalling(model_cls, output_type='xxx', round=1):

    save_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-description-{output_type}-round{round}.npz'
    if os.path.exists(save_npz): return

    ## => (gt_labels, whole_messages)
    whole_names, whole_messages = [], []
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        ###########################
        if args.debug and ii == 2: break # debug
        ###########################
        name = row['name']
        audio_path = os.path.join(config.PATH_TO_RAW_AUDIO[output_type], name+'.wav')
        video_path = os.path.join(config.PATH_TO_RAW_VIDEO[output_type], name+'.mp4')
        prompt = f"""Please provide a detailed description to a given video, especially focusing on their emotions."""
        message = model_cls.generate_message(audio_path, video_path, prompt, args.input_type)
        whole_names.append(name)
        whole_messages.append(message)
        
    ## 2. name2description
    whole_responses = model_cls.func_calling(whole_messages)
    name2description = {}
    for (name, response) in zip(whole_names, whole_responses):
        name2description[name] = response
    np.savez_compressed(save_npz,
                        name2description=name2description)


def cot_step2_description_batchcalling(model_cls, output_type='xxx', round=1):

    description_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-description-{output_type}-round{round}.npz'
    answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-answer-{output_type}-round{round}.npz'
    if os.path.exists(answer_npz): return
    
    gt_labels, whole_messages = [], []
    name2description = np.load(description_npz, allow_pickle=True)['name2description'].tolist()
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        ###########################
        if args.debug and ii == 2: break # debug
        ###########################
        name = row['name']
        a1 = row['a1'].replace('\n', ' ').replace('\t', ' ').strip()
        a2 = row['a2'].replace('\n', ' ').replace('\t', ' ').strip()
        preference = row['preference']
        description = name2description[name].replace('\n', ' ').replace('\t', ' ').strip()
        gt_labels.append(preference)
        
        audio_path = os.path.join(config.PATH_TO_RAW_AUDIO[output_type], name+'.wav')
        video_path = os.path.join(config.PATH_TO_RAW_VIDEO[output_type], name+'.mp4')
        prompt = f"""We provide a ground truth description: {description} We also provide two predicted descriptions: \
a1: \"{a1}\". \
a2: \"{a2}\". \
Please determinate which one is better aligned with the ground truth description. \
If both of them equally align with the input content, please output 'same'. \
Therefore, the output should be a1, a2, or same. Please direct output the answer without additional reasoning process."""
        message = model_cls.generate_message(audio_path, video_path, prompt, args.input_type)
        whole_messages.append(message)

    ## 2. (whole_messages) => (pred_labels)
    whole_response = model_cls.func_calling(whole_messages)
    pred_labels = func_postprocess_preference(whole_response)
    np.savez_compressed(answer_npz,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels)


## 因为这部分函数是一样的，所以先设置为共用的形式，没必要重复写这个
from main_dpo_sample import cot_step3_description_batchcalling
from main_dpo_sample import cot_step4_description_batchcalling
from main_dpo_sample import cot_step5_description_batchcalling


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MLLM as the judger for DPO Preference")
    ## main params
    parser.add_argument("--model",  default='xxx', type=str, help="qwen25vl, qwen25omni")
    parser.add_argument("--prompt", default='normal', type=str, help="normal, cot, cot2")
    parser.add_argument("--totalround", default=1, type=int, help="repeat calling times")
    parser.add_argument("--input_type",  default='xxx', type=str, help="audio, video, audiovideo")
    parser.add_argument("--output_type",  default='xxx', type=str, help="preferencestrict, preferenceab")
    ## other params
    parser.add_argument("--llm",  default='xxx', type=str, help="qwen25, qwen3_8b, qwen3_14b")
    parser.add_argument("--saveroot", default='output-matching', type=str, help="save root") # 所有结果按照 npz 格式存储下来
    parser.add_argument("--debug", default=False, type=bool, help="True or False")
    args = parser.parse_args()
    print (args)

    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot)

    ## 模型放在这里，避免重复加载
    if args.prompt in ['cot2', 'cot3']: # cot2 只采用 llm 进行二次处理
        if args.llm == 'qwen25':
            from utils.qwen25 import QWEN25
            llm_cls = QWEN25(config.model2path[args.llm])
        elif args.llm in ['qwen3_8b', 'qwen3_14b']:
            from utils.qwen3 import QWEN3
            llm_cls = QWEN3(config.model2path[args.llm])

    elif args.prompt in ['normal', 'cot']:
        if args.model.startswith('qwen25vl'):
            from utils.qwen25vl import QWEN25VL
            model_cls = QWEN25VL(config.model2path[args.model])
        elif args.model.startswith('qwen25omni'):
            from utils.qwen25omni import QWEN25OMNI
            model_cls = QWEN25OMNI(config.model2path[args.model])
            assert args.input_type in ['video', 'audio', 'audiovideo']

    if args.model in config.model2input: # 如果模型在这个里面，则基于这个确定读取类型
        args.input_type = config.model2input[args.model]

    ## 开始运行并获取结果
    for round in range(args.totalround):
        if args.prompt == 'normal':
            normal_batchcalling(model_cls, args.output_type, round=round)
        elif args.prompt == 'cot':
            cot_step1_description_batchcalling(model_cls, args.output_type, round=round)
            cot_step2_description_batchcalling(model_cls, args.output_type, round=round)
        elif args.prompt == 'cot2':
            cot_step3_description_batchcalling(args, llm_cls, args.output_type, round=round)
        elif args.prompt == 'cot3':
            cot_step4_description_batchcalling(args, llm_cls, args.output_type, round=round)
            cot_step5_description_batchcalling(args, llm_cls, args.output_type, round=round)
