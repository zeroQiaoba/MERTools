import os
import glob
import numpy as np
from utils.common import *
from toolkit.utils.functions import *
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


########################################
## 全局变量
########################################
model_mapping = {
    'otter': 'Otter \\citep{li2023otter}',
    'videochat2': 'VideoChat2 \\citep{li2024mvbench}',
    'chatunivi_7b': 'Chat-UniVi \\citep{jin2024chat}',
    'videochatgpt': 'Video-ChatGPT \\citep{maaz2024video}',
    'mplugowl': 'mPLUG-Owl \\citep{ye2023mplugowl}',
    'videollava': 'Video-LLaVA \\citep{lin2024video}',
    'llamavid': 'LLaMA-VID \\citep{li2024llama}',
    'videochat': 'VideoChat \\citep{li2023videochat}',
    'pllava_7b': 'PLLAVA \\citep{xu2024pllava}',
    'vita_15': 'VITA-1.5 \\citep{fu2025vita}',
    'llavanextvideo_7b': 'LLaVA-Next-Video \\citep{li2024llava}',
    'gpt_4o': 'GPT-4o \\citep{openai24gpt4o}',
    'gemini_20_flash': 'Gemini-2.0-Flash \\citep{gemini20flash}',
    'qwen2audio': 'Qwen2-Audio \\citep{chu2024qwen2}',
    'gemini_25_flash': 'Gemini-2.5-Flash \\citep{gemini25flash}',
    'gpt_41': 'GPT-4.1 \\citep{achiam2023gpt}',
    'gemini_15_pro': 'Gemini-1.5-Pro \\citep{team2024gemini}',
    'qwen25vl_7b': 'Qwen2.5-VL \\citep{bai2025qwen2}',
    'gemini_15_flash': 'Gemini-1.5-Flash \\citep{team2024gemini}',
    'qwen25omni_7b': 'Qwen2.5-Omni \\citep{xu2025qwen2}',
    'affectgpt_mercaptionplus': 'AffectGPT \\citep{lian2025affectgpt}',
}

model_open  = ['otter','videochat2','chatunivi_7b','videochatgpt','mplugowl','videollava','llamavid','videochat',
               'pllava_7b','vita_15','llavanextvideo_7b','qwen2audio','qwen25vl_7b','qwen25omni_7b','affectgpt_mercaptionplus']

model_close = ['gpt_4o','gemini_20_flash','gemini_25_flash','gpt_41','gemini_15_pro','gemini_15_flash']


prompt_namemapping = {
    'normal': 'S1',
    'cot':    'S2',
    'cot2':   'S3',
    'cot3':   'S4',
}

########################################
## 计算统计结果
########################################
def main_statistic(basefile='xxx'):
    
    whole_store = {}

    for (model, input_type) in [
        ('qwen25vl_7b',       'video'),
        ('qwen25omni_7b',     'audiovideo'),
        ('qwen2audio',        'audio'),
        ('llavanextvideo_7b', 'video'),
        ('videollava',        'video'),
        ('llamavid',          'video')
    ]:
        
        save_npz = f'output-matching/{model}-{input_type}-normal-{basefile}-round0.npz'
        waf_twoclass, acc_twoclass = func_preference_metric(save_npz, metric='twoclass')
        print (model_mapping[model], f'& {(waf_twoclass*100):.2f} & {(acc_twoclass*100):.2f} \\\\')
            
# main_statistic(basefile='MER2026Track3')