import os
import glob
import tqdm
import argparse
import numpy as np
import pandas as pd

from utils.common import * 
import config 

import warnings
warnings.filterwarnings('ignore') # 不显示 warning

#############################################################
# os.environ['CUDA_VISIBLE_DEVICES'] = '3' # debug 时临时设置的
#############################################################
def normal_samplecalling(model_cls, output_type='xxx', round=1):

    save_npz = f'{args.saveroot}/{args.model}-{args.input_type}-normal-{output_type}-round{round}.npz'
    if os.path.exists(save_npz): return

    gt_labels, whole_responses = [], []
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in tqdm.tqdm(enumerate(df.iterrows())):
        name = row['name']
        a1 = row['a1'].replace('\n', ' ').replace('\t', ' ').strip()
        a2 = row['a2'].replace('\n', ' ').replace('\t', ' ').strip()
        # preference = row['preference']
        preference = 'tie'
        gt_labels.append(preference)
        ###########################
        ## debug
        if args.debug and ii == 2: break
        ###########################
        
        audio_path = os.path.join(config.PATH_TO_RAW_AUDIO[output_type], name+'.wav')
        video_path = os.path.join(config.PATH_TO_RAW_VIDEO[output_type], name+'.mp4')
        prompt = f"""We provide two descriptions for a given input: \
a1: \"{a1}\". \
a2: \"{a2}\". \
Please determinate which one is better aligned with the input content. \
If both of them equally align with the input content, please output 'same'. \
Therefore, the output should be a1, a2, or same. Please direct output the answer without additional reasoning process."""
        response = model_cls.func_calling_sample(audio_path, video_path, prompt, args.input_type)
        whole_responses.append(response)

    ## 2. (whole_messages) => (pred_labels)
    pred_labels = func_postprocess_preference(whole_responses)
    np.savez_compressed(save_npz,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels)
    

## Solution: cot 形式，先给出 description, 然后计算 answer
def cot_step1_description_samplecalling(model_cls, output_type='xxx', round=1):

    save_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-description-{output_type}-round{round}.npz'
    if os.path.exists(save_npz): return

    ## => (gt_labels, whole_messages)
    whole_names, whole_responses = [], []
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        ###########################
        if args.debug and ii == 2: break # debug
        ###########################
        name = row['name']
        audio_path = os.path.join(config.PATH_TO_RAW_AUDIO[output_type], name+'.wav')
        video_path = os.path.join(config.PATH_TO_RAW_VIDEO[output_type], name+'.mp4')
        prompt = f"""Please provide a detailed description to a given input, especially focusing on their emotions."""
        response = model_cls.func_calling_sample(audio_path, video_path, prompt, args.input_type)
        whole_names.append(name)
        whole_responses.append(response)
        
    ## 2. name2description
    name2description = {}
    for (name, response) in zip(whole_names, whole_responses):
        name2description[name] = response
    np.savez_compressed(save_npz,
                        name2description=name2description)


def cot_step2_description_samplecalling(model_cls, output_type='xxx', round=1):

    description_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-description-{output_type}-round{round}.npz'
    answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-answer-{output_type}-round{round}.npz'
    if os.path.exists(answer_npz): return

    gt_labels, whole_responses = [], []
    name2description = np.load(description_npz, allow_pickle=True)['name2description'].tolist()
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        ###########################
        if args.debug and ii == 2: break # debug
        ###########################
        name = row['name']
        a1 = row['a1'].replace('\n', ' ').replace('\t', ' ').strip()
        a2 = row['a2'].replace('\n', ' ').replace('\t', ' ').strip()
        # preference = row['preference']
        preference = 'tie'
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
        response = model_cls.func_calling_sample(audio_path, video_path, prompt, args.input_type)
        whole_responses.append(response)

    ## 2. (whole_messages) => (pred_labels)
    pred_labels = func_postprocess_preference(whole_responses)
    np.savez_compressed(answer_npz,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels)


# 用其他 LLM 进行处理
def cot_step3_description_batchcalling(args, llm_cls, output_type='xxx', round=1):

    description_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-description-{output_type}-round{round}.npz'
    # answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-answer2-{output_type}-round{round}.npz'
    answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-answer2-{output_type}-{args.llm}-round{round}.npz'
    if os.path.exists(answer_npz): return
    
    # 1. => (gt_labels, whole_prompts)
    gt_labels, whole_prompts = [], []
    name2description = np.load(description_npz, allow_pickle=True)['name2description'].tolist()
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        name = row['name']
        a1 = row['a1'].replace('\n', ' ').replace('\t', ' ').strip()
        a2 = row['a2'].replace('\n', ' ').replace('\t', ' ').strip()
        # preference = row['preference']
        preference = 'tie'
        description = name2description[name].replace('\n', ' ').replace('\t', ' ').strip()
        gt_labels.append(preference)
        
        prompt = f"""We provide a ground truth description: {description} We also provide two predicted descriptions: \
a1: \"{a1}\". \
a2: \"{a2}\". \
Please determinate which one is better aligned with the ground truth description. \
If both of them equally align with the input content, please output 'same'. \
Therefore, the output should be a1, a2, or same. Please direct output the answer without additional reasoning process."""
        whole_prompts.append(prompt)
    
    # 2. => whole_responses
    whole_responses = []
    batches_prompts = split_list_into_batch(whole_prompts, batchsize=8)
    for batch_prompts in batches_prompts:
        batch_response = llm_cls.get_completion_qwen_bacth(batch_prompts)
        whole_responses.extend(batch_response)
    
    ## 3. (whole_responses) => (pred_labels)
    pred_labels = func_postprocess_preference(whole_responses)
    np.savez_compressed(answer_npz,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels)


## 增加额外的 reasoning 过程
def cot_step4_description_batchcalling(args, llm_cls, output_type='xxx', round=1):

    description_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-description-{output_type}-round{round}.npz'
    # answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-reasoning-{output_type}-round{round}.npz'
    answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-reasoning-{output_type}-{args.llm}-round{round}.npz'
    if os.path.exists(answer_npz): return

    # 1. => (gt_labels, whole_prompts)
    whole_names, whole_prompts = [], []
    name2description = np.load(description_npz, allow_pickle=True)['name2description'].tolist()
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        name = row['name']
        a1 = row['a1'].replace('\n', ' ').replace('\t', ' ').strip()
        a2 = row['a2'].replace('\n', ' ').replace('\t', ' ').strip()
        description = name2description[name].replace('\n', ' ').replace('\t', ' ').strip()
        prompt = f"""We provide a ground truth description: {description} We also provide two predicted descriptions: \
a1: \"{a1}\". \
a2: \"{a2}\". \
Please determinate which one is better aligned with the ground truth description. \
If both of them equally align with the input content, please output 'same'. \
Please output the answer along with the reasoning process."""
        whole_names.append(name)
        whole_prompts.append(prompt)
    
    # 2. => whole_responses
    whole_responses = []
    batches_prompts = split_list_into_batch(whole_prompts, batchsize=8)
    for batch_prompts in batches_prompts:
        batch_response = llm_cls.get_completion_qwen_bacth(batch_prompts)
        whole_responses.extend(batch_response)
    
    # 3. => name2reason
    name2reason = {}
    for (name, reason) in zip(whole_names, whole_responses):
        name2reason[name] = reason
    np.savez_compressed(answer_npz,
                        name2reason=name2reason)

# (answer + reason) -> answer
def cot_step5_description_batchcalling(args, llm_cls, output_type='xxx', round=1):

    # description_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-reasoning-{output_type}-round{round}.npz'
    # answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-answer3-{output_type}-round{round}.npz'
    description_npz = f'{args.saveroot}/{args.model}-{args.input_type}-cot-reasoning-{output_type}-{args.llm}-round{round}.npz'
    answer_npz      = f'{args.saveroot}/{args.model}-{args.input_type}-cot-answer3-{output_type}-{args.llm}-round{round}.npz'
    if os.path.exists(answer_npz): return

    # 1. => (gt_labels, whole_prompts)
    gt_labels, whole_prompts = [], []
    name2reason = np.load(description_npz, allow_pickle=True)['name2reason'].tolist()
    df = pd.read_csv(config.PATH_TO_LABEL[output_type])
    for ii, (_, row) in enumerate(df.iterrows()):
        name = row['name']
        # preference = row['preference']
        preference = 'tie'
        reason = name2reason[name].replace('\n', ' ').replace('\t', ' ').strip()
        gt_labels.append(preference)
        
        prompt = f"""Based on the provided decription:  \"{reason}\", please determinate which one is better aligned with the ground truth description. \
        The output should be a1, a2, or same. Please direct output the answer without additional reasoning process."""
        whole_prompts.append(prompt)
    
    # 2. => whole_responses
    whole_responses = []
    batches_prompts = split_list_into_batch(whole_prompts, batchsize=8)
    for batch_prompts in batches_prompts:
        batch_response = llm_cls.get_completion_qwen_bacth(batch_prompts)
        whole_responses.extend(batch_response)
    
    ## 3. (whole_responses) => (pred_labels)
    pred_labels = func_postprocess_preference(whole_responses)
    np.savez_compressed(answer_npz,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels)
    

###################################################
## 目标1：是否可以用 zero-shot 方式，实现对偏好的拟合？
## 目标2：如何使用模型，能够实现更好的拟合效果？
###################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MLLM as the judger for DPO Preference")
    ## main params
    parser.add_argument("--model",  default='xxx', type=str, help="qwen25vl, qwen25omni")
    parser.add_argument("--prompt", default='normal', type=str, help="normal, cot, cot2")
    parser.add_argument("--totalround", default=1, type=int, help="repeat calling times")
    parser.add_argument("--input_type",  default='xxx', type=str, help="audio, video, audiovideo")
    parser.add_argument("--output_type",  default='xxx', type=str, help="preferencestrict, preferenceab")
    ## other params
    parser.add_argument("--llm",  default='xxx', type=str, help="qwen25, qwen3")
    parser.add_argument("--saveroot", default='output-matching', type=str, help="save root")
    parser.add_argument("--debug", default=False, type=bool, help="True or False")
    args = parser.parse_args()
    print (args)

    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot)

    #########################################################
    ## 结果计算
    #########################################################
    ## 模型放在这里，避免重复加载
    if args.prompt in ['cot2', 'cot3']: # cot2是纯文本的测试
        if args.llm == 'qwen25':
            from utils.qwen25 import QWEN25
            llm_cls = QWEN25(config.model2path[args.llm])
        elif args.llm in ['qwen3_8b', 'qwen3_14b', 'qwen3_32b']:
            from utils.qwen3 import QWEN3
            llm_cls = QWEN3(config.model2path[args.llm])

    elif args.prompt in ['normal', 'cot']:
        ## load GEMINI models
        if args.model.startswith('gemini'):
            from utils.gemini import GEMINI
            model_cls = GEMINI(config.model2path[args.model])
        ## load GPT models
        elif args.model.startswith('gpt'):
            from utils.gpt import GPT
            model_cls = GPT(config.model2path[args.model])
        ## load other open-sourced models
        elif args.model == 'videollava':
            from utils.videollava import VideoLLAVA
            model_cls = VideoLLAVA()
        elif args.model == 'videochatgpt':
            from utils.videochatgpt import VideoChatGPT
            model_cls = VideoChatGPT()
        elif args.model == 'qwenaudio':
            from utils.qwenaudio import QWENAUDIO
            model_cls = QWENAUDIO()
        elif args.model == 'qwen2audio':
            from utils.qwen2audio import QWEN2AUDIO
            model_cls = QWEN2AUDIO(config.model2path[args.model])
        elif args.model == 'salmonn':
            from utils.salmonn import SALMONNLZ
            model_cls = SALMONNLZ()
        elif args.model.startswith('chatunivi'):
            from utils.chatunivi import CHATUNIVI
            model_cls = CHATUNIVI(config.model2path[args.model])
        elif args.model == 'mplugowl':
            from utils.mplugowl import MPLUGOWL
            model_cls = MPLUGOWL()
        elif args.model == 'otter':
            from utils.otter import OTTER
            model_cls = OTTER()
        elif args.model == 'llamavid':
            from utils.llamavid import LLAMAVID
            model_cls = LLAMAVID()
        elif args.model == 'videochat':
            from utils.videochat import VIDEOCHAT
            model_cls = VIDEOCHAT()
        elif args.model == 'videochat2':
            from utils.videochat2 import VIDEOCHAT2
            model_cls = VIDEOCHAT2()
        elif args.model.startswith('llavanextvideo'):
            from utils.llavanextvideo import LLAVANEXTVIDEO
            model_cls = LLAVANEXTVIDEO(config.model2path[args.model])
        elif args.model == 'vita_15':
            from utils.vita import VITA
            model_cls = VITA(config.model2path[args.model])
        # 有些包需要重新安装，用 pip install -U xxx 的形式去解决环境冲突的问题
        elif args.model.startswith('pllava'):
            from utils.pllava import PLLAVA
            model_cls = PLLAVA(config.model2path[args.model])

    args.input_type = config.model2input[args.model]

    ## 开始运行
    for round in range(args.totalround):
        if args.prompt == 'normal':
            normal_samplecalling(model_cls, args.output_type, round=round)
        elif args.prompt == 'cot':
            cot_step1_description_samplecalling(model_cls, args.output_type, round=round)
            cot_step2_description_samplecalling(model_cls, args.output_type, round=round) # 用原生的模型
        elif args.prompt == 'cot2': # 用其他LLM模型进行分析
            cot_step3_description_batchcalling(args, llm_cls, args.output_type, round=round)
        elif args.prompt == 'cot3': # 加上 reasoning 分析过程，以及最终 answer 抽取
            cot_step4_description_batchcalling(args, llm_cls, args.output_type, round=round)
            cot_step5_description_batchcalling(args, llm_cls, args.output_type, round=round)
