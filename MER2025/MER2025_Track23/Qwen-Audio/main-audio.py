
import torch
torch.manual_seed(1234)
import os
import argparse
import numpy as np

import sys
sys.path.append('../')

import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='xxx', help="evaluate dataset") # dataset can set to "hybird"
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()


    ###########################################
    ## Load model
    ###########################################
    tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_MLLM['qwen-audio-chat'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config.PATH_TO_MLLM['qwen-audio-chat'], device_map="cuda", trust_remote_code=True).eval()


    ##########################################################
    ## Main Process
    ##########################################################
    process_datasets = args.dataset.split(',')
    print ('process datasets: ', process_datasets)


    for dataset in process_datasets:
        print (f'======== Process for {dataset} ========')

        # 1. read dataset info
        dataset_cls = get_name2cls(dataset)
        test_names = dataset_cls.read_test_names()
        name2subtitle = dataset_cls.name2subtitle
        video_root = config.PATH_TO_RAW_VIDEO[dataset]
        audio_root = config.PATH_TO_RAW_AUDIO[dataset]
        face_root  = config.PATH_TO_RAW_FACE[dataset]

        # 2. main process
        name2reason = {}
        for ii, name in enumerate(test_names):
            subtitle = name2subtitle[name]
            print ('=======================================================')
            print (f'process on {ii}|{len(test_names)}: {name} | {subtitle}')

            # get path
            sample = {'name': name}
            video_path, image_path, audio_path, face_npy = None, None, None, None
            if hasattr(dataset_cls, '_get_video_path'): video_path = dataset_cls._get_video_path(sample)
            if hasattr(dataset_cls, '_get_audio_path'): audio_path = dataset_cls._get_audio_path(sample)
            if hasattr(dataset_cls, '_get_face_path'):  face_npy   = dataset_cls._get_face_path(sample)
            if hasattr(dataset_cls, '_get_image_path'): image_path = dataset_cls._get_image_path(sample)

            # inference
            if args.subtitle_flag == 'subtitle':
                query = tokenizer.from_list_format([
                    {'audio': audio_path}, # Either a local path or an url
                    {'text': f"Subtitle content of the audio: {subtitle}; As an expert in the field of emotions, please focus on the acoustic information and subtitle content in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the audio."},
                ])
            elif args.subtitle_flag == 'nosubtitle':
                query = tokenizer.from_list_format([
                    {'audio': audio_path}, # Either a local path or an url
                    {'text': f"As an expert in the field of emotions, please focus on the acoustic information and subtitle content in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the audio."},
                ])

            response, history = model.chat(tokenizer, query=query, history=None)
            response = response.replace('\n', ' ').replace('\t', ' ').strip()
            print (response)
            
            name2reason[name] = response

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/Qwen-Audio'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)

