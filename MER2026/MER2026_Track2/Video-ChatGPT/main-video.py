import os
import json
import time
import datetime
import argparse
import numpy as np

from video_chatgpt.video_conversation import (default_conversation) # 设置对话模版，
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
# from video_chatgpt.demo.gradio_patch import Chatbot as grChatbot
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.demo.chat import Chat
# from video_chatgpt.demo.template import tos_markdown, css, title, disclaimer, Seafoam
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *

import sys
sys.path.append('../')
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls


def add_text(state, text, image, first_run):
   
    text = text[:1536] # Hard cut-off
    if first_run: # text -> text\n<video>
        text = text[:1200]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
        text = (text, image)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return state


# 实际上，这个函数上传的是video
def upload_image(image, state):
    '''
    state = Conversation(system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", \
            roles=('Human', 'Assistant'), \
            messages=[['Human', 'What are the key differences between renewable and non-renewable energy sources?'], \
                      ['Assistant', 'Renewable energy sources are those that can be replenished naturally.\n']], \
                      offset=2, sep_style=<SeparatorStyle.SINGLE: 1>, sep='###', sep2=None, version='Unknown', skip_next=False)
    '''
    state = default_conversation.copy()
    img_list = []
    first_run = True
    chat.upload_video(image, img_list)
    return state, img_list, first_run


if __name__ == "__main__":

    ## load args
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true") # False
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--conv_mode", type=str, default="video-chatgpt_v1")
    parser.add_argument("--dataset", default='xxx', help="evaluate dataset") # dataset can set to "hybird"
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()
    disable_torch_init()
    args.model_name = config.PATH_TO_MLLM['LLaVA-7B-Lightening-v1-1']
    args.mm_vision_tower = config.PATH_TO_MLLM['clip-vit-large-patch14']
    args.projection_path = config.PATH_TO_MLLM['video_chatgpt-7B']
    
    ###################################################
    ## Load model
    ###################################################
    # 外部的两个路径，提供了初始化策略
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(model_name = args.model_name, 
                                                                                        mm_vision_tower = args.mm_vision_tower, 
                                                                                        projection_path = args.projection_path)
    # Create replace token, this will replace the <video> in the prompt.
    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

    # Create chat for the demo
    chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token)
    print('Initialization Finished')

    temperature = 0.2
    max_output_tokens = 512


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
            state = []
            img_list = []

            if args.subtitle_flag == 'subtitle':
                prompt = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, acoustic information, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video. Subtitle content of the video: {subtitle} "
            elif args.subtitle_flag == 'nosubtitle':
                prompt = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, acoustic information, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video. "
            
            state, img_list, first_run = upload_image(video_path, state)
            state = add_text(state, prompt, video_path, first_run)
            response = chat.answer(state, img_list, temperature, max_output_tokens, first_run)
            response = response.replace('\n', ' ').replace('\t', ' ').strip()
            print (response)
            
            name2reason[name] = response

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/Video-ChatGPT'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)

