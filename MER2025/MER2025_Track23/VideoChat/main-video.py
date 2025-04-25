
import os
import glob
import argparse
import numpy as np
from conversation import Chat

# videochat
import torch
from utils.config import Config
from utils.easydict import EasyDict
from models.videochat import VideoChat

import sys
sys.path.append('../')

import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls


# ========================================
#             Gradio Setting
# ========================================
def upload_img(gr_video, chat_state, num_segments):
    chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []
    _, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, num_segments)
    return chat_state, img_list
    

def gradio_ask(user_message, chat_state):
    chat_state = chat.ask(user_message, chat_state)
    return chat_state


def gradio_answer(chat_state, img_list, num_beams, temperature):
    llm_message, llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    return llm_message


# ========================================
#             Main Process
# ========================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='xxx', help="evaluate dataset") # dataset can set to "hybird"
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()

    
    ###################################################
    ## Load model
    ###################################################
    print('Initializing VideoChat')
    config_file = "configs/config_7b.json"
    cfg = Config.from_file(config_file)
    cfg.model.vit_model_path=config.PATH_TO_MLLM['eva_vit_g']
    cfg.model.q_former_model_path = config.PATH_TO_MLLM['blip2_pretrained_flant5xxl']
    cfg.model.llama_model_path = config.PATH_TO_MLLM['vicuna-7b-v0']
    cfg.model.videochat_model_path = config.PATH_TO_MLLM['videochat_7b']

    model = VideoChat(config=cfg.model)
    model.llama_model = model.llama_model.float() # cpu 专享
    model = model.to(torch.device(cfg.device))
    model = model.eval()
    chat = Chat(model, device=cfg.device)
    print('Initialization Finished')


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
            num_beams = 1
            temperature = 1.0
            num_segments = 8
            up_video = video_path
            
            if args.subtitle_flag == 'subtitle':
                text_input = f"Subtitle content of the video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video. "
            elif args.subtitle_flag == 'nosubtitle':
                text_input = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video. "
            
            img_list = []
            chat_state = []
            chat_state, img_list = upload_img(up_video, chat_state, num_segments)
            chat_state = gradio_ask(text_input, chat_state)
            response = gradio_answer(chat_state, img_list, num_beams, temperature)
            response = response.replace('\n', ' ').replace('\t', ' ').strip()
            print (response)
            
            name2reason[name] = response

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/VideoChat'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)
