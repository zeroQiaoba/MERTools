import sys
sys.path.append('../')
from utils.common import *

import os
import glob
import argparse
import numpy as np
from VideoChat.conversation import Chat

# videochat
import torch
from VideoChat.utils.config import Config
from VideoChat.utils.easydict import EasyDict
from VideoChat.models.videochat import VideoChat

class VIDEOCHAT:
    def __init__(self, ):
        print ('initial videochat model')

        config_file = "VideoChat/configs/config_7b.json"
        cfg = Config.from_file(config_file)
        cfg.model.vit_model_path='AffectGPT-master/models/eva_vit_g.pth'
        cfg.model.q_former_model_path = 'AffectGPT-master/models/blip2_pretrained_flant5xxl.pth'
        cfg.model.llama_model_path = 'AffectGPT-master/models/vicuna-7b-v0'
        cfg.model.videochat_model_path = 'video_chat/pretrained_models/videochat_7b.pth'

        model = VideoChat(config=cfg.model)
        model.llama_model = model.llama_model.float() # cpu 专享
        model = model.to(torch.device(cfg.device))
        model = model.eval()
        chat = Chat(model, device=cfg.device)

        self.chat = chat


    def upload_img(self, gr_video, chat_state, num_segments):
        chat_state = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        img_list = []
        _, img_list, chat_state = self.chat.upload_video(gr_video, chat_state, img_list, num_segments)
        return chat_state, img_list
        

    def gradio_ask(self, user_message, chat_state):
        chat_state = self.chat.ask(user_message, chat_state)
        return chat_state

    def gradio_answer(self, chat_state, img_list, num_beams, temperature):
        llm_message, llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
        llm_message = llm_message.replace("<s>", "") # handle <s>
        return llm_message


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, text_input, input_type):
        
        # inference
        num_beams = 1
        temperature = 1.0
        num_segments = 8
        up_video = video_path
        
        img_list = []
        chat_state = []
        chat_state, img_list = self.upload_img(up_video, chat_state, num_segments)
        chat_state = self.gradio_ask(text_input, chat_state)
        response = self.gradio_answer(chat_state, img_list, num_beams, temperature)
        response = response.replace('\n', ' ').replace('\t', ' ').strip()
        print (response)

        return response
