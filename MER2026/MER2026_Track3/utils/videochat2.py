import sys
sys.path.append('../')
from utils.common import *

import os
import glob
import torch
import argparse
import numpy as np

# videochat
from VideoChat2.conversation import Chat
from VideoChat2.utils.config import Config
from VideoChat2.utils.easydict import EasyDict
from VideoChat2.models.videochat2_it import VideoChat2_it
from peft import get_peft_model, LoraConfig, TaskType



class VIDEOCHAT2:
    def __init__(self, ):
        print ('initial videochat2 model')

        self.num_beams = 1
        self.temperature = 1.0
        self.num_segments = 8

        config_file = "VideoChat2/configs/config.json"
        cfg = Config.from_file(config_file)
        cfg.model.vit_blip_model_path = 'video_chat2/pretrained_models/umt_l16_qformer.pth'
        cfg.model.llama_model_path = 'AffectGPT-master/models/vicuna-7b-v0'
        cfg.model.videochat2_model_path = 'video_chat2/pretrained_models/videochat2_7b_stage2.pth'

        cfg.model.vision_encoder.num_frames = 4
        model = VideoChat2_it(config=cfg.model)
        model = model.to(torch.device(cfg.device))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.
        )
        model.llama_model = get_peft_model(model.llama_model, peft_config)
        model.llama_model = model.llama_model.float() # cpu 专享

        state_dict = torch.load('video_chat2/pretrained_models/videochat2_7b_stage3.pth', "cpu")
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model = model.eval()

        self.chat = Chat(model, device=cfg.device)
        
    
    

    # ========================================
    #             Gradio Setting
    # ========================================
    def upload_img(self,gr_video, chat_state, num_segments):
        print(gr_video)
        chat_state = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        img_list = []
        _, img_list, chat_state = self.chat.upload_video(gr_video, chat_state, img_list, num_segments)
        return chat_state, img_list
        
    def gradio_ask(self,user_message, chat_state):
        chat_state = self.chat.ask(user_message, chat_state)
        return chat_state

    def gradio_answer(self,chat_state, img_list, num_beams, temperature):
        llm_message, llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
        llm_message = llm_message.replace("<s>", "") # handle <s>
        return llm_message


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, text_input, input_type):
        
        # inference
        chat_state = []
        img_list = []
        chat_state, img_list = self.upload_img(video_path, chat_state, self.num_segments)
        chat_state = self.gradio_ask(text_input, chat_state)
        response = self.gradio_answer(chat_state, img_list, self.num_beams, self.temperature)
        response = response.replace('\n', ' ').replace('\t', ' ').strip()
        print (response)

        return response
