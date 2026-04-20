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

import sys
sys.path.append('../')

from video_chatgpt.video_conversation import (default_conversation) # 设置对话模版，
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.demo.chat import Chat
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *


class VideoChatGPT:
    def __init__(self, ):
        print ('initial videochatgpt model')
        self.conv_mode = 'video-chatgpt_v1'
        self.model_name = 'tools/transformers/LLaVA-7B-Lightening-v1-1'
        self.mm_vision_tower = 'tools/transformers/clip-vit-large-patch14'
        self.projection_path = 'Video-ChatGPT-main/models/video_chatgpt-7B.bin'

        self.temperature = 0.2
        self.max_output_tokens = 512

        model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(model_name = self.model_name, 
                                                                                            mm_vision_tower = self.mm_vision_tower, 
                                                                                            projection_path = self.projection_path)
        # Create replace token, this will replace the <video> in the prompt.
        replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
        replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

        # Create chat for the demo
        self.chat = Chat(self.model_name, self.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token)
        print('Initialization Finished')


    def add_text(self, state, text, image, first_run):
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
    def upload_image(self, image, state):
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
        self.chat.upload_video(image, img_list)
        return state, img_list, first_run


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        # inference
        state = []
        img_list = []

        print (prompt)
        state, img_list, first_run = self.upload_image(video_path, state)
        state = self.add_text(state, prompt, video_path, first_run)
        response = self.chat.answer(state, img_list, self.temperature, self.max_output_tokens, first_run)
        response = response.replace('\n', ' ').replace('\t', ' ').strip()
        print (response)
        
        return response
