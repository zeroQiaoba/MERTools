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

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import config

class VideoLLAVA:
    def __init__(self, ):
        print ('initial videollava model')

        # default model
        disable_torch_init()
        model_path = config.model2path['videollava']
        cache_dir = None
        device = 'cuda' # cpu 专享 
        load_4bit, load_8bit = False, False
        model_name = get_model_name_from_path(model_path) # model_name='Video-LLaVA-7B'
        tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
        video_processor = processor['video']
        
        self.model = model
        self.tokenizer = tokenizer
        self.video_processor = video_processor


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, inp, input_type):
        
        # read video_path
        video_tensor = self.video_processor(video_path, return_tensors='pt')['pixel_values'] # [1, 3, 8, 224, 224] => 采样数为8帧
        if type(video_tensor) is list:
            tensor = [video.to(self.model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(self.model.device, dtype=torch.float16)
        
        # read prompt
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print ('===============')
        print('\n', prompt, '\n')
        print ('===============')
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('\n', ' ').replace('\t', ' ').strip()
        if outputs.endswith('</s>'): outputs = outputs[:-len('</s>')]
        print (outputs)
        return outputs
