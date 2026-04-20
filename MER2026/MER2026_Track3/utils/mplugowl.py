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

from transformers import AutoTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

class MPLUGOWL:
    def __init__(self, ):
        print ('initial mplug-owl model')

        pretrained_ckpt = 'mPLUG-Owl/mplug-owl-llama-7b-video'
        model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        )
        image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        processor = MplugOwlProcessor(image_processor, tokenizer)
        model = model.cuda()

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
  

    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        video_list = [video_path] 
        prompts = [
        f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <|video|>
        Human: {prompt}
        AI: ''']
        # generate_kwargs = {
        #     'do_sample': True,
        #     'top_k': 5,
        #     'max_length': 512
        # }
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 1024
        }
        inputs = self.processor(text=prompts, videos=video_list, num_frames=4, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                res = self.model.generate(**inputs, **generate_kwargs)
            response = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            response = response.replace('\n', ' ').replace('\t', ' ').strip()
        except: # 很少的样本，视频的字幕太长会报错
            response = ""
        print (response)

        return response
