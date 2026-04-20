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

import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

class LLAVANEXTVIDEO:
    def __init__(self, model_root):
        print ('initial LLaVA-NeXT-Video model')
        
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_root, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            # use_flash_attention_2=True,
        ).to('cuda')

        processor = LlavaNextVideoProcessor.from_pretrained(model_root)

        self.model = model
        self.processor = processor


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(messages, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        # output = self.model.generate(**inputs, max_new_tokens=128)
        output = self.model.generate(**inputs, max_new_tokens=512)
        # output = self.processor.decode(output[0][2:], skip_special_tokens=True)
        output = self.processor.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True) # only output answers
        print (output)

        return output







