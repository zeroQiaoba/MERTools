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
from sklearn.metrics import f1_score, accuracy_score
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.common import *

from transformers import AutoModelForCausalLM, AutoTokenizer

class QWENAUDIO:
    def __init__(self, ):
        print ('initial qwenaudio model')

        model_root = 'Qwen-main/qwen-audio-chat'
        tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_root, device_map="cuda", trust_remote_code=True).eval()

        self.tokenizer = tokenizer
        self.model = model

    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        # inference
        query = self.tokenizer.from_list_format([
            {'audio': audio_path}, # Either a local path or an url
            {'text': prompt},
        ])

        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        response = response.replace('\n', ' ').replace('\t', ' ').strip()
        print (response)
        
        return response
