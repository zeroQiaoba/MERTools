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

from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from utils.common import *

class QWEN2AUDIO:
    def __init__(self, model_path):
        print ('initial qwen2audio model')
        processor = AutoProcessor.from_pretrained(model_path)
        # model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="cuda")

        self.model = model
        self.processor = processor
    
    def func_read_audio(self, audio_path):
        # "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"
        # BytesIO(urlopen(ele['audio_url']).read())

        with open(audio_path, 'rb') as f:
            audio_data = BytesIO(f.read())
        return audio_data
    

    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]
            
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                self.func_read_audio(ele['audio_url']), 
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        # inputs.input_ids = inputs.input_ids.to("cuda")
        inputs = inputs.to("cuda")

        # generate_ids = self.model.generate(**inputs, max_length=256)
        # generate_ids = self.model.generate(**inputs, max_length=512) # 输入 prompt 长度太长，我们将这块增加一些
        generate_ids = self.model.generate(**inputs, max_new_tokens=512) # 输入 prompt 长度太长，修改为这个试试
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace('\n', ' ').replace('\t', ' ').strip()
        print (response)
        
        return response






        