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

from SALMONN.model import SALMONN

class SALMONNLZ:
    def __init__(self, ):
        print ('initial salmonn model')
    
        whisper_path = 'SALMONN/models/whisper-large-v2'
        beats_path = 'SALMONN/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
        ckpt_path = 'SALMONN/models/salmonn_7b.pth'
        vicuna_path = 'SALMONN/models/vicuna-7b-v1.5'

        model = SALMONN(
            ckpt=ckpt_path,
            whisper_path=whisper_path,
            beats_path=beats_path,
            vicuna_path=vicuna_path,
            low_resource=False
        )
        model.to('cuda')
        model.eval()

        self.model = model

    # sample-wise calling: salmonn 不支持很长的 prompt 输入，导致 normal 运行有问题
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        try:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                response = self.model.generate(audio_path, max_length=300,
                        prompt=prompt, 
                        device='cuda')[0]
                response = response.replace('\n', ' ').replace('\t', ' ').strip()
        except:
            response = ""
        
        print (response)
        
        return response
