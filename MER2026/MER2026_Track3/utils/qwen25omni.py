import os
import time
import math
import glob
import torch
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from utils.common import *

class QWEN25OMNI:
    def __init__(self, model_root):
        print ('initial qwen25omni model')

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_root,
            torch_dtype="auto",
            # device_map="auto",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_root)

        self.model = model
        self.processor = processor


    # def generate_message(self, audio_path, video_path, prompt, input_type):
        
    #     assert input_type in ['audio', 'video', 'audiovideo']

    #     if input_type == 'audio':
    #         message = [
    #             {
    #                 "role": "system",
    #                 "content": [
    #                     {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    #                 ],
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "audio", "audio": audio_path},
    #                     {"type": "text",  "text": prompt},
    #                 ],
    #             }
    #         ]
    #         self.USE_AUDIO_IN_VIDEO = True

    #     elif input_type == 'video':
    #         message = [
    #             {
    #                 "role": "system",
    #                 "content": [
    #                     {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    #                 ],
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "video", "video": video_path},
    #                     {"type": "text", "text": prompt},
    #                 ],
    #             }
    #         ]
    #         self.USE_AUDIO_IN_VIDEO = False
            
    #     elif input_type == 'audiovideo':
    #         message = [
    #             {
    #                 "role": "system",
    #                 "content": [
    #                     {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    #                 ],
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "video", "video": video_path},
    #                     {"type": "text", "text": prompt},
    #                 ],
    #             }
    #         ]
    #         self.USE_AUDIO_IN_VIDEO = True

    #     return message


    ####################################################################################
    ## 切换到 rebuttal 模式: 这部分只是为了 ablation study 临时测试的，上面的才是正常的 code
    ####################################################################################
    def generate_message(self, audio_path, video_path, prompt, input_type):
        
        if input_type in ['audio', 'audiotext']:
            message = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text",  "text": prompt},
                    ],
                }
            ]
            self.USE_AUDIO_IN_VIDEO = True

        elif input_type in ['video', 'videotext']:
            message = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            self.USE_AUDIO_IN_VIDEO = False
            
        elif input_type in ['audiovideo', 'audiovideotext']:
            message = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            self.USE_AUDIO_IN_VIDEO = True
        
        elif input_type in ['text']:
            message = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            self.USE_AUDIO_IN_VIDEO = False

        return message


    def func_calling(self, whole_messages, temperature=None):
        whole_responses = []

        # batches_messages = split_list_into_batch(whole_messages, batchsize=8)
        batches_messages = split_list_into_batch(whole_messages, batchsize=4) # 上面 OOM，减少 batchsize
        for batch_messages in batches_messages:
            #############################################
            ## 老是 OOM，尝试 Debug 定位错误
            for message in batch_messages:
                print (message[0])
            #############################################
            text = self.processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(batch_messages, use_audio_in_video=self.USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=self.USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            ## 根据 rebuttal 要求，额外测试温度系数的影响
            if temperature is None: # 默认情况
                generated_ids = self.model.generate(**inputs, use_audio_in_video=self.USE_AUDIO_IN_VIDEO, return_audio=False)
            elif temperature == 'case1':
                generated_ids = self.model.generate(**inputs, use_audio_in_video=self.USE_AUDIO_IN_VIDEO, return_audio=False, temperature=0.7)
            elif temperature == 'case2':
                generated_ids = self.model.generate(**inputs, use_audio_in_video=self.USE_AUDIO_IN_VIDEO, return_audio=False, temperature=1.0)
            elif temperature == 'case3':
                generated_ids = self.model.generate(**inputs, use_audio_in_video=self.USE_AUDIO_IN_VIDEO, return_audio=False, temperature=1.3)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ] # 提出 out_ids 部分内容
            batch_responses = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ) # 解码出文字
            whole_responses.extend(batch_responses)

            print (text[0], batch_responses[0])
        return whole_responses

