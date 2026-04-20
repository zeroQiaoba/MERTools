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


class QWEN25VL:
    def __init__(self, model_root):
        print (f'initialize {model_root}')

        # default model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            # "Qwen-main/Qwen2.5-VL-3B-Instruct",
            model_root,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            # device_map="cuda",
        )

        # default processor
        # processor = AutoProcessor.from_pretrained("Qwen-main/Qwen2.5-VL-3B-Instruct")
        processor = AutoProcessor.from_pretrained(model_root)
        processor.tokenizer.padding_side = "left" # for batch calling 

        self.model = model
        self.processor = processor


    def generate_message(self, audio_path, video_path, prompt, input_type):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages


    def func_calling(self, whole_messages):
        whole_responses = []
        batches_messages = split_list_into_batch(whole_messages, batchsize=8)
        for batch_messages in tqdm.tqdm(batches_messages):
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference # [他这个vl默认的限制到 128 output tokens 上] 
            # generated_ids = model.generate(**inputs, max_new_tokens=128)
            # generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ] # 提出 out_ids 部分内容
            batch_responses = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ) # 解码出文字
            
            whole_responses.extend(batch_responses)
            print (texts[0], batch_responses[0])
        return whole_responses

