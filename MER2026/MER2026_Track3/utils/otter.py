import os
import cv2
import sys
import glob
import numpy as np

import mimetypes
import requests
import torch
import argparse
import transformers
from PIL import Image
from typing import Union

import sys
sys.path.append('../')

# make sure you can properly access the otter folder
from Otter.otter.modeling_otter import OtterForConditionalGeneration

from utils.common import *

# Disable warnings
requests.packages.urllib3.disable_warnings()

class OTTER:
    def __init__(self, ):
        print ('initial otter model')

        load_bit = "fp32"
        if load_bit == "fp16":
            precision = {"torch_dtype": torch.float16}
        elif load_bit == "bf16":
            precision = {"torch_dtype": torch.bfloat16}
        elif load_bit == "fp32":
            precision = {"torch_dtype": torch.float32}

        # This model version is trained on MIMIC-IT DC dataset.
        model = OtterForConditionalGeneration.from_pretrained('Otter-main/OTTER-Video-LLaMA7B-DenseCaption', device_map="auto", **precision)
        tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]
        model.text_tokenizer.padding_side = "left"
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        model.eval()

        self.tensor_dtype = tensor_dtype
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model


    # ------------------- Utility Functions -------------------
    def get_content_type(self, file_path):
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type


    # ------------------- Image and Video Handling Functions -------------------
    def extract_frames(self, video_path, num_frames=16):
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = total_frames // num_frames
        frames = []

        for i in range(num_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert("RGB")
                frames.append(frame)

        video.release()
        return frames


    def get_image(self, url: str) -> Union[Image.Image, list]:
        video_path = url
        frames = self.extract_frames(video_path) # 默认抽取16帧
        return frames
        

    # ------------------- OTTER Prompt and Response Functions -------------------
    def get_formatted_prompt(self, prompt: str) -> str:
        return f"<image>User: {prompt} GPT:<answer>"

    def get_response(self, input_data, prompt: str, model=None, image_processor=None, tensor_dtype=None) -> str:
        if isinstance(input_data, Image.Image):
            vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        elif isinstance(input_data, list):  # list of video frames
            vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

        lang_x = model.text_tokenizer(
            [
                self.get_formatted_prompt(prompt),
            ],
            return_tensors="pt",
        )

        bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text = model.generate(
            vision_x=vision_x.to(model.device, dtype=tensor_dtype),
            lang_x=lang_x["input_ids"].to(model.device),
            attention_mask=lang_x["attention_mask"].to(model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_id,
        )
        parsed_output = (
            model.text_tokenizer.decode(generated_text[0])
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        return parsed_output


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        frames_list = self.get_image(video_path) # 16 帧输入
        response = self.get_response(frames_list, prompt, self.model, self.image_processor, self.tensor_dtype)
        response = response.replace('\n', ' ').replace('\t', ' ').strip()
        print (response)

        return response
