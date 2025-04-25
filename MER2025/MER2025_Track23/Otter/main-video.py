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

# make sure you can properly access the otter folder
from otter.modeling_otter import OtterForConditionalGeneration


import sys
sys.path.append('../')
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls

# Disable warnings
requests.packages.urllib3.disable_warnings()


# ------------------- Utility Functions -------------------
def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------
def extract_frames(video_path, num_frames=16):
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


def get_image(url: str) -> Union[Image.Image, list]:
    video_path = url
    frames = extract_frames(video_path) # 默认抽取16帧
    return frames
    

# ------------------- OTTER Prompt and Response Functions -------------------
def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"

def get_response(input_data, prompt: str, model=None, image_processor=None, tensor_dtype=None) -> str:
    if isinstance(input_data, Image.Image):
        vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    elif isinstance(input_data, list):  # list of video frames
        vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='xxx', help="evaluate dataset") # dataset can set to "hybird"
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()

    load_bit = "fp32"
    if load_bit == "fp16":
        precision = {"torch_dtype": torch.float16}
    elif load_bit == "bf16":
        precision = {"torch_dtype": torch.bfloat16}
    elif load_bit == "fp32":
        precision = {"torch_dtype": torch.float32}

    # This model version is trained on MIMIC-IT DC dataset.
    model = OtterForConditionalGeneration.from_pretrained(config.PATH_TO_MLLM['otter'], device_map="auto", **precision)
    tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    
    ##########################################################
    ## Main Process
    ##########################################################
    process_datasets = args.dataset.split(',')
    print ('process datasets: ', process_datasets)

    for dataset in process_datasets:
        print (f'======== Process for {dataset} ========')

        # 1. read dataset info
        dataset_cls = get_name2cls(dataset)
        test_names = dataset_cls.read_test_names()
        name2subtitle = dataset_cls.name2subtitle
        video_root = config.PATH_TO_RAW_VIDEO[dataset]
        audio_root = config.PATH_TO_RAW_AUDIO[dataset]
        face_root  = config.PATH_TO_RAW_FACE[dataset]

        # 2. main process
        name2reason = {}
        for ii, name in enumerate(test_names):
            subtitle = name2subtitle[name]
            print ('=======================================================')
            print (f'process on {ii}|{len(test_names)}: {name} | {subtitle}')

            # get path
            sample = {'name': name}
            video_path, image_path, audio_path, face_npy = None, None, None, None
            if hasattr(dataset_cls, '_get_video_path'): video_path = dataset_cls._get_video_path(sample)
            if hasattr(dataset_cls, '_get_audio_path'): audio_path = dataset_cls._get_audio_path(sample)
            if hasattr(dataset_cls, '_get_face_path'):  face_npy   = dataset_cls._get_face_path(sample)
            if hasattr(dataset_cls, '_get_image_path'): image_path = dataset_cls._get_image_path(sample)

            # inference
            frames_list = get_image(video_path) # 16 帧输入
            if args.subtitle_flag == 'subtitle':
                prompts_input = f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
            elif args.subtitle_flag == 'nosubtitle':
                prompts_input = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
            response = get_response(frames_list, prompts_input, model, image_processor, tensor_dtype)
            response = response.replace('\n', ' ').replace('\t', ' ').strip()
            print (response)
            
            name2reason[name] = response

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/Otter'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)







        
