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

import os
import torch
import argparse
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

import sys
sys.path.append('../')

from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

class CHATUNIVI:
    def __init__(self, model_path):
        print ('initial chat-univi model')

        self.max_frames = 100
        self.video_framerate = 1

        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = "ChatUniVi"
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor

        if model.config.config["use_cluster"]:
            for n, m in model.named_modules():
                m = m.to(dtype=torch.float16)
        
        self.image_processor = image_processor
        self.model = model
        self.tokenizer = tokenizer

        print ('Loading model finish!!')
    

    # 这个是1s采样一帧，而不是固定采样的
    def _get_rawvideo_dec(self, video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224, video_framerate=1, s=None, e=None):
        # speed up video decode via decord.

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1

        if os.path.exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

            patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
            slice_len = patch_images.shape[0]

            return patch_images, slice_len
        else:
            print("video path: {} error.".format(video_path))


    # sample-wise calling: salmonn 不支持很长的 prompt 输入，导致 normal 运行有问题
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        
        # get prompt
        user_message = prompt
        
        # Sampling Parameter
        conv_mode = "simple"
        temperature = 0.2
        top_p = None
        num_beams = 1

        # read video: video_frames: [nframe, 3, 224, 224]; slice_len=nframe
        video_frames, slice_len = self._get_rawvideo_dec(video_path, self.image_processor, max_frames=self.max_frames, video_framerate=self.video_framerate)

        if self.model.config.mm_use_im_start_end:
            user_message = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + user_message
        else:
            user_message = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + user_message
        '''
        qs = <image> * slice_len \n Describe the video.
        '''

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_frames.half().cuda(),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        response = outputs.replace('\n', ' ').replace('\t', ' ').strip()
        
        print (response)
        
        return response
