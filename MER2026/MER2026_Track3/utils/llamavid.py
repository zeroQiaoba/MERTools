
import sys
sys.path.append('../')

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu
import torch

import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from utils.common import *

import config

class LLAMAVID:
    def __init__(self, ):
        print ('initial llamavid model')

        disable_torch_init()
        model_path = config.model2path['llamavid']
        model_base = None
        load_8bit = False
        load_4bit = False
        model_name = get_model_name_from_path(model_path) # model_name='llama-vid-7b-full-224-video-fps-1'
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.conv_mode = None


    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    # 1s 采样 1帧
    def load_video(self, video_path, fps=1):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames
    

    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, inp, input_type):
        
        # => conv_mode = 'llava_v1'
        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower() or "vid" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        if video_path is not None:
            if '.mp4' in video_path or '.avi' in video_path or '.mkv' in video_path:
                image = self.load_video(video_path)
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda() # [nframe, 3, 224, 224]
                image_tensor = [image_tensor]
            else:
                image = self.load_image(video_path)
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        else:
            image_tensor = None
        
        self.model.update_prompt([[inp]])

        if video_path is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        '''
        prompt:
            A chat between a curious human and an artificial intelligence assistant. 
            The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nPlease describe the video in detail. ASSISTANT:
        '''

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() # cpu 专享
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) 
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode(): # for inference only
            output_ids = self.model.generate(input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.5,
                top_p=0.7,
                max_new_tokens=512,
                streamer=streamer, # show results in a stream manner
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        conv.messages[-2][-1] = conv.messages[-2][-1].replace(DEFAULT_IMAGE_TOKEN+'\n','')
        outputs = outputs.replace('\n', ' ').replace('\t', ' ').strip()
        if outputs.endswith('</s>'): outputs = outputs[:-len('</s>')]

        print (outputs)

        return outputs
