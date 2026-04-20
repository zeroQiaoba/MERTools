import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

import sys
sys.path.append('../')

from decord import VideoReader, cpu
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init
from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

from utils.common import *


# 他的目标是输入 (audio or text query) + video -> 输出结果;
# 因此，他只能理解 video 的信息，只不过将输入扩展到 audio 输入，同时输出也加上了 TTS 转换的模块
class VITA:
    def __init__(self, model_path):
        print ('initial vita model')

        # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
        # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
        self.max_frames = MAX_IMAGE_LENGTH  # 100

        # The number of frames retained per second in the video.
        self.video_framerate = 1

        # Sampling Parameter
        self.temperature = 0.01
        self.top_p = None
        self.num_beams = 1

        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, 'qwen2p5_instruct'
        )

        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor

        audio_encoder = model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        audio_processor = audio_encoder.audio_processor

        model.eval()

        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.model = model
        self.tokenizer = tokenizer


    # load raw video
    def _get_rawvideo_dec(
        self,
        video_path,
        image_processor,
        max_frames=MAX_IMAGE_LENGTH,
        min_frames=4,
        image_resolution=384,
        video_framerate=1,
        s=None,
        e=None,
        image_aspect_ratio="pad",
    ):
        # speed up video decode via decord.

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0.0 else 0.0
            end_time = end_time if end_time >= 0.0 else 0.0
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
                sample_pos = [
                    all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
                ]
            elif len(all_pos) < min_frames:
                sample_pos = [
                    all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
                ]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

            if image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                patch_images = [
                    expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                    for i in patch_images
                ]
                patch_images = [
                    image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                    for i in patch_images
                ]
            else:
                patch_images = [
                    image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                    for i in patch_images
                ]

            patch_images = torch.stack(patch_images)
            slice_len = patch_images.shape[0]

            return patch_images, slice_len
        else:
            print("video path: {} error.".format(video_path))


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, qs, input_type):
        
        
        ## 结合 qs 去回答 video 的问题
        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio_for_llm_lens = 60
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios = dict()
        audios["audios"] = audio.half().cuda()
        audios["lengths"] = audio_length.half().cuda()
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()


        # Check if the video exists
        video_frames, slice_len = self._get_rawvideo_dec(
            video_path,
            self.image_processor,
            max_frames=self.max_frames,
            video_framerate=self.video_framerate,
            image_aspect_ratio=getattr(self.model.config, "image_aspect_ratio", None),
        )
        image_tensor = video_frames.half().cuda()
        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
        modality = "video"
        
        conv = conv_templates['qwen2p5_instruct'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                do_sample=False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                shared_v_pid_stride=None#2#16#8#4#1#None,
            )
        output_ids = output_ids.sequences
        # outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0] 
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0][1:]

        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print(outputs)
        
        return outputs
