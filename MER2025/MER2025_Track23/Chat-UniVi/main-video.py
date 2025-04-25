import os
import torch
import argparse
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

import sys
sys.path.append('../')
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls


# 这个是1s采样一帧，而不是固定采样的
def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224, video_framerate=1, s=None, e=None):
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

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).numpy()]

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default='xxx',  help="evaluate dataset") # dataset can set to "hybird"
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()
    assert args.subtitle_flag in ['subtitle', 'nosubtitle']

    ###################################################
    ## Load model
    ###################################################
    model_path = config.PATH_TO_MLLM['Chat-UniVi']
    max_frames = 100
    video_framerate = 1

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
    
    print ('Loading model finish!!')



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

            # get prompt
            if args.subtitle_flag == 'subtitle':
                user_message = f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video and recognize all possible emotional states of the individual."
            elif args.subtitle_flag == 'nosubtitle':
                user_message = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video and recognize all possible emotional states of the individual."
            
            # Sampling Parameter
            conv_mode = "simple"
            temperature = 0.2
            top_p = None
            num_beams = 1

            # read video: video_frames: [nframe, 3, 224, 224]; slice_len=nframe
            video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=max_frames, video_framerate=video_framerate)

            if model.config.mm_use_im_start_end:
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
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
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
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.replace('\n', ' ').replace('\t', ' ').strip()
            print (prompt + '\n')
            print (outputs)

            name2reason[name] = outputs

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/Chat-UniVi'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)

