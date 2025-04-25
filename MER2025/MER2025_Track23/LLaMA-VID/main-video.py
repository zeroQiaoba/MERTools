import argparse
import torch

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu

import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import sys
sys.path.append('../')
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# 1s 采样 1帧
def load_video(video_path, fps=1):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).numpy()
    return spare_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.5) # set to 0.5 for video
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true") # default=False
    parser.add_argument("--load-4bit", action="store_true") # default=False
    parser.add_argument("--debug", action="store_true")     # default=False
    parser.add_argument("--dataset", default='xxx', help="evaluate dataset") # dataset can set to "hybird"
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()
    print (args)
    args.model_path = config.PATH_TO_MLLM['llama-vid']
    assert args.subtitle_flag in ['subtitle', 'nosubtitle']

    ###################################################
    ## Load model
    ###################################################
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path) # model_name='llama-vid-7b-full-224-video-fps-1'
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)


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
            # => conv_mode = 'llava_v1'
            if 'llama-2' in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower() or "vid" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            args.image_file = video_path
            if args.image_file is not None:
                if '.mp4' in args.image_file or '.avi' in args.image_file or '.mkv' in args.image_file:
                    image = load_video(args.image_file)
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda() # cpu 专享
                    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'] # [nframe, 3, 224, 224]
                    image_tensor = [image_tensor]
                else:
                    image = load_image(args.image_file)
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda() # cpu 专享
                    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            else:
                image_tensor = None
            
            if args.subtitle_flag == 'subtitle':
                inp = f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
            elif args.subtitle_flag == 'nosubtitle':
                inp = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
            
            model.update_prompt([[inp]])

            if args.image_file is not None:
                # first message
                if model.config.mm_use_im_start_end:
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

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() # cpu 专享
            # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) 
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode(): # for inference only
                output_ids = model.generate(input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer, # show results in a stream manner
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            conv.messages[-2][-1] = conv.messages[-2][-1].replace(DEFAULT_IMAGE_TOKEN+'\n','')
            outputs = outputs.replace('\n', ' ').replace('\t', ' ').strip()
            if outputs.endswith('</s>'): outputs = outputs[:-len('</s>')]
            print (prompt)
            print (outputs)

            name2reason[name] = outputs

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/LLaMA-VID'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)

