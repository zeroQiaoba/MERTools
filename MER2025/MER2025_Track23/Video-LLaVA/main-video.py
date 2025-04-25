import torch
import argparse
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import os
import numpy as np

import sys
sys.path.append('../')

import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='xxx', help="evaluate dataset") # dataset can set to "hybird"
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()


    ###################################################
    ## Load model
    ###################################################
    disable_torch_init()
    model_path = config.PATH_TO_MLLM['Video-LLaVA']
    cache_dir = None
    device = 'cuda' # cpu 专享 
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path) # model_name='Video-LLaVA-7B'
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']


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
            if args.subtitle_flag == 'subtitle':
                inp = f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
            elif args.subtitle_flag == 'nosubtitle':
                inp = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
            
            video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values'] # [1, 3, 8, 224, 224] => 采样数为8帧
            if type(video_tensor) is list:
                tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
            else:
                tensor = video_tensor.to(model.device, dtype=torch.float16)
            # tensor = tensor.float() # cpu 专享
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles
            inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            print ('===============')
            print('\n', prompt, '\n')
            print ('===============')
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() # cpu 专享
            # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            outputs = outputs.replace('\n', ' ').replace('\t', ' ').strip()
            if outputs.endswith('</s>'): outputs = outputs[:-len('</s>')]
            print (outputs)
            
            name2reason[name] = outputs

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/Video-LLaVA'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)

