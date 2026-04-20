import torch
import os
import argparse
import numpy as np
from transformers import AutoTokenizer

# mplug_owl_video
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

import sys
sys.path.append('../')

import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_affectgpt.datasets.builders.image_text_pair_builder import get_name2cls # 加载所有dataset cls

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='xxx', help="evaluate dataset")
    parser.add_argument("--subtitle_flag", default='xxx', help="evaluate dataset")
    args = parser.parse_args()


    ###################################################
    ## Load model
    ###################################################
    pretrained_ckpt = config.PATH_TO_MLLM['mplug-owl']
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model = model.cuda()


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
            video_list = [video_path] 

            if args.subtitle_flag == 'subtitle':
                user_message = f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
            elif args.subtitle_flag == 'nosubtitle':
                user_message = f"As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
               
            prompts = [
            f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <|video|>
            Human: {user_message}
            AI: ''']
            generate_kwargs = {
                'do_sample': True,
                'top_k': 5,
                'max_length': 512
            }
            inputs = processor(text=prompts, videos=video_list, num_frames=4, return_tensors='pt')
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            # inputs = {k: v.float() if v.dtype == torch.float else v for k, v in inputs.items()} # cpu 专享
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    res = model.generate(**inputs, **generate_kwargs)
                response = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
                response = response.replace('\n', ' ').replace('\t', ' ').strip()
            except: # 很少的样本，视频的字幕太长会报错
                response = ""
            print (response)
            
            name2reason[name] = response

            # if ii == 0: break # for debug

        save_root = f'../output/results-{dataset.lower()}/mPLUG-Owl'
        if not os.path.exists(save_root): os.makedirs(save_root)
        save_path = f'{save_root}/results-{args.subtitle_flag}.npz'
        np.savez_compressed(save_path, name2reason=name2reason)
    
