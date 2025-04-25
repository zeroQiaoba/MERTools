import os
import tqdm
import copy
import random
import pandas as pd
from typing import Dict, Optional, Sequence, Iterable

import torch
from torch.utils.data import Dataset, ConcatDataset
from my_affectgpt.models.tokenizer import load_tokenizer_from_LLM

import torch
from PIL import Image
import numpy as np

import transformers
from my_affectgpt.processors.video_processor import load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
import config

class BaseDataset():
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None, model_cfg=None, dataset_cfg=None,
                vis_root=None, ann_path=None, wav_root=None, face_root=None, img_root=None):
        
        ####################################
        ## part1: common ones
        self.vis_root = vis_root
        self.img_root = img_root
        self.wav_root = wav_root
        self.ann_path = ann_path
        self.face_root = face_root
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor
        self.img_processor = img_processor
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg

        self.image_caption_prompt_candidates = ["Describe this image in detail.",
                                                "Take a look at this image and describe what you notice.",
                                                "Please provide a detailed description of the picture.",
                                                "Could you describe the contents of this image for me?"]

        self.audio_caption_prompt_candidates = ["Describe this audio in detail.",
                                                "Listen to this audio and describe what you hear.",
                                                "Please provide a detailed description of this audio.",
                                                "Could you describe the contents of this audio for me?"]

        ####################################
        ## part2: (model_cfg, dataset_cfg) specific ones
        if model_cfg is None or dataset_cfg is None: return
        
        self.max_length = model_cfg.max_length
        self.num_video_query_token = model_cfg.num_video_query_token
        self.num_audio_query_token = model_cfg.num_audio_query_token
        self.num_multi_query_token = model_cfg.num_multi_query_token
        self.num_image_query_token = model_cfg.num_image_query_token

        ## 控制视频采样的帧数
        self.n_frms = model_cfg.vis_processor.train.n_frms

        # 这里token的设置和 affectgpt.py 中的一致 (所以这部分调用改成全局调用了)
        self.tokenizer = load_tokenizer_from_LLM(model_cfg.llama_model)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_AUDIO_PATCH_TOKEN]
        self.FRAME_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_FRAME_PATCH_TOKEN]
        self.FACE_PATCH_TOKEN_ID  = self.tokenizer.get_vocab()[config.DEFAULT_FACE_PATCH_TOKEN]
        self.MULTI_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_MULTI_PATCH_TOKEN]
        
        # 让模型只读取一定比例的文件
        if 'ratio' in dataset_cfg and dataset_cfg.ratio < 1:
            self.annotation = self.func_random_sample_subset(self.annotation, ratio=dataset_cfg.ratio)
            print(f'after sampled sample number: {len(self.annotation)}')

        ####################################
        ## part3: debug
        sample1 = self.__getitem__(random.randint(0, len(self)-1))
        sample2 = self.__getitem__(random.randint(0, len(self)-1))
        sample3 = self.__getitem__(random.randint(0, len(self)-1))
        self.func_visualize_samples(sample1)
        self.func_visualize_samples(sample2)
        self.func_visualize_samples(sample3)
        samples = [sample1, sample2, sample3]
        self.collater(samples)

        ## debug2: for all datasets (whether contains errors)
        # print ('Debug: whether all data are readable?')
        # for index in tqdm.tqdm(range(len(self))):
        #     sample = self.__getitem__(index)
        #     self.func_visualize_samples(sample)
        #     # print (sample['raw_audio'].shape)

        ## debug3: short version, only length
        print ('training sample number: ', len(self))
        ####################################

    def __len__(self):
        return len(self.annotation)
    
    def func_visualize_samples(self, sample):
        text_input = copy.deepcopy(sample['text_input'])
        input_convert = self.tokenizer.decode(text_input)
        print (input_convert)

        label = copy.deepcopy(sample['label'])
        label[label==config.IGNORE_INDEX] = self.tokenizer.bos_token_id
        output_convert = self.tokenizer.decode(label)
        print (output_convert)
    
    # to_token_ids: 开头不增加特殊符号，裁剪输入保证不超过 max_length
    def to_token_ids(self, text, max_length):
        input_ids = self.tokenizer(text, return_tensors="pt", padding="longest", max_length=max_length, 
                                truncation=True, add_special_tokens=False).input_ids[0]
        return input_ids


    def func_map_valence_to_emotion(self, valence):
        if valence > 0:
            return 'positive'
        elif valence < 0:
            return 'negative'
        else:
            return 'neutral'
        

    def get_cur_label_type(self, label_type_candidates, label_type):
        if label_type == 'hybird':
            index = random.randint(0, len(label_type_candidates) -1)
            return label_type_candidates[index]
        else:
            assert label_type in label_type_candidates, f'error label type: {label_type} not in {label_type_candidates}'
            return label_type
        
    
    def func_random_prompts(self, candidates):
        index = random.randint(0, len(candidates) - 1)
        prompt = candidates[index]
        return prompt
    
    
    # 随机采样一个 annotations
    def func_random_sample_subset(self, annotations, ratio=0.1):
        annotations_subset = random.sample(annotations, int(len(annotations)*ratio))
        return annotations_subset


    ###########################################################
    ## 数据读取部分操作
    ###########################################################
    # all types: {audio, frame, face, image}
    def get_needed_data(self, face_or_frame):
        if face_or_frame == 'faceframe': # (face, frame, audio, text)
            needed_data = ['audio', 'frame', 'face']
        elif face_or_frame == 'face': # (face, audio, text)
            needed_data = ['audio', 'face']
        elif face_or_frame == 'frame': # (frame, audio, text)
            needed_data = ['audio', 'frame']
        elif face_or_frame == 'audioonly': # (audio)
            needed_data = ['audio']
        elif face_or_frame == 'textonly':  # (text)
            needed_data = []
        elif face_or_frame == 'faceonly':  # (face)
            needed_data = ['face']
        elif face_or_frame == 'frameonly': # (frame)
            needed_data = ['frame']
        elif face_or_frame == 'multiface_text': # (multi, text)
            needed_data = ['face', 'audio']
        elif face_or_frame == 'multiface_audio_face_text': # (multi, face, audio, text)
            needed_data = ['face', 'audio']
        elif face_or_frame == 'image': # (image)
            needed_data = ['image']
        elif face_or_frame == 'multiframe_audio_frame_text': # (multi, face, audio, text)
            needed_data = ['frame', 'audio']
        elif face_or_frame == 'multiface_audio_face_frame_text': # (multi, face, audio, text)
            needed_data = ['frame', 'face', 'audio']
        elif face_or_frame == 'audio_text': # (audio, text)
            needed_data = ['audio']
        elif face_or_frame == 'face_text': # (face, text)
            needed_data = ['face']
        elif face_or_frame == 'frame_text': # (frame, text)
            needed_data = ['frame']
        return needed_data
    

    def read_frame_face_audio_text(self, video_path=None, face_npy=None, audio_path=None, image_path=None):

        sample_data = {}

        # step1: read (raw_frame, frame)
        frame, raw_frame = None, None
        if video_path is not None and 'frame' in self.needed_data:
            raw_frame, msg = load_video(
                video_path=video_path,
                n_frms = self.n_frms,
                height = 224,
                width  = 224,
                sampling ="uniform",
                return_msg = True
            )
            frame = self.vis_processor.transform(raw_frame) # [3, 8, 224, 224] # 建议可视化，看看这部分数据扩增是否合适
        sample_data['frame'] = frame
        sample_data['raw_frame'] = raw_frame
        # print (sample_data)

        # step2: read (raw_face, face)
        face, raw_face = None, None
        if face_npy is not None and 'face' in self.needed_data:
            raw_face, msg = load_face(
                face_npy=face_npy,
                n_frms = self.n_frms,
                height = 224,
                width  = 224,
                sampling ="uniform",
                return_msg=True
            )
            face = self.vis_processor.transform(raw_face) # [3, 8, 224, 224] # 建议可视化，看看这部分数据扩增是否合适
        sample_data['face'] = face
        sample_data['raw_face'] = raw_face
        # print (sample_data)

        # step3: read audio [需要针对没有 audio track 的 video 进行额外处理]
        audio, raw_audio = None, None
        if audio_path is not None and 'audio' in self.needed_data:
            raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0] # [8, 1, 16000*2s]
            audio = transform_audio(raw_audio, "cpu") # [8, 1, 128, 204]
        sample_data['audio'] = audio
        sample_data['raw_audio'] = raw_audio
        # print (sample_data)
        
        # step4: read image
        image, raw_image = None, None
        if image_path is not None and 'image' in self.needed_data:
            ###### 支持两种类型的 image_path 输入 ######
            if not isinstance(image_path, Image.Image):
                raw_image = Image.open(image_path)
            else:
                raw_image = image_path
            ##########################################
            ## image process
            image = self.img_processor(raw_image.convert("RGB")) # [3, 224, 224] 这是 vis processor 默认下的处理，正常情况其实也不需要这个内容
            image = image.unsqueeze(dim=1) # [3, 1, 224, 224]
            raw_image = torch.from_numpy(np.array(raw_image.resize((224, 224)))) # [H, W, C] => 可能因为llava中的图片有些并不是一样大小的，使得转换过程中有些
            raw_image = raw_image.permute(2, 0, 1).unsqueeze(dim=1).float() # (C, T=1, H, W)
        sample_data['image'] = image
        sample_data['raw_image'] = raw_image
        # print (sample_data)

        return sample_data


    ###########################################################
    ## QA 获取
    ###########################################################
    ## 建立一个 qa 读取器，用于后续统一化的处理
    def func_get_qa_description(self, sample, question_only=False):
        question = "Please infer the person's emotional state and provide your reasoning process."

        if question_only:
            return question
        else:
            return {
                'question': question, 
                'answer':sample['description'],
                }
    
    def func_get_qa_ovlabel(self, sample, question_only=False):
        question = "Please recognize all possible emotional states of the character."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's emotional state is {sample['ovlabel']}."
                }
    
    def func_get_qa_onehot_w_candidates(self, sample, question_only=False):
        question = f"Please select the label that can best describe the person's emotional state from the provided candidate labels: {self.candidate_labels}."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':   f"The most likely label is {sample['onehot']}."
                }

    def func_get_qa_onehot_wo_candidates(self, sample, question_only=False):
        question = "Please recognize the character's most likely emotional state."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's emotional state is {sample['onehot']}."
                }

    def func_get_qa_valence(self, sample, question_only=False):
        question = f"Please identify the overall positive or negative emotional polarity of the main characters. " \
                 + f"The output should be a ﬂoating-point number ranging from {self.minval} to {self.maxval}. " \
                 + f"Here, {self.minval} indicates extremely negative emotions, 0 indicates neutral emotions, and {self.maxval} indicates extremely positive emotions. " \
                 + f"Please provide your judgment as a ﬂoating-point number."
        
        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer': 'The valence score is %.2f.' %(sample['valence']),
                }

    def func_get_qa_sentiment(self, sample, question_only=False):
        question = "Please select the most likely sentiment label that can best describe the person's emotional state: positive, negative, neutral."
        
        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's sentiment state is {sample['sentiment']}.",
                }

    def func_get_qa_direct(self, sample):
        return {
            'question': sample['question'],
            'answer':   sample['answer'],
            }
    
    def func_get_qa_caption(self, sample, modality):
        if modality == 'image':
            return {
            'question': self.func_random_prompts(self.image_caption_prompt_candidates),
            'answer':   sample['caption'],
            }
        elif modality == 'audio':
            return {
            'question': self.func_random_prompts(self.audio_caption_prompt_candidates),
            'answer':   sample['caption'],
            }
    
    def func_get_qa_preference(self, sample):

        a1 = sample['preference']['a1']
        a2 = sample['preference']['a2']
        p  = sample['preference']['p']

        question = f"We provide two descriptions. a1: {a1} \t\t\t a2: {a2} Please select the one that best matches the video content."
        
        assert p in ['a1', 'a2', 'same']
        if p in ['a1', 'a2']:
            answer = f"The best one is {p}."
        else:
            answer = f'These two sentences describe the content of the video with the same accuracy.'

        return {
            'question': question,
            'answer':   answer,
            }

    # this (q, a) is used to determinate the reward value
    def func_get_description_reward(self, sample):
        reason = sample['description']
        reward = sample['reward']

        question = f"We have provided a description: {reason} \t\t\t Please evaluate and decide whether to accept or reject this description based on its alignment with the video content."

        assert reward in ['accept', 'reject']
        answer = f'{reward} this sentence.'

        return {
            'question': question,
            'answer':   answer,
        }

    ## 获取 <question, answer> 用于后续训练
    def get_qa_pairs(self, dataset, label_type, sample):
        
        '''
        self.  -> 数据集全局的内容
        sample -> 样本局部的内容
        '''
        # EMERFine 指的是 (training set) 那 332 samples，同时包含 ovlabel/description
        if dataset in ['EMERCoarse', 'EMERFine']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
            }
        
        elif dataset in ['EMERCoarseFilter']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
            }
        
        elif dataset in ['MERCaptionPlus', 'OVMERD']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
            }
        
        elif dataset in ['Preference']: # 带 preference 优化
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
                'preference':  self.func_get_qa_preference(sample),
            }

        elif dataset in ['Preference2', 'Preference4']: # 不带 preference 优化
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
            }
        
        elif dataset in ['Preference3']: # 不带 preference 优化
            candidates = {
                'reward': self.func_get_description_reward(sample),
            }
        
        ## case1: Zebang's labels
        elif dataset in ['MERRCoarse', 'MERRFine', 'MAFW']:
            candidates = {
                'description': self.func_get_qa_description(sample),
            }

        ## case2: onehot labels
        elif dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour']:
            candidates = {
                'onehot_w_candidates':  self.func_get_qa_onehot_w_candidates(sample),
                'onehot_wo_candidates': self.func_get_qa_onehot_wo_candidates(sample),
            }

        ## case3: valence scores
        elif dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']:
            candidates = {
                'valence':   self.func_get_qa_valence(sample),
                'sentiment': self.func_get_qa_sentiment(sample),
            }

        ## case4: instruction dataset
        elif dataset in ['VideoChat', 'LLaVA', 'EmoVIT']:
            candidates = {
                'qa':  self.func_get_qa_direct(sample),
            }

        elif dataset in ['MiniGPT4']:
            candidates = {
                'caption': self.func_get_qa_caption(sample, 'image'),
            }

        elif dataset in ['WavCaps', 'TextrolSpeech', 'PromptSpeech']:
            candidates = {
                'caption': self.func_get_qa_caption(sample, 'audio'),
            }

        return candidates[label_type] # 包含 question, answer 两部分内容


    def get_prompt_for_multimodal(self, face_or_frame, subtitle, user_message):

        # step5: get prompts for differet cases [可能存在三种数据加载情况，从而能够扩展至4种模态输入]
        if face_or_frame == 'faceframe': # (face, frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"Additionally, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'face': # (face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'frame': # (frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'audioonly': # (audio)
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'textonly':  # (text)
            assert subtitle is not None
            prompt = f"###Human: The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'faceonly':  # (face)
            prompt = f"###Human: We uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'frameonly': # (frame)
            prompt = f"###Human: We uniformly sample raw frames from the video: <Video><FrameHere></Video>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'image': # (image)
            prompt = f"###Human: The image content is as follows: <Image><ImageHere></Image>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        
        ## 这部分是为了和其他 MLLM 进行公平比较，所进行的 ablation study 部分
        elif face_or_frame == 'audio_text': # (audio, text)
            assert subtitle is not None
            prompt =  f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'face_text': # (audio, text)
            assert subtitle is not None
            prompt =  f"We uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'frame_text': # (audio, text)
            assert subtitle is not None
            prompt =  f"we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
            
        ## 后面都是增加 <Multi> token 后的结果    
        elif face_or_frame == 'multiface_text': # (multi, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_text': # (multi, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiframe_audio_frame_text': # (multi, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_frame_text': # (multi, frame, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        return prompt
    
    ## 替换 <FaceHere> / <FrameHere> / <AudioHere> / <ImageHere> / <MultiToken>
    def replace_token_for_multimodal(self, prompt):

        replace_token = config.DEFAULT_FRAME_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FRAME_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_FACE_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FACE_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_AUDIO_PATCH_TOKEN * self.num_audio_query_token
        prompt = prompt.replace(config.DEFAULT_AUDIO_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_MULTI_PATCH_TOKEN * self.num_multi_query_token
        prompt = prompt.replace(config.DEFAULT_MULTI_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_IMAGE_PATCH_TOKEN * self.num_image_query_token
        prompt = prompt.replace(config.DEFAULT_IMAGE_PATCH_TOKEN, replace_token)
        return prompt


    ####################################################################################
    ## 读取一个样本 (read one sample)
    ####################################################################################
    def __getitem__(self, index):
        num_retries = 10 # skip error or too long videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                cur_label_type = self.get_cur_label_type(self.label_type_candidates, self.label_type)
                # print ('cur_label_type: ', cur_label_type)

                # step1: read needed data
                video_path, image_path, audio_path, face_npy = None, None, None, None
                if hasattr(self, '_get_video_path'): video_path = self._get_video_path(sample)
                if hasattr(self, '_get_image_path'): image_path = self._get_image_path(sample)
                if hasattr(self, '_get_audio_path'): audio_path = self._get_audio_path(sample)
                if hasattr(self, '_get_face_path'):  face_npy   = self._get_face_path(sample)
                # print (video_path, image_path, audio_path, face_npy)
                sample_data = self.read_frame_face_audio_text(video_path, face_npy, audio_path, image_path)

                # step2: read (question, answer)
                # => 如果 sample 中缺少 qa 对应内容的信息，结果是会报错的
                qa_pair = self.get_qa_pairs(self.dataset, cur_label_type, sample)
                # print (qa_pair)

                # step4: generate (text_input, label)
                if 'subtitle' not in sample: sample['subtitle'] = None
                prompt = self.get_prompt_for_multimodal(self.face_or_frame, sample['subtitle'], qa_pair['question']) # get prompt
                prompt = self.replace_token_for_multimodal(prompt) # replace specific tokens
                # print (prompt)

                ## tokenizer [每部分内容不能超过 self.max_length, 且两部分内容的和也不能超过 self.max_length]
                prompt_id = self.to_token_ids(prompt, self.max_length) # => 避免 GPU OOM
                
                target = qa_pair['answer'] + '###'
                # print (target)
                target_id = self.to_token_ids(target, self.max_length)

                text_input = torch.cat([prompt_id, target_id])
                label = torch.cat([torch.ones([len(prompt_id)], dtype=text_input.dtype) * -100, target_id])
                assert len(text_input) == len(label)
                if len(text_input) > self.max_length:
                    raise RuntimeError("too long text_input")
            except Exception as error:
                print(f'Error: {error}')
                print(f"Failed to load data {self.dataset} {sample['name']}. We will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        return {
            "face": sample_data['face'],           # [c=3, frame=8, 224, 224] [这个经过了transformer变换]
            "raw_face": sample_data['raw_face'],   # [c=3, frame=8, 224, 224]

            "frame": sample_data['frame'],         # [c=3, frame=8, 224, 224] [这个经过了transformer变换]
            "raw_frame": sample_data['raw_frame'], # [c=3, frame=8, 224, 224]

            "audio": sample_data['audio'],          # [frame=8, c=1, 128, 204]
            "raw_audio": sample_data['raw_audio'],  # [frame=8, c=1, 16000*2采样点]

            "image": sample_data['image'],
            "raw_image": sample_data['raw_image'],

            "label": label,
            "text_input": text_input,
            'dataset': self.dataset.lower(),
            'face_or_frame': self.face_or_frame,
        }

        
    ####################################################################################
    ## batch 级别数据合并
    ####################################################################################
    def collater(self, instances):
        '''
        llama token ids:
            <unk>: 0
            bos|<s>: 1
            eos|pad|</s>: 2
            <ImageHere>: 32000
            <AudioHere>: 32001

        data_dict:  input_ids:[###Human: <Image> <ImageHere>*32 /<Image> xxx  ...   ###Assistant: xxx###Human: xxx###Assistant: xxx###]
                    labels:   [-100..., -100, ....,                                 ...           xxx###-100...,        ...     xxx###]

        data_dict:  input_ids:[<bos>###Human: <Image> <ImageHere>*32 /<Image> xxx  ...   ###Assistant: xxx###Human: xxx###Assistant: xxx###, <eos>,    ...]
                    labels:   [-100..., -100, ....,                                 ...                xxx###-100...,        ...     xxx###, -100, ...]
                    images:   [bs=3, c=3, 224, 224]
        '''
        labels = []
        input_ids = []
        for instance in instances:
            label = instance['label']
            input_id = instance['text_input']
            label    = torch.cat([torch.ones([1], dtype=input_id.dtype) * config.IGNORE_INDEX, label,
                                  torch.ones([1], dtype=input_id.dtype) * self.tokenizer.eos_token_id]) # (-100  xxx <eos>)
            input_id = torch.cat([torch.ones([1], dtype=input_id.dtype) * self.tokenizer.bos_token_id, input_id,
                                  torch.ones([1], dtype=input_id.dtype) * self.tokenizer.eos_token_id]) # (<bos> xxx <eos>)
            labels.append(label)
            input_ids.append(input_id)

        # pad bacth input into the same length 
        # => input_ids: <bos> xxx <eos> <pad>
        # => label    : -100  xxx <eos> -100
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, 
                                                    batch_first=True, 
                                                    padding_value=self.tokenizer.pad_token_id)
        labels    = torch.nn.utils.rnn.pad_sequence(labels,    
                                                    batch_first=True, 
                                                    padding_value=config.IGNORE_INDEX)
        batch = dict(
            labels=labels,
            input_ids=input_ids,
            attention_masks=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 后面跟着的是 dataset 中所有数据类型
        # => 只有符合约束，才把这部分数据存储在 batch 里面，如果有问题，直接就不存储
        for sample_type in ['face', 'raw_face', 'frame', 'raw_frame', 'audio', 'raw_audio', 'image', 'raw_image']:
            batch_type = sample_type + 's'

            if sample_type in instances[0]:
                datas = [instance[sample_type] for instance in instances]
                if all(x is not None and x.shape == datas[0].shape for x in datas):
                    batch[batch_type] = torch.stack(datas)
        
        batch['dataset'] = instances[0]['dataset']
        batch['face_or_frame'] = instances[0]['face_or_frame']
        return batch
    

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
