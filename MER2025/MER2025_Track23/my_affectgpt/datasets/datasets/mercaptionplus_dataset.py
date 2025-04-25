import os
import tqdm
import random
import numpy as np
import pandas as pd

import decord
from decord import VideoReader

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from my_affectgpt.processors import transforms_video, AlproVideoTrainProcessor
from my_affectgpt.conversation.conversation_video import Conversation,SeparatorStyle
from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
from my_affectgpt.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
from toolkit.utils.functions import string_to_list

import config

# 要让模型同时支持audio, video, text三部分输入信息才行
class MERCaptionPlus_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                    dataset_cfg=None, model_cfg=None):
        
        # filter 包含两部分，一个是merg_eng只包括文本；而的ov label也只包括 textonly 抽取的 ov 标签
        self.dataset = 'MERCaptionPlus'
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data) # ['audio', 'frame', 'face']
        
        ################# 直接手动指定所有信息的存储路径 #################
        ov_path = os.path.join(config.DATA_DIR[self.dataset], 'track2_train_mercaptionplus.csv')
        name2openset = {}
        df = pd.read_csv(ov_path)
        for _, row in df.iterrows():
            name = row['name']
            openset = row['openset']
            openset = string_to_list(openset)
            if len(openset) == 0: openset = ['neutral']
            name2openset[name] = ", ".join(openset)
        self.name2openset = name2openset

        description_path = os.path.join(config.DATA_DIR[self.dataset], 'track3_train_mercaptionplus.csv')
        name2reason = {}
        df = pd.read_csv(description_path)
        for _, row in df.iterrows():
            name = row['name']
            reason = row['reason']
            name2reason[name] = reason
        self.name2reason = name2reason

        name2subtitle = {}
        subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset]
        df = pd.read_csv(subtitle_csv)
        for _, row in df.iterrows():
            name = row['name']
            subtitle = row['english']
            if pd.isna(subtitle): subtitle=""
            name2subtitle[name] = subtitle
        self.name2subtitle = name2subtitle
        
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
        face_root= config.PATH_TO_RAW_FACE[self.dataset]
        
        # you can process on filter / whole samples
        self.annotation = []
        for name in name2openset:
            self.annotation.append({'name': name, 
                                    'subtitle': name2subtitle[name], 
                                    'description': name2reason[name], 
                                    'ovlabel': name2openset[name],
                                    })
        self.label_type_candidates = ['description', 'ovlabel']
        ##################################################################

        # use base model initialize approach
        super().__init__(vis_processor=vis_processor, 
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=vis_root,
                         ann_path='',
                         face_root=face_root,
                         wav_root=wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
        
        
    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.vis_root, sample['name'] + '.mp4')
        return full_video_fp

    def _get_audio_path(self, sample):
        full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
        return full_audio_fp
    
    def _get_face_path(self, sample):
        full_face_fp = os.path.join(self.face_root, sample['name'], sample['name'] + '.npy')
        return full_face_fp
