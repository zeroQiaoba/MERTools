import re
import os
import sys
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd

import cv2 # pip install opencv-python
import config

# split audios from videos
def split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + '.wav')
        if os.path.exists(audio_path): continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path)
        os.system(cmd)

# preprocess dataset-release
def normalize_dataset_format(data_root, save_root):
    ## input path
    train_data  = os.path.join(data_root, 'train')
    train_label = os.path.join(data_root, 'train-label.csv')
    test1_data  = os.path.join(data_root, 'test1')
    test2_data  = os.path.join(data_root, 'test2')
    test3_data  = os.path.join(data_root, 'test3')

    ## output path
    save_video = os.path.join(save_root, 'video')
    save_label = os.path.join(save_root, 'label-6way.npz')
    if not os.path.exists(save_root): os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## move all videos into save_video
    for temp_root in [train_data, test1_data, test2_data, test3_data]:
        video_paths = glob.glob(temp_root + '/*')
        for video_path in tqdm.tqdm(video_paths):
            video_name = os.path.basename(video_path)
            new_path = os.path.join(save_video, video_name)
            shutil.copy(video_path, new_path)

    ## generate label path
    train_corpus = {}
    df_label = pd.read_csv(train_label)
    for _, row in df_label.iterrows(): ## read for each row
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        train_corpus[name] = {'emo': emo, 'val': val}

    test1_corpus = {}
    for video_path in glob.glob(test1_data + '/*'):
        video_name = os.path.basename(video_path)[:-4]
        test1_corpus[video_name] = {'emo': 'neutral', 'val': -10} # for convenience

    test2_corpus = {}
    for video_path in glob.glob(test2_data + '/*'):
        video_name = os.path.basename(video_path)[:-4]
        test2_corpus[video_name] = {'emo': 'neutral', 'val': -10} # for convenience

    test3_corpus = {}
    for video_path in glob.glob(test3_data + '/*'):
        video_name = os.path.basename(video_path)[:-4]
        test3_corpus[video_name] = {'emo': 'neutral', 'val': -10} # for convenience

    np.savez_compressed(save_label,
                        train_corpus=train_corpus,
                        test1_corpus=test1_corpus,
                        test2_corpus=test2_corpus,
                        test3_corpus=test3_corpus)

# generate transcription files using asr
def generate_transcription_files_asr(audio_root, save_path):
    import torch
    import wenetruntime as wenet
    decoder = wenet.Decoder('./tools/wenet/wenetspeech_u2pp_conformer_libtorch', lang='chs')

    names = []
    sentences = []
    for audio_path in tqdm.tqdm(glob.glob(audio_root + '/*')):
        name = os.path.basename(audio_path)[:-4]
        sentence = decoder.decode_wav(audio_path)
        sentence = sentence.split('"')[5]
        names.append(name)
        sentences.append(sentence)

    ## write to csv file
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_path, index=False)

# add punctuation to transcripts
def refinement_transcription_files_asr(old_path, new_path):
    from paddlespeech.cli.text.infer import TextExecutor
    text_punc = TextExecutor()

    ## read 
    names, sentences = [], []
    df_label = pd.read_csv(old_path)
    for _, row in df_label.iterrows():
        names.append(row['name'])
        sentence = row['sentence']
        if pd.isna(sentence):
            sentences.append('')
        else:
            sentence = text_punc(text=sentence)
            sentences.append(sentence)
        print (sentences[-1])

    ## write
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(new_path, index=False)

if __name__ == '__main__':
    import fire
    fire.Fire()