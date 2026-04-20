import os
import time
import glob
import shutil
import argparse
import numpy as np

# import config
import sys
sys.path.append('../../')
import config

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def extract(audio_files, save_dir, feature_level, finetune):

    start_time = time.time()

    # 如果有gpu的话，模型会自动加载
    if finetune:
        inference_pipeline = pipeline(task=Tasks.emotion_recognition,
                                      model="/share/home/lianzheng/tools/emotion2vec_base_finetuned")
    else:
        inference_pipeline = pipeline(task=Tasks.emotion_recognition, 
                                    model="/share/home/lianzheng/tools/emotion2vec_base", 
                                    model_revision="v2.0.4")
        
    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')

        if feature_level == 'UTTERANCE':
            rec_result = inference_pipeline(audio_file, granularity="utterance")
        else:
            rec_result = inference_pipeline(audio_file, granularity="frame")

        ## store values
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        feature = rec_result[0]['feats']
        np.save(csv_file, feature)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


'''
python extract_emotion2vec.py --dataset='MER2024' --feature_level=UTTERANCE
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MER2023', help='input dataset')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='FRAME or UTTERANCE')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether use finetuned model')
    args = parser.parse_args()

    # analyze input
    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir  = config.PATH_TO_FEATURES[args.dataset]

    # audio_files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # save_dir
    if args.finetune:
        dir_name = f'emotion2vec-finetune-{args.feature_level[:3]}'
    else:
        dir_name = f'emotion2vec-{args.feature_level[:3]}'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # extract features
    extract(audio_files, save_dir, args.feature_level, args.finetune)
