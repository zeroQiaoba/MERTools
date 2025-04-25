# *_*coding:utf-8 *_*
"""
wav2vec: https://arxiv.org/abs/1904.05862
official github repo: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
"""
import os
import time
import glob
import torch
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from fairseq.models.wav2vec import Wav2VecModel # Note: use fairseq version of 0.10.1, error occurred when using the newest officical script and version of 0.10.2 (pip install fairseq==0.10.1)

# import config
import sys
sys.path.append('../../')
import config

def write_feature_to_npy(feature, csv_file, feature_level):
    if feature_level == 'UTTERANCE':
        feature = np.array(feature).squeeze() # [C,]
        if len(feature.shape) != 1: # change [T, C] => [C,]
            feature = np.mean(feature, axis=0)
        np.save(csv_file, feature)
    else:
        np.save(csv_file, feature)

def extract(audio_files, feature_level, model, save_dir, overwrite=False, gpu=None):
    start_time = time.time()
    device = torch.device(f'cuda:{gpu}')

    # create folders
    dir_name = 'wav2vec-large'
    out_dir_z = os.path.join(save_dir, f'{dir_name}-z-{feature_level[:3]}') # features output by feature encoder
    if not os.path.exists(out_dir_z):
        os.makedirs(out_dir_z)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    out_dir_c = os.path.join(save_dir, f'{dir_name}-c-{feature_level[:3]}') # features output by context network
    if not os.path.exists(out_dir_c):
        os.makedirs(out_dir_c)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # iterate audios
    for idx, wav_file in enumerate(audio_files, 1):
        file_name = os.path.basename(wav_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')
        # load audio
        audio, sampling_rate = sf.read(wav_file)
        audio = audio.astype('float32')[np.newaxis, :]
        audio = torch.from_numpy(audio)
        audio = audio.to(device)
        assert sampling_rate == 16000, f'Error: sampling rate ({sampling_rate}) != 16k!'
        with torch.no_grad():
            z = model.feature_extractor(audio) # (1, C, T), stride: 10ms (100Hz), receptive field: 30ms
            c = model.feature_aggregator(z) # (1, C, T), stride: 10ms (100Hz), receptive field: 801ms (for large version)

        # save
        z_feature = z.detach().squeeze().t().cpu().numpy()
        c_feature = c.detach().squeeze().t().cpu().numpy()
        z_csv_file = os.path.join(out_dir_z, f'{vid}.npy')
        c_csv_file = os.path.join(out_dir_c, f'{vid}.npy')
        write_feature_to_npy(z_feature, z_csv_file, feature_level)
        write_feature_to_npy(c_feature, c_csv_file, feature_level)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=7, help='index of gpu')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    args = parser.parse_args()

    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir  = config.PATH_TO_FEATURES[args.dataset]

    # in: get audios (assert file extension is '.wav')
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # load model
    device = torch.device(f'cuda:{args.gpu}')
    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'wav2vec/wav2vec_large.pt')
    cp = torch.load(model_file)
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    model.to(device)
    model.eval()

    # extract features
    extract(audio_files, feature_level=args.feature_level, model=model, save_dir=save_dir, overwrite=args.overwrite, gpu=args.gpu)
