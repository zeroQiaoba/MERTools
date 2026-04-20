import os
import math
import time
import glob
import torch
import argparse
import numpy as np
import soundfile as sf

# import config
import sys
sys.path.append('../../')
import config

from transformers import AutoModel
from transformers import WhisperFeatureExtractor, Wav2Vec2FeatureExtractor


# supported models
################## ENGLISH ######################
WAV2VEC2_BASE = 'wav2vec2-base-960h' # https://huggingface.co/facebook/wav2vec2-base-960h
WAV2VEC2_LARGE = 'wav2vec2-large-960h' # https://huggingface.co/facebook/wav2vec2-large-960h
DATA2VEC_AUDIO_BASE = 'data2vec-audio-base-960h' # https://huggingface.co/facebook/data2vec-audio-base-960h
DATA2VEC_AUDIO_LARGE = 'data2vec-audio-large' # https://huggingface.co/facebook/data2vec-audio-large

################## CHINESE ######################
HUBERT_BASE_CHINESE = 'chinese-hubert-base' # https://huggingface.co/TencentGameMate/chinese-hubert-base
HUBERT_LARGE_CHINESE = 'chinese-hubert-large' # https://huggingface.co/TencentGameMate/chinese-hubert-large
WAV2VEC2_BASE_CHINESE = 'chinese-wav2vec2-base' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-base
WAV2VEC2_LARGE_CHINESE = 'chinese-wav2vec2-large' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-large

################## Multilingual #################
WAVLM_BASE = 'wavlm-base' # https://huggingface.co/microsoft/wavlm-base
WAVLM_LARGE = 'wavlm-large' # https://huggingface.co/microsoft/wavlm-large
WHISPER_BASE = 'whisper-base' # https://huggingface.co/openai/whisper-base
WHISPER_LARGE = 'whisper-large-v2' # https://huggingface.co/openai/whisper-large-v2

## Target: avoid too long inputs
# input_values: [1, wavlen], output: [bsize, maxlen]
def split_into_batch(input_values, maxlen=16000*10):
    if len(input_values[0]) <= maxlen:
        return input_values
    
    bs, wavlen = input_values.shape
    assert bs == 1
    tgtlen = math.ceil(wavlen / maxlen) * maxlen
    batches = torch.zeros((1, tgtlen))
    batches[:, :wavlen] = input_values
    batches = batches.view(-1, maxlen)
    return batches

def extract(model_name, audio_files, save_dir, feature_level, gpu):

    start_time = time.time()

    # load model
    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    
    if model_name in [WHISPER_BASE, WHISPER_LARGE]:
        model = AutoModel.from_pretrained(model_file)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_file)
    else:
        model = AutoModel.from_pretrained(model_file)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)

    if gpu != -1:
        device = torch.device(f'cuda:{gpu}')
        model.to(device)
    model.eval()

    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')

        ## process for too short ones
        samples, sr = sf.read(audio_file)
        assert sr == 16000, 'currently, we only test on 16k audio'
        
        ## model inference
        with torch.no_grad():
            if model_name in [WHISPER_BASE, WHISPER_LARGE]:
                layer_ids = [-1]
                input_features = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_features # [1, 80, 3000]
                decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
                if gpu != -1: input_features = input_features.to(device)
                if gpu != -1: decoder_input_ids = decoder_input_ids.to(device)
                last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
                assert last_hidden_state.shape[0] == 1
                feature = last_hidden_state[0].detach().squeeze().cpu().numpy() # (2, D)
            else:
                layer_ids = [-4, -3, -2, -1]
                input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values # [1, wavlen]
                input_values = split_into_batch(input_values) # [bsize, maxlen=10*16000]
                if gpu != -1: input_values = input_values.to(device)
                hidden_states = model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
                feature = torch.stack(hidden_states)[layer_ids].sum(dim=0)  # (B, T, D) # -> compress waveform channel
                bsize, segnum, featdim = feature.shape
                feature = feature.view(-1, featdim).detach().squeeze().cpu().numpy() # (B*T, D)

        ## store values
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if feature_level == 'UTTERANCE':
            feature = np.array(feature).squeeze()
            if len(feature.shape) != 1:
                feature = np.mean(feature, axis=0)
            np.save(csv_file, feature)
        else:
            np.save(csv_file, feature)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu')
    parser.add_argument('--model_name', type=str, default='chinese-hubert-large', help='feature extractor')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='FRAME or UTTERANCE')
    parser.add_argument('--dataset', type=str, default='MER2023', help='input dataset')
    # ------ 临时测试SNR对于结果的影响 ------
    parser.add_argument('--noise_case', type=str, default=None, help='extract feature of different noise conditions')
    # ------ 临时测试 tts audio 对于结果的影响 -------
    parser.add_argument('--tts_lang', type=str, default=None, help='extract feature from tts audio, [chinese, english]')
    args = parser.parse_args()

    # analyze input
    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]
    if args.noise_case is not None:
        audio_dir += '_' + args.noise_case
    if args.tts_lang is not None:
        audio_dir += '-' + f'tts{args.tts_lang[:3]}16k'

    # audio_files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # save_dir
    if args.noise_case is not None:
        dir_name = f'{args.model_name}-noise{args.noise_case}-{args.feature_level[:3]}'
    elif args.tts_lang is not None:
        dir_name = f'{args.model_name}-tts{args.tts_lang[:3]}-{args.feature_level[:3]}'
    else:
        dir_name = f'{args.model_name}-{args.feature_level[:3]}'

    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # extract features
    extract(args.model_name, audio_files, save_dir, args.feature_level,gpu=args.gpu)

