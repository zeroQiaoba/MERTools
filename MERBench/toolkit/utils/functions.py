import os
import cv2
import tqdm
import glob
import random
import argparse
import torchaudio
from PIL import Image

import torch
import numpy as np

from toolkit.globals import *
from toolkit.utils.chatgpt import get_translate_eng2chi, get_translate_chi2eng
from toolkit.utils.read_files import func_write_key_to_csv
from toolkit.utils.read_files import func_read_key_from_csv

def func_avi_to_mp4(video_root):
    for video_path in tqdm.tqdm(glob.glob(video_root + '/*')):
        if video_path.endswith('.mp4'): continue
        assert video_path.endswith('.avi'), f'unknown video type for {video_path}'

        try:
            save_path = video_path[:-4] + '.mp4'
            cmd = '%s -y -i \"%s\" -loglevel quiet -y \"%s\"' %(config.PATH_TO_FFMPEG, video_path, save_path)
            os.system(cmd)

            cmd = 'rm -rf \"%s\"' %(video_path)
            os.system(cmd)
        except:
            print (f'error videos: {video_path}')

# gain video paths
def func_gain_videopath(video_root, vid_name):
    video_path1 = os.path.join(video_root, vid_name + '.avi')
    video_path2 = os.path.join(video_root, vid_name + '.mp4')
    if os.path.exists(video_path1):
        return video_path1
    else:
        return video_path2

def func_gain_audiopath(video_root, vid_name):
    audio_path = os.path.join(video_root, vid_name + '.wav')
    return audio_path
    
# read name2trans
def func_gain_name2trans(trans_path):
    name2trans = {}
    names = func_read_key_from_csv(trans_path, 'name')
    chis  = func_read_key_from_csv(trans_path, 'chinese')
    for ii in range(len(names)):
        name2trans[names[ii]] = chis[ii]
    return name2trans

def func_read_audio_second(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    duration = len(waveform) /  sr
    return duration

def func_opencv_to_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def func_decord_to_image(img):
    img = Image.fromarray(img)
    return img

################################################
## read face from openface outputs
def func_opencv_to_decord(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def func_video_to_face(vname):
    npy_path = os.path.join(config.PATH_TO_RAW_FACE[config.dataset], vname, vname+'.npy')
    assert os.path.exists(npy_path), f'error video has no face: {vname}'
    frames = np.load(npy_path)
    return frames

def load_video_from_npy(vname, n_frms=8, height=224, width=224, readtype='uniform', return_raw=False):

    frames = func_video_to_face(vname)
    vlen = len(frames)
    start, end = 0, vlen
    
    if readtype == 'all':
        indices = np.arange(start, end, 1).astype(int).tolist()
    elif readtype == 'uniform':
        n_frms_update = min(n_frms, vlen) # for vlen < n_frms, only read vlen
        indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    elif readtype == 'continuous': # for videomae
        ii = np.random.randint(start, max(start+1, end-n_frms))
        indices = np.arange(ii, min(end, ii+n_frms)).astype(int).tolist()
    elif readtype == 'continuous_polish': # for videomae [删除前后1s, 间隔4帧采样一次]
        start += 25 # 删除前后1s数据
        end   -= 25 
        ii = np.random.randint(start, max(start+1, end-n_frms*4))
        indices = np.linspace(ii, min(end, ii+n_frms*4), n_frms).astype(int).tolist()
 
    # whether compress into 'n_frms'
    if n_frms != 0:
        while len(indices) < n_frms:
            indices.append(indices[-1])
        indices = indices[:n_frms]
        assert len(indices) == n_frms, f'{indices}, {vlen}, {n_frms}'
    
    # indices -> images
    temp_frms = frames[indices] # [n_frms=8, 112, 112, 3]
    tgt_dim = (width, height)
    temp_frms = [cv2.resize(frm, tgt_dim)   for frm in temp_frms] # [n_frms=8, 224, 224, 3]
    temp_frms = [func_opencv_to_decord(frm) for frm in temp_frms] # [n_frms=8, 224, 224, 3]
    temp_frms = np.array(temp_frms) # [n_frms=8, 224, 224, 3]
     
    if return_raw is False:
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)
        return frms
    else:
        return temp_frms

## 临时测试，测试indexes的选择合理性 [ok]
# python toolkit/utils/functions.py debug_on_index_selection
def debug_on_index_selection(n_frms=16):
    for readtype in ['uniform', 'continuous']:
        for vlen in range(1, 100):
            print (f'{readtype}: vlen: {vlen}')
            start, end = 0, vlen
            
            if readtype == 'uniform':
                n_frms_update = min(n_frms, vlen)
                indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
            elif readtype == 'continuous':
                ii = np.random.randint(start, max(start+1, end-n_frms))
                indices = np.arange(ii, min(end, ii+n_frms)).astype(int).tolist()
            
            # get_batch -> T, H, W, C
            while len(indices) < n_frms:
                indices.append(indices[-1])

            assert len(indices) == n_frms
            print (indices)

################################################
# config -> args [只把config里存在，但是args中不存在或者为None的部分赋值]
def merge_args_config(args, config):
    args_dic = vars(args) # convert to map version
    for key in config:
        if key not in args_dic or args_dic[key] is None:
            args_dic[key] = config[key]
    args_new = argparse.Namespace(**args_dic) # change to namespace
    return args_new

# random select one param from model-tune.yaml
def func_random_select(config):
    for key in config:
        values = config[key]
        index = random.randint(0, len(values)-1)
        value = values[index]
        config[key] = value
    return config


## store into outputs
def func_update_storage(inputs, prefix, outputs):
    for key in inputs:
        val = inputs[key]
        # update key and value
        newkey = f'{prefix}_{key}'
        newval = val
        # store into outputs
        assert newkey not in outputs
        outputs[newkey] = newval


## 将openface压缩到.npy方便移动和操作
# python toolkit/utils/functions.py func_compress_openface_into_npy (face_root, save_root)
def func_compress_openface_into_npy(face_root, save_root):

    for face_dir in tqdm.tqdm(glob.glob(face_root + '/*')):
        frame_names = sorted(os.listdir(face_dir))

        frames = []
        for ii in range(len(frame_names)):
            frame_path = os.path.join(face_dir, frame_names[ii])
            frame = cv2.imread(frame_path)
            frames.append(frame)

        videoname = os.path.basename(face_dir)
        save_dir = os.path.join(save_root, videoname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, videoname+'.npy')
        np.save(save_path, frames)


# 将英文字幕翻译为中文
def func_translate_transcript(trans_path, save_path):

    names = func_read_key_from_csv(trans_path, 'name')
    engs  = func_read_key_from_csv(trans_path, 'english')
    print (f'whole sample number: {len(names)}')
    
    # translate eng2chi
    chis = []
    for eng in tqdm.tqdm(engs):
        chi = get_translate_eng2chi(eng, model='gpt-3.5-turbo-16k-0613')
        # chi = get_translate_eng2chi(eng, model='gpt-4-0613')
        chis.append(chi)

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [chis[ii], engs[ii]]
    func_write_key_to_csv(save_path, names, name2key, ['chinese', 'english'])


# 适用于先 -> english.csv -> eng2chi.csv的情况
def func_translate_transcript_polish(trans_path, save_path, polish_path):
    names1 = func_read_key_from_csv(trans_path, 'name')
    engs1  = func_read_key_from_csv(trans_path, 'english')
    names2 = func_read_key_from_csv(save_path, 'name')
    engs2  = func_read_key_from_csv(save_path, 'english')
    chis2  = func_read_key_from_csv(save_path, 'chinese')
    print (f'sample number: {len(names1)}, {len(engs1)}')
    print (f'sample number: {len(names2)}, {len(engs2)}, {len(chis2)}')

    # 将没翻译的内容再次翻译一遍
    for ii, chi2 in tqdm.tqdm(enumerate(chis2)):
        name1, name2 = names1[ii], names2[ii]
        eng1,  eng2  = engs1[ii], engs2[ii]
        assert name1 == name2 and eng1 == eng2
        if len(chi2) == 0 and len(eng2) >= 2:
            print (f'error in {names1[ii]}')
            chi2_new = get_translate_eng2chi(eng2, model='gpt-3.5-turbo-16k-0613')
            chis2[ii] = chi2_new
        if chi2.find('\n')!=-1:
            print (f'error in {names1[ii]}')

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names1):
        name2key[name] = [chis2[ii], engs2[ii]]
    func_write_key_to_csv(polish_path, names2, name2key, ['chinese', 'english'])


# 适用于 直接是 eng2chi.csv 的情况
def func_translate_transcript_polish_merge(trans_path, polish_path):
    names = func_read_key_from_csv(trans_path, 'name')
    engs  = func_read_key_from_csv(trans_path, 'english')
    chis  = func_read_key_from_csv(trans_path, 'chinese')
    print (f'process sample number: {len(names)}')

    for ii, chi in tqdm.tqdm(enumerate(chis)):
        eng = engs[ii]
        if len(chi) == 0 and len(eng) != 0:
            print (f'error in {names[ii]}')
            chis[ii] = get_translate_eng2chi(eng, model='gpt-3.5-turbo-16k-0613')
        if len(eng) == 0 and len(chi) != 0:
            print (f'error in {names[ii]}')
            engs[ii] = get_translate_chi2eng(chi, model='gpt-3.5-turbo-16k-0613')

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [chis[ii], engs[ii]]
    func_write_key_to_csv(polish_path, names, name2key, ['chinese', 'english'])


# 将audio从video中分割出来
def func_split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + '.wav')
        if os.path.exists(audio_path): continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path)
        os.system(cmd)

# 找到 split 出错的 audio data
# run -d toolkit/utils/functions.py func_find_false_audio 'dataset/meld-process/subvideo' 'dataset/meld-process/subaudio'
def func_find_false_audio(video_root, audio_root):
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(audio_root, videoname + '.wav')
        if not os.path.exists(audio_path):
            print (audio_path)

# feature -> tgt feature name
def func_name_conversion(audio, suffix):
    audio1 = featname_mapping_reverse[audio] + f'_{suffix}'
    audio2 = featname_mapping_reverse[audio] + f'-{suffix}'
    if os.path.exists(os.path.join(config.PATH_TO_FEATURES['SIMS'], audio1)):
        audio = audio1
    else:
        audio = audio2
    return audio

def func_check_feature_completeness(feature_root, intermerdia, suffix):
    # select features using (intermerdia, suffix)
    feature_names = os.listdir(feature_root)
    if suffix is not None:
        feature_names = [feature_name for feature_name in feature_names if feature_name.endswith(suffix)]
    if intermerdia is not None:
        feature_names = [feature_name for feature_name in feature_names if feature_name.find(intermerdia)!=-1]
    print (f'extracted feature numbers: {len(feature_names)}')

    for feature_name in feature_names:
        feature_dir = os.path.join(feature_root, feature_name)
        samples = glob.glob(feature_dir + '/*')
        sample_num = len(samples)
        if sample_num != 0:
            index = random.randint(0, sample_num-1)
            feature_shape = np.load(samples[index]).shape
        else:
            feature_shape = (0, 0)
        print (f'{feature_name} => shape: {feature_shape}  number: {sample_num}')


# suffix: 只统计以 suffix 为后缀的特征名
# intermerdia: 只统计内部包含 intermerdia 的特征
def check_feature_completeness(intermerdia=None, suffix=None):
    for dataset in config.PATH_TO_FEATURES:
        print (f'====== {dataset} ======')
        feature_root = config.PATH_TO_FEATURES[dataset]
        if feature_root is not None and os.path.exists(feature_root):
            func_check_feature_completeness(feature_root, intermerdia, suffix)


def func_discrte_label_distribution(labels):
    print (f'label number: {len(set(labels))}')

    label2count = {}
    for label in labels:
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1

    print ('label distribution')
    for label in sorted(label2count):
        print (label, ':', label2count[label])



#######################################
## add noise into audio
#######################################
import cv2
import wave
import array
def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params)
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()

def func_add_noise_to_audio(argv=None, clean_path=None, noise_paths=None, snr=None, save_path=None):

    clean_path, noise_paths, snr, save_path = argv

    # read clean
    clean_wav = wave.open(clean_path, "r")
    clean_amp = cal_amp(clean_wav) # (采样点, )

    # read noise [ensure len(noise) > len(audio)]
    noise_amp = []
    while True:
        noise_path = noise_paths[random.randint(0, len(noise_paths)-1)]
        noise_wav = wave.open(noise_path, "r")
        noise_amp.extend(cal_amp(noise_wav))
        if len(noise_amp) > len(clean_amp): break
    start = random.randint(0, len(noise_amp) - len(clean_amp))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    divided_noise_amp = np.array(divided_noise_amp)
    assert len(clean_amp) == len(divided_noise_amp)

    # clean + noise with snr
    clean_rms = cal_rms(clean_amp)
    noise_rms = cal_rms(divided_noise_amp)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
    mixed_amp = clean_amp + adjusted_noise_amp
    
    # avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
            reduction_rate = max_int16 / mixed_amp.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed_amp.min(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)

    save_waveform(save_path, clean_wav.getparams(), mixed_amp)


# choice_snrs = [0], [10], [20], [0, 10, 20] 这种混合snr
import multiprocessing
def add_noise_multiprocess(audio_root, choice_snrs):
    
    audio_paths = glob.glob(audio_root + '/*') # audio paths
    noise_paths = glob.glob(PATH_TO_NOISE + '/*') # noise path
    print (f'process audio: {len(audio_paths)}')
    print (f'candidate noises: {len(noise_paths)}')
    if len(choice_snrs) == 1:
        save_root = audio_root + f'_snr{choice_snrs[0]}'
    elif len(choice_snrs) == 3:
        save_root = audio_root + f'_snrmix'
    else:
        print ('Error: unsupported choice_snrs!!')
    if not os.path.exists(save_root): os.makedirs(save_root)

    params = []
    for audio_path in tqdm.tqdm(audio_paths):
        snr = choice_snrs[random.randint(0, len(choice_snrs)-1)]
        audio_name = os.path.basename(audio_path)
        save_path  = os.path.join(save_root, audio_name)
        params.append((audio_path, noise_paths, snr, save_path))

    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(func_add_noise_to_audio, params), total=len(params)))

def whether_contains_chieng():
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
    # for dataset in ['SIMSv2']:
        print (f'process dataset: {dataset}')
        label_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        chis = func_read_key_from_csv(label_path, 'chinese')
        engs = func_read_key_from_csv(label_path, 'english')
        print (f'sample number: {len(chis)}  {len(engs)}')
        print (f'eng: {engs[100]}')
        print (f'chi: {chis[100]}')


def generate_cmds_for_feature_extraction():
    whole_cmds = []
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
        for model in ['Llama-2-13b-hf', 'bloom-7b1', 'Baichuan-13B-Base', 'falcon-7b']:
            cmd = f"python extract_text_huggingface.py --dataset={dataset} --feature_level='UTTERANCE' --model_name={model} --language='english' --gpu=0"
            whole_cmds.append(cmd)
    indices = np.arange(len(whole_cmds))
    random.shuffle(indices)
    whole_cmds = np.array(whole_cmds)[indices]
    print (f'whole cmds numbers: {len(whole_cmds)}')
    for cmd in whole_cmds: print (cmd)
    return whole_cmds

def func_none_or_str(value):
    if value.lower() == "none":
        return None
    return value
