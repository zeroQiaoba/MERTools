import re
import os
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
import numpy as np
import pandas as pd
import multiprocessing

import config

#######################################
## add noise into audio + video
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

def add_noise_to_audio(clean_path, noise_paths, snr, save_path):

    # read clean
    clean_wav = wave.open(clean_path, "r")
    clean_amp = cal_amp(clean_wav)

    # read noise
    noise_amp = []
    noise_info = []
    while True:
        noise_path = noise_paths[random.randint(0, len(noise_paths)-1)]
        noise_wav = wave.open(noise_path, "r")
        noise_amp.extend(cal_amp(noise_wav))
        noise_info.append(noise_path)
        if len(noise_amp) > len(clean_amp): break
    start = random.randint(0, len(noise_amp) - len(clean_amp))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    divided_noise_amp = np.array(divided_noise_amp)

    # clean + noise
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
    info = {'noise_info': noise_info, 'snr': snr}
    return info

def func_blur(frame, blur_rate):
    blur_time = int(math.log2(blur_rate))
    for ii in range(blur_time):
        frame = cv2.pyrDown(frame)
    for ii in range(blur_time):
        frame = cv2.pyrUp(frame)
    return frame

def add_noise_to_video(video_path, blur_rate, save_path):

    ## read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    assert ret == True, f'video should be readable'
    blur_frame = func_blur(frame, blur_rate)
    height, width, _ = blur_frame.shape
    cap.release()

    ## blur images
    cap = cv2.VideoCapture(video_path)
    videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (int(width), int(height)), True)
    while True:
        ret, frame = cap.read()
        if ret == False: break
        blur_frame = func_blur(frame, blur_rate)
        videoWriter.write(blur_frame)
    cap.release()
    videoWriter.release()

    info = {'blur_rate': blur_rate}
    return info

def split_audio_from_video(video_path, audio_path):
    cmd = '%s -i \"%s\" -ac 1 -ar 16000 -loglevel quiet -y \"%s\"' %(config.PATH_TO_FFMPEG, video_path, audio_path)
    os.system(cmd)

def merge_audio_and_video(audio_path, video_path, save_path):
    cmd = '%s -i \"%s\" -i \"%s\" -vcodec copy -acodec copy -c:v copy -c:a aac -loglevel quiet -y \"%s\"' %(config.PATH_TO_FFMPEG, audio_path, video_path, save_path)
    os.system(cmd)

## convert to multiprocess version
def func_mixture(argv=None, video_path=None, choice_snrs=None, choice_blurs=None, noise_paths=None, save_root=None):

    if argv != None:
        video_path, choice_snrs, choice_blurs, noise_paths, save_root = argv

    snr = choice_snrs[random.randint(0, len(choice_snrs)-1)]
    blur = choice_blurs[random.randint(0, len(choice_blurs)-1)]

    ## process for one video
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    save_path1 = os.path.join(save_root, f"{video_name}.wav")
    save_path2 = os.path.join(save_root, f"{video_name}-noise.wav")
    save_path3 = os.path.join(save_root, f"{video_name}-noise.avi")
    save_path4 = os.path.join(save_root, f"{video_name}.mp4")
    split_audio_from_video(video_path, save_path1) # video -> audio [16k + 1channel audio]
    info_a_noise = add_noise_to_audio(save_path1, noise_paths, snr, save_path2) # add noise to audio
    info_v_noise = add_noise_to_video(video_path, blur, save_path3) # add noise to video
    merge_audio_and_video(save_path2, save_path3, save_path4) # add noise to video
    os.system('rm -rf \"%s\" \"%s\" \"%s\"' %(save_path1, save_path2, save_path3)) # remove unnecessary files

    noise_info = {}
    noise_info['video_name']   = video_name
    noise_info['audio_snr']    = info_a_noise['snr']
    noise_info['audio_noises'] = info_a_noise['noise_info']
    noise_info['video_blur']   = info_v_noise['blur_rate']
    return noise_info

def main_mixture_multiprocess(video_root, save_root, debug=False):
    choice_blurs = [1, 2, 4]
    choice_snrs = np.arange(5, 11)
    video_paths = glob.glob(video_root + '/*')
    noise_paths = glob.glob(config.PATH_TO_NOISE + '/*')
    if not os.path.exists(save_root): os.makedirs(save_root)
    if debug: video_paths = video_paths[:100]

    params = []
    for video_path in video_paths:
        params.append((video_path, choice_snrs, choice_blurs, noise_paths, save_root))

    noise_info = []
    with multiprocessing.Pool(processes=8) as pool:
        noise_info = list(tqdm.tqdm(pool.imap(func_mixture, params), total=len(params)))
    return noise_info

if __name__ == '__main__':
    import fire
    fire.Fire()