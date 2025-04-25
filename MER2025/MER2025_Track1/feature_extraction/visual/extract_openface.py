import os
import cv2
import glob
import shutil
import pathlib
import argparse
import numpy as np
from util import read_hog, read_csv

import sys
sys.path.append('../../')
import config

def generate_face_faceDir(input_root, save_root, savetype='image'):
    if savetype == 'image':
        for dir_path in sorted(glob.glob(input_root + '/*_aligned')): # 'xx/xx/000100_guest_aligned'
            frame_names = os.listdir(dir_path) # ['xxx.bmp']
            if len(frame_names) != 1: continue
            frame_path = os.path.join(dir_path, frame_names[0]) # 'xx/xx/000100_guest_aligned/xxx.bmp'
            name = os.path.basename(dir_path)[:-len('_aligned')] # '000100_guest'
            save_path = os.path.join(save_root, name + '.bmp')
            shutil.copy(frame_path, save_path)
    elif savetype == 'npy':
        frames = []
        for dir_path in sorted(glob.glob(input_root + '/*_aligned')): # 'xx/xx/000100_guest_aligned'
            frame_names = os.listdir(dir_path) # ['xxx.bmp']
            if len(frame_names) != 1: continue
            frame_path = os.path.join(dir_path, frame_names[0]) # 'xx/xx/000100_guest_aligned/xxx.bmp'
            frame = cv2.imread(frame_path)
            frames.append(frame)
        videoname = os.path.basename(input_root)
        save_path = os.path.join(save_root, videoname+'.npy')
        np.save(save_path, frames)
        
def generate_face_videoOne(input_root, save_root, savetype='image'):
    for dir_path in glob.glob(input_root + '/*_aligned'): # 'xx/xx/000100_guest_aligned'
        frame_names = sorted(os.listdir(dir_path)) # ['xxx.bmp']
        if savetype == 'image':
            for ii in range(len(frame_names)):
                frame_path = os.path.join(dir_path, frame_names[ii]) # 'xx/xx/000100_guest_aligned/xxx.bmp'
                frame_name = os.path.basename(frame_path)
                save_path = os.path.join(save_root, frame_name)
                shutil.copy(frame_path, save_path)
        elif savetype == 'npy':
            frames = []
            for ii in range(len(frame_names)):
                frame_path = os.path.join(dir_path, frame_names[ii])
                frame = cv2.imread(frame_path)
                frames.append(frame)
            videoname = os.path.basename(input_root)
            save_path = os.path.join(save_root, videoname+'.npy')
            np.save(save_path, frames)
            
def generate_hog(input_root, save_root):
    for hog_path in glob.glob(input_root + '/*.hog'):
        csv_path = hog_path[:-4] + '.csv'
        if os.path.exists(csv_path):
            hog_name = os.path.basename(hog_path)[:-4]
            _, feature = read_hog(hog_path)
            save_path = os.path.join(save_root, hog_name + '.npy')
            np.save(save_path, feature)

def generate_csv(input_root, save_root, startIdx):
    for csv_path in glob.glob(input_root + '/*.csv'):
        csv_name = os.path.basename(csv_path)[:-4]
        feature = read_csv(csv_path, startIdx)
        save_path = os.path.join(save_root, csv_name + '.npy')
        np.save(save_path, feature)


# name_npy: only process on names in 'name_npy'
def extract(input_dir, process_type, save_dir, face_dir, hog_dir, pose_dir, name_npy=None):

    # => process_names
    if name_npy is not None: # 指定特定的文件进行处理
        process_names = np.load(name_npy)
    else: # 处理所有视频文件
        vids = os.listdir(input_dir)
        process_names = [vid.rsplit('.', 1)[0] for vid in vids]
    print(f'processing names: {len(process_names)}')

    # process folders
    vids = os.listdir(input_dir)
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        saveVid = vid.rsplit('.', 1)[0] # unify folder and video names
        if saveVid not in process_names: continue

        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        input_root = os.path.join(input_dir, vid) # exists
        save_root  = os.path.join(save_dir, saveVid)
        face_root  = os.path.join(face_dir, saveVid)
        hog_root   = os.path.join(hog_dir, saveVid)
        pose_root  = os.path.join(pose_dir, saveVid)
        if os.path.exists(face_root): continue
        if not os.path.exists(save_root): os.makedirs(save_root)
        if not os.path.exists(face_root): os.makedirs(face_root)
        if not os.path.exists(hog_root):  os.makedirs(hog_root)
        if not os.path.exists(pose_root): os.makedirs(pose_root)
        if process_type == 'faceDir':
            exe_path = os.path.join(config.PATH_TO_OPENFACE_Win, 'FaceLandmarkImg.exe')
            commond = '%s -fdir \"%s\" -out_dir \"%s\"' % (exe_path, input_root, save_root)
            os.system(commond)
            # generate_face_faceDir(save_root, face_root, savetype='image') # more subtle files
            generate_face_faceDir(save_root, face_root, savetype='npy') # compress frames into npy
            # generate_hog(save_root, hog_root) # not used
            # generate_csv(save_root, pose_root, startIdx=2) # not used
            ## delete temp folder
            dir_path = pathlib.Path(save_root)
            shutil.rmtree(dir_path)
        elif process_type == 'videoOne':
            exe_path = os.path.join(config.PATH_TO_OPENFACE_Win, 'FeatureExtraction.exe')
            commond = '%s -f \"%s\" -out_dir \"%s\"' % (exe_path, input_root, save_root)
            os.system(commond)
            # generate_face_videoOne(save_root, face_root, savetype='image') # more subtle files
            generate_face_videoOne(save_root, face_root, savetype='npy') # compress frames into npy
            # generate_hog(save_root, hog_root)
            # generate_csv(save_root, pose_root, startIdx=5)
            ## delete temp folder
            dir_path = pathlib.Path(save_root)
            shutil.rmtree(dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset',  type=str, default='MER2023', help='input dataset')
    parser.add_argument('--name_npy', type=str, default=None, help='process name lists')
    parser.add_argument('--type', type=str, default='videoOne', choices=['faceDir', 'videoOne'], help='faceDir: process on facedirs; videoOne: process on one video')
    params = parser.parse_args()
    
    print(f'==> Extracting openface features...')

    # in: face dir
    dataset = params.dataset
    process_type = params.type
    input_dir = config.PATH_TO_RAW_FACE_Win[dataset]

    # out: feature csv dir
    save_dir = os.path.join(config.PATH_TO_FEATURES_Win[dataset], 'openface_all')
    hog_dir  = os.path.join(config.PATH_TO_FEATURES_Win[dataset], 'openface_hog')
    pose_dir = os.path.join(config.PATH_TO_FEATURES_Win[dataset], 'openface_pose')
    face_dir = os.path.join(config.PATH_TO_FEATURES_Win[dataset], 'openface_face')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(hog_dir):  os.makedirs(hog_dir)
    if not os.path.exists(pose_dir): os.makedirs(pose_dir)
    if not os.path.exists(face_dir): os.makedirs(face_dir)
    
    # process
    extract(input_dir, process_type, save_dir, face_dir, hog_dir, pose_dir, params.name_npy)

    print(f'==> Finish')