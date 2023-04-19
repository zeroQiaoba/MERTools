# *_*coding:utf-8 *_*
import os
import sys
import socket


############ For LINUX ##############
DATA_DIR = {
	'MER2023': '/share/home/lianzheng/MER2023-Baseline-master/dataset-process',
}
PATH_TO_RAW_AUDIO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
}
PATH_TO_RAW_FACE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features'),
}
PATH_TO_LABEL = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
}

PATH_TO_PRETRAINED_MODELS = '/share/home/lianzheng/MER2023-Baseline-master/tools'
PATH_TO_OPENSMILE = '/share/home/lianzheng/MER2023-Baseline-master/tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = '/share/home/lianzheng/MER2023-Baseline-master/tools/ffmpeg-4.4.1-i686-static/ffmpeg'
PATH_TO_NOISE = '/share/home/lianzheng//MER2023-Baseline-master/tools/musan/audio-select'

SAVED_ROOT = os.path.join('./saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows (openface-win) ##############
DATA_DIR_Win = {
	'MER2023': 'H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\dataset-process',
}

PATH_TO_RAW_FACE_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'video'),
}

PATH_TO_FEATURES_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'features'),
}

PATH_TO_OPENFACE_Win = "H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\tools\\openface_win_x64"