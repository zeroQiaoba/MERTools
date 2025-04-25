# *_*coding:utf-8 *_*
import os
import sys
import socket


############ For LINUX ##############
# path
DATA_DIR = {
	'MER2023': '/share/home/lianzheng/chinese-mer-2023/dataset/mer2023-dataset-process',
    'MER2023_UNLABEL': '/share/home/lianzheng/chinese-mer-2023/dataset/mer2023-dataset-unlabel',
    'IEMOCAPFour':  '/share/home/lianzheng/chinese-mer-2023/dataset/iemocap-process',
    'IEMOCAPSix':  '/share/home/lianzheng/chinese-mer-2023/dataset/iemocap-process',
    'CMUMOSI':  '/share/home/lianzheng/chinese-mer-2023/dataset/cmumosi-process',
    'CMUMOSEI': '/share/home/lianzheng/chinese-mer-2023/dataset/cmumosei-process',
    'SIMS': '/share/home/lianzheng/chinese-mer-2023/dataset/sims-process',
    'MELD': '/share/home/lianzheng/chinese-mer-2023/dataset/meld-process',
    'SIMSv2': '/share/home/lianzheng/chinese-mer-2023/dataset/simsv2-process',
    'AFFWILD2': '/share/home/lianzheng/emotion-data/affwild2',
    'MER2024': '/share/home/lianzheng/MER2024-baseline/mer2024-dataset-process',
}
PATH_TO_RAW_AUDIO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
    'MER2023_UNLABEL': None,
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
    'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subaudio'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'audio'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subaudio'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'audio'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'video'),
    'MER2023_UNLABEL': os.path.join(DATA_DIR['MER2023'], 'video'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideo-tgt'),
    'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subvideo-tgt'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subvideo'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subvideo'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'video'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subvideo'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'video'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),
}
PATH_TO_RAW_FACE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
    'MER2023_UNLABEL': os.path.join(DATA_DIR['MER2023_UNLABEL'], 'openface_face'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'openface_face'),
    'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'openface_face'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'openface_face'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'openface_face'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'openface_face'),
    'AFFWILD2': os.path.join(DATA_DIR['AFFWILD2'], 'openface_face'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription-engchi-polish.csv'),
    'MER2023_UNLABEL': None,
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription-engchi-polish.csv'),
    'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'transcription-engchi-polish.csv'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription-engchi-polish.csv'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription-engchi-polish.csv'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'transcription-engchi-polish.csv'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'transcription-engchi-polish.csv'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'transcription-engchi-polish.csv'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'transcription-merge.csv'),
}
PATH_TO_FEATURES = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features'),
    'MER2023_UNLABEL': None,
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'features'),
    'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'features'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'features'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'features'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'features'),
    'AFFWILD2': os.path.join(DATA_DIR['AFFWILD2'], 'features'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'features'),
}
PATH_TO_LABEL = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
    'MER2023_UNLABEL': None,
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'label_4way.npz'),
    'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'label_6way.npz'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'label.npz'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'label.npz'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'label.npz'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'label.npz'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'label.npz'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'label-6way.npz'),
}

# pre-trained models, including supervised and unsupervised
PATH_TO_PRETRAINED_MODELS = 'tools'
PATH_TO_OPENSMILE = 'tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = 'tools/ffmpeg-4.4.1-i686-static/ffmpeg'
PATH_TO_WENET = 'tools/wenet/wenetspeech_u2pp_conformer_libtorch'

# dir
SAVED_ROOT = os.path.join('./saved')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows ##############
DATA_DIR_Win = {
	'CMUMOSI': 'E:\\Dataset\\CMU-MOSI\\Raw',
	'CMUMOSEI': 'E:\\Dataset\\CMU-MOSEI',
	'MER2023': 'H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\mer2023-dataset-process',
    'MER2023_UNLABEL': 'G:\\mer-challenge\\mer2023-dataset',
    'SIMS': 'F:\\CH-SIMS-process',
    'IEMOCAP': 'E:\\Dataset\\iemocap-process',
    'MELD': 'E:\\Dataset\\meld-process',
    'SIMSv2': 'E:\Dataset\simsv2-process',
    'MER2024': 'H:\\desktop\\Multimedia-Transformer\\MER2024-baseline\\mer2024-dataset-process',
}

PATH_TO_RAW_FACE_Win = {
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'Video\\Segmented'),
	'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'], 'video'),
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'video'),
    'MER2023_UNLABEL': os.path.join(DATA_DIR_Win['MER2023_UNLABEL'], 'test3'),
    'SIMS': os.path.join(DATA_DIR_Win['SIMS'], 'video'),
    'IEMOCAP': os.path.join(DATA_DIR_Win['IEMOCAP'], 'subvideo-tgt'),
    'MELD': os.path.join(DATA_DIR_Win['MELD'], 'subvideo'),
    'SIMSv2': os.path.join(DATA_DIR_Win['SIMSv2'], 'video'),
    'MER2024': os.path.join(DATA_DIR_Win['MER2024'], 'video'),
}

PATH_TO_FEATURES_Win = {
	'CMUMOSI':   os.path.join(DATA_DIR_Win['CMUMOSI'],   'features'),
	'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'],  'features'),
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'features'),
    'MER2023_UNLABEL': os.path.join(DATA_DIR_Win['MER2023_UNLABEL'], 'features'),
    'SIMS': os.path.join(DATA_DIR_Win['SIMS'], 'features'),
    'IEMOCAP': os.path.join(DATA_DIR_Win['IEMOCAP'], 'features'),
    'MELD': os.path.join(DATA_DIR_Win['MELD'], 'features'),
    'SIMSv2': os.path.join(DATA_DIR_Win['SIMSv2'], 'features'),
    'MER2024': os.path.join(DATA_DIR_Win['MER2024'], 'features'),
}

PATH_TO_OPENFACE_Win = "H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\openface_win_x64"
PATH_TO_FFMPEG_Win = "H:\\desktop\\Multimedia-Transformer\\tools\\ffmpeg-3.4.1-win32-static\\bin\\ffmpeg"
