# *_*coding:utf-8 *_*
import os

AFFECTGPT_ROOT = './'
EMOTION_WHEEL_ROOT = './emotion_wheel'
OUTSIDE_WHEEL_MAPPING = os.path.join(EMOTION_WHEEL_ROOT, 'wheel_mapping.npz')
RESULT_ROOT = os.path.join(AFFECTGPT_ROOT, 'output/results')

###########################################
## 所有模型的存储路径 [放在一个路径下]
###########################################
PATH_TO_LLM = {
    'Qwen25': 'models/Qwen2.5-7B-Instruct',
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': 'models/clip-vit-large-patch14',
}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  'models/chinese-hubert-large',
}

PATH_TO_MLLM = {
    ## For Qwen-Audio
    'qwen-audio-chat':            '../models/qwen-audio-chat',
    ## For SALMONN
    'salmonn_7b':                 '../models/salmonn_7b.pth',
    'vicuna-7b-v1.5':             '../models/vicuna-7b-v1.5',
    'BEATs':                      '../models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', 
    'whisper-large-v2':           '../models/whisper-large-v2', 
    ## For Video-ChatGPT
    'video_chatgpt-7B':           '../models/video_chatgpt-7B.bin',
    'LLaVA-7B-Lightening-v1-1':   '../models/LLaVA-7B-Lightening-v1-1',
    'clip-vit-large-patch14':     '../models/clip-vit-large-patch14',
    ## For Video-LLaMA
    'llama-2-7b-chat-hf':         '../models/llama-2-7b-chat-hf',
    'imagebind_huge':             '../models/imagebind_huge.pth',
    'video_llama_vl':             '../models/VL_LLaMA_2_7B_Finetuned.pth',
    'video_llama_al':             '../models/AL_LLaMA_2_7B_Finetuned.pth',
    'blip2_pretrained_flant5xxl': '../models/blip2_pretrained_flant5xxl.pth',
    'bert-base-uncased':          '../models/bert-base-uncased',
    'eva_vit_g':                  '../models/eva_vit_g.pth',
    ## For Chat-UniVi
    'Chat-UniVi':                 '../models/Chat-UniVi',
    ## For LLaMA-VID
    'llama-vid':                  '../models/llama-vid-7b-full-224-video-fps-1',
    ## For mPLUG-Owl
    'mplug-owl':                  '../models/mplug-owl-llama-7b-video',
    ## For Otter
    'otter':                      '../models/OTTER-Video-LLaMA7B-DenseCaption',
    ## For VideoChat
    'vicuna-7b-v0':               '../models/vicuna-7b-v0',
    'videochat_7b':               '../models/videochat_7b.pth',
    ## For VideoChat2
    'umt_l16_qformer':            '../models/umt_l16_qformer.pth',
    'videochat2_7b_stage2':       '../models/videochat2_7b_stage2.pth',
    'videochat2_7b_stage3':       '../models/videochat2_7b_stage3.pth',
    ## For Video-LLaVA
    'Video-LLaVA':                '../models/Video-LLaVA-7B',
}


###################################################
## 所有数据集的存储路径 [所有标签都在 MER2026 路径下]
###################################################
DATA_DIR = {
    'MER2026':          'xxx/dataset/mer2026-dataset',
}
PATH_TO_RAW_AUDIO = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'audio'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'audio'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'video'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'video'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'video'),
}
PATH_TO_RAW_FACE = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'openface_face'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'openface_face'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
}
PATH_TO_LABEL = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'track2_train_human.csv'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'track2_train_mercaptionplus.csv'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'track2_test.csv'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
