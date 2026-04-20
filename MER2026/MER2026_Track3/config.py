# *_*coding:utf-8 *_*
import os

DATA_DIR = {
    'MER2026':          'xxx/dataset/mer2026-dataset',
}
PATH_TO_RAW_AUDIO = {
    'MER2026Track3': os.path.join(DATA_DIR['MER2026'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2026Track3': os.path.join(DATA_DIR['MER2026'], 'video'),
}
PATH_TO_RAW_FACE = {
    'MER2026Track3': os.path.join(DATA_DIR['MER2026'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2026Track3': os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
}
PATH_TO_LABEL = {
    'MER2026Track3': os.path.join(DATA_DIR['MER2026'], 'track3_test.csv'),
}

## model -> model path
model2path = {
   
    'qwen3_8b':  "models/Qwen3-8B",

    'qwen25':    "models/Qwen2.5-7B-Instruct",

    'qwen25vl_7b':  "models/Qwen2.5-VL-7B-Instruct",

    'qwen25omni_7b': "models/Qwen2.5-Omni-7B",

    'qwen2audio': 'models/Qwen2-Audio-7B-Instruct',

    'llavanextvideo_7b':     'models/LLaVA-NeXT-Video-7B-hf',

    'videollava': 'models/Video-LLaVA-7B',

    'llamavid': 'models/llama-vid-7b-full-224-video-fps-1',
    
}

## model -> default input type
model2input = {

    'qwen25vl_7b':  'video',

    'qwen2audio': 'audio',

    'llavanextvideo_7b':     'video',

    'videollava':      'video',

    'llamavid':        'video',


}
