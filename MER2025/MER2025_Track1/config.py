# *_*coding:utf-8 *_*
import os

PATH_TO_PRETRAINED_MODELS = 'xxx/tools'

#######################
## 所有数据集的存储路径
#######################
DATA_DIR = {
    'MER2025Raw':   'xxx/dataset/mer2025-dataset',
    'MER2025':      'xxx/dataset/mer2025-dataset-process',
}
PATH_TO_RAW_AUDIO = {
    'MER2025Raw':  os.path.join(DATA_DIR['MER2025Raw'], 'audio'),
    'MER2025':     os.path.join(DATA_DIR['MER2025'],    'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2025Raw':  os.path.join(DATA_DIR['MER2025Raw'], 'video'),
    'MER2025':     os.path.join(DATA_DIR['MER2025'],    'video'),
}
PATH_TO_RAW_FACE = {
    'MER2025Raw':  os.path.join(DATA_DIR['MER2025Raw'], 'openface_face'),
    'MER2025':     os.path.join(DATA_DIR['MER2025'],    'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2025Raw':  os.path.join(DATA_DIR['MER2025Raw'], 'subtitle_chieng.csv'),
    'MER2025':     os.path.join(DATA_DIR['MER2025'],    'track1_subtitle_chieng.csv'),
}
PATH_TO_LABEL = {
    'MER2025Raw':  os.path.join(DATA_DIR['MER2025Raw'], 'xxx'),
    'MER2025':     os.path.join(DATA_DIR['MER2025'],    'track1_label_6way.npz'),
}
PATH_TO_FEATURES = {
    'MER2025Raw':  os.path.join(DATA_DIR['MER2025Raw'], 'xxx'),
    'MER2025':     os.path.join(DATA_DIR['MER2025'],    'features'),
}
