# *_*coding:utf-8 *_*
import os

PATH_TO_PRETRAINED_MODELS = 'xxx/tools'


#######################
## 所有数据集的存储路径
#######################
DATA_DIR = {
    'MER2026Raw':   'xxx/dataset/mer2026-dataset',
    'MER2026':      'xxx/dataset/mer2026-dataset-process',
}
PATH_TO_RAW_AUDIO = {
    'MER2026Raw':  os.path.join(DATA_DIR['MER2026Raw'], 'audio'),
    'MER2026':     os.path.join(DATA_DIR['MER2026'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2026Raw':  os.path.join(DATA_DIR['MER2026Raw'], 'video'),
    'MER2026':     os.path.join(DATA_DIR['MER2026'], 'video'),
}
PATH_TO_RAW_FACE = {
    'MER2026Raw':  os.path.join(DATA_DIR['MER2026Raw'], 'openface_face'),
    'MER2026':     os.path.join(DATA_DIR['MER2026'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2026Raw':  os.path.join(DATA_DIR['MER2026Raw'], 'subtitle_chieng.csv'),
    'MER2026':     os.path.join(DATA_DIR['MER2026'], 'track1_subtitle_chieng.csv'),
}
PATH_TO_LABEL = {
    'MER2026Raw':  os.path.join(DATA_DIR['MER2026Raw'], 'xxx'),
    'MER2026':     os.path.join(DATA_DIR['MER2026'],    'track1_label_6way.npz'),
}
PATH_TO_FEATURES = {
    'MER2026Raw':  os.path.join(DATA_DIR['MER2026Raw'], 'xxx'),
    'MER2026':     os.path.join(DATA_DIR['MER2026'],    'features'),
}
