import itertools
import torchaudio

from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import numpy as np

from toolkit.globals import *
from toolkit.utils.functions import *
from sklearn.metrics import f1_score, accuracy_score

def get_top10_mean_std(data):
    data = np.array(data) * 100 
    data = np.sort(data)[-10:]
    mean_val = '%.2f' %(np.mean(data))
    std_val  = '%.2f' %(np.std(data))
    # np.round(np.mean(data), 2)
    # std_val = np.round(np.std(data, ddof=1), 2)
    return mean_val, std_val


def calculate_statistics_for_unimodal():

    feat2print = {}
    for feature_name in [
        'senet50face_UTT',
        'resnet50face_UTT',
        'emonet_UTT',
        'chinese-roberta-wwm-ext-UTT',
        'chinese-roberta-wwm-ext-large-UTT',
        'chinese-macbert-base-UTT',
        'chinese-macbert-large-UTT',
        'clip-vit-base-patch32-UTT',
        'clip-vit-large-patch14-UTT',
        'videomae-base-UTT',
        'videomae-large-UTT',
        'chinese-hubert-base-UTT',
        'chinese-hubert-large-UTT',
        'chinese-wav2vec2-base-UTT',
        'chinese-wav2vec2-large-UTT',
        'wavlm-base-UTT',
    ]:
        feature_whole = ""
        for set_name in ['cv', 'test1']:
            candidates = glob.glob(f"saved-unimodal/result/{set_name}_features:{feature_name}_dataset:MER2026_model:attention+utt+None_f1:*_acc:*.npz")
            # print (feature_name, set_name, len(candidates))
            ## calculate statistics
            whole_acc, whole_f1 = [], []
            pattern = r"f1:([0-9.]+)_acc:([0-9.]+)"
            for filename in candidates:
                match = re.search(pattern, filename)
                if match:
                    f1 = float(match.group(1))
                    acc = float(match.group(2))
                    whole_acc.append(acc)
                    whole_f1.append(f1)
            # return statistics
            mean_f1, std_f1 = get_top10_mean_std(whole_f1)
            mean_acc, std_acc = get_top10_mean_std(whole_acc)
            feature_whole += f"& {mean_f1}$\pm${std_f1} & {mean_acc}$\pm${std_acc} "
        
        feat2print[feature_name[:-4]] = feature_whole + '\\\\'
    
    # print via classification
    for featname in feat2print:
        if featname in WHOLE_IMAGE:
            print (featname_mapping[featname], feat2print[featname])
    print ('========================================')

    for featname in feat2print:
        if featname in WHOLE_AUDIO:
            print (featname_mapping[featname], feat2print[featname])
    print ('========================================')

    for featname in feat2print:
        if featname in WHOLE_TEXT:
            print (featname_mapping[featname], feat2print[featname])
    print ('========================================')

# calculate_statistics_for_unimodal()


def calculate_statistics_for_multimodal():

    for topn in [1, 2]:
        feature_whole = ""
        for set_name in ['cv', 'test1']:
            candidates = glob.glob(f"saved-multitop-others/result/{set_name}_features:_dataset:MER2026_model:attention_topn+utt+None_fusiontopn:{topn}_modality:AVT_f1:*_acc:*.npz")

            ## calculate statistics
            whole_acc, whole_f1 = [], []
            pattern = r"f1:([0-9.]+)_acc:([0-9.]+)"
            for filename in candidates:
                match = re.search(pattern, filename)
                if match:
                    f1 = float(match.group(1))
                    acc = float(match.group(2))
                    whole_acc.append(acc)
                    whole_f1.append(f1)
            # return statistics
            mean_f1, std_f1 = get_top10_mean_std(whole_f1)
            mean_acc, std_acc = get_top10_mean_std(whole_acc)
            feature_whole += f"& {mean_f1}$\pm${std_f1} & {mean_acc}$\pm${std_acc} "
        
        print (topn, feature_whole)

# calculate_statistics_for_multimodal()
