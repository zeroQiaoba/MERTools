import itertools
import torchaudio

from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import numpy as np

from toolkit.globals import *
from toolkit.utils.functions import *
from sklearn.metrics import f1_score, accuracy_score


def generate_submission(result_npz, save_csv):

    # 1. read preds
    emo_probs = np.load(result_npz, allow_pickle=True)['emo_probs'].tolist()
    emo_preds = np.argmax(emo_probs, 1)
    emo_preds = [idx2emo_mer[idx] for idx in emo_preds]

    # 2. names
    # label_csv = os.path.join(config.DATA_DIR['MER2025Raw'], 'track1_test_dis.csv') # w/ gt 
    label_csv = os.path.join(config.DATA_DIR['MER2025Raw'], 'track_all_candidates.csv') # w/o gt 
    names = func_read_key_from_csv(label_csv, 'name')
    emos  = func_read_key_from_csv(label_csv, 'discrete')

    # 3. save_csv
    name2key = {}
    for (name, pred) in zip(names, emo_preds):
        name2key[name] = pred
    func_write_key_to_csv(save_csv, names, name2key, ['discrete'])


if __name__ == '__main__':
    import fire
    fire.Fire()
