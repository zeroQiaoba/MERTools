import os
import re
import cv2
import copy
import math
import tqdm
import glob
import shutil
import random
import argparse
import itertools
import torchaudio

from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing


def func_label_distribution(labels):
    label2count = {}
    for label in labels:
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1
    return label2count

def func_majoremo_majorcount(labels):
    label2count = func_label_distribution(labels)
    maxcount = max([label2count[label] for label in label2count])

    for label in label2count:
        if label2count[label] == maxcount:
            return maxcount, label

def func_majority_vote(whole_preds):
    voted_preds = []
    merge_maxcounts = []
    whole_preds = np.array(whole_preds)

    for sample_idx in range(len(whole_preds[0])):
        sample_preds = whole_preds[:, sample_idx]
        maxcount, label = func_majoremo_majorcount(sample_preds)
        voted_preds.append(label)
        merge_maxcounts.append(maxcount)
    return voted_preds

def func_whether_two_list_are_same(list1, list2):
    if len(list1) != len(list2):
        return False

    for ii in range(len(list1)):
        if list1[ii] != list2[ii]:
            return False
    
    return True

def func_two_list_same_precentage(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("two lists must be in the same length")

    same_counts = 0
    for ii in range(len(list1)):
        if list1[ii] == list2[ii]:
            same_counts += 1
    same_precentage = same_counts / len(list1)
    return same_precentage

