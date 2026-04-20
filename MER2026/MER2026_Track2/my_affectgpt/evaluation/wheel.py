import os
import glob
import scipy
import numpy as np
import pandas as pd

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
import config


#########################################################################
######## 采用得到的mapping，去度量 gt and pred openset 之间的重叠度 ########
## => 所有评价指标，都是把不在词表中的元素直接剔除掉 [统一化处理，方便后续比较]
##########################################################################
def func_get_name2reason(reason_root):
    name2reason = {}
    for reason_npy in glob.glob(reason_root + '/*.npy'):
        name = os.path.basename(reason_npy)[:-4]
        reason = np.load(reason_npy).tolist()
        name2reason[name] = reason
    return name2reason


# case1: 计算只依赖于 format_mapping 下的结果
def func_backward_case1(label, format_mapping, raw_mapping=None, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    return stage1_unique


# case2: 核心是保证 backward 过程中的唯一性
def func_backward_case2(label, format_mapping, raw_mapping, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    stage2_labels = raw_mapping[stage1_unique]
    assert isinstance(stage2_labels, list)
    stage2_unique = sorted(stage2_labels)[0]
    return stage2_unique


## 函数3：引入 emotion wheel 进行评价
def func_backward_case3(label, format_mapping, raw_mapping, wheel_map):
    if label not in format_mapping:
        return ""
    
    level1_whole = []
    for format in format_mapping[label]:
        for raw in raw_mapping[format]:
            level1_whole.append(raw)
    
    for level1 in sorted(level1_whole): # 保证了结果唯一性
        if level1 in wheel_map:
            return wheel_map[level1]
    return ""


# metric: ['case1', 'case2', 'case3'] 展示的是一种层级化的聚类结果，从而看出每层的必要性
def func_map_label_to_synonym(mlist, format_mapping, raw_mapping, wheel_map, metric='case1'):
    new_mlist = []
    for label in mlist:
        if metric.startswith('case1'): label = func_backward_case1(label, format_mapping)
        if metric.startswith('case2'): label = func_backward_case2(label, format_mapping, raw_mapping)
        if metric.startswith('case3'): label = func_backward_case3(label, format_mapping, raw_mapping, wheel_map)
        if label == '': continue # 如果找不到 backward 的词，就把他剔除
        new_mlist.append(label)
    return new_mlist


def calculate_openset_overlap_rate(gt_root=None, gt_csv=None, name2gt=None, 
                                   openset_root=None, openset_npz=None, name2pred=None, 
                                   process_names=None, 
                                   metric='case1',
                                   inter_print=True):

    # read name2gt
    if name2gt is None:
        if gt_root is not None:
            name2gt = func_get_name2reason(gt_root)
        elif gt_csv is not None:
            name2gt = {}
            names = func_read_key_from_csv(gt_csv, 'name')
            gts   = func_read_key_from_csv(gt_csv, 'openset')
            for (name, gt) in zip(names, gts):
                name2gt[name] = gt

    # read name2pred
    if name2pred is None:
        if openset_root is not None:
            name2pred = func_get_name2reason(openset_root)
        elif openset_npz is not None:
            names = np.load(openset_npz)['filenames']
            items = np.load(openset_npz)['fileitems']
            name2pred = {}
            for (name, item) in zip(names, items):
                name2pred[name] = item

    # process_names => (whole) / (subset)
    if process_names is None:
        process_names = [name for name in name2gt]

    # read all mapping # 改为默认从外部读取吧
    mapping_path = config.OUTSIDE_WHEEL_MAPPING
    format_mapping  = np.load(mapping_path, allow_pickle=True)['format_mapping'].tolist()  # level3 -> level2
    raw_mapping     = np.load(mapping_path, allow_pickle=True)['raw_mapping'].tolist()     # level2 -> level1
    wheel_map_whole = np.load(mapping_path, allow_pickle=True)['wheel_map_whole'].tolist() # level1 -> level0
    if metric.startswith('case3'):
        _, wheelname, levelname = metric.split('_')
        wheel_map = wheel_map_whole[wheelname][levelname]
    else:
        wheel_map = None

    # calculate (accuracy, recall) two values
    accuracy, recall = [], []
    for name in process_names:      
       
        # 删除 gt and pred 中的同义词
        gt = string_to_list(name2gt[name])
        gt = [item.lower().strip() for item in gt]
        gt = set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric))

        pred = string_to_list(name2pred[name])
        pred = [item.lower().strip() for item in pred]
        pred = set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric))

        if len(gt) == 0: continue
        if len(pred) == 0:
            accuracy.append(0)
            recall.append(0)
        else:
            accuracy.append(len(gt & pred)/len(pred))
            recall.append(len(gt & pred)/len(gt))
    if inter_print: print ('process number (after filter): ', len(accuracy))

    ## for special case
    if len(accuracy) != 0:
        avg_accuracy = np.mean(accuracy)
    else:
        avg_accuracy = 0
    
    if len(recall) != 0:
        avg_recall = np.mean(recall)
    else:
        avg_recall = 0

    ## print results
    if inter_print: print (f'avg acc: {avg_accuracy} avg recall: {avg_recall}')
    return  avg_accuracy, avg_recall


def wheel_metric_calculation(gt_root=None, gt_csv=None, name2gt=None, 
                             openset_root=None, openset_npz=None, name2pred=None, 
                             process_names=None, inter_print=True, level='level1'):

    if level == 'level1':
        candidate_metrics = [
                            'case3_wheel1_level1',
                            'case3_wheel2_level1',
                            'case3_wheel3_level1',
                            'case3_wheel4_level1',
                            'case3_wheel5_level1',
                            ]
    elif level == 'level2':
        candidate_metrics = [
                            'case3_wheel1_level2',
                            'case3_wheel2_level2',
                            'case3_wheel3_level2',
                            'case3_wheel4_level2',
                            'case3_wheel5_level2',
                            ]

    # 计算每个metric的这个值
    whole_scores = []
    for metric in candidate_metrics:
        precision, recall = calculate_openset_overlap_rate(gt_root=gt_root,
                                                           gt_csv=gt_csv,
                                                           name2gt=name2gt,
                                                           openset_root=openset_root, 
                                                           openset_npz=openset_npz, 
                                                           name2pred=name2pred,
                                                           process_names=process_names, 
                                                           metric=metric,
                                                           inter_print=inter_print)
        # 非常少的概率下，会出现这个情况
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * (precision * recall) / (precision + recall)
        whole_scores.append([fscore, precision, recall])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores