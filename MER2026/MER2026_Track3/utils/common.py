
import math
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def split_list_into_batch(items, split_num=None, batchsize=None):
    
    assert (split_num is not None) or (batchsize is not None)

    if split_num is None:
        split_num = math.ceil(len(items)/batchsize)

    # split infos into subset
    batches = []
    # items = np.array(items)
    each_split = math.ceil(len(items)/split_num)
    for ii in range(split_num):
        batch = items[ii*each_split:(ii+1)*each_split]
        if len(batch)!=0: 
            batches.append(batch)

    return batches

def func_label_distribution(labels):
    label2count = {}
    for label in labels:
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1
    return label2count

# 目标：通过这个处理后，输出结果只会包含 [a1, a2, same] 三种情况
def func_postprocess_preference(whole_response):
    pred_labels = []
    for ii, response in enumerate(whole_response):
        print (len(pred_labels), ' => ', response)
        response = response.lower()
        if response.startswith('a1'):
            pred_labels.append('a1')
        elif response.startswith('a2'):
            pred_labels.append('a2')
        elif response.startswith('same'):
            pred_labels.append('same')
        elif response.find('a1') != -1 or response.find('a2') != -1 or response.find('same') != -1:
            keys  = ['a1', 'a2', 'same']
            poses = [response.find('a1'), response.find('a2'), response.find('same')]
            poses = [10000 if item == -1 else item for item in poses]
            pred_labels.append(keys[np.argmin(poses)])
        else: # 其他无法判断的情况
            pred_labels.append('same')
    return pred_labels

def func_postprocess_matching(whole_response):
    pred_labels = []
    for ii, response in enumerate(whole_response):
        print (len(pred_labels), ' => ', response)
        # print (len(pred_labels), text[ii], ' => ', response)
        assert response.startswith('Yes') or response.startswith('No')
        if response.startswith('Yes'):
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    return pred_labels


def func_three_class_preference_metric(result_npz=None, gt_labels=None, pred_labels=None):
    label2idx = {
        'a1': 0,
        'a2': 1,
        'same': 2,
    }
    if result_npz is not None:
        gt_labels   = np.load(result_npz, allow_pickle=True)['gt_labels'].tolist()
        pred_labels = np.load(result_npz, allow_pickle=True)['pred_labels'].tolist()

    # score calculation
    gt_labels   = [label2idx[label] for label in gt_labels]
    pred_labels = [label2idx[label] for label in pred_labels]
    fscore   = f1_score(gt_labels, pred_labels, average='weighted')
    accuracy = accuracy_score(gt_labels, pred_labels)
    return fscore, accuracy


# result_npz 会把 (gt, pred) 两部分内容存下来，真聪明呀
def func_two_class_preference_metric(result_npz=None, gt_labels=None, pred_labels=None):
    label2idx = {
        'a1': 0,
        'a2': 1,
    } 
    if result_npz is not None:
        gt_labels   = np.load(result_npz, allow_pickle=True)['gt_labels'].tolist()
        pred_labels = np.load(result_npz, allow_pickle=True)['pred_labels'].tolist()

    # score calculation
    gt_labels_new, pred_labels_new = [], []
    for (gt_label, pred_label) in zip(gt_labels, pred_labels):
        if gt_label == 'same': continue
        if pred_label == 'same': pred_label = 'a1'
        gt_labels_new.append(label2idx[gt_label])
        pred_labels_new.append(label2idx[pred_label])
    fscore   = f1_score(gt_labels_new, pred_labels_new, average='weighted')
    accuracy = accuracy_score(gt_labels_new, pred_labels_new)
    return fscore, accuracy


def func_preference_metric(result_npz=None, gt_labels=None, pred_labels=None, metric='twoclass'):
    if metric == 'twoclass':
        fscore, accuracy = func_two_class_preference_metric(result_npz, gt_labels, pred_labels)
    elif metric == 'threeclass':
        fscore, accuracy = func_three_class_preference_metric(result_npz, gt_labels, pred_labels)
    else:
        raise ValueError(f"{metric} must in [twoclass, threeclass]")
    return fscore, accuracy
