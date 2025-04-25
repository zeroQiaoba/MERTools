import os
import glob
import numpy as np

from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from toolkit.utils.chatgpt import *


def func_get_name2reason(reason_root):
    name2reason = {}
    for reason_npy in glob.glob(reason_root + '/*.npy'):
        name = os.path.basename(reason_npy)[:-4]
        reason = np.load(reason_npy).tolist()
        name2reason[name] = reason
    return name2reason


def generate_openset_synonym_mer2024(gt_csv, pred_csv, synonym_root, gptmodel):

    ## read gt openset
    name2gt = {}
    names = func_read_key_from_csv(gt_csv, 'name')
    opensets = func_read_key_from_csv(gt_csv, 'openset')
    for (name, openset) in zip(names, opensets):
        name2gt[name] = openset

    ## read pred openset
    name2pred = {}
    names = func_read_key_from_csv(pred_csv, 'name')
    opensets = func_read_key_from_csv(pred_csv, 'openset')
    for (name, openset) in zip(names, opensets):
        name2pred[name] = openset
    
    ## define store root
    if not os.path.exists(synonym_root):
        os.makedirs(synonym_root)

    ## process
    for name in name2gt:
        list1 = string_to_list(name2gt[name])
        list2 = string_to_list(name2pred[name])
        # all convert to lower
        list1 = [item.lower() for item in list1]
        list2 = [item.lower() for item in list2]
        # find synonym
        response = get_openset_synonym(list1, list2, model=gptmodel)
        save_path = os.path.join(synonym_root, f'{name}.npy')
        np.save(save_path, response)



def calculate_openset_overlap_rate_mer2024(gt_csv, pred_csv, synonym_root):

    ## read gt openset
    name2gt = {}
    names = func_read_key_from_csv(gt_csv, 'name')
    opensets = func_read_key_from_csv(gt_csv, 'openset')
    for (name, openset) in zip(names, opensets):
        name2gt[name] = openset

    ##  read pred openset
    name2pred = {}
    names = func_read_key_from_csv(pred_csv, 'name')
    opensets = func_read_key_from_csv(pred_csv, 'openset')
    for (name, openset) in zip(names, opensets):
        name2pred[name] = openset

    ## read mapping
    name2mapping = func_get_name2reason(synonym_root)
    
    # main process
    accuracy, recall = [], []
    for name in name2mapping:
        # => synonym_map
        synonym_map = {}
        mapping = name2mapping[name]
        multi_lists = listlist_to_list(mapping)
        for one_list in multi_lists:
            for ii in range(len(one_list)):
                synonym_map[one_list[ii]] = one_list[0]
        
        # map gt and pred to group id
        gt = string_to_list(name2gt[name])
        gt = [item.lower() for item in gt]
        gt = set([synonym_map[item] if item in synonym_map else item for item in gt])

        pred = string_to_list(name2pred[name])
        pred = [item.lower() for item in pred]
        pred = set([synonym_map[item] if item in synonym_map else item for item in pred])

        if len(pred) == 0:
            accuracy.append(0)
            recall.append(0)
        else:
            accuracy.append(len(gt & pred)/len(pred))
            recall.append(len(gt & pred)/len(gt))
    print ('process number (after filter): ', len(accuracy))
    return np.mean(accuracy), np.mean(recall)


gptmodel = 'gpt-3.5-turbo-16k-0613'
def main_metric(gt_csv, pred_csv):
    base_root = os.path.split(gt_csv)[0]

    ## 1. calculate synonym
    synonym_root = os.path.join(base_root, 'openset-synonym')
    generate_openset_synonym_mer2024(gt_csv, pred_csv, synonym_root, gptmodel)

    ## 2. calculate set-wise metric
    accuracy, recall = calculate_openset_overlap_rate_mer2024(gt_csv, pred_csv, synonym_root)
    print (f'set level accuracy: {accuracy}; recall: {recall}')
    avg_score = np.mean([accuracy, recall])
    print (f'avg score: {avg_score}')
    '''
    if you use provided openset-synonym.zip, you can get following results:
    
        set level accuracy: 0.581777108433735; recall: 0.4977911646586345
        avg score: 0.5397841365461847
    '''


# python main-ov.py main_metric --gt_csv='ov_store/check-openset.csv' --pred_csv='ov_store/predict-openset.csv'
if __name__ == '__main__':
    import fire
    fire.Fire()
