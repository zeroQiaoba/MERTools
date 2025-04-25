import os
import glob
import shutil
from toolkit.globals import *
from toolkit.utils.read_files import *
from toolkit.utils.functions import *


## 解析 train_label
def read_train_label(train_video, train_label):
    video_names = os.listdir(train_video)
    video_names = [item.rsplit('.')[0] for item in video_names]

    name2emo = {}
    names = func_read_key_from_csv(train_label, 'name')
    emos  = func_read_key_from_csv(train_label, 'discrete')
    for (name, emo) in zip(names, emos):
        name2emo[name] = emo
    
    train_names = video_names
    train_emos = [name2emo[name] for name in train_names]
    print(f'train: {len(video_names)}')
    return train_names, train_emos


## 解析 test_label
def read_test_label(test_video, test_label):
    video_names = os.listdir(test_video)
    video_names = [item.rsplit('.')[0] for item in video_names]
    emos = ['neutral'] * len(video_names)
    print(f'test: {len(emos)}')
    return video_names, emos


def normalize_dataset_format(data_root, save_root):

    ## input path
    train_video, train_label = os.path.join(data_root, 'video-labeled'),   os.path.join(data_root, 'label-disdim.csv')
    test_video,  test_label  = os.path.join(data_root, 'video-unlabeled-with-test2noise'), None

    ## step1: 将semi分成两部分，一部分作为test1, 一部分作为test2
    train_names, train_emos = read_train_label(train_video, train_label)
    test_names,  test_emos  = read_test_label(test_video, test_label)

    ## output path
    save_video = os.path.join(save_root, 'video')
    save_label = os.path.join(save_root, 'label-6way.npz')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## generate label path
    whole_corpus = {}
    for (subset, video_root, names, labels) in [ ('train', train_video, train_names, train_emos),
                                                 ('test1', test_video,  test_names,  test_emos)]:
        
        whole_corpus[subset] = {}
        print (f'{subset}: sample number: {len(names)}')
        for (name, label) in zip(names, labels):
            whole_corpus[subset][name] = {'emo': label}
            # copy video
            video_path = glob.glob(os.path.join(video_root, f'{name}.*'))[0]
            assert os.path.exists(video_path), f'video does not exist.'
            video_name = os.path.basename(video_path)
            new_path = os.path.join(save_video, video_name)
            shutil.copy(video_path, new_path)

    ## 需要对视频进行加噪处理
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        test1_corpus=whole_corpus['test1'])


# run -d toolkit/preprocess/mer2024.py
if __name__ == '__main__':

    data_root = '/share/home/lianzheng/MER2024-baseline/mer2024-dataset'
    save_root = '/share/home/lianzheng/MER2024-baseline/mer2024-dataset-process'
    normalize_dataset_format(data_root, save_root)

