import os
from toolkit.globals import *
from toolkit.utils.read_files import *
from toolkit.utils.functions import *


## 解析 train_label
def read_train_label_csv(train_label):
    names = func_read_key_from_csv(train_label, 'name')
    emos  = func_read_key_from_csv(train_label, 'discrete')
    print(f'train: {len(names)}')
    return names, emos


## 解析 test_label
def read_test_label_csv(test_label, with_gt=True):
    ## for baselines
    if with_gt:
        names = func_read_key_from_csv(test_label, 'name')
        emos  = func_read_key_from_csv(test_label, 'discrete')
        print(f'test: {len(names)}')
        return names, emos

    ## for real testing cases
    else:
        names = func_read_key_from_csv(test_label, 'name')
        emos  = ['neutral'] * len(names)
        print(f'test: {len(names)}')
        return names, emos


## 主要数据预处理流程
def normalize_dataset_format(data_root, save_root, with_gt=True):

    ## save path
    save_video    = config.PATH_TO_RAW_VIDEO['MER2026']
    save_audio    = config.PATH_TO_RAW_AUDIO['MER2026']
    save_openface = config.PATH_TO_RAW_FACE['MER2026']
    if not os.path.exists(save_root): os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)
    if not os.path.exists(save_audio): os.makedirs(save_audio)
    if not os.path.exists(save_openface): os.makedirs(save_openface)

    ## => (save_label)
    train_label = os.path.join(data_root, 'track1_train.csv')
    train_names, train_emos = read_train_label_csv(train_label)

    if with_gt: # with gt
        test_label  = os.path.join(data_root, 'track1_test.csv')
        test_names,  test_emos  = read_test_label_csv(test_label, with_gt)
    else: # without gt
        test_label  = os.path.join(data_root, 'track1_track2_candidate.csv')
        test_names,  test_emos  = read_test_label_csv(test_label, with_gt)
        
    whole_corpus = {}
    for (subset, names, labels) in [ ('train', train_names, train_emos),
                                     ('test1', test_names,  test_emos)]:
        whole_corpus[subset] = {}
        print (f'{subset}: sample number: {len(names)}')
        for (name, label) in zip(names, labels):
            whole_corpus[subset][name] = {'emo': label}
    save_label = os.path.join(save_root, 'track1_label_6way.npz')
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        test1_corpus=whole_corpus['test1'])
    
    # ## => (save_video)
    process_names = train_names + test_names
    print ('process_names: ', len(process_names))
    for name in tqdm.tqdm(process_names):
        raw_path  = os.path.join(config.PATH_TO_RAW_VIDEO['MER2026Raw'], name+'.mp4')
        save_path = os.path.join(config.PATH_TO_RAW_VIDEO['MER2026'],    name+'.mp4')
        assert os.path.exists(raw_path)
        shutil.copy(raw_path, save_path)
    
    ## => (save_audio)
    for name in tqdm.tqdm(process_names):
        raw_path  = os.path.join(config.PATH_TO_RAW_AUDIO['MER2026Raw'], name+'.wav')
        save_path = os.path.join(config.PATH_TO_RAW_AUDIO['MER2026'],    name+'.wav')
        assert os.path.exists(raw_path)
        shutil.copy(raw_path, save_path)
    
    ## => (openface_face)
    for name in tqdm.tqdm(process_names):
        raw_path  = os.path.join(config.PATH_TO_RAW_FACE['MER2026Raw'], name)
        save_path = os.path.join(config.PATH_TO_RAW_FACE['MER2026'],    name)
        assert os.path.exists(raw_path)
        shutil.copytree(raw_path, save_path)
    
    # => (subtitle)
    name2key = {}
    raw_subtitle = config.PATH_TO_TRANSCRIPTIONS['MER2026Raw']
    df = pd.read_csv(raw_subtitle)
    for _, row in tqdm.tqdm(df.iterrows()):
        name = row['name']
        chinese = row['chinese']
        english = row['english']
        if pd.isna(chinese): chinese=""
        if pd.isna(english): english=""
        if name in process_names:
            name2key[name] = [chinese, english]
    save_subtitle = config.PATH_TO_TRANSCRIPTIONS['MER2026']
    func_write_key_to_csv(save_subtitle, process_names, name2key, ['chinese', 'english'])


## Move 'MER2026Raw' -> 'MER2026'
if __name__ == '__main__':
    data_root = config.DATA_DIR['MER2026Raw']
    save_root = config.DATA_DIR['MER2026']
    # normalize_dataset_format(data_root, save_root, with_gt=True)  # w/  gt files
    normalize_dataset_format(data_root, save_root, with_gt=False) # w/o gt files 
