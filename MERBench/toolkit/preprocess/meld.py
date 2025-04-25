import os
import shutil
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import *

emos = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii
    idx2emo[ii]  = emo


def read_labels(label_path):

    dia_ids = func_read_key_from_csv(label_path, 'Dialogue_ID')
    utt_ids = func_read_key_from_csv(label_path, 'Utterance_ID')
    labels  = func_read_key_from_csv(label_path, 'Emotion')
    engs    = func_read_key_from_csv(label_path, 'Utterance')

    names = []
    for ii in range(len(dia_ids)):
        names.append(f'dia{dia_ids[ii]}_utt{utt_ids[ii]}')
    
    labels = [emo2idx[label] for label in labels]

    return names, labels, engs


def normalize_dataset_format(data_root, save_root):

    # gain paths
    train_label_path = os.path.join(data_root, 'train_sent_emo.csv')
    train_video_root = os.path.join(data_root, 'train')
    val_label_path   = os.path.join(data_root, 'dev_sent_emo.csv')
    val_video_root   = os.path.join(data_root, 'dev')
    test_label_path  = os.path.join(data_root, 'test_sent_emo.csv')
    test_video_root  = os.path.join(data_root, 'test')

    # gain (names, labels)
    train_names, train_labels, train_engs = read_labels(train_label_path)
    val_names,   val_labels,   val_engs   = read_labels(val_label_path)
    test_names,  test_labels,  test_engs  = read_labels(test_label_path)
    print (f'train number: {len(train_names)}')
    print (f'val   number: {len(val_names)}')
    print (f'test  number: {len(test_names)}')
    
    ## output path
    save_video = os.path.join(save_root, 'subvideo')
    save_label = os.path.join(save_root, 'label.npz')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## generate label path
    name2eng = {}
    whole_corpus = {}
    for datatype, names, labels, engs, video_root in [('train', train_names, train_labels, train_engs, train_video_root),
                                                      ('val',   val_names,   val_labels,   val_engs,   val_video_root),
                                                      ('test',  test_names,  test_labels,  test_engs,  test_video_root)]:
        whole_corpus[datatype] = {}
        for ii, name in enumerate(names):
            newname = f'{datatype}_{name}'
            whole_corpus[datatype][newname] = {'emo': labels[ii], 'val': -10} # save labels
            name2eng[newname] = engs[ii] # save trans
            
            # move video
            video_path = os.path.join(video_root, name+'.mp4')
            save_path  = os.path.join(save_video, newname+'.mp4')
            if os.path.exists(save_path): continue
            try:
                shutil.copy(video_path, save_path)
            except:
                print (f'ERROR: {video_path} does not exist!')
            
    # save labels
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['val'],
                        test_corpus=whole_corpus['test'])

    # save trans
    names = [name for name in name2eng]
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [name2eng[name]]
    func_write_key_to_csv(save_trans, names, name2key, ['english'])


if __name__ == '__main__':

    data_root = 'E:\\Dataset\\MELD'
    save_root = 'E:\\Dataset\\meld-process'
    normalize_dataset_format(data_root, save_root)

    # data_root = 'E:\\Dataset\\meld-process'
    # trans_path  = os.path.join(data_root, 'transcription.csv')
    # polish_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    # func_translate_transcript_polish_merge(trans_path, polish_path) # 再次检测一下遗漏的部分
