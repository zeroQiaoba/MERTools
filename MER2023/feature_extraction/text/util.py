# *_*coding:utf-8 *_*
import os
import re
import pandas as pd
import numpy as np
import unicodedata


def write_feature_to_csv(embeddings, timestamps, words, csv_file, log_file=None, embedding_dim=None):
    # get label file
    vid = os.path.basename(os.path.splitext(csv_file)[0])
    # label_dir = os.path.abspath(os.path.join(os.path.dirname(csv_file), '../../label_segments/arousal'))
    # assert os.path.exists(label_dir), f'Error:  label dir "{label_dir}" does not exist!'
    save_dir = os.path.dirname(csv_file)
    task_id = int(re.search('c(\d)_muse_', save_dir).group(1))  # infer the task id from save_dir (naive/unelegant approach)
    if task_id == 2:  # for task "c2"
        rel_path = '../au'  # use csv file in "au" feature as reference beacause of there is no timestamp in the label file
    elif task_id == 4:  # for task "c4"
        rel_path = '../../label_segments/anno12_EDA'  # no arousal label for this task
    else:
        rel_path = '../../label_segments/arousal'
    label_dir = os.path.abspath(os.path.join(save_dir, rel_path))
    assert os.path.exists(label_dir), f'Error:  label dir "{label_dir}" does not exist!'
    label_file = os.path.join(label_dir, f'{vid}.csv')
    df_label = pd.read_csv(label_file)
    meta_columns = ['timestamp', 'segment_id']
    metas = df_label[meta_columns].values
    label_timestamps = metas[:,0]
    # align word, timestamp & embedding
    # embedding_dim = len(embeddings[0]) # use the argument "embedding_dim" instead, in case of embeddings is []
    n_frames = len(label_timestamps)
    aligned_embeddings = np.zeros((n_frames, embedding_dim))
    aligned_timestamps = np.empty((n_frames, 2), dtype=np.object)
    aligned_words = np.empty((n_frames,), dtype=np.object)
    label_timestamp_idxs = np.arange(n_frames)
    hit_count = 0
    for i, (s_t, e_t) in enumerate(timestamps):
        idxs = label_timestamp_idxs[np.where((label_timestamps >= s_t) & (label_timestamps < e_t))]
        if len(idxs) > 0:
            aligned_embeddings[idxs] = embeddings[i]
            aligned_timestamps[idxs] = [int(s_t), int(e_t)]
            aligned_words[idxs] = words[i]
            hit_count += len(idxs)
    print(f'Video "{vid}" hit rate: {hit_count/n_frames:.1%}.')
    # write csv file
    columns = meta_columns + [str(i) for i in range(embedding_dim)]
    data = np.column_stack([metas, aligned_embeddings])
    df = pd.DataFrame(data=data, columns=columns)
    df[meta_columns] = df[meta_columns].astype(np.int64)
    df.to_csv(csv_file, index=False)
    # write log file
    if log_file is not None:
        log_columns = meta_columns + ['start', 'end', 'word']
        log_data = np.column_stack([metas, aligned_timestamps, aligned_words])
        log_df = pd.DataFrame(data=log_data, columns=log_columns)
        log_df[meta_columns] = log_df[meta_columns].astype(np.int64)
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        log_df.to_csv(log_file, index=False)
    return data




def load_glove(embedding_file):
    embeddings = {}
    with open(embedding_file, 'r') as f:
        for line in f.readlines():
            splited_line = line.split(' ')
            word = splited_line[0]
            embedding = np.array([float(val) for val in splited_line[1:]])  # to numpy
            embeddings[word] = embedding
    embedding_dim = len(embedding)
    return embeddings, embedding_dim


def load_word2vec(embedding_file):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    # embeddings = dict(zip(model.vocab, model.vectors)) # for Gensim 3.x
    embedding_dim = model.vector_size
    return model, embedding_dim


# strip accent in unicode string
def strip_accent(string):
    return ''.join(
        character for character in unicodedata.normalize('NFD', string)
        if unicodedata.category(character) != 'Mn'
    )




if __name__ == '__main__':
    main()