import os
import tqdm
import glob
import numpy as np
import pandas as pd

import config
from toolkit.utils.read_files import *

# python main-asr.py generate_transcription_files_asr ./dataset-process/audio ./dataset-process/transcription.csv
def generate_transcription_files_asr(audio_root, save_path):
    import torch
    import wenetruntime as wenet # must load torch first
    # from paddlespeech.cli.text.infer import TextExecutor
    # text_punc = TextExecutor()
    decoder = wenet.Decoder(config.PATH_TO_WENET, lang='chs')

    names = []
    sentences = []
    for audio_path in tqdm.tqdm(glob.glob(audio_root + '/*')):
        name = os.path.basename(audio_path)[:-4]
        sentence = decoder.decode_wav(audio_path)
        sentence = sentence.split('"')[5]
        # if len(sentence) > 0: sentence = text_punc(text=sentence)
        names.append(name)
        sentences.append(sentence)

    ## write to csv file
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_path, index=False)


# python main-asr.py refinement_transcription_files_asr(old_path, new_path)
def refinement_transcription_files_asr(old_path, new_path):
    from paddlespeech.cli.text.infer import TextExecutor
    text_punc = TextExecutor()

    ## read 
    names, sentences = [], []
    df_label = pd.read_csv(old_path)
    for _, row in df_label.iterrows(): ## read for each row
        names.append(row['name'])
        sentence = row['sentence']
        if pd.isna(sentence):
            sentences.append('')
        else:
            sentence = text_punc(text=sentence)
            sentences.append(sentence)
        print (sentences[-1])

    ## write
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(new_path, index=False)


# python main-asr.py merge_trans_with_checked dataset/mer2024-dataset-process/transcription.csv dataset/mer2024-dataset/label-transcription.csv dataset/mer2024-dataset-process/transcription-merge.csv
def merge_trans_with_checked(new_path, check_path, merge_path):

    # read new_path 7369
    name2new = {}
    names = func_read_key_from_csv(new_path, 'name')
    trans = func_read_key_from_csv(new_path, 'sentence')
    for (name, tran) in zip(names, trans):
        name2new[name] = tran
    print (f'new sample: {len(name2new)}')
    
    # read check_path 5030
    name2check = {}
    names = func_read_key_from_csv(check_path, 'name')
    trans = func_read_key_from_csv(check_path, 'chinese')
    for (name, tran) in zip(names, trans):
        name2check[name] = tran
    print (f'check sample: {len(name2check)}')
    
    # 生成新的merge结果
    name2merge = {}
    for name in name2new:
        if name in name2check:
            name2merge[name] = [name2check[name]]
        else:
            name2merge[name] = [name2new[name]]
    print (f'merge sample: {len(name2merge)}')
    
    # 存储 name2merge
    names = [name for name in name2merge]
    keynames = ['chinese']
    func_write_key_to_csv(merge_path, names, name2merge, keynames)


if __name__ == '__main__':
    import fire
    fire.Fire()
    