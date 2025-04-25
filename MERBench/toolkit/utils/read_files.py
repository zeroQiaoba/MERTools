import os
import json
import math
import random
import numpy as np
import pandas as pd

# 功能1：给每个 names[ii] 的 keyname 设置 values[ii]
def func_labelstudio_init_key(keyname, names, values):
    whole_json = []
    maxlen = max([len(value) for value in values])
    print (f'max len of values {maxlen}')
    for ii, name in enumerate(names):
        s3_path = f's3://zeroqiaoba-first/video3/{name}.webm'

        onefile_json = {}
        onefile_json['id'] = ii
        onefile_json['data'] = {}
        onefile_json['data']['video'] = s3_path
        ###############################################
        # values may contain multiple values
        for jj in range(0, len(values[ii])):
            onefile_json['data'][keyname+f'{jj}'] = values[ii][jj]
        for jj in range(len(values[ii]), maxlen):
            onefile_json['data'][keyname+f'{jj}'] = ''
        ###############################################
        onefile_json['annotations'] = []
        onefile_json['predictions'] = []
        whole_json.append(onefile_json)
    print (f'whole sample number: {len(whole_json)}')
    return whole_json


# 功能1：给一个json文件增加一个key
def func_labelstudio_update_key(json_path, val_name, name2val):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        video = item['data']['video']
        videoname = os.path.basename(video).rsplit('.', 1)[0]
        item['data'][val_name] = name2val[videoname]
    
    with open(json_path, "w") as f:
        json.dump(data, f)
        

# 功能：将一个json分割到多个json，并存储在store_root中
def func_labelstudio_split_json(json_path, store_root, split_num=8, shuffle=True):
    if not os.path.exists(store_root):
        os.makedirs(store_root)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if shuffle:
        data = func_shuffle_list_data(data)

    subset_number = math.ceil(len(data)/split_num)
    for ii in range(split_num):
        sub_data = data[ii*subset_number:(ii+1)*subset_number]

        save_path = os.path.join(store_root, f'split-{ii}.json')
        with open(save_path, "w") as f:
            json.dump(sub_data, f)

# 功能：将一个list文件分成多份，存储在store_root中
def func_split_list_data(data, store_root, split_num=8, shuffle=True):
    if not os.path.exists(store_root):
        os.makedirs(store_root)

    if shuffle:
        data = func_shuffle_list_data(data)

    subset_number = math.ceil(len(data)/split_num)
    for ii in range(split_num):
        sub_data = data[ii*subset_number:(ii+1)*subset_number]

        save_path = os.path.join(store_root, f'split-{ii}.npy')
        np.save(save_path, sub_data)


# 功能2：读取key值对应的 name2key [因为可能存在多个values，所以返回的values都变成list格式了]
def func_labelstudio_read_key(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    
    name2val = {}
    for item in data:
        values = []
        videopath = item['data']['video']
        videoname = os.path.basename(videopath).rsplit('.', 1)[0]
        # 对 videoname 进行后处理
        # case1: sample_00001189.webm
        # case2: def5d5b7-sample_00001189.webm
        videoname_split = videoname.split('-', 1)
        if len(videoname_split) == 2:
            videoname = videoname_split[1]
        elif len(videoname_split) == 1:
            videoname = videoname_split[0]
        else:
            print (videoname)
            raise ValueError('videoname has some errors!!')
        # 分析标注结果
        annotations = item['annotations']
        if len(annotations) == 0:
            values = []
        elif len(annotations) == 1:
            result = annotations[0]['result']

            if len(result) == 0:
                values = []
            else: # 对于存在多个results情况，那么就逐个解析
                for ii in range(len(result)):
                    # 分析 choices 内容
                    if 'choices' in result[ii]['value']:
                        item = result[ii]['value']['choices']
                        if len(item) == 1:
                            values.append(item[0].strip())
                        else:
                            print (videoname)
                            raise ValueError('item has more than one values!!')
                    # 分析 text 内容
                    elif 'text' in result[ii]['value']:
                        item = result[ii]['value']['text']
                        if len(item) == 1:
                            values.append(item[0].strip())
                        else:
                            print (videoname)
                            raise ValueError('item has more than one values!!')
        else:
            print ('some errors may exist in annotations')
        name2val[videoname] = values
    return name2val


def func_shuffle_list_data(whole_json):
	indices = np.arange(len(whole_json))
	random.shuffle(indices)

	new_json = []
	for index in indices:
		new_json.append(whole_json[index])
	return new_json


# 功能3：从csv中读取特定的key对应的值
def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            if pd.isna(value): value=""
            values.append(value)
    return values


# names[ii] -> keys=name2key[names[ii]], containing keynames
def func_write_key_to_csv(csv_path, names, name2key, keynames):
    ## specific case: only save names
    if len(name2key) == 0 or len(keynames) == 0:
        df = pd.DataFrame(data=names, columns=['name'])
        df.to_csv(csv_path, index=False)
        return

    ## other cases:
    if isinstance(keynames, str):
        keynames = [keynames]
    assert isinstance(keynames, list)
    columns = ['name'] + keynames

    values = []
    for name in names:
        value = name2key[name]
        values.append(value)
    values = np.array(values)
    # ensure keynames is mapped
    if len(values.shape) == 1:
        assert len(keynames) == 1
    else:
        assert values.shape[-1] == len(keynames)
    data = np.column_stack([names, values])

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(csv_path, index=False)
    
    
def func_read_text_file(file_path):
    with open(file_path, encoding='utf8') as f: lines = [line.strip() for line in f]
    lines = [line for line in lines if len(line)!=0]
    return lines
