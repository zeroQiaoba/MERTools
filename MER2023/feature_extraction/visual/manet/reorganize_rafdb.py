import os
import glob
import pandas as pd
import shutil


rafdb_path = '/data1/sunlicai/Affective Computing/Dataset/RAF-DB/basic'
src_path = os.path.join(rafdb_path, 'Image/aligned')
tgt_path = os.path.join(rafdb_path, 'Image/aligned_c') # split/class_id/img_file
label_file = os.path.join(rafdb_path, 'EmoLabel/list_patition_label.txt')
df = pd.read_csv(label_file, header=None, delimiter=' ')
file_names, label_ids = df[0].values, df[1].values
print(f'Number of images: {len(df)}.')
name_to_label = dict(zip(file_names, label_ids))
img_files = glob.glob(os.path.join(src_path, '*.jpg'))

for src_file in img_files:
    img_name = os.path.basename(src_file).replace('_aligned', '')
    label = name_to_label[img_name]
    split = img_name.split('_')[0]
    saved_path = os.path.join(tgt_path, split, str(label))
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    tgt_file = os.path.join(saved_path, img_name)
    shutil.copyfile(src_file, tgt_file)
    print(f'Copy "{src_file}" to "{tgt_file}".')