# *_*coding:utf-8 *_*
import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image

import torch 
import timm # pip install timm==0.9.7
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor

# import config
import sys
sys.path.append('../../')
import config

##################### Pretrained models #####################
CLIP_VIT_BASE = 'clip-vit-base-patch32'  # https://huggingface.co/openai/clip-vit-base-patch32
CLIP_VIT_LARGE = 'clip-vit-large-patch14' # https://huggingface.co/openai/clip-vit-large-patch14
EVACLIP_VIT = 'eva02_base_patch14_224.mim_in22k' # https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k
DATA2VEC_VISUAL = 'data2vec-vision-base-ft1k' # https://huggingface.co/facebook/data2vec-vision-base-ft1k
VIDEOMAE_BASE = 'videomae-base' # https://huggingface.co/MCG-NJU/videomae-base
VIDEOMAE_LARGE = 'videomae-large' # https://huggingface.co/MCG-NJU/videomae-large
DINO2_LARGE = 'dinov2-large' # https://huggingface.co/facebook/dinov2-large
DINO2_GIANT = 'dinov2-giant' # https://huggingface.co/facebook/dinov2-giant


def func_opencv_to_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def func_opencv_to_numpy(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def func_read_frames(face_dir, vid):
    npy_path  = os.path.join(face_dir, vid, f'{vid}.npy')
    assert os.path.exists(npy_path), f'Error: {vid} does not have frames.npy!'
    frames = np.load(npy_path)
    return frames

# 策略2：VideoMAE修订，采用孙总提供的采样代码
def func_get_indexes_for_videomae(total_frames):
    
    clip_len = 16
    test_num_segment = 2
    frame_sample_rate = 4
    
    # sampling according to 'frame_sample_rate'
    all_index = [x for x in range(0, total_frames, frame_sample_rate)]
    while len(all_index) < clip_len:
        all_index.append(all_index[-1])

    # get start end index according to 'segment_id' => 至少可以保证后面有 16 frames
    whole_segment_index = []
    for segment_id in range(test_num_segment):
        temporal_step = max(1.0 * (len(all_index) - clip_len)  / (test_num_segment - 1), 0)
        temporal_start = int(segment_id * temporal_step)
        segment_index = all_index[temporal_start:temporal_start + clip_len]
        print(f"segment '{segment_id+1}' index: {segment_index}")
        whole_segment_index.append(segment_index)
    return whole_segment_index

def resample_frames_sunlicai(frames):
    batches = []
    whole_segment_index = func_get_indexes_for_videomae(len(frames))
    for segment_index in whole_segment_index:
        assert len(segment_index) == 16
        batches.append(np.array(frames)[segment_index])
    return batches

# 策略3：相比于上面采样更加均匀 [将videomae替换并重新测试]
def resample_frames_uniform(frames, nframe=16):
    vlen = len(frames)
    start, end = 0, vlen
    
    n_frms_update = min(nframe, vlen) # for vlen < n_frms, only read vlen
    indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    
    # whether compress into 'n_frms'
    while len(indices) < nframe:
        indices.append(indices[-1])
    indices = indices[:nframe]
    assert len(indices) == nframe, f'{indices}, {vlen}, {nframe}'
    return frames[indices]
    

def split_into_batch(inputs, bsize=32):
    batches = []
    for ii in range(math.ceil(len(inputs)/bsize)):
        batch = inputs[ii*bsize:(ii+1)*bsize]
        batches.append(batch)
    return batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MER2023', help='input dataset')
    parser.add_argument('--model_name', type=str, default=None, help='name of pretrained model')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--videomae_type', type=str, default=None, help='videomae input type: [None or sunlicai]')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    params = parser.parse_args()

    print(f'==> Extracting {params.model_name} embeddings...')
    model_name = params.model_name.split('.')[0]
    face_dir = config.PATH_TO_RAW_FACE[params.dataset]

    # gain save_dir
    if params.videomae_type is None:
        save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f'{model_name}-{params.feature_level[:3]}')
    else: 
        assert params.videomae_type == 'sunlicai'
        save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f'sunlicai-{model_name}-{params.feature_level[:3]}')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model
    if params.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE, DATA2VEC_VISUAL, VIDEOMAE_BASE, VIDEOMAE_LARGE]: # from huggingface
        model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{params.model_name}')
        model = AutoModel.from_pretrained(model_dir)
        processor  = AutoFeatureExtractor.from_pretrained(model_dir)
    elif params.model_name in [DINO2_LARGE, DINO2_GIANT]:
        model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{params.model_name}')
        model = AutoModel.from_pretrained(model_dir)
        processor  = AutoImageProcessor.from_pretrained(model_dir)
    elif params.model_name in [EVACLIP_VIT]: # from timm
        model_path = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'timm/{params.model_name}/model.safetensors')
        model = timm.create_model(params.model_name, pretrained=True, num_classes=0,pretrained_cfg_overlay=dict(file=model_path))
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    # 有 gpu 才会放在cuda上
    if params.gpu != -1:
        torch.cuda.set_device(params.gpu)
        model.cuda()
    model.eval()

    # extract embedding video by video
    vids = os.listdir(face_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # save_file = os.path.join(save_dir, f'{vid}.npy')
        # if os.path.exists(save_file): continue

        # forward process [different model has its unique mode, it is hard to unify them as one process]
        # => split into batch to reduce memory usage
        with torch.no_grad():
            frames = func_read_frames(face_dir, vid)
            if params.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE]:
                frames = [func_opencv_to_image(frame) for frame in frames]
                inputs = processor(images=frames, return_tensors="pt")['pixel_values']
                if params.gpu != -1: inputs = inputs.to("cuda")
                batches = split_into_batch(inputs, bsize=32)
                embeddings = []
                for batch in batches:
                    embeddings.append(model.get_image_features(batch)) # [58, 768]
                embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]

            elif params.model_name in [DATA2VEC_VISUAL]:
                frames = [func_opencv_to_image(frame) for frame in frames]
                inputs = processor(images=frames, return_tensors="pt")['pixel_values'] # [nframe, 3, 224, 224]
                if params.gpu != -1: inputs = inputs.to("cuda")
                batches = split_into_batch(inputs, bsize=32)
                embeddings = []
                for batch in batches: # [32, 3, 224, 224]
                    hidden_states = model(batch, output_hidden_states=True).hidden_states # [58, 196 patch + 1 cls, feat=768]
                    embeddings.append(torch.stack(hidden_states)[-1].sum(dim=1)) # [58, 768]
                embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]

            elif params.model_name in [DINO2_LARGE, DINO2_GIANT]:
                frames = resample_frames_uniform(frames, nframe=64) # 加速特征提起：这种方式更加均匀的采样64帧
                frames = [func_opencv_to_image(frame) for frame in frames]
                inputs = processor(images=frames, return_tensors="pt")['pixel_values'] # [nframe, 3, 224, 224]
                if params.gpu != -1: inputs = inputs.to("cuda")
                batches = split_into_batch(inputs, bsize=32)
                embeddings = []
                for batch in batches: # [32, 3, 224, 224]
                    hidden_states = model(batch, output_hidden_states=True).hidden_states # [58, 196 patch + 1 cls, feat=768]
                    embeddings.append(torch.stack(hidden_states)[-1].sum(dim=1)) # [58, 768]
                embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]

            elif params.model_name in [VIDEOMAE_BASE, VIDEOMAE_LARGE]:
                # videoVAE: only supports 16 frames inputs
                if  params.videomae_type == 'sunlicai':
                    batches = resample_frames_sunlicai(frames)
                else:
                    batches = [resample_frames_uniform(frames)] # convert to list of batches
                embeddings = []
                for batch in batches:
                    frames = [func_opencv_to_numpy(frame) for frame in batch] # 16 * [112, 112, 3]
                    inputs = processor(list(frames), return_tensors="pt")['pixel_values'] # [1, 16, 3, 224, 224]
                    if params.gpu != -1: inputs = inputs.to("cuda")
                    outputs = model(inputs).last_hidden_state # [1, 1586, 768]
                    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2 # 14*14
                    outputs = outputs.view(16 // model.config.tubelet_size, num_patches_per_frame, -1) # [seg_number, patch, featdim]
                    embeddings.append(outputs.mean(dim=1)) # [seg_number, featdim]
                embeddings = torch.cat(embeddings, axis=0)

            elif params.model_name in [EVACLIP_VIT]:
                frames = [func_opencv_to_image(frame) for frame in frames]
                inputs = torch.stack([transforms(frame) for frame in frames]) # [117, 3, 224, 224]
                if params.gpu != -1: inputs = inputs.to("cuda")
                batches = split_into_batch(inputs, bsize=32)
                embeddings = []
                for batch in batches:
                    embeddings.append(model(batch)) # [58, 768]
                embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]
    
        embeddings = embeddings.detach().squeeze().cpu().numpy()
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        # save into npy
        save_file = os.path.join(save_dir, f'{vid}.npy')
        if params.feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, EMBEDDING_DIM))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(save_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((EMBEDDING_DIM, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(save_file, embeddings)
 