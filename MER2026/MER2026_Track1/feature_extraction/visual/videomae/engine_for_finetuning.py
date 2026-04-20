import os
import torch
import numpy as np

@torch.no_grad()
def final_test(data_loader, model, feature_level, device):
    
    model.eval()

    # me: for saving feature in the last layer
    saved_features = {}
    for batchid, batch in enumerate(data_loader): # 每个视频，无论多长，都采样4次

        if batchid % 10 == 0: # 查看运行进度
            print (f'process on {batchid}|{len(data_loader)}')

        videos = batch[0] # [64, 3, 16, 160, 160]
        frmids = batch[1] # [64], 对于句子级别全是0，对于帧级别会有差异

        videos = videos.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output, saved_feature = model(videos, save_feature=True) # saved_feature: [64, 768]
            
        for i in range(output.size(0)): # range(64)
            if frmids[i] not in saved_features:
                saved_features[frmids[i]] = []
            saved_features[frmids[i]].append(saved_feature.data[i].cpu().numpy().tolist())
    
    # store features
    embeddings = []
    for frame_id in sorted(saved_features):
        features_block = saved_features[frame_id]
        assert len(features_block) == 4 # 每个视频采样了4个block
        embedding = np.mean(features_block, axis=0)
        embeddings.append(embedding)

    # save into npy
    if feature_level == 'FRAME':
        embeddings = np.array(embeddings).squeeze()
        if len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
    else:
        embeddings = np.array(embeddings).squeeze()
        if len(embeddings.shape) == 2:
            embeddings = np.mean(embeddings, axis=0)
    return embeddings