import os
import argparse
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# import config
import sys
sys.path.append('../../')
import config
from dataset import FaceDataset
from manet.model.manet import manet

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, names in data_loader:
            images = images.cuda()
            embedding = model(images, return_embedding=True)
            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    params = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    print(f'==> Extracting manet embedding...')
    face_dir = config.PATH_TO_RAW_FACE[params.dataset]
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f'manet_{params.feature_level[:3]}')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model
    model = manet(num_classes=7).cuda()
    checkpoint_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'manet/[02-08]-[21-19]-model_best-acc88.33.pth')
    checkpoint = torch.load(checkpoint_file)
    pre_trained_dict = {k.replace('module.', ''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(pre_trained_dict)

    # transform
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    # extract embedding video by video
    vids = os.listdir(face_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")

        # forward
        dataset = FaceDataset(vid, face_dir, transform=transform)
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            embeddings, framenames = [], []
        else:
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=32,
                                                      num_workers=4,
                                                      pin_memory=True)
            embeddings, framenames = extract(data_loader, model)

        # save results
        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes]
        framenames = framenames[indexes]
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

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
