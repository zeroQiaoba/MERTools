import os
import six
import sys
import argparse
import numpy as np

import torch
import torch._utils
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import torchvision.transforms as transforms

import sys
sys.path.append('../../')
import config
from dataset import FaceDataset

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3
    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition
    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod

def load_model(model_name, model_dir, pretrained_dir):
    """Load imoprted PyTorch model by name
    Args:
        model_name (str): the name of the model to be loaded
    Return:
        nn.Module: the loaded network
    """
    model_def_path = os.path.join(model_dir, model_name + '.py')
    weights_path = os.path.join(pretrained_dir, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net

def compose_transforms(meta):
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    
    transform_list = [transforms.Resize(256),
                      transforms.CenterCrop(size=(im_size[0], im_size[1])),
                      transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]:  # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)

def get_feature(model, layer_name, image):
    bs = image.size(0)
    layer = model._modules.get(layer_name)
    if layer_name == 'fc7':
        my_embedding = torch.zeros(bs, 4096)
    elif layer_name == 'fc8':
        my_embedding = torch.zeros(bs, 7)
    elif layer_name == 'pool5' or layer_name == 'pool5_full':
        my_embedding = torch.zeros([bs, 512, 7, 7])
    elif layer_name == 'pool4':
        my_embedding = torch.zeros([bs, 512, 14, 14])
    elif layer_name == 'pool3':
        my_embedding = torch.zeros([bs, 256, 28, 28])
    elif layer_name == 'pool5_7x7_s1':  # available
        my_embedding = torch.zeros([bs, 2048, 1, 1])
    elif layer_name == 'conv5_3_3x3_relu': # available
        my_embedding = torch.zeros([bs, 512, 7, 7])
    else:
        raise Exception(f'Error: not supported layer "{layer_name}".')

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    h = layer.register_forward_hook(copy_data)
    _ = model(image)
    h.remove()
    if layer_name == 'pool5' or layer_name == 'conv5_3_3x3_relu':
        GAP_layer = nn.AvgPool2d(kernel_size=[7, 7], stride=(1, 1))
        my_embedding = GAP_layer(my_embedding)

    my_embedding = F.relu(my_embedding.squeeze())
    if my_embedding.size(0) != bs:
        my_embedding = my_embedding.unsqueeze(0)
    my_embedding = my_embedding.detach().cpu().numpy().tolist()
    return my_embedding

def extract(data_loader, model, layer_name):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for imgs, ids in data_loader:
            imgs = imgs.cuda()
            batch_features = get_feature(model, layer_name, imgs)
            features.extend(batch_features)
            timestamps.extend(ids)
        features, timestamps = np.array(features), np.array(timestamps)
        return features, timestamps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset',       type=str, default=None,        help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--model_name',    type=str, default=None,        choices=['resnet50_ferplus_dag', 'senet50_ferplus_dag'])
    parser.add_argument('--layer_name',    type=str, default='conv5_3_3x3_relu', help='which layer used to extract feature')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    params = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu

    print(f'==> Extracting ferplus embedding...')
    face_dir = config.PATH_TO_RAW_FACE[params.dataset]
    save_name = f"{params.model_name.split('_')[0]}face_{params.feature_level[:3]}"
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], save_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load pre-trained model
    pretrained_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'ferplus') # directory of pre-trained models
    model_dir = './pytorch-benchmarks/model'
    model = load_model(params.model_name, model_dir, pretrained_dir)
    meta = model.meta
    model = model.to(torch.device("cuda"))

    # transform
    transform = compose_transforms(meta)

    # extract embedding video by video
    vids = os.listdir(face_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # csv_file = os.path.join(save_dir, f'{vid}.npy')
        # if os.path.exists(csv_file): continue
        
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
            embeddings, framenames = extract(data_loader, model, params.layer_name)

        # save results
        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes]
        framenames = framenames[indexes]
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if params.feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, EMBEDDING_DIM))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(csv_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((EMBEDDING_DIM, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)
