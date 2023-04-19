import re
import os
import sys
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing

import sklearn
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import config

emos = ['neutral', 'angry', 'happy', 'sad', 'worried',  'surprise']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo

########################################################
############## multiprocess read features ##############
########################################################
def func_read_one(argv=None, feature_root=None, name=None):

    feature_root, name = argv
    feature_dir = glob.glob(os.path.join(feature_root, name+'*'))
    assert len(feature_dir) == 1
    feature_path = feature_dir[0]

    feature = []
    if feature_path.endswith('.npy'):
        single_feature = np.load(feature_path)
        single_feature = single_feature.squeeze()
        feature.append(single_feature)
    else:
        facenames = os.listdir(feature_path)
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_path, facename))
            feature.append(facefeat)

    single_feature = np.array(feature).squeeze()
    if len(single_feature) == 0:
        print ('feature has errors!!')
    elif len(single_feature.shape) == 2:
        single_feature = np.mean(single_feature, axis=0)
    return single_feature
    
def read_data_multiprocess(label_path, feature_root, task='emo', data_type='train', debug=False):

    ## gain (names, labels)
    names, labels = [], []
    assert task in  ['emo', 'aro', 'val', 'whole']
    assert data_type in ['train', 'test1', 'test2', 'test3']
    if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    if data_type == 'test1': corpus = np.load(label_path, allow_pickle=True)['test1_corpus'].tolist()
    if data_type == 'test2': corpus = np.load(label_path, allow_pickle=True)['test2_corpus'].tolist()
    if data_type == 'test3': corpus = np.load(label_path, allow_pickle=True)['test3_corpus'].tolist()
    for name in corpus:
        names.append(name)
        if task in ['aro', 'val']:
            labels.append(corpus[name][task])
        if task == 'emo':
            labels.append(emo2idx[corpus[name]['emo']])
        if task == 'whole':
            corpus[name]['emo'] = emo2idx[corpus[name]['emo']]
            labels.append(corpus[name])

    ## ============= for debug =============
    if debug: 
        names = names[:100]
        labels = labels[:100]
    ## =====================================

    ## names => features
    params = []
    for ii, name in tqdm.tqdm(enumerate(names)):
        params.append((feature_root, name))

    features = []
    with multiprocessing.Pool(processes=8) as pool:
        features = list(tqdm.tqdm(pool.imap(func_read_one, params), total=len(params)))
    feature_dim = np.array(features).shape[-1]

    ## save (names, features)
    print (f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats, name2labels = {}, {}
    for ii in range(len(names)):
        name2feats[names[ii]]  = features[ii]
        name2labels[names[ii]] = labels[ii]
    return name2feats, name2labels, feature_dim


########################################################
##################### data loader ######################
########################################################
class MERDataset(Dataset):

    def __init__(self, label_path, audio_root, text_root, video_root, data_type, debug=False):
        assert data_type in ['train', 'test1', 'test2', 'test3']
        self.name2audio, self.name2labels, self.adim = read_data_multiprocess(label_path, audio_root, task='whole', data_type=data_type, debug=debug)
        self.name2text,  self.name2labels, self.tdim = read_data_multiprocess(label_path, text_root,  task='whole', data_type=data_type, debug=debug)
        self.name2video, self.name2labels, self.vdim = read_data_multiprocess(label_path, video_root, task='whole', data_type=data_type, debug=debug)
        self.names = [name for name in self.name2audio if 1==1]

    def __getitem__(self, index):
        name = self.names[index]
        return torch.FloatTensor(self.name2audio[name]),\
               torch.FloatTensor(self.name2text[name]),\
               torch.FloatTensor(self.name2video[name]),\
               self.name2labels[name]['emo'],\
               self.name2labels[name]['val'],\
               name

    def __len__(self):
        return len(self.names)

    def get_featDim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

## for five-fold cross-validation on Train&Val
def get_loaders(args, config):
    train_dataset = MERDataset(label_path = config.PATH_TO_LABEL[args.train_dataset],
                               audio_root = os.path.join(config.PATH_TO_FEATURES[args.train_dataset], args.audio_feature),
                               text_root  = os.path.join(config.PATH_TO_FEATURES[args.train_dataset], args.text_feature),
                               video_root = os.path.join(config.PATH_TO_FEATURES[args.train_dataset], args.video_feature),
                               data_type  = 'train',
                               debug      = args.debug)

    # gain indices for cross-validation
    whole_folder = []
    whole_num = len(train_dataset.names)
    indices = np.arange(whole_num)
    random.shuffle(indices)

    # split indices into five-fold
    num_folder = args.num_folder
    each_folder_num = int(whole_num / num_folder)
    for ii in range(num_folder-1):
        each_folder = indices[each_folder_num*ii: each_folder_num*(ii+1)]
        whole_folder.append(each_folder)
    each_folder = indices[each_folder_num*(num_folder-1):]
    whole_folder.append(each_folder)
    assert len(whole_folder) == num_folder
    assert sum([len(each) for each in whole_folder if 1==1]) == whole_num

    ## split into train/eval
    train_eval_idxs = []
    for ii in range(num_folder):
        eval_idxs = whole_folder[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii: train_idxs.extend(whole_folder[jj])
        train_eval_idxs.append([train_idxs, eval_idxs])

    ## gain train and eval loaders
    train_loaders = []
    eval_loaders = []
    for ii in range(len(train_eval_idxs)):
        train_idxs = train_eval_idxs[ii][0]
        eval_idxs  = train_eval_idxs[ii][1]
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=SubsetRandomSampler(train_idxs),
                                  num_workers=args.num_workers,
                                  pin_memory=False)
        eval_loader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SubsetRandomSampler(eval_idxs),
                                 num_workers=args.num_workers,
                                 pin_memory=False)
        train_loaders.append(train_loader)
        eval_loaders.append(eval_loader)


    test_loaders = []
    for test_set in args.test_sets:
        test_dataset = MERDataset(label_path = config.PATH_TO_LABEL[args.test_dataset],
                                  audio_root = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.audio_feature),
                                  text_root  = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.text_feature),
                                  video_root = os.path.join(config.PATH_TO_FEATURES[args.test_dataset], args.video_feature),
                                  data_type  = test_set,
                                  debug      = args.debug)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=False)
        test_loaders.append(test_loader)

    ## return loaders
    adim, tdim, vdim = train_dataset.get_featDim()
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim


########################################################
##################### build model ######################
########################################################
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(MLP, self).__init__()

        self.all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        self.module = nn.Sequential(*self.all_layers)
        self.fc_out_1 = nn.Linear(layers[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers[-1], output_dim2)
        
    def forward(self, inputs):
        features = self.module(inputs)
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        return features, emos_out, vals_out


class Attention(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(Attention, self).__init__()

        self.audio_mlp = self.MLP(audio_dim, layers, dropout)
        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.video_mlp = self.MLP(video_dim, layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 3
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 3)
        self.fc_out_1 = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers_list[-1], output_dim2)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, audio_feat, text_feat, video_feat):
        audio_hidden = self.audio_mlp(audio_feat) # [32, 128]
        text_hidden  = self.text_mlp(text_feat)   # [32, 128]
        video_hidden = self.video_mlp(video_feat) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        emos_out  = self.fc_out_1(fused_feat)
        vals_out  = self.fc_out_2(fused_feat)
        return fused_feat, emos_out, vals_out


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1)
        target = target.squeeze().long()
        loss = self.loss(pred, target) / len(pred)
        return loss

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss


########################################################
########### main training/testing function #############
########################################################
def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False):
    
    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    embeddings = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        ## analyze dataloader
        audio_feat, text_feat, visual_feat = data[0], data[1], data[2]
        emos, vals = data[3], data[4].float()
        vidnames += data[-1]
        multi_feat = torch.cat([audio_feat, text_feat, visual_feat], dim=1)
        
        ## add cuda
        emos = emos.cuda()
        vals = vals.cuda()
        audio_feat  = audio_feat.cuda()
        text_feat   = text_feat.cuda()
        visual_feat = visual_feat.cuda()
        multi_feat  = multi_feat.cuda()

        ## feed-forward process
        if args.model_type == 'mlp':
            features, emos_out, vals_out = model(multi_feat)
        elif args.model_type == 'attention':
            features, emos_out, vals_out = model(audio_feat, text_feat, visual_feat)
        emo_probs.append(emos_out.data.cpu().numpy())
        val_preds.append(vals_out.data.cpu().numpy())
        emo_labels.append(emos.data.cpu().numpy())
        val_labels.append(vals.data.cpu().numpy())
        embeddings.append(features.data.cpu().numpy())

        ## optimize params
        if train:
            loss1 = cls_loss(emos_out, emos)
            loss2 = reg_loss(vals_out, vals)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    embeddings = np.concatenate(embeddings)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')

    ## evaluate on dimensional labels
    val_preds  = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    val_mse = mean_squared_error(val_labels, val_preds)

    save_results = {}
    # item1: statistic results
    save_results['val_mse'] = val_mse
    save_results['emo_fscore'] = emo_fscore
    save_results['emo_accuracy'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['val_preds'] = val_preds
    save_results['emo_labels'] = emo_labels
    save_results['val_labels'] = val_labels
    save_results['names'] = vidnames
    # item3: latent embeddings
    if args.savewhole: save_results['embeddings'] = embeddings
    return save_results


########################################################
############# metric and save results ##################
########################################################
def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score

def average_folder_results(folder_save, testname):
    name2preds = {}
    num_folder = len(folder_save)
    for ii in range(num_folder):
        names    = folder_save[ii][f'{testname}_names']
        emoprobs = folder_save[ii][f'{testname}_emoprobs']
        valpreds = folder_save[ii][f'{testname}_valpreds']
        for jj in range(len(names)):
            name = names[jj]
            emoprob = emoprobs[jj]
            valpred = valpreds[jj]
            if name not in name2preds: name2preds[name] = []
            name2preds[name].append({'emo': emoprob, 'val': valpred})

    ## gain average results
    name2avgpreds = {}
    for name in name2preds:
        preds = np.array(name2preds[name])
        emoprobs = [pred['emo'] for pred in preds if 1==1]
        valpreds = [pred['val'] for pred in preds if 1==1]

        avg_emoprob = np.mean(emoprobs, axis=0)
        avg_emopred = np.argmax(avg_emoprob)
        avg_valpred = np.mean(valpreds)
        name2avgpreds[name] = {'emo': avg_emopred, 'val': avg_valpred, 'emoprob': avg_emoprob}
    return name2avgpreds

def gain_name2feat(folder_save, testname):
    name2feat = {}
    assert len(folder_save) >= 1
    names      = folder_save[0][f'{testname}_names']
    embeddings = folder_save[0][f'{testname}_embeddings']
    for jj in range(len(names)):
        name = names[jj]
        embedding = embeddings[jj]
        name2feat[name] = embedding
    return name2feat

def write_to_csv_pred(name2preds, save_path):
    names, emos, vals = [], [], []
    for name in name2preds:
        names.append(name)
        emos.append(idx2emo[name2preds[name]['emo']])
        vals.append(name2preds[name]['val'])

    columns = ['name', 'discrete', 'valence']
    data = np.column_stack([names, emos, vals])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)

def report_results_on_test1_test2(test_label, test_pred):

    # read target file (few for test3)
    name2label = {}
    df_label = pd.read_csv(test_label)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        name2label[name] = {'emo': emo2idx[emo], 'val': val}
    print (f'labeled samples: {len(name2label)}')

    # read prediction file (more for test3)
    name2pred = {}
    df_label = pd.read_csv(test_pred)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        name2pred[name] = {'emo': emo2idx[emo], 'val': val}
    print (f'predict samples: {len(name2pred)}')
    assert len(name2pred) == len(name2label), f'make sure len(name2pred)=len(name2label)'

    emo_labels, emo_preds, val_labels, val_preds = [], [], [], []
    for name in name2label:
        emo_labels.append(name2label[name]['emo'])
        val_labels.append(name2label[name]['val'])
        emo_preds.append(name2pred[name]['emo'])
        val_preds.append(name2pred[name]['val'])

    # analyze results
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    print (f'emo results (weighted f1 score): {emo_fscore:.4f}')
    val_mse = mean_squared_error(val_labels, val_preds)
    print (f'val results (mse): {val_mse:.4f}')
    final_metric = overall_metric(emo_fscore, val_mse)
    print (f'overall metric: {final_metric:.4f}')
    return emo_fscore, val_mse, final_metric


## only fscore for test3
def report_results_on_test3(test_label, test_pred):

    # read target file (few for test3)
    name2label = {}
    df_label = pd.read_csv(test_label)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        name2label[name] = {'emo': emo2idx[emo]}
    print (f'labeled samples: {len(name2label)}')

    # read prediction file (more for test3)
    name2pred = {}
    df_label = pd.read_csv(test_pred)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        name2pred[name] = {'emo': emo2idx[emo]}
    print (f'predict samples: {len(name2pred)}')
    assert len(name2pred) >= len(name2label)

    emo_labels, emo_preds = [], []
    for name in name2label: # on few for test3
        emo_labels.append(name2label[name]['emo'])
        emo_preds.append(name2pred[name]['emo'])

    # analyze results
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    print (f'emo results (weighted f1 score): {emo_fscore:.4f}')
    return emo_fscore, -100, -100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--train_dataset', type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--test_dataset',  type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--test_sets', type=str, default='test1,test2', help='process on which test sets, [test1, test2, test3]')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether save latent embeddings')

    ## Params for model
    parser.add_argument('--layers', type=str, default='256,128', help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=-1, help='number of classes [defined by args.label_path]')
    parser.add_argument('--num_folder', type=int, default=-1, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type for training [mlp or attention]')

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    args = parser.parse_args()

    args.n_classes = 6
    args.num_folder = 5
    args.test_sets = args.test_sets.split(',')

    if args.dataset is not None:
        args.train_dataset = args.dataset
        args.test_dataset  = args.dataset
    assert args.train_dataset is not None
    assert args.test_dataset  is not None

    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    if len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    torch.cuda.set_device(args.gpu)
    print(args)

    
    print (f'====== Reading Data =======')
    train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_loaders(args, config)      
    assert len(train_loaders) == args.num_folder, f'Error: folder number'
    assert len(eval_loaders)   == args.num_folder, f'Error: folder number'
    
    
    print (f'====== Training and Evaluation =======')
    folder_save = []
    folder_evalres = []
    for ii in range(args.num_folder):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]
        start_time = time.time()
        name_time  = time.time()

        print (f'Step1: build model (each folder has its own model)')
        if args.model_type == 'mlp':
            model = MLP(input_dim=adim + tdim + vdim,
                        output_dim1=args.n_classes,
                        output_dim2=1,
                        layers=args.layers)
        elif args.model_type == 'attention':
            model = Attention(audio_dim=adim,
                              text_dim=tdim,
                              video_dim=vdim,
                              output_dim1=args.n_classes,
                              output_dim2=1,
                              layers=args.layers)
        reg_loss = MSELoss()
        cls_loss = CELoss()
        model.cuda()
        reg_loss.cuda()
        cls_loss.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        print (f'Step2: training (multiple epoches)')
        eval_metrics = []
        eval_fscores = []
        eval_valmses = []
        test_save = []
        for epoch in range(args.epochs):

            store_values = {}

            ## training and validation
            train_results = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, optimizer=optimizer, train=True)
            eval_results  = train_or_eval_model(args, model, reg_loss, cls_loss, eval_loader,  optimizer=None,      train=False)
            eval_metric = overall_metric(eval_results['emo_fscore'], eval_results['val_mse']) # bigger -> better
            eval_metrics.append(eval_metric)
            eval_fscores.append(eval_results['emo_fscore'])
            eval_valmses.append(eval_results['val_mse'])
            store_values['eval_emoprobs'] = eval_results['emo_probs']
            store_values['eval_valpreds'] = eval_results['val_preds']
            store_values['eval_names']    = eval_results['names']
            print ('epoch:%d; train_fscore:%.4f; eval_metric:%.4f' %(epoch+1, train_results['emo_fscore'], eval_metric))

            ## testing and saving
            for jj, test_loader in enumerate(test_loaders):
                test_set = args.test_sets[jj]
                test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, optimizer=None, train=False)
                store_values[f'{test_set}_emoprobs']   = test_results['emo_probs']
                store_values[f'{test_set}_valpreds']   = test_results['val_preds']
                store_values[f'{test_set}_names']      = test_results['names']
                if args.savewhole: store_values[f'{test_set}_embeddings'] = test_results['embeddings']
            test_save.append(store_values)
            
        print (f'Step3: saving and testing on the {ii+1} folder')
        best_index = np.argmax(np.array(eval_metrics))
        best_save  = test_save[best_index]
        best_evalfscore = eval_fscores[best_index]
        best_evalvalmse = eval_valmses[best_index]
        folder_save.append(best_save)
        folder_evalres.append([best_evalfscore, best_evalvalmse])
        end_time = time.time()
        print (f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')


    print (f'====== Gain predition on test data =======')
    assert len(folder_save)     == args.num_folder
    assert len(folder_evalres)  == args.num_folder
    save_modelroot = os.path.join(args.save_root, 'model')
    save_predroot  = os.path.join(args.save_root, 'prediction')
    if not os.path.exists(save_predroot): os.makedirs(save_predroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    feature_name = f'{args.audio_feature}+{args.text_feature}+{args.video_feature}'

    ## analyze cv results
    cv_fscore, cv_valmse = np.mean(np.array(folder_evalres), axis=0)
    cv_metric = overall_metric(cv_fscore, cv_valmse)
    res_name = f'f1:{cv_fscore:.4f}_val:{cv_valmse:.4f}_metric:{cv_metric:.4f}'
    save_path = f'{save_modelroot}/cv_features:{feature_name}_{res_name}_{name_time}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path, args=np.array(args, dtype=object))

    for setname in args.test_sets:
        pred_path  = f'{save_predroot}/{setname}-pred-{name_time}.csv'
        label_path = f'./dataset-release/{setname}-label.csv'
        name2preds = average_folder_results(folder_save, setname)
        if args.savewhole: name2feats = gain_name2feat(folder_save, setname)
        write_to_csv_pred(name2preds, pred_path)

        res_name = 'nores'
        if os.path.exists(label_path):
            if setname in ['test1', 'test2']: emo_fscore, val_mse, final_metric = report_results_on_test1_test2(label_path, pred_path)
            if setname in ['test3']:          emo_fscore, val_mse, final_metric = report_results_on_test3(label_path, pred_path)
            res_name = f'f1:{emo_fscore:.4f}_val:{val_mse:.4f}_metric:{final_metric:.4f}'

        save_path = f'{save_modelroot}/{setname}_features:{feature_name}_{res_name}_{name_time}.npz'
        print (f'save results in {save_path}')

        if args.savewhole:
            np.savez_compressed(save_path,
                            name2preds=name2preds,
                            name2feats=name2feats,
                            args=np.array(args, dtype=object))
        else:
            np.savez_compressed(save_path,
                                name2preds=name2preds,
                                args=np.array(args, dtype=object))
