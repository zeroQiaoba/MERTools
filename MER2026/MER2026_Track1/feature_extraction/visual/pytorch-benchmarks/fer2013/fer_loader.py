# -*- coding: utf-8 -*-
"""Contains two data loaders. One is for the Fer 2013 emotion dataset
described in the paper:

Goodfellow, Ian J., et al. "Challenges in representation learning:
A report on three machine learning contests." International Conference on
Neural Information Processing. Springer, Berlin, Heidelberg, 2013.
https://arxiv.org/abs/1307.0414

The second is for the "Fer 2013 plus" dataset, described in the paper:

Barsoum, Emad, Cha Zhang, Cristian Canton Ferrer, and Zhengyou Zhang.
"Training deep networks for facial expression recognition with
crowd-sourced label distribution." In Proceedings of the 18th ACM
International Conference on Multimodal Interaction, pp. 279-283. ACM, 2016.
https://arxiv.org/abs/1608.01041

"""

import os
import csv
import tqdm
import torch
import pickle
import numpy as np
from copy import deepcopy
import PIL.Image

from os.path import join as pjoin

class Fer2013Dataset(torch.utils.data.Dataset):
    """Dataset class helper for the Fer2013 dataset. Converts the csv
    files used to distribute the dataset into a pickle format

    Args:
        data_dir (str): Directory where the original csv files distributed
            with the dataset are found.
        mode (str): The subset of the dataset to use
        transform (torch.transforms): a transformaton that can be applied
            to images on loading
        include_train (bool) [False]: whether to include the training set
            in the loader (it's not required for benchmarking purposes).
    """
    def __init__(self, data_dir, mode='val', transform=None,
                 include_train=False):
        self.data_dir = data_dir
        self.mode = mode
        self.include_train = include_train
        self._transform = transform
        self.pkl_path = pjoin(data_dir, 'pytorch', 'data.pkl')

        if not os.path.isfile(self.pkl_path):
            self.prepare_data()

        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        """Retreive the sample at the given index.

        Args:
            index (int): the index of the sample to be retrieved

        Returns:
            (torch.Tensor): the image
            (int): the label
        """
        im_data = self.data['images'][self.mode][index].astype('uint8')
        image = PIL.Image.fromarray(im_data)
        label = self.data['labels'][self.mode][index]
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def prepare_data(self):
        """Transform raw data from csv format into a dict.

        Args:
            phase, str: 'train'/'val'/'test'.
            size, int. Size of the dataset.
        """
        print('preparing data...')
        with open(pjoin(self.data_dir, 'fer2013.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader) # skip header
            rows = [row for row in reader]

        train_ims, val_ims, test_ims = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        for row in tqdm.tqdm(rows):
            subset = row[2]
            raw_im = np.array([int(x) for x in row[1].split(' ')])
            im = np.repeat(raw_im.reshape(48,48)[:,:,np.newaxis], 3, axis=2)
            if subset == 'Training':
                train_labels.append(int(row[0]))
                train_ims.append(im)
            elif subset == 'PublicTest':
                val_labels.append(int(row[0]))
                val_ims.append(im)
            elif subset == 'PrivateTest':
                test_labels.append(int(row[0]))
                test_ims.append(im)
            else:
                raise ValueError('unrecognised subset: {}'.format(subset))

        data = {'labels': {}, 'images': {}}
        data['labels']['val'] = np.array(val_labels)
        data['labels']['test'] = np.array(test_labels)

        data['images']['val'] = np.array(val_ims)
        data['images']['test'] = np.array(test_ims)

        if self.include_train:
            data['labels']['train'] = np.array(train_labels)
            data['images']['train'] = np.array(train_ims)

        for key in 'images', 'labels':
            assert len(data[key]['val']) == 3589, 'unexpected length'
            assert len(data[key]['test']) == 3589, 'unexpected length'
            if self.include_train:
                assert len(data[key]['train']) == 28709, 'unexpected length'

        if not os.path.exists(os.path.dirname(self.pkl_path)):
                os.makedirs(os.path.dirname(self.pkl_path))

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data, f)

    def __len__(self):
        """Return the total number of images in the datset.

        Return:
            (int) the number of images.
        """
        return self.data['labels'][self.mode].size

class Fer2013PlusDataset(Fer2013Dataset):
    """Dataset class helper for the Fer2013plus dataset. Converts the csv
    files used to distribute the dataset into a pickle format
    """

    def __init__(self, *args, **kwargs):
        super(Fer2013PlusDataset, self).__init__(*args, **kwargs)
        self.update_labels()

    def update_labels(self):
        """Update dataset to use FerPlus labels, rather Fer2013 dataset labels

        Aim to reproduce the Microsoft CNTK cleaning process. These are based
        on some heuristics about the level of ambiguity in the annotator labels
        that should be tolerated to ensure that the dataset is moderately
        clearn. We generate hard labels, rather than soft ones for evaluation.
        """
        with open(pjoin(self.data_dir, 'fer2013new.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader) # skip header
            rows = [row for row in reader]

        set_map = {'Training': 1, 'PublicTest': 2, 'PrivateTest': 3}
        sets = np.atleast_2d([set_map[x[0]] for x in rows]).T
        labels = [np.atleast_2d([int(x) for x in r[2:]]) for r in rows]
        labels = np.concatenate(labels, axis=0)
        orig_labels = deepcopy(labels)
        outliers = (labels <=1)
        labels[outliers] = 0 # drop outliers
        dropped = 1 - (labels.sum() / orig_labels.sum())
        print('dropped {:.1f}%% of votes as outliers'.format(dropped * 100))
        num_votes = np.sum(labels, 1)
        # following CNTK processing - there are three reasons to drop examples:
        # (1) If the majority votes for either "unknown-face" or "not-face"
        # (2) If more than three votes share the maximum voting value
        # (3) If the max votes do not account for more than half of the votes
        to_drop = np.zeros((labels.shape[0], 1))
        for ii in tqdm.tqdm(range(labels.shape[0])):
            max_vote = np.max(labels[ii,:])
            max_vote_emos = np.where(labels[ii,:] == max_vote)[0]
            drop = any([x in [8, 9] for x in max_vote_emos])
            num_max_votes = max_vote_emos.size
            drop = drop or num_max_votes >= 3
            drop = drop or (num_max_votes * max_vote <= 0.5 * num_votes[ii])
            to_drop[ii] = drop

        # TODO(samuel): verify that this is correct
        assert to_drop.sum() == 3079, 'unexpected number of dropped votes'
        # NOTE: use slightly different "keep" indicies, depending on how data
        # is accessed.
        val_keep_ims = np.logical_not(to_drop[sets == 2])
        test_keep_ims = np.logical_not(to_drop[sets == 3])
        val_keep_labels = np.logical_and(sets == 2,
                                      np.logical_not(to_drop)).flatten()
        test_keep_labels = np.logical_and(sets == 3,
                                      np.logical_not(to_drop)).flatten()
        val_labels = labels[val_keep_labels, :]
        test_labels = labels[test_keep_labels, :]
        print('val size: ', len(val_labels), 'test size: ', len(test_labels))
        # update images in place
        self.data['images']['val'] = \
                            self.data['images']['val'][val_keep_ims,:,:,:]
        self.data['images']['test'] = \
                            self.data['images']['test'][test_keep_ims,:,:,:]

        # create "hard labels" with voting
        self.data['labels']['val'] = np.argmax(val_labels, 1)
        self.data['labels']['test'] = np.argmax(test_labels, 1)
