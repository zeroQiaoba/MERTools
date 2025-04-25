import os
import logging
import warnings

from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
from my_affectgpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from my_affectgpt.datasets.datasets.mer2025ov_dataset import MER2025OV_Dataset
from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
from my_affectgpt.datasets.datasets.ovmerd_dataset import OVMERD_Dataset


# get name -> dataset_cls
def get_name2cls(dataset):
    if dataset == 'OVMERD': return OVMERD_Dataset()
    if dataset == 'MER2025OV': return MER2025OV_Dataset()
    print ('dataset cls not provided!')
    return None


@registry.register_builder("mer2025ov")
class MER2025OV_Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2025OV_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2025OV_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    

@registry.register_builder("mercaptionplus")
class MERCaptionPlus_Builder(BaseDatasetBuilder):
    train_dataset_cls = MERCaptionPlus_Dataset

    def build_datasets(self):
        logging.info("Building datasets MERCaptionPlus_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    

@registry.register_builder("ovmerd")
class OVMERD_Builder(BaseDatasetBuilder):
    train_dataset_cls = OVMERD_Dataset

    def build_datasets(self):
        logging.info("Building datasets OVMERD_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    
