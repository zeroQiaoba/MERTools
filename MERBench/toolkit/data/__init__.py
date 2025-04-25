from torch.utils.data import Dataset

from .feat_data import Data_Feat

# 目标：输入 (names, labels, data_type)，得到所有特征与标签
class get_datasets(Dataset):

    def __init__(self, args, names, labels):

        MODEL_DATASET_MAP = {
            
            # 解析特征
            'attention': Data_Feat,
            'lf_dnn': Data_Feat,
            'lmf': Data_Feat,
            'misa': Data_Feat,
            'mmim': Data_Feat,
            'tfn': Data_Feat,
            'mfn': Data_Feat,
            'graph_mfn': Data_Feat,
            'ef_lstm': Data_Feat, 
            'mfm': Data_Feat,
            'mctn': Data_Feat,
            'mult': Data_Feat,

        }

        self.dataset_class = MODEL_DATASET_MAP[args.model]
        self.dataset = self.dataset_class(args, names, labels)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def collater(self, instances):
        return self.dataset.collater(instances)
         
    def get_featdim(self):
        return self.dataset.get_featdim()