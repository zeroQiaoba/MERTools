"""
get_models: get models and load default configs; 
link: https://github.com/thuiar/MMSA-FET/tree/master
"""
import torch

from .tfn import TFN
from .lmf import LMF
from .mfn import MFN
from .mfm import MFM
from .mult import MULT
from .misa import MISA
from .mctn import MCTN
from .mmim import MMIM
from .graph_mfn import Graph_MFN
from .attention import Attention

class get_models(torch.nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高

        MODEL_MAP = {
            
            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            'attention': Attention,
            'lmf': LMF,
            'misa': MISA,
            'mmim': MMIM,
            'tfn': TFN,
            
            # 只支持align
            'mfn': MFN, # slow
            'graph_mfn': Graph_MFN, # slow
            'mfm': MFM, # slow
            'mctn': MCTN, # slow

            # 支持align/unalign
            'mult': MULT, # slow

        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
