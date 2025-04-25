import os
import glob
import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

import videomae.modeling_finetune
from videomae.datasets import build_dataset
from videomae.engine_for_finetuning import final_test
import videomae.utils as utils

# import config
import sys
sys.path.append('../../')
import config

def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--dataset', default='MER2023', type=str)
    parser.add_argument('--feature_level', default='UTTERANCE', type=str)
    parser.add_argument('--debug', action='store_true', default=False)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    ## me: for more attention types
    parser.add_argument('--attn_type', default='joint', choices=['joint', 'factorised', 'tokenfuser', 'perceiver', 'only_spatial',
                                                                 'part_window', 'temporal_window', 'factorised2',
                                                                 'temporal_pyramid', 'st_pyramid', 'local_global'],
                        type=str, help='attention type for spatiotemporal modeling')
    ### factorised
    parser.add_argument('--fst_share_st_attn', action='store_true',
                        help='use shared spatial/temporal layer for factorised spatiotemporal attention')
    parser.add_argument('--fst_temporal_first', action='store_true',
                        help='use temporal layer before spatial layer for factorised spatiotemporal attention')
    ### token fuser
    parser.add_argument('--tf_start_layer', type=int, default=7, help='start layer for token fuser')
    parser.add_argument('--tf_num_tokens', type=int, default=8, help='number of tokens for token fuser')
    parser.add_argument('--tf_bottleneck_dim', type=int, default=64, help='bottleneck dim for token fuser')
    ### perceiver
    parser.add_argument('--p_start_layer', type=int, default=7, help='start layer for perceiver')
    parser.add_argument('--p_num_latents', type=int, default=16, help='number of latents for perceiver')
    parser.add_argument('--p_num_layer', type=int, default=3, help='number of perceiver blocks')
    parser.add_argument('--p_num_latent_layer', type=int, default=2, help='number of self-attn layers in perceiver block')
    parser.add_argument('--p_dim', type=int, default=768, help='hidden dim for perceiver')
    parser.add_argument('--p_num_cross_heads', type=int, default=4, help='number of cross-attn heads for perceiver')
    ### part window
    parser.add_argument('--part_win_size', type=int, nargs='+', default=(2,2,10), help='window size (t,h,w) for part window attn')
    parser.add_argument('--part_cls_type', type=str, default='org', help='classification type for part window attn')
    parser.add_argument('--part_local_first', action='store_true', help='perform global part attn after local window attn for part window attn')
    ### temporal window
    parser.add_argument('--tem_win_size', type=int, nargs='+', default=(1,2,4,8),
                        help='temporal window size for each temporal window attn block')
    parser.add_argument('--tem_win_depth', type=int, nargs='+', default=(6,6,6,6),
                        help='depth for each temporal window attn block')
    ### temporal pyramid
    parser.add_argument('--tem_pyr_depth', type=int, nargs='+', default=(8,8,8),
                        help='depth for each temporal pyramid block')
    parser.add_argument('--tem_pyr_type', type=str, default='conv', choices=['avg', 'max', 'conv'],
                        help='downsample type for temporal pyramid attention')
    parser.add_argument('--tem_pyr_kernel_size', type=int, default=2,
                        help='kernel_size for temporal pyramid downsample')
    parser.add_argument('--tem_pyr_stride', type=int, default=2,
                        help='stride for temporal pyramid downsample')
    parser.add_argument('--tem_pyr_type_up', type=str, default='repeat', choices=['repeat', 'conv'],
                        help='upsample type for temporal pyramid attention')
    parser.add_argument('--tem_pyr_no_use_multiscale_feature', action='store_true',
                        help='do not use multiscale features for final classification')
    ### st pyramid
    parser.add_argument('--st_pyr_depth', type=int, nargs='+', default=(8,8,8),
                        help='depth for each spatial temporal pyramid block')
    parser.add_argument('--st_pyr_type', type=str, default='conv', choices=['avg', 'max', 'conv'],
                        help='downsample type for spatial temporal pyramid attention')
    parser.add_argument('--st_pyr_kernel_size', type=int, default=2,
                        help='kernel_size for spatial temporal pyramid downsample')
    parser.add_argument('--st_pyr_stride', type=int, default=2,
                        help='stride for spatial temporal pyramid downsample')
    parser.add_argument('--st_pyr_type_up', type=str, default='repeat', choices=['repeat', 'conv'],
                        help='upsample type for spatial temporal pyramid attention')
    parser.add_argument('--st_tf_num_tokens', type=int, nargs='+',  default=(8,), help='number of tokens for spatial downsampling in spatial temporal pyramid attention')
    parser.add_argument('--st_tf_bottleneck_dim', type=int, default=128, help='bottleneck dim for spatial downsampling in spatial temporal pyramid attention')
    parser.add_argument('--st_pyr_no_multiscale_feature',  action='store_true', help='do not use multiscale features for final classification')
    parser.add_argument('--st_pyr_spatial_only_cross_attn', action='store_true', help='only use cross attn for spatial upsample in spatial temporal pyramid attention')
    parser.add_argument('--st_pyr_window_size', type=int, nargs='+', default=None, help='window size for applying 3d shifted window attention in the first stage')
    parser.add_argument('--st_pyr_disable_temporal_pyr', action='store_true', help='do not use temporal pyramid in spatial temporal pyramid attention')
    parser.add_argument('--st_pyr_disable_spatial_pyr', action='store_true', help='do not use spatial pyramid in spatial temporal pyramid attention')
    ### local_global
    parser.add_argument('--lg_region_size', type=int, nargs='+', default=(2,2,10),
                        help='region size (t,h,w) for local_global attention')
    parser.add_argument('--lg_first_attn_type', type=str, default='self', choices=['cross', 'self'],
                        help='the first attention layer type for local_global attention')
    parser.add_argument('--lg_third_attn_type', type=str, default='cross', choices=['cross', 'self'],
                        help='the third attention layer type for local_global attention')
    parser.add_argument('--lg_attn_param_sharing_first_third', action='store_true',
                        help='share parameters of the first and the third attention layers for local_global attention')
    parser.add_argument('--lg_attn_param_sharing_all', action='store_true',
                        help='share all the parameters of three attention layers for local_global attention')
    parser.add_argument('--lg_classify_token_type', type=str, default='org', choices=['org', 'region', 'all'],
                        help='the token type in final classification for local_global attention')
    parser.add_argument('--lg_no_second', action='store_true',
                        help='no second (inter-region) attention for local_global attention')
    parser.add_argument('--lg_no_third', action='store_true',
                        help='no third (local-global interaction) attention for local_global attention')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
  

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=2)
    parser.add_argument('--test_num_crop', type=int, default=2)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=6, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # me:
    parser.add_argument('--val_metric', type=str, default='acc1', choices=['acc1', 'acc5', 'war', 'uar', 'weighted_f1', 'micro_f1', 'macro_f1'],
                        help='validation metric for saving best ckpt')
    # me: for more flexible depth, NOTE: only works when 'no_depth' model is used!
    parser.add_argument('--depth', default=None, type=int,
                        help='specify model depth, NOTE: only works when no_depth model is used!')

    known_args, _ = parser.parse_known_args()
    return parser.parse_args()


def main(args):
    print(args)

    # -------------------- define face_dir and save_dir ---------------------- #
    ## feature extraction
    face_dir = config.PATH_TO_RAW_FACE[args.dataset]

    # gain save_dir
    model_name, epoch_name = args.finetune.split('/')[-2:]
    epoch_name = epoch_name[:-4].split('-')[1]
    model_name = f'{model_name}-{epoch_name}'
    save_dir = os.path.join(config.PATH_TO_FEATURES[args.dataset], f'{model_name}-{args.feature_level[:3]}')
    # save_dir = os.path.join('./', f'sunlicai-{model_name}-{args.feature_level[:3]}') # 临时测试，保存在当前目录下
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    # -------------------- load model and weight ---------------------- #
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # load model
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        # me: for more attention types
        attn_type=args.attn_type,
        fst_share_st_attn=args.fst_share_st_attn, # for factorised
        fst_temporal_first=args.fst_temporal_first,
        tf_start_layer=args.tf_start_layer,  # for token fuser
        tf_num_tokens=args.tf_num_tokens,
        tf_bottleneck_dim=args.tf_bottleneck_dim,
        p_start_layer=args.p_start_layer, p_num_latents=args.p_num_latents, p_num_layer=args.p_num_layer, # for perceiver
        p_num_latent_layer=args.p_num_latent_layer, p_dim=args.p_dim, p_num_cross_heads=args.p_num_cross_heads,
        part_win_size=args.part_win_size, part_cls_type=args.part_cls_type, part_local_first=args.part_local_first,  # for part window attention
        tem_win_size=args.tem_win_size, tem_win_depth=args.tem_win_depth,  # for temporal window attention
        tem_pyr_depth=args.tem_pyr_depth, tem_pyr_kernel_size=args.tem_pyr_kernel_size, # for temporal pyramid
        tem_pyr_stride=args.tem_pyr_stride, tem_pyr_type=args.tem_pyr_type, tem_pyr_type_up=args.tem_pyr_type_up,
        tem_pyr_no_use_multiscale_feature=args.tem_pyr_no_use_multiscale_feature,
        st_pyr_depth=args.st_pyr_depth, st_pyr_kernel_size=args.st_pyr_kernel_size, st_pyr_stride=args.st_pyr_stride, # for spatial temporal pyramid
        st_pyr_type=args.st_pyr_type, st_pyr_type_up=args.st_pyr_type_up,
        st_tf_num_tokens=args.st_tf_num_tokens, st_tf_bottleneck_dim=args.st_tf_bottleneck_dim,
        st_pyr_no_multiscale_feature=args.st_pyr_no_multiscale_feature,
        st_pyr_spatial_only_cross_attn=args.st_pyr_spatial_only_cross_attn,
        st_pyr_window_size=args.st_pyr_window_size,
        st_pyr_disable_temporal_pyr=args.st_pyr_disable_temporal_pyr,
        st_pyr_disable_spatial_pyr=args.st_pyr_disable_spatial_pyr,
        lg_region_size=args.lg_region_size, lg_first_attn_type=args.lg_first_attn_type, # for local_global
        lg_third_attn_type=args.lg_third_attn_type,
        lg_attn_param_sharing_first_third=args.lg_attn_param_sharing_first_third,
        lg_attn_param_sharing_all=args.lg_attn_param_sharing_all,
        lg_classify_token_type=args.lg_classify_token_type,
        lg_no_second=args.lg_no_second, lg_no_third=args.lg_no_third,
    )
    patch_size = model.patch_embed.patch_size
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1]) # (8, 10, 10)
    args.patch_size = patch_size # (16, 16)

    ## load pretrained weight
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        utils.load_state_dict(model, checkpoint_model, prefix='')

    ## convert to cuda
    model.to(device)

    # -------------------- feature extractor ---------------------- #
    face_npys = glob.glob(face_dir + '/*/*.npy')
    ## 打乱vid，方便进行双管齐下的特征提取
    indices = np.arange(len(face_npys))
    random.shuffle(indices)
    face_npys = np.array(face_npys)[indices]
    ## debug下只测试10个样本
    if args.debug:
        face_npys = face_npys[:10]
    print (f'process sample number: {len(face_npys)}')

    for ii, face_npy in enumerate(face_npys):
        print (f'process on {ii}|{len(face_npys)}: {face_npy}')

        vid = os.path.basename(face_npy).rsplit('.', 1)[0]
        save_file = os.path.join(save_dir, f'{vid}.npy')
        if os.path.exists(save_file):
            continue

        # load test dataset
        dataset_test = build_dataset(args, face_npy)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # feature extraction
        embeddings = final_test(data_loader_test, model, args.feature_level, device)
        np.save(save_file, embeddings)

        # # compare with pre-features
        # yy = np.load('/share/home/lianzheng/chinese-mer-2023/dataset/mer2023-dataset-process/features/pretrain-videomae-base-VoxCeleb2-99/' + f'{vid}.npy')
        # print (yy - embeddings)


'''
# command line [debug version] => [ok], 修改为每个视频处理后会产生些许差异，但是差别很小 => 我感觉还算可以接受吧
python extract_sun_videomae.py --dataset MER2023 --feature_level UTTERANCE --batch_size 64 
                               --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune videomae/pretrain_models/videomae-base-VoxCeleb2/checkpoint-99.pth
                               --debug

# 实现 FRAME 级别特征抽取 [debug version] => [ok]
python extract_sun_videomae.py --dataset MER2023 --feature_level FRAME --batch_size 64 
                               --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune videomae/pretrain_models/videomae-base-VoxCeleb2/checkpoint-99.pth
                               --debug

# 实现 FRAME 级别特征抽取 for AFFWILD2
CUDA_VISIBLE_DEVICES=0 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune videomae/pretrain_models/videomae-base-VoxCeleb2/checkpoint-99.pth        >1.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_224 --input_size 224 --short_side_size 224 --finetune videomae/pretrain_models/videomae-base-K400-mer2023/checkpoint-299.pth    >2.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_224 --input_size 224 --short_side_size 224 --finetune videomae/pretrain_models/videomae-large-K400-VoxCeleb2/checkpoint-35.pth  >3.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_224 --input_size 224 --short_side_size 224 --finetune videomae/pretrain_models/videomae-large-K400-mer2023/checkpoint-49.pth    >4.out &

# 数据量太大了，采用双管齐下的策略，进行特征提取 [test on videomae-base-VoxCeleb2/checkpoint-99]
CUDA_VISIBLE_DEVICES=0 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune videomae/pretrain_models/videomae-base-VoxCeleb2/checkpoint-99.pth        >1.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune videomae/pretrain_models/videomae-base-VoxCeleb2/checkpoint-99.pth        >2.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune videomae/pretrain_models/videomae-base-VoxCeleb2/checkpoint-99.pth        >3.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u extract_sun_videomae.py --dataset AFFWILD2 --feature_level FRAME --batch_size 64 --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune videomae/pretrain_models/videomae-base-VoxCeleb2/checkpoint-99.pth        >4.out &

## 测试过程中，好像large模型有些问题，需要进一步debug

'''
if __name__ == '__main__':
    opts = get_args()
    main(opts)
