import glob
import torch
import warnings
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import videomae.video_transforms as video_transforms 
import videomae.volume_transforms as volume_transforms


# only process one single file
class VideoClsDatasetFrame(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, face_npy, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.face_npy = face_npy
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args

        self.frames = np.load(face_npy)
        self.nframes = len(self.frames)
        self.feature_level = args.feature_level
        
        assert mode == 'test'
        self.data_resize = video_transforms.Compose([
            video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear')
        ])
        self.data_transform = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])

        # generate test dataloader
        self.test_seg = []

        if self.feature_level == 'UTTERANCE':
            for ck in range(self.test_num_segment): # 2
                for cp in range(self.test_num_crop): # 2
                    self.test_seg.append((ck, cp))

        elif self.feature_level == 'FRAME': # 每一帧取他前后窗长为16的一块，计算特征
            for ck in range(self.test_num_segment): # 2
                for cp in range(self.test_num_crop): # 2
                    for frmid in range(self.nframes): # 第A个样本
                        self.test_seg.append((ck, cp, frmid))
                    
        if args.debug:
            for _ in range(2):
                self.__getitem__(0)


    def __getitem__(self, index):
        
        assert self.mode == 'test'

        if self.feature_level == 'UTTERANCE':
            frmid = 0 # 用于return时统一
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_video(self.frames) # 每个视频按照间隔4采样帧，不足16帧补到16帧
        elif self.feature_level == 'FRAME':
            chunk_nb, split_nb, frmid = self.test_seg[index]
            # convert frmid -> (start, end)
            if frmid < 10:
                start, end = 0, 16
            elif frmid > self.nframes - 10:
                start, end = self.nframes-17, self.nframes-1
            else:
                start, end = frmid-8, frmid+8
            # gain chunk
            buffer = self.load_video(self.frames, start, end) # 每个视频按照间隔4采样帧，不足16帧补到16帧

        buffer = self.data_resize(buffer)
        if isinstance(buffer, list):
            buffer = np.stack(buffer, 0) # [nfrm, 160, 160, 3]

        ## 给一个视频采样4块
        spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                        / (self.test_num_crop - 1) # 0
        temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                            / (self.test_num_segment - 1), 0) # 6
        temporal_start = int(chunk_nb * temporal_step)
        spatial_start = int(split_nb * spatial_step)
        if buffer.shape[1] >= buffer.shape[2]:
            buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                        spatial_start:spatial_start + self.short_side_size, :, :]
        else:
            buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                        :, spatial_start:spatial_start + self.short_side_size, :]

        buffer = self.data_transform(buffer) # [3, 16, 160, 160]
        return buffer, '%010d'%(frmid)
        

    # sample: npy face path => 4帧为间隔均匀采样
    def load_video(self, frames, start=None, end=None):
        """Load video content using Decord"""
       
        if self.feature_level == 'UTTERANCE': # 间隔四帧采样
            # gain sample idx: 4帧为间隔均匀采样 [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84]
            idxs = [x for x in range(0, len(frames), self.frame_sample_rate)]
        elif self.feature_level == 'FRAME': # 用start, end直接读取
            idxs = [x for x in range(start, end)]
        # at least self.clip_len
        while len(idxs) < self.clip_len:
            idxs.append(idxs[-1])
        
        # frames -> sampled frames
        buffer = [Image.fromarray(frames[min(idx, len(frames) - 1), :, :, :]).convert('RGB') for idx in idxs] # 按照index读取数据，并转成videomae支持的类型
        return buffer

    def __len__(self):
        return len(self.test_seg)
