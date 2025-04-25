
from videomae.transforms import *
from videomae.kinetics import VideoClsDatasetFrame

def build_dataset(args, face_npy):
    
    dataset = VideoClsDatasetFrame(
        face_npy=face_npy,
        data_path='/',
        mode='test', # 'train'
        clip_len=args.num_frames, # 16
        frame_sample_rate=args.sampling_rate, # 4
        num_segment=1,
        test_num_segment=args.test_num_segment, # 2
        test_num_crop=args.test_num_crop, # 2
        num_crop=3,
        keep_aspect_ratio=True,
        crop_size=args.input_size, # 160
        short_side_size=args.short_side_size, #
        new_height=256, 
        new_width=320,
        args=args,
    )
    
    return dataset
