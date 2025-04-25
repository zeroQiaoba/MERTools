# *_*coding:utf-8 *_*
import os
import time
import glob
import shutil
import argparse

import numpy as np
from tqdm import tqdm
import concurrent.futures
from handcrafted_feature_func import OPENSMILE, Librosa

# import config
import sys
sys.path.append('../../')
import config

class Worker(object):
    def __init__(self):
        self.count = 0

    def process_one_audio(
            self,
            audio_file,
            save_dir,
            output_dir,
            feature_extractor,
            feature_set,
            label_interval,
            feature_level,
            frame_mode_functionals_param
        ):
        vid = os.path.basename(audio_file)[:-4]
        if feature_extractor.name == 'opensmile':
            output_file = os.path.join(output_dir, f'{vid}.csv')
            feature = feature_extractor.extract_acoustic_feature(
                input_file=audio_file,
                output_file=output_file,
                feature_set=feature_set,
                feauture_level=feature_level,
                frame_mode_functionals_param=frame_mode_functionals_param
            )
        elif feature_extractor.name == 'librosa':
            feature = feature_extractor.extract_acoustic_feature(
                input_file=audio_file,
                feature_set=feature_set
            )
        else:
            raise Exception('Not supported acoustic feature extractor!')

        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if feature_level == 'UTTERANCE':
            feature = np.array(feature).squeeze() # [C,]
            if len(feature.shape) != 1: # change [T, C] => [C,]
                feature = np.mean(feature, axis=0)
            np.save(csv_file, feature)
        else:
            np.save(csv_file, feature)
            
        return audio_file


    def construct_feature_extractor(self, name):
        if name == 'opensmile':
            feature_extractor = OPENSMILE(config.PATH_TO_OPENSMILE)
        elif name == 'Librosa':
            feature_extractor = Librosa()
        else:
            raise Exception(f'Error: not supported feature extractor "{name}"!')
        return feature_extractor

    def run(
            self,
            audio_files,
            save_dir,
            feature_extractor,
            feature_set,
            label_interval,
            feature_level='FRAME',
            frame_mode_functionals_param=None,
            del_opensmile_feature_file=True,
            multi_process=False,
        ):
        # feature extractor
        feature_extractor = self.construct_feature_extractor(feature_extractor)
        tmp_dir = None
        if feature_extractor.name == 'opensmile':
            # make tmp dir in the current
            tmp_dir = f'./saved/tmp_dir'  # used to store the feature file output by opensmile
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)

        # extract features for each audio file
        n_files = len(audio_files)
        start_time = time.time()
        count = 0
        if multi_process: # using multi process to extract acoustic features for each audio file
            with concurrent.futures.ProcessPoolExecutor() as executor:
                tasks = [executor.submit(self.process_one_audio,
                                         audio_file,
                                         save_dir,
                                         tmp_dir,
                                         feature_extractor,
                                         feature_set,
                                         label_interval,
                                         feature_level,
                                         frame_mode_functionals_param) \
                         for audio_file in audio_files]
                for task in concurrent.futures.as_completed(tasks):
                    try:
                        audio_file = task.result()
                        count += 1
                    except Exception as e:
                        print('When process "{}", exception "{}" occurred!'.format(audio_file, e))
                    else:
                        print(f'"{audio_file:<50}" done, rate of progress: {100.0 * count / n_files:3.0f}% ({count}/{n_files})')
        else: # sequentially process [only for test]
            for audio_file in tqdm(audio_files):
                self.process_one_audio(audio_file, save_dir, tmp_dir, feature_extractor, feature_set, label_interval,
                                       feature_level=feature_level, frame_mode_functionals_param=frame_mode_functionals_param)
                break
        end_time = time.time()
        print('Time used for acoustic feature extraction: {:.1f} s'.format(end_time - start_time))

        # del tmp dir
        if feature_extractor.name == 'opensmile' and del_opensmile_feature_file:
            shutil.rmtree(tmp_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--feature_extractor', type=str, default='opensmile', help='name of feature extractor')
    parser.add_argument('--feature_set', type=str, default='IS09', help='name of feature set')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    # ------ 临时测试SNR对于结果的影响 ------
    parser.add_argument('--noise_case', type=str, default=None, help='extract feature of different noise conditions')
    # -----------------------------------------
    args = parser.parse_args()

    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]
    if args.noise_case is not None:
        audio_dir += '_' + args.noise_case

    feature_extractor = args.feature_extractor
    feature_set = args.feature_set
    feature_level = args.feature_level

    print(f'==> Extracting "{feature_set}" using "{feature_extractor}"...')

    # infer label interval for specified tasks from audio_dir
    assert feature_level in ['FRAME', 'UTTERANCE']
    if feature_level == 'FRAME':     label_interval = 50
    if feature_level == 'UTTERANCE': label_interval = 500
    print(f'==> Note: for "{audio_dir}", the label interval is {label_interval}ms.')

    # in: get audios
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
    print(f'Find total "{len(audio_files)}" audio files.')

    # out: save dir
    if args.noise_case is None:
        dir_name = '%s_%s' %(feature_set, feature_level[:3])
    else:
        dir_name = '%s_noise%s_%s' %(feature_set, args.noise_case, feature_level[:3])
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # multi-processor feature extractor
    my_worker = Worker()
    my_worker.run(
        audio_files=audio_files,
        save_dir=save_dir,
        feature_extractor=feature_extractor,
        feature_set=feature_set,
        label_interval=label_interval,
        multi_process=True,
        feature_level=feature_level,
    )
