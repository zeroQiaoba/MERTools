# *_*coding:utf-8 *_*
"""
VGGish: https://arxiv.org/abs/1609.09430
official github repo: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
"""

import os
import glob
import time
import argparse
import numpy as np

from vggish import vggish_input
from vggish import vggish_params
from vggish import vggish_slim
import tensorflow.compat.v1 as tf # version: 1.15.0 (gpu)
tf.disable_v2_behavior()

# import config
import sys
sys.path.append('../../')
import config

def extract(audio_files, save_dir, feature_level, batch_size=2048):
    start_time = time.time()
    
    if feature_level == 'FRAME':     label_interval = 50.0
    if feature_level == 'UTTERANCE': label_interval = 500.0

    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'vggish/vggish_model.ckpt')
        vggish_slim.load_vggish_slim_checkpoint(sess, model_file)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)   # get one layer
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME) # get one layer

        for i, audio_file in enumerate(audio_files, 1):
            print(f'Processing "{os.path.basename(audio_file)}" ({i}/{len(audio_files)})...')
            vid = os.path.basename(audio_file)[:-4]
            samples = vggish_input.wavfile_to_examples(audio_file, label_interval / 1000.0) # (segment_num, height(96), width(64))
            sample_size = samples.shape[0]

            # model inference (max sample size: 6653, will cause OOM. Need to chunk samples.)
            embeddings = []
            num_batches =  int(np.ceil(sample_size / batch_size))
            for i in range(num_batches):
                examples_batch = samples[i*batch_size:min((i+1)*batch_size, sample_size)]
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})
                embeddings.append(embedding_batch) 
            embeddings = np.row_stack(embeddings) # (segment_num, featdim=128)

            # save feature
            csv_file = os.path.join(save_dir, f'{vid}.npy')
            if feature_level == 'UTTERANCE':
                embeddings = np.array(embeddings).squeeze()
                if len(embeddings.shape) != 1:
                    embeddings = np.mean(embeddings, axis=0) # (featdim=128)
                np.save(csv_file, embeddings)
            else:
                np.save(csv_file, embeddings)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='feature_level: FRAME or UTTERANCE')
    parser.add_argument('--dataset', type=str, default='MER2023', help='input dataset')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]

    # in: get audios
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # out: check dir
    dir_name = f'vggish_{args.feature_level[:3]}'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # extract features
    extract(audio_files, save_dir, args.feature_level)
