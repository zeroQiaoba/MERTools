# *_*coding:utf-8 *_*
import os
import re
import pandas as pd
import numpy as np
import struct

## for OPENFACE
## reference: https://gist.github.com/btlorch/6d259bfe6b753a7a88490c0607f07ff8
def read_hog(filename, batch_size=5000):
    """
    Read HoG features file created by OpenFace.
    For each frame, OpenFace extracts 12 * 12 * 31 HoG features, i.e., num_features = 4464. These features are stored in row-major order.
    :param filename: path to .hog file created by OpenFace
    :param batch_size: how many rows to read at a time
    :return: is_valid, hog_features
        is_valid: ndarray of shape [num_frames]
        hog_features: ndarray of shape [num_frames, num_features]
    """
    all_feature_vectors = []
    with open(filename, "rb") as f:
        num_cols, = struct.unpack("i", f.read(4)) # 12
        num_rows, = struct.unpack("i", f.read(4)) # 12
        num_channels, = struct.unpack("i", f.read(4)) # 31

        # The first four bytes encode a boolean value whether the frame is valid
        num_features = 1 + num_rows * num_cols * num_channels
        feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
        feature_vector = np.array(feature_vector).reshape((1, num_features)) # [1, 4464+1]
        all_feature_vectors.append(feature_vector)

        # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
        num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
        # Read in batches of given batch_size
        num_floats_to_read = num_floats_per_feature_vector * batch_size
        # Multiply by 4 because of float32
        num_bytes_to_read = num_floats_to_read * 4

        while True:
            bytes = f.read(num_bytes_to_read)
            # For comparison how many bytes were actually read
            num_bytes_read = len(bytes)
            assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
            num_floats_read = num_bytes_read // 4
            assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
            num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

            feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
            # Convert to array
            feature_vectors = np.array(feature_vectors).reshape((num_feature_vectors_read, num_floats_per_feature_vector))
            # Discard the first three values in each row (num_cols, num_rows, num_channels)
            feature_vectors = feature_vectors[:, 3:]
            # Append to list of all feature vectors that have been read so far
            all_feature_vectors.append(feature_vectors)

            if num_bytes_read < num_bytes_to_read:
                break

        # Concatenate batches
        all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

        # Split into is-valid and feature vectors
        is_valid = all_feature_vectors[:, 0]
        feature_vectors = all_feature_vectors[:, 1:]

        return is_valid, feature_vectors


## for OPENFACE
def read_csv(filename, startIdx):
    data = pd.read_csv(filename)
    all_feature_vectors = []
    for index in data.index:
        features = np.array(data.iloc[index][startIdx:])
        all_feature_vectors.append(features)
    all_feature_vectors = np.array(all_feature_vectors)
    return all_feature_vectors

