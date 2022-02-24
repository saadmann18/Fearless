from paderbox.notebook import *
import paderbox as pb
import padercontrib as pc
from padertorch.data.segment import get_segment_boundaries
from padertorch.data.utils import collate_fn
from padercontrib.database.chime5.database import activity_time_to_frequency

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data_preparation(data, dataset, batch_size=10, shuffle=False):
    
    """Accessing audio stream from train stream"""
    data = data
    dataset = dataset
    mfcc = pb.transform.mfcc

    """Adding a activity dictionary to existing dataset"""
    def activity(dataset):
        get_activity = data.get_activity(dataset)
        dataset['activity'] = get_activity
        dataset['audio_path'] = dataset['audio_path']['observation']
        return dataset
    
    """zero extension"""
    def zero_extension(dataset):
        x = np.arange(dataset['num_samples'])
        desired_length = 32000
        num_splits = np.ceil(x.shape[0]/desired_length)
        padding = int(num_splits*desired_length - x.shape[0])
        x_pad = np.pad(x, (0,padding), 'constant', constant_values=0)
        dataset['padded_num_samples'] = len(x_pad)
        dataset['activity'].shape = (len(x_pad),)
        return dataset
    
    """Segment the audio in dataset"""
    def segmentation(dataset, chunk_size=32000):
        segment_audio = []
        boundaries = get_segment_boundaries(dataset['padded_num_samples'], chunk_size)
        for start,stop in boundaries:
            audio_chunk = dataset.copy()
            audio_chunk.update(audio_start = start)
            audio_chunk.update(audio_stop = stop)
            audio_chunk.update(label = (activity_time_to_frequency(dataset['activity'][start:stop], 
                                            stft_window_length=400, 
                                            stft_shift=160, 
                                            stft_fading=None)))
            segment_audio.append(audio_chunk)
        return segment_audio

    """Read the audio file"""

    def mfcc_feature(dataset):
        start = dataset['audio_start']
        stop = dataset['audio_stop']
        if dataset['num_samples'] % 32000 == 0:
            audio = dataset['audio_path']
            feature = mfcc(pb.io.load_audio(audio, start=start, stop=stop), stft_shift=160)
        else:
            audio = pb.io.load_audio(dataset['audio_path']).copy()
            y = np.arange(len(audio))
            desired_length = 32000
            num_splits = np.ceil(y.shape[0]/desired_length)
            padding = int(num_splits*desired_length - y.shape[0])
            y_pad = len(np.pad(y, (0,padding), 'constant', constant_values=0))
            audio.resize(y_pad)
            feature = mfcc(audio[start:stop], stft_shift=160)
        dataset['features'] = feature.astype(np.float32)
        dataset['label'] = np.asarray(dataset['label']).astype(np.float32)
        return dataset

    """Keeping only needed dicitionary"""
    def new_dataset(dataset):
        dic = dict()
        dic['example_id'] = dataset['example_id']
        dic['features'] = np.expand_dims(dataset['features'], axis=0)
        dic['features_shape'] = dic['features'].shape
        dic['label'] = dataset['label']
        return dic

    """Stacking all the batch of features and class label to nparray"""
    def conv_list_nparray(dataset):
        dataset['features'] = np.stack(dataset['features'])
        dataset['label'] = np.vstack(dataset['label'])
        return dataset


    """ Mapping, shuffling, Prefetch, unbatch, batch_map, batch and collate_fn"""
    dataset = dataset.map(activity)
    dataset = dataset.map(zero_extension)  
    dataset = dataset.map(segmentation)
    
#    if shuffle:
#        dataset = dataset.shuffle()
    dataset = dataset.prefetch(num_workers=8, buffer_size=8).unbatch()
    dataset = dataset.map(mfcc_feature)
    dataset = dataset.map(new_dataset)
    dataset = dataset.batch(batch_size).map(collate_fn)
    dataset = dataset.map(conv_list_nparray)
    
    return dataset