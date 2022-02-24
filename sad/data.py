from paderbox.notebook import *
import paderbox as pb
import padercontrib as pc
from padertorch.data.segment import get_segment_boundaries
from padertorch.data.utils import collate_fn

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

    """Chunk the audio data"""
    
    def chunk_4s_with_label(audio, chunk_size=4*8000):
        concat_audio = {}
        feature = []
        label = []
        boundaries = get_segment_boundaries(audio['num_samples'], chunk_size)
        audio_chunk = audio.copy()
        for start,stop in boundaries:
            audio_chunk.update(audio_start = start)
            audio_chunk.update(audio_stop = stop)
            feature.append(mfcc(pb.io.load_audio(audio['audio_path'], start=start, stop=stop)))
            
            audio_chunk.update(feature = feature)
            label.append(int(any(audio['activity'][start:stop] &1)))
            audio_chunk.update(label = label)
            concat_audio.update(audio_chunk)
        return concat_audio


    """Keeping only needed dicitionary"""
    def final(dataset):
        dic = dict()
        dic['example_id'] = dataset['example_id']
        dic['feature'] = np.expand_dims(dataset['feature'], axis=0).reshape(-1,1,199,13)
        dic['feature_shape'] = dic['feature'].shape
        dic['label'] = dataset['label']

        return dic
    

    """ Mapping, shuffling, Prefetch, unbatch, batch_map, batch and collate_fn"""

    dataset = dataset.map(activity)
    dataset = dataset.map(chunk_4s_with_label)
    dataset = dataset.map(final)
    dataset = dataset.prefetch(num_workers=8, buffer_size=8) #comment unbatch

    return dataset