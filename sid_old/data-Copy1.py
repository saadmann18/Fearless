from padercontrib.database.fearless import Fearless
import padertorch as pt
import paderbox as pb
import numpy as np
import paderbox as pb
from padercontrib.database.iterator import AudioReader
from paderbox.transform import stft,fbank
import scipy
import pydub
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import pandas as pd
from pydub import AudioSegment
import torch
from padertorch.data.utils import collate_fn
import lazy_dataset


def prepare_data(example, batch_size, shuffle = False):
    

    ds = example
    
    def prep_features(example):
        padded_audio = []
        fbank_data = []
        
        """ Obtain audio segments from the dataset"""
        """ If segments smaller than 4secs, pad with silence. Else, extract 4secs from larger audio segments """
        audio_sam = AudioSegment.from_wav(example['audio_path']['observation'])
        if audio_sam.duration_seconds < 4:
            pad_ms = (4 - audio_sam.duration_seconds)*1000
            silence = AudioSegment.silent(duration=pad_ms) # milliseconds of silence needed
            padded = audio_sam + silence
            
        elif audio_sam.duration_seconds >= 4:
            pad_ms = 0
            audio_sam = audio_sam[0:4000]
            silence = AudioSegment.silent(duration=pad_ms)
            padded = audio_sam + silence        
  
        a = padded.get_array_of_samples()
        b = np.array(a)
        padded_audio.append(b)
        
        """ Compute the 64 dimensional filter banks for the 4secs fixed length audio segments"""
    
        f_banks = fbank(padded_audio, sample_rate=8000, window_length=400, stft_shift=160, number_of_filters=64,
                        stft_size=512,lowest_frequency=0,highest_frequency=None, preemphasis_factor=0.97,
                        window=scipy.signal.windows.hamming, denoise=False)
        fbank_data.append(f_banks)
        float_fbank = np.float32(fbank_data)
        float_fbank = np.squeeze(float_fbank,0)
        
        example['features'] = (float_fbank)
    
        return example
    
    def prep_label(example):
        #label_dict = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/labels_argmax.json')
        #label_hot = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/labels_2.json')
        label_dict3 = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/labels_167.json')
        if example['speaker_id'] in label_dict3.keys():
        
            pos = label_dict3[example['speaker_id']]
            example['label_array'] = np.array(pos)
            
    #    if example['speaker_id'] in label_hot.keys():
    #    
    #        pos = label_hot[example['speaker_id']]
    #        example['label_hot'] = np.array(pos)  
    
        return example
        
    
    def stack(example):
        example['features'] = np.stack(example['features'])
        example['label_array'] = np.stack(example['label_array'])
        #example['label_hot'] = np.stack(example['label_hot'])
        return example
    
    example = example.map(prep_features)
    example = example.map(prep_label)
    if shuffle:
        example = example.shuffle()
   
    example = example.batch(batch_size).map(collate_fn)
    example = example.map(stack)
    example = example.prefetch(buffer_size=8, num_workers=8)
              
            
            
    return example

def prepare_data_2(example, batch_size, shuffle = False):
    
    #db = Fearless()
    ds = example
    #tr_dict = {}
    #tr_dict['features'] = {} 
    #tr_dict['label_array'] = {}
    
    def prep_features(example):
        
        padded_audio = []
        fbank_data = []
        """ Obtain audio segments from the dataset"""
        """ If segments smaller than 4secs, pad with silence. Else, extract 4secs from larger audio segments """
   #     audio_sam = AudioSegment.from_wav(example['audio_path']['observation'])
   #     if audio_sam.duration_seconds < 4:
   #         pad_ms = (4 - audio_sam.duration_seconds)*1000
   #         silence = AudioSegment.silent(duration=pad_ms) # milliseconds of silence needed
   #         padded = audio_sam + silence
   #         
   #     elif audio_sam.duration_seconds >= 4:
   #         pad_ms = 0
   #         audio_sam = audio_sam[0:4000]
   #         silence = AudioSegment.silent(duration=pad_ms)
   #         padded = audio_sam + silence        
  #
   #     a = padded.get_array_of_samples()
   #     b = np.array(a)
   #     padded_audio.append(b)
   #     
        """ Compute the 64 dimensional filter banks for the 4secs fixed length audio segments"""
        audio = pb.io.load_audio(example['audio_path']['observation'])
        f_banks = fbank(audio, sample_rate=8000, window_length=400, stft_shift=160, number_of_filters=64,
                        stft_size=512,lowest_frequency=0,highest_frequency=None, preemphasis_factor=0.97,
                        window=scipy.signal.windows.hamming, denoise=False)
        fbank_data.append(f_banks)
        tens_fbank = torch.FloatTensor(fbank_data)
        #tens_fbank = torch.squeeze(tens_fbank,0)
        #tr_dict['features'] = tens_fbank.to(dev)
        example['features'] = (tens_fbank)
        
        return example
    
    def prep_label(example):
        label_dict = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/labels_argmax.json')
        #label_dict2 = pb.io.load_json(path = 'labels_2.json')
        #print (example)
        if example['speaker_id'] in label_dict.keys():
        
            pos = label_dict[example['speaker_id']]
            example['label_array'] = np.array(pos) 
            
       # if example['speaker_id'] in label_dict2.keys():
       # 
       #     pos = label_dict2[example['speaker_id']]
       #     example['label_hot'] = np.array(pos)
        return example
        
    
    def stack(example):
        example['features'] = np.stack(example['features'])
        example['label_array'] = np.stack(example['label_array'])
       # example['label_hot'] = np.stack(example['label_hot'])
        
        return example
    
    example = example.map(prep_features)
    example = example.map(prep_label)
    if shuffle:
        example = example.shuffle()
   
    example = example.batch(batch_size).map(collate_fn)
    example = example.map(stack)
    example = example.prefetch(buffer_size=8, num_workers=8)
            
            
    return example

def prepare_data_3(example, batch_size, shuffle = False):
    
    #db = Fearless()
    ds = example
    #tr_dict = {}
    #tr_dict['features'] = {} 
    #tr_dict['label_array'] = {}
    
    def prep_features(example):
        
        padded_audio = []
        fbank_data = []
        """ Obtain audio segments from the dataset"""
        """ If segments smaller than 4secs, pad with silence. Else, extract 4secs from larger audio segments """
        audio_sam = AudioSegment.from_wav(example['audio_path']['observation'])
        if audio_sam.duration_seconds < 4:
            pad_ms = (4 - audio_sam.duration_seconds)*1000
            silence = AudioSegment.silent(duration=pad_ms) # milliseconds of silence needed
            padded = audio_sam + silence
            
        elif audio_sam.duration_seconds >= 4:
            pad_ms = 0
            audio_sam = audio_sam[0:4000]
            silence = AudioSegment.silent(duration=pad_ms)
            padded = audio_sam + silence        
  
        a = padded.get_array_of_samples()
        b = np.array(a)
        padded_audio.append(b)
        
        """ Compute the 64 dimensional filter banks for the 4secs fixed length audio segments"""
    
        mfcc = pb.transform.mfcc(padded_audio, sample_rate=8000, window_length=400, stft_shift=160,numcep=13,
                        stft_size=512,lowest_frequency=0,highest_frequency=None, preemphasis_factor=0.97, 
                        window=scipy.signal.windows.hamming)
        fbank_data.append(mfcc)
        tens_fbank = torch.FloatTensor(fbank_data)
        tens_fbank = torch.squeeze(tens_fbank,0)
        #tr_dict['features'] = tens_fbank.to(dev)
        example['features'] = (tens_fbank)
        
        return example
    
    def prep_label(example):
        #label_dict = pb.io.load_json(path = 'labels_argmax.json')
        #label_dict2 = pb.io.load_json(path = 'labels_2.json')
        label_dict3 = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/labels_167.json')
        #print (example)
    #    if example['speaker_id'] in label_dict.keys():
    #    
    #        pos = label_dict[example['speaker_id']]
    #        example['label_array'] = np.array(pos) 
    #        
    #    if example['speaker_id'] in label_dict2.keys():
    #    
    #        pos = label_dict2[example['speaker_id']]
    #        example['label_hot'] = np.array(pos)
    #    return example
    
        if example['speaker_id'] in label_dict3.keys():
        
            pos = label_dict3[example['speaker_id']]
            example['label_array'] = np.array(pos)
        return example
        
    
    def stack(example):
        example['features'] = np.stack(example['features'])
        example['label_array'] = np.stack(example['label_array'])
        #example['label_hot'] = np.stack(example['label_hot'])
        
        return example
    
    example = (example
               .map(prep_features)
               .map(prep_label)
               .batch(batch_size).map(collate_fn)
               .map(stack)
               .prefetch(buffer_size=8, num_workers=8))
            
            
    return example