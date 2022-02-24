import paderbox as pb
import padertorch as pt
import numpy as np 


def segmenter(example,segment_size=32000):
    
    seg_unbatch = []
    ds = example
    for j in ds:
        
        audio = pb.io.load_audio(j['audio_path']['observation'],dtype=np.int16)
        j['array_samples'] = audio
        seg = []
        if len(audio)>=segment_size:
            
            segmenter = pt.data.segment.Segmenter(length=segment_size,include_keys='array_samples',padding=True,
                                      mode='constant')
            seg.append(segmenter(j))
            
        for i in seg:
            for j in i:
                seg_unbatch.append(j)
                
    seg_unbatch = [{k: v for k, v in d.items() if k != 'segment_start'} for d in seg_unbatch]
    seg_unbatch = [{k: v for k, v in d.items() if k != 'segment_stop'} for d in seg_unbatch] 
    
    for o in ds:
        audio = pb.io.load_audio(o['audio_path']['observation'],dtype=np.int16)
        if len(audio)<segment_size:
            p = segment_size-len(audio)            
            audio = np.concatenate((audio,np.zeros(p)))
            o['array_samples'] = audio
            
            seg_unbatch.append(o)
    
    seg_unbatch = sorted(seg_unbatch, key=lambda k: k['audio_path']['observation'])
    
    return(seg_unbatch)


