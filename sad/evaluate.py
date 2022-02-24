"""
Example call:
python -m SAD.evaluate with exp_dir=/path/to/exp_dir
"""
import numpy as np
import itertools
import torch
import padertorch as pt
import padercontrib as pc
import paderbox as pb

from pathlib import Path
from padertorch import Model
from sacred import Experiment, commands
from sklearn import metrics
from tqdm import tqdm
from sacred.observers import FileStorageObserver
from .eval_data import get_data_preparation
from paderbox.io.new_subdir import get_new_subdir
from paderbox.io import load_json, dump_json
from pprint import pprint
from padercontrib.database.chime5.database import activity_frequency_to_time
from collections import Counter
from padertorch.contrib.jensheit.eval_sad import get_tp_fp_tn_fn
from padertorch.contrib.jensheit.eval_sad import smooth_vad

ex = Experiment('Speaker_Activity_Detection')


@ex.config
def config():
    subset = 'stream'
    debug = False
    exp_dir = ''
    assert len(exp_dir) > 0, 'Set the model path on the command line.'
    storage_dir = str(get_new_subdir(
        Path(exp_dir) / 'eval', id_naming='time', consider_mpi=True
    ))
    database_json = load_json(Path(exp_dir) / 'config.json')["database_json"]
    num_workers = 8
    batch_size = 1
    device = 0
    ckpt_name = 'ckpt_best_loss.pth'
    sample_rate = 8000
    dump_audio = True
    
@ex.automain
def main(
        _run, exp_dir, storage_dir, database_json, ckpt_name,
        num_workers, batch_size, device, subset, dump_audio):
    commands.print_config(_run)

    exp_dir = Path(exp_dir)
    storage_dir = Path(storage_dir)

    config = load_json(exp_dir / 'config.json')

    model = Model.from_storage_dir(
        exp_dir, consider_mpi=True, checkpoint_name=ckpt_name
    )
    model.to(device)
    model.eval()
    data = pc.database.fearless.Fearless()
    validation_stream = data.get_dataset_validation(subset=subset)   
    validation_data = get_data_preparation(data, validation_stream, batch_size)
    
    
    def thresholding(y_pred, threshold=0.5):
        y_pred = y_pred.copy()
        y_pred[y_pred > threshold] = 1.
        y_pred[y_pred < 1] = 0.
        return y_pred

    outputs = []
    to_counter = []
    rre_audio = []
    with torch.no_grad():
        for example in tqdm(validation_data):
            example = model.example_to_device(example, device)
            y = model(example)
            y = y['predictions']
            outputs.append((
                #smooth_vad(y.cpu().detach().numpy(), threshold=0.1, window=3), #try smoothing in TD, w=25
                thresholding(y.cpu().detach().numpy()),
                example['label'].cpu().detach().numpy(),
            ))
            to_counter.append(example['example_id'])

        item_counter = Counter(itertools.chain.from_iterable(to_counter))
        item_counter = np.concatenate([[0], np.cumsum(list(item_counter.values()))])
        y_freq_aud_all, _ = list(zip(*outputs))
        y_freq_aud_all = np.concatenate(y_freq_aud_all)
  
        for j in range(len(item_counter)-1):
            y_freq = y_freq_aud_all[item_counter[j]:item_counter[j+1]]          
            y_tim = []
            for k in range(len(y_freq)):
                y_time =(activity_frequency_to_time(y_freq[k], 
                                                stft_window_length=400, 
                                                stft_shift=160, 
                                                stft_fading=None)).astype(int)
                y_tim.append(y_time[:-80])
            y_dump = list(itertools.chain(*y_tim))

            if len(y_dump) == 14976000:
                y_to_dump = y_dump[:-20706] #removes padded zeros
            elif len(y_dump) == 15392000:
                y_to_dump = y_dump[:-20000] #removes padded zeros
            else:
                y_to_dump = y_dump
                
            if j < 9:
                path = f'/net/vol/saadmann/dump_audio_test_WS/FS02_dev_00{j+1}.wav'
            else:
                path = f'/net/vol/saadmann/dump_audio_test_WS/FS02_dev_0{j+1}.wav'
             
            re_audio = pb.io.dump_audio(y_to_dump,
                path,
                sample_rate=8000,
                dtype=np.int16,
                start=None,
                normalize=True,
                format=None,)
            rre_audio.append(len(pb.io.load_audio(path)))
            
    
    targets = []
    scores = []
    tn = []
    fp = []
    fn = []
    tp = []
    for i in tqdm(range(len(item_counter)-1)):
        activity = data.get_activity(validation_stream[i])
        targets.append(activity[0:validation_stream[i]['num_samples']].astype(int))
        if i<9:
            path = f'/net/vol/saadmann/dump_audio_test_WS/FS02_dev_00{i+1}.wav'
            rre_audio = np.round(pb.io.load_audio(path)).astype(int)
            scores.append(rre_audio)
        else:
            path = f'/net/vol/saadmann/dump_audio_test_WS/FS02_dev_0{i+1}.wav'
            rre_audio = np.round(pb.io.load_audio(path)).astype(int)
            scores.append(rre_audio)
        tp_, fp_, tn_, fn_ = get_tp_fp_tn_fn(targets[i], scores[i])
        tp.append(tp_)
        fp.append(fp_)
        tn.append(tn_)
        fn.append(fn_)
    
    fnr, fpr = (sum(fn)/(sum(tp)+sum(fn))), (sum(fp)/(sum(tn)+sum(fp)))
    dcf = 0.75*fnr + 0.25*fpr
    p = sum(tp)/(sum(tp)+sum(fp))
    r = sum(tp)/(sum(tp)+sum(fn)) 
    
    f1s = 2*(p*r) / (p+r)
    metrics_info = {
        'validation': {
            'mP': p,
            'mR': r,
            'mf1': f1s,
            'mdcf': dcf
        }
    }

    dump_json(
        metrics_info, storage_dir/'metrics_info.json', indent=4, sort_keys=False
    )
    dump_json(
        rre_audio, storage_dir/'dump_audio.json'
    )
    pprint(metrics_info)