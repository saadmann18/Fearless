""" Example Call : python -m fearless.sid.train with database_json=/net/vol/jenkins/jsons/fearless.json """


import os
from pathlib import Path
import padertorch as pt
import padercontrib as pc 
import paderbox as pb
from padertorch.io import get_new_storage_dir
from padertorch.train.optimizer import SGD, Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment, commands
from sacred.observers import FileStorageObserver

from fearless.sid.model_2 import PT_ConvNet_2
#from fearless.sid.model_mfcc import PT_ConvNet
#from fearless.sid.model_asm import PT_ConvNet
from fearless.sid.model_asm import ConvAngularPen
#from fearless.sid.simple_models import Simple_Conv
from fearless.sid.model_resnet_SID import ResNet_SID
#from fearless.sid.model_resnet_SID_50 import ResNet50
from fearless.sid.model_rnn import RNN
from fearless.sid.data import prepare_data
from fearless.sid.data import prepare_data_2
from fearless.sid.data import prepare_data_3
from fearless.sid.data import prepare_data_4
from fearless.sid.segmenter import segmenter
import lazy_dataset
import torch
import tensorflow as tf
import datetime

ex = Experiment("Speaker_Identification")

@ex.config
def config():
    
    subset = "segment"
    debug = False
    batch_size = 16
    
    """Interactive trainer configuration parameters"""
        
    trainer = {
        "model": {
            'factory': ResNet_SID
            #'factory': ConvAngularPen
            
        },
        "storage_dir":get_new_storage_dir(
            'SID', id_naming='time', mkdir=False
        ),
        "optimizer": {
            "factory": SGD,
            "lr": 0.1,
            #"nesterov" : True,
            #"gradient_clipping": 100.0,
            #"momentum": 0.9,
            #"dampening": 0.9
        },
        "summary_trigger": (1000, "iteration"),
        "stop_trigger": (50, "epoch"),
        "checkpoint_trigger": (1000, "iteration")
        }
    pt.Trainer.get_config(trainer)
    if trainer['storage_dir'] is None:
        trainer['storage_dir'] = get_new_storage_dir(experiment_name)

    ex.observers.append(FileStorageObserver(
        Path(trainer['storage_dir']) / 'sacred')
    )
    trainer = Trainer.get_config(trainer)
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))
        

        
@ex.automain
def main(_run, _log, trainer, database_json, subset, batch_size, resume=False):
    commands.print_config(_run)
    trainer = Trainer.from_config(trainer)
    storage_dir = Path(trainer.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    commands.save_config(_run.config, _log, config_filename=str(storage_dir/'config.json'))
        
        
    """Train and validation stream"""
    data = pc.database.fearless.Fearless()
    
    #train_json = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/train_167.json')
    #dev_json = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/dev_167.json')  
    #train_seg = segmenter(train_json['Train_167'])
    #dev_seg = segmenter(dev_json['Dev_167'])
    
    train_segment = data.get_dataset('Train_SID')
    validation_segment = data.get_dataset('Dev_SID')
    train_seg = segmenter(train_segment,segment_size=32000)
    dev_seg = segmenter(validation_segment,segment_size=32000)
    
    
    train_lazy = lazy_dataset.new(train_seg)
    dev_lazy = lazy_dataset.new(dev_seg)
    
 
    
    """Data preparation"""    
    training_data = prepare_data_4(train_lazy, batch_size, shuffle=True)
    
    validation_data = prepare_data_4(dev_lazy, batch_size=1)
    
    
    """Training , Validation and Testing"""
   
    
    trainer.register_validation_hook(validation_data, max_checkpoints=None)
    trainer.test_run(training_data,validation_data)
    trainer.register_hook(pt.train.hooks.LRSchedulerHook(
        torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer.optimizer, milestones=[15,30], gamma=0.1)))
    trainer.train(training_data)


