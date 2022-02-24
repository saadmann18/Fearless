"""
Example call:
python -m SAD.train with database_json=/net/vol/jenkins/jsons/fearless.jsonclear
"""

import os
from pathlib import Path
import padertorch as pt
import padercontrib as pc
import torch
from padertorch.io import get_new_storage_dir
from padertorch.train.optimizer import SGD
from padertorch.train.trainer import Trainer
from sacred import Experiment, commands
from sacred.observers import FileStorageObserver

from .ResNetModel import ResNet
#from .model import Sad
from .data import get_data_preparation

ex = Experiment("Speaker_Activity_Detection")

@ex.config
def config():
    subset = 'stream'
    debug = False
    batch_size = 10
    
    """Interactive trainer configuration parameters"""
    
    trainer = {
        "model": {
            'factory': ResNet #Sad #        
        },
        "storage_dir":get_new_storage_dir(
            'sad', id_naming='time', mkdir=False
        ),
        "optimizer": {
            "factory": SGD,
            "lr" : 0.01
        },
        'summary_trigger': (1000, 'iteration'),
        'checkpoint_trigger': (10_000, 'iteration'),
        'stop_trigger': (200_000, 'iteration'),
        }
    #pt.Trainer.get_config(trainer)
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
    train_stream = data.get_dataset_train(subset=subset)
    validation_stream = data.get_dataset_validation(subset=subset)
    
    """Data preparation"""    
    training_data = get_data_preparation(data, train_stream, batch_size, shuffle=True)
    
    validation_data = get_data_preparation(data, validation_stream, batch_size)
        
    trainer.register_validation_hook(
        validation_data) #, early_stopping_patience=3
    trainer.test_run(training_data, validation_data)
    trainer.register_hook(pt.train.hooks.LRSchedulerHook(
        torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer.optimizer, milestones=[10,20], gamma=0.1)))
    trainer.train(training_data)

