""" Example call : python -m fearless.sid.evaluate with exp_dir=/net/vol/dheerajpr/models/SID/2021-03-16-11-21-09"""
""" Example call : python -m fearless.sid.evaluate with exp_dir=/net/vol/dheerajpr/models/SID/2021-04-07-18-28-48"""
""" Example call : python -m fearless.sid.evaluate with exp_dir=/net/vol/dheerajpr/models/SID/2021-06-20-14-38-46"""


from pathlib import Path
import numpy as np
import torch
import padertorch as pt
import padercontrib as pc
import paderbox as pb
from padertorch import Model
from sacred import Experiment, commands
from sklearn import metrics
from tqdm import tqdm
from paderbox.io.new_subdir import get_new_subdir
from paderbox.io import load_json, dump_json
from pprint import pprint
from fearless.sid.data import prepare_data
from scipy.special import softmax
from scipy.spatial.distance import euclidean as euc
from statistics import mean
import tensorflow as tf
import pb_sed.evaluation.instance_based as sed
import lazy_dataset

ex = Experiment("Speaker_Identification")

@ex.config
def config():
    exp_dir = ''
    assert len(exp_dir) > 0, 'Set the model path on the command line.'
    storage_dir = str(get_new_subdir(
        Path(exp_dir) / 'eval', id_naming='time', consider_mpi=True
    ))
    database_json = load_json(Path(exp_dir) / 'config.json')["database_json"]
    subset = 'segment'
    batch_size = 1
    device = 0
    ckpt_name = 'ckpt_best_loss.pth'
    
    
@ex.automain
def main(_run, exp_dir, storage_dir, database_json, ckpt_name, subset, batch_size, device):
    
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
    
    train_json = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/train_167.json')
    dev_json = pb.io.load_json(path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/dev_167.json')
    
    train_lazy = lazy_dataset.new(train_json['Train_167'])
    dev_lazy = lazy_dataset.new(dev_json['Dev_167'])
    
    #train_segment = data.get_dataset('Train_SID')
    #validation_segment = data.get_dataset('Dev_SID')
    
    train_data = prepare_data(train_lazy,batch_size)
    validation_data = prepare_data(dev_lazy, batch_size)
    
    with torch.no_grad():
        metric = {'Accuracy':[],'Top_5_Accuracy':[]}
        Accuracy = []
        Precision = []
        Recall = []
        F1 = []
        Top_5 = []
        scores = []
        targets = []
        for example in tqdm(validation_data):
            example = model.example_to_device(example, device)
            output = model(example)
            pred = output['prediction'].cpu().detach().numpy()
            prediction_soft = softmax(pred)
            prediction_max = np.argmax(pred, axis=-1)
            target = example['label_array'].cpu().detach().numpy()
            target_hot = example['label_hot'].cpu().detach().numpy()
            #print(target_hot)
            accuracy = (prediction_max == target).mean()            
           
            Accuracy.append(accuracy)  
            top_5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(target,prediction_soft, k=5)
            top_5 = np.array(top_5).mean()
            Top_5.append(top_5)
            
            precision = metrics.precision_score(target,prediction_max, average='macro', zero_division=0)
            Precision.append(precision)
            recall = metrics.recall_score(target,prediction_max, average='macro', zero_division=0)
            Recall.append(recall)
            f1 = metrics.f1_score(target,prediction_max, average='macro', zero_division=0)
            F1.append(f1)
            
            targets.append(target_hot)            
            scores.append(prediction_soft)
            
        targets_con = np.concatenate((targets))
        scores_con = np.concatenate((scores))
        thr,met = sed.get_optimal_thresholds(targets_con,scores_con,metric='f1')       
        #decisions = scores > thr
        #print(decisions)
        #f1, p, r = sed.fscore(targets, decisions, event_wise=False)
        
        auc = metrics.roc_auc_score(targets_con, scores_con, None)
        
            
        metric['Accuracy'] = np.mean(Accuracy)
        metric['Top_5_Accuracy'] = np.mean(Top_5)
        metric['Precision'] = np.mean(Precision)
        metric['Recall'] = np.mean(Recall)
        metric['F1_score'] = np.mean(F1)
        metric['AUC'] = np.mean(auc)
        metric['Thresholds'] = thr
        
    #with torch.no_grad():
            
        
        for example in tqdm(validation_data):
            new_thr = []
            example = model.example_to_device(example, device)
            output = model(example)
            pred = output['prediction'].cpu().detach().numpy()
            #print(pred,pred.shape)
            target = example['label_array'].cpu().detach().numpy()
            #print(target)
            pred_soft = softmax(pred)
            pred_max = np.max(pred_soft) 
            #print(pred_max.shape)
            for i in thr:
                if pred_max < i:
                    new_thr.append(0)
                else:
                    new_thr.append(i)
              #  dis = euc(pred_max,i)
              #  new_thr.append(dis)
              #  if pred_max < i:
              #      new_thr.append(0)
              #  else:
              #      new_thr.append(pred_max - i)          
            
            #print(new_thr)
            pred_min = np.argmax(new_thr,axis=-1) #max and relative dist measure
            print(pred_min)
            accuracy = (pred_min == target).mean()            
            Accuracy.append(accuracy)
                        
        metric['Acc_thr']=np.mean(Accuracy)
                                      
        dump_json(metric, storage_dir/'overall.json', indent=4, sort_keys=False)
        dump_json(metric, path = '/net/home/dheerajpr/my_project/fearless/fearless/sid/eval/result_newce2.json', indent = 4, sort_keys = False)
        
    pprint(metric)