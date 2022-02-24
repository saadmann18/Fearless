""" A Convolutonal NN"""

import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import metrics
import padertorch as pt
from scipy.special import softmax
import tensorflow as tf

class Simple_Conv(pt.base.Model):
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')    
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.fc1 = nn.Linear(128, 256) 
        self.fc2 = nn.Linear(256,218)
    def forward(self, data):
       
        x = data['features']
        out = self.layer1(x)
        
        #print(out.shape)
        out = torch.mean(out,(2,3))
        
        #print(out.shape)
        #out = torch.std(out,(2))
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        
        
        #print(out.shape)
        
        out = self.fc1(out)
        
        #out = self.fc2(out)  
        #out = self.fc3(out)
    
        return dict(prediction=out)
    
    def review(self, data, outputs):
        
        labels = data['label_array']
        label_hot = data['label_hot']
        pred = outputs['prediction']
       
        ce = self.loss_fn(pred,labels)
        
        review = dict(
            loss = ce.mean(),
            buffers = dict(
                labels = labels.data.cpu().numpy(),
                predictions = pred.data.cpu().numpy(),
            )  
        )
        return review
    
    def modify_summary(self,summary):
        #Precision, Recall computation
        
        if 'labels' in summary['buffers']:
            labels = np.concatenate(summary['buffers'].pop('labels'))
            predictions = np.concatenate(summary['buffers'].pop('predictions'))       
            
           
            prediction_soft = softmax(predictions)
            prediction_max = np.argmax(prediction_soft, axis=-1)
        
            summary['scalars']['accuracy'] = (prediction_max == labels).mean()
            summary['scalars']['top_k'] = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels,prediction_soft, k=5)
            
            summary['scalars']['precision_score'] = metrics.precision_score(labels,prediction_max, average='macro', zero_division=0)
            summary['scalars']['recall_score'] = metrics.recall_score(labels,prediction_max, average='macro', zero_division=0)
            summary['scalars']['f1_score'] = metrics.f1_score(labels,prediction_max, average='macro', zero_division=0)
            
        summary = super().modify_summary(summary)
              
        
        return summary
