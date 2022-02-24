import padertorch as pt
from padertorch.base import Model
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import metrics
import padertorch as pt
from scipy.special import softmax
import tensorflow as tf

class RNN(pt.base.Model):
    ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,stride=2)
        self.avg2d = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,64,kernel_size=5,stride=2)
        self.rnn = nn.RNN(64,1024,num_layers=3,batch_first=True)
     
        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,167)
        
    def forward(self, data):
        x = data['features']
        x = self.conv1(x)
        x = self.conv2(x)     
        x = self.conv2(x)
        x = self.avg2d(x)
        
        x = torch.mean(x,3)
       
        x = torch.transpose(x,1,2)
        
        x,_ = self.rnn(x)
   
        x = torch.mean(x,1)
        x = x.view(x.size(0), -1)        
        x = self.linear1(x)
        x = self.linear2(x)
        
        return dict(prediction=x)
        
    def review(self, data, outputs):
        
        labels = data['label_array']
        pred = outputs['prediction']
        
        ce = self.ce_loss(pred,labels)
        review = dict(
            loss = ce.mean(),
            buffers = dict(
                labels = labels.data.cpu().numpy(),
                #predictions = asm[0].data.cpu().numpy(),
                predictions = pred.data.cpu().numpy(),
            )  
        )
        return review
    
    def modify_summary(self,summary):
        
        if 'labels' in summary['buffers']:
            labels = np.concatenate(summary['buffers'].pop('labels'))
            predictions = np.concatenate(summary['buffers'].pop('predictions'))
            #prediction_soft = softmax(predictions)
            prediction_max = np.argmax(predictions, axis=-1)
            summary['scalars']['accuracy'] = (prediction_max == labels).mean()
            summary['scalars']['top_k'] = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels,predictions, k=5)
        summary = super().modify_summary(summary)
        
        return summary