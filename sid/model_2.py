""" A Convolutonal NN"""
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import metrics
import padertorch as pt
from scipy.special import softmax
import tensorflow as tf


class PT_ConvNet_2(pt.base.Model):
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2))
            
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.drop_out = nn.Dropout(0.5)        
        self.fc1 = nn.Linear(256, 128)                
        self.fc2 = nn.Linear(128,167)
        
        #self.soft = nn.Softmax(1)

    def forward(self, data):
       
        x = data['features']
        out = self.layer1(x)
        out = self.layer2(out)        
        out = self.layer3(out)        
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = torch.mean(out,(2,3))
        
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        
        
        #print(out.shape)
        
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)  
        #out = self.fc3(out)
    
        return dict(prediction=out)
    
    def review(self, data, outputs):
        
        labels = data['label_array']
        pred = outputs['prediction']
        #asm = self.adms_loss(pred, labels) #output from this function.
        ce = self.loss_fn(pred,labels)
        review = dict(
            #loss = asm[1].mean(),
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
            prediction_soft = softmax(predictions)
            prediction_max = np.argmax(prediction_soft, axis=-1)
            summary['scalars']['f1_score'] = metrics.f1_score(prediction_max,labels,average='weighted')
            summary['scalars']['accuracy'] = (prediction_max == labels).mean()
            summary['scalars']['top_k'] = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels,prediction_soft, k=5)
        summary = super().modify_summary(summary)
        
        return summary
 