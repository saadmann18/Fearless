
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import metrics
import padertorch as pt
from scipy.special import softmax
import tensorflow as tf


class PT_ConvNet(pt.base.Model):
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=1))
            
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        #self.pool1 = nn.AdaptiveMaxPool2d(1)
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
        out = self.layer5(out)
        out = torch.mean(out,(2,3))
        
        out = out.reshape(out.size(0), -1)
          
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)  
    
        return dict(prediction=out)
    
    def review(self, data, outputs):
        
        labels = data['label_array']
        pred = outputs['prediction']
        #print(pred.shape, labels.shape)
        ce = self.loss_fn(pred,labels)
        #print(pred.shape, labels.shape)
        review = dict(
            loss = ce.mean(),
            buffers = dict(
                labels= labels.data.cpu().numpy(),
                predictions = pred.data.cpu().numpy(),
            )  #use buffers instead of images
            #buffers = dict(predictions=pred.data.cpu().numpy(), targets = labels.data.cpu().numpy())
        )
        return review
    
    def modify_summary(self,summary):
        #Precision, Recall computation
        
        if 'labels' in summary['buffers']:
            labels = np.concatenate(summary['buffers'].pop('labels'))
            predictions = np.concatenate(summary['buffers'].pop('predictions'))
            #print(labels.shape, predictions.shape)
            
            #soft = nn.Softmax(dim=0)
            prediction_soft = softmax(predictions)
            #print(prediction_soft)
            prediction_max = np.argmax(prediction_soft, axis=-1)
            
            #label_max = np.argmax(labels,axis=-1)
            
            #print(labels, prediction_max)
            
            
            summary['scalars']['accuracy'] = (prediction_max == labels).mean()
            summary['scalars']['top_5'] = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels,prediction_soft, k=5)
            
            summary['scalars']['precision_score'] = metrics.precision_score(labels,prediction_max, average='macro', zero_division=0)
            summary['scalars']['recall_score'] = metrics.recall_score(labels,prediction_max, average='macro', zero_division=0)
            summary['scalars']['f1_score'] = metrics.f1_score(labels,prediction_max, average='macro', zero_division=0)
            
        summary = super().modify_summary(summary)
              
        
        return summary