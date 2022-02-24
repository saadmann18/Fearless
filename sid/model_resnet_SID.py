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

class Block(pt.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 1 # No expansion between ResNet blocks
        
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        
        self.conv2 = nn.Conv2d(
            intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
       
        self.relu = nn.LeakyReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet_SID(pt.Model):
    
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    block = Block
    def __init__(self, layers=[3,4,6,3], image_channels=1, num_classes=218):
        super(ResNet_SID, self).__init__()
        
        self.in_channels = 32
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        #self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, dropout=0.5, bidirectional=True)

        self.layer1 = self._make_layer(
            self.block, layers[0], intermediate_channels=32, stride=1
        )
        self.layer2 = self._make_layer(
            self.block, layers[1], intermediate_channels=64, stride=1
        )
        self.layer3 = self._make_layer(
            self.block, layers[2], intermediate_channels=128, stride=1
        )
        self.layer4 = self._make_layer(
            self.block, layers[3], intermediate_channels=256, stride=1
        )

        self.drop_out = nn.Dropout(0.5)        
        self.fc1 = nn.Linear(256, 128)               
        self.fc2 = nn.Linear(128,num_classes)
        
    def forward(self, data):
        x = data['features']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
       # print(x.shape)
        x = self.layer2(x)
        x = self.maxpool(x)
        #print(x.shape)
        #x = self.avgpool2d(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        #x = self.avgpool2d(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        #x = self.avgpool2d(x)
        x = torch.mean(x,(2,3))
        
        #x,_ = torch.max(x,3)
        #x,_ = torch.max(x,2)
       
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return dict(prediction=x)
   

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 1:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 1,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(intermediate_channels * 1),
            )

        layers.append(
            self.block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = intermediate_channels * 1

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    
     
    def review(self, data, outputs):
        
        labels = data['label_array']
        pred = outputs['prediction']
        #asm = self.adms_loss(pred, labels) #output from this function.
        ce = self.ce_loss(pred,labels)
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
            summary['scalars']['top_2'] = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels,prediction_soft, k=2)
        summary = super().modify_summary(summary)
        
        return summary