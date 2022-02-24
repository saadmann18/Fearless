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
import torch.nn.functional as F


class Bottleneck(pt.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(pt.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        identity = x.clone()
  
        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
  
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x

        
        
class ResNet(pt.Model):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    def __init__(self, ResBlock, layer_list, num_classes=218, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 32
        
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=32)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=64, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=128, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256*ResBlock.expansion, 128)   
        self.drop_out = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128,num_classes)
        
    def forward(self, data):
        x = data['features']
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #x = self.avgpool(x)
        x = torch.mean(x,(2,3))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        
        return dict(prediction=x)
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

         
    def review(self, data, outputs):
        
        labels = data['label_array'].to('cuda')
        pred = outputs['prediction'].to('cuda')
        #asm = self.adms_loss(pred, labels)
        #print(asm)
        ce = self.loss_fn(pred,labels)
        review = dict(
            # loss = asm[1].mean(),
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
            summary['scalars']['top_2'] = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels,predictions, k=2)
        summary = super().modify_summary(summary)
        
        return summary
    

        
def ResNet50(num_classes=218, channels=1):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)