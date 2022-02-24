import padertorch as pt
from padertorch.base import Model
import torch
import torch.nn as nn
import numpy as np

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

class ResNet(pt.Model):
    loss_func = nn.CrossEntropyLoss(reduction='None')
    block = Block
    def __init__(self, layers=[3,4,6,3], image_channels=1, num_classes=167):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        #self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, dropout=0.5, bidirectional=True)

        self.layer1 = self._make_layer(
            self.block, layers[0], intermediate_channels=32, stride=2
        )
        self.layer2 = self._make_layer(
            self.block, layers[1], intermediate_channels=64, stride=2
        )
        self.layer3 = self._make_layer(
            self.block, layers[2], intermediate_channels=128, stride=2
        )
        self.layer4 = self._make_layer(
            self.block, layers[3], intermediate_channels=256, stride=2
        )

        self.drop_out = nn.Dropout(0.5)        
        self.fc1 = nn.Linear(256, 128)               
        self.fc2 = nn.Linear(128,num_classes)
        
    def forward(self, x):
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
        return x
   

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
    
     
    def review():
        pass