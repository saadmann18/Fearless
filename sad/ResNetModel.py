import padertorch
from padertorch.base import Model
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from einops import rearrange
from padertorch.contrib.jensheit.eval_sad import smooth_vad
class Block(nn.Module):
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
       
        self.relu = nn.ReLU()
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

class ResNet(Model):
    loss_func = nn.BCELoss()
    block = Block
    def __init__(self, layers=[2,2,2,2], image_channels=1, num_classes=199):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, dropout=0.5, bidirectional=True)

        self.layer1 = self._make_layer(
            self.block, layers[0], intermediate_channels=16, stride=1
        )
        self.layer2 = self._make_layer(
            self.block, layers[1], intermediate_channels=32, stride=1
        )
        self.layer3 = self._make_layer(
            self.block, layers[2], intermediate_channels=64, stride=1
        )
        self.layer4 = self._make_layer(
            self.block, layers[3], intermediate_channels=128, stride=1
        )

        self.fc = nn.Linear(128 * 25 , num_classes)
        self.nonlin = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x['features'])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.maxpool(x)
        x = self.avgpool2d(x)
        x = self.layer3(x)
        #x = self.maxpool(x)
        x = self.avgpool2d(x)
        x = self.layer4(x)
        #x = self.maxpool(x)
        x = self.avgpool2d(x)
        x = torch.mean(x, 3)
        x = rearrange(x, 't b c -> t c b')
        x,_ = self.lstm(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.nonlin(x)

        return dict(
            predictions = x
        )
    
    def review(self, data, outputs):
        targets = data['label']
        y = outputs['predictions']
        bce = self.loss_func(y, targets)
        
        review = dict(
            loss=bce,
            scalars=dict(),
            histograms=dict(),
            #buffers=dict(predictions=(y.data.cpu().numpy() > 0.5) * 1, targets=targets.data.cpu().numpy()),
            buffers=dict(predictions=(smooth_vad(y.data.cpu().numpy())), targets=targets.data.cpu().numpy()),
       )
        return review

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
    
    def modify_summary(self, summary):
        # compute precision, recall and fscore
        if f'predictions' in summary['buffers']:
            predictions = np.concatenate(summary['buffers'].pop('predictions'))
            targets = np.concatenate(summary['buffers'].pop('targets'))
            
            if (targets.sum(0) > 1).all():
                summary['scalars']['map'] = metrics.average_precision_score(targets, predictions)
                summary['scalars']['mauc'] = metrics.roc_auc_score(targets, predictions)
                multilabel_c_m = metrics.multilabel_confusion_matrix(targets, predictions)
                res = sum(multilabel_c_m[:])
                tn = res[0][0]
                fp = res[0][1]
                fn = res[1][0]
                tp = res[1][1]
                fnr, fpr = (fn/(tp+fn)), (fp/(tn+fp))
                summary['scalars']['mdcf'] = 0.75*fnr + 0.25*fpr

        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)
        return summary
