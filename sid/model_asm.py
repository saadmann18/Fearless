""" A Convolutonal NN"""


from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import metrics
import padertorch as pt
from scipy.special import softmax
import tensorflow as tf
from fearless.sid.loss_functions2 import AngularPenaltySMLoss2
from fearless.sid.model_3 import PT_ConvNet
from fearless.sid.model_2 import PT_ConvNet_2
from fearless.sid.model_resnet import ResNet
#from fearless.sid.model_resnet_2 import ResNet

class ConvAngularPen(pt.base.Model):
    
    def __init__(self, num_classes=167):
        super(ConvAngularPen, self).__init__()
        self.convlayers = PT_ConvNet()
        #self.convlayers_small = PT_ConvNet_2()
        self.adms_loss = AngularPenaltySMLoss2(100, num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.resnet = ResNet()
        
    def forward(self, x, labels=None, embed=False):
        #x = self.convlayers(x['features'])
        x = self.resnet(x['features'])
        if embed:
            return x
        return dict(prediction=x)
    
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