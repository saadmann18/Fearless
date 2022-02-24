import padertorch
from padertorch.base import Model
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics

class Sad(Model):
    loss_func = nn.BCELoss()
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3,stride=2, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=3,stride=2, padding=1),
            nn.ReLU(),    
            )
        self.lin = nn.Sequential(
                nn.Linear(16*25*2,120),
                nn.Linear(120, 1)
                
            )
        self.nonlin = nn.Sigmoid()
    
    def forward(self, data):
        x = data['features']
        x = x.float()
        out = self.seq(x)
        out = out.view(out.size(0), -1)
        out = self.lin(out)
        out = self.nonlin(out)
        return dict(
            predictions = out
        )
    
    def review(self, data, outputs):
        targets = data['label']
        y = outputs['predictions']
        bce = self.loss_func(y, targets)
        
        review = dict(
            loss=bce,
            scalars=dict(),
            histograms=dict(),
            buffers=dict(predictions=y.data.cpu().numpy().round(), targets=targets.data.cpu().numpy()),
       )
        return review
        
    
    def modify_summary(self, summary):
        # compute precision, recall and fscore
        if f'predictions' in summary['buffers']:
            predictions = np.concatenate(summary['buffers'].pop('predictions'))
            targets = np.concatenate(summary['buffers'].pop('targets'))
            
            if (targets.sum(0) > 1).all():
                summary['scalars']['map'] = metrics.average_precision_score(targets, predictions)
                summary['scalars']['mauc'] = metrics.roc_auc_score(targets, predictions)
                tn, fp, fn, tp = metrics.confusion_matrix(targets, predictions).ravel()
                fnr, fpr = (fn/(tp+fn)), (fp/(tn+fp))
                summary['scalars']['mdcf'] = 0.75*fnr + 0.25*fpr
        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)
        return summary